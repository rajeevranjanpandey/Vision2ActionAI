"""Physics-based scenario simulator used to bootstrap and unit-test the risk head.

Why a simulator at all, in a project whose headline contribution is a real dataset?

* GuardianBench clips are expensive (a sighted safety escort walks every route) and
  hazard *onsets* are rare -- maybe 3 per kilometre. Training a temporal head on that
  alone overfits the handful of positive onsets.
* The quantity we supervise -- "will this object breach the ego cylinder in the next
  2 s" -- has an exact ground truth in simulation and only a human-annotated proxy on
  real video. Pre-training on exact labels and fine-tuning on the noisy real ones is
  strictly better than fitting the proxy from scratch.
* It makes the whole learning stack runnable, deterministic, and CI-testable with no
  data download. Every number in the results table can be regenerated from a seed.

The simulator is deliberately *not* a renderer. It models the part the head actually
sees: noisy metric tracks in the ego frame. Appearance realism would be a distraction;
measurement-noise realism is the thing that transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RiskConfig
from ..types import Track
from .features import HORIZON_S, FrameContext, track_features

DT = 0.1  # 10 Hz fast path


@dataclass(slots=True)
class Sequence:
    """One simulated track: causal features, exact labels, and diagnostics."""

    clip_id: str
    scenario: str
    features: np.ndarray      # (T, D)
    labels: np.ndarray        # (T,) in {0, 1}: breach within HORIZON_S
    ttc_true: np.ndarray      # (T,) exact time-to-breach, +inf when no breach
    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # (T, 4) x,z,vx,vz


# --------------------------------------------------------------------------- scenarios
# Each returns a (T, 4) array of TRUE ego-frame states (x, z, vx, vz) plus a class name
# and an object width. Ego motion is already folded into the relative velocity, which is
# what the tracker produces on-device after ego compensation.


def _integrate(x, z, vx, vz, steps, accel=None, jitter=0.0, rng=None):
    out = np.zeros((steps, 4), dtype=np.float64)
    for i in range(steps):
        out[i] = (x, z, vx, vz)
        ax, az = accel(i * DT, x, z, vx, vz) if accel else (0.0, 0.0)
        if jitter and rng is not None:
            ax += rng.normal(0.0, jitter)
            az += rng.normal(0.0, jitter)
        vx += ax * DT
        vz += az * DT
        x += vx * DT
        z += vz * DT
    return out


def _crossing_cyclist(rng: np.random.Generator, steps: int):
    """Lateral crosser on a collision course. Constant velocity gets this one right."""
    side = rng.choice([-1.0, 1.0])
    z = rng.uniform(6.0, 11.0)
    speed = rng.uniform(3.0, 5.5)
    x = side * rng.uniform(3.5, 6.0)
    # Aim at where the ego will be, with a small miss so not every clip is a hit.
    t_hit = z / max(rng.uniform(1.0, 1.6), 1e-3)
    vx = (-x + rng.normal(0.0, 0.6)) / max(t_hit, 1e-3)
    vz = -z / max(t_hit, 1e-3)
    scale = speed / max(np.hypot(vx, vz), 1e-6)
    return _integrate(x, z, vx * scale, vz * scale, steps), "bicycle", 0.6


def _yielding_pedestrian(rng: np.random.Generator, steps: int):
    """Starts on a collision course, then decelerates and stops short of the corridor.

    This is the scenario the constant-velocity kernel *must* get wrong: for the first
    second it is indistinguishable from a crosser. The head's only chance is the
    deceleration signature inside the 0.8 s window.
    """
    side = rng.choice([-1.0, 1.0])
    x = side * rng.uniform(2.5, 4.5)
    z = rng.uniform(4.0, 8.0)
    vx = -np.sign(x) * rng.uniform(1.1, 1.7)
    vz = -rng.uniform(0.4, 1.0)
    brake_at = rng.uniform(0.6, 1.4)

    def accel(t, *_):
        return (-vx / brake_at, -vz / brake_at) if t >= brake_at else (0.0, 0.0)

    states = _integrate(x, z, vx, vz, steps, accel=accel, jitter=0.05, rng=rng)
    # Hard stop once nearly at rest, so it does not creep backwards through the origin.
    speeds = np.hypot(states[:, 2], states[:, 3])
    stopped = speeds < 0.12
    if stopped.any():
        i = int(np.argmax(stopped))
        states[i:, 0] = states[i, 0]
        states[i:, 1] = states[i, 1]
        states[i:, 2:] = 0.0
    return states, "person", 0.5


def _static_obstacle(rng: np.random.Generator, steps: int):
    """Bollard/A-board in the corridor; only ego motion closes the gap."""
    x = rng.normal(0.0, 0.35)
    z = rng.uniform(5.0, 9.0)
    ego_speed = rng.uniform(1.1, 1.5)
    return _integrate(x, z, 0.0, -ego_speed, steps), "bollard", rng.uniform(0.2, 0.9)


def _parallel_walker(rng: np.random.Generator, steps: int):
    """Someone walking the same way, offset laterally. The classic false-alarm source."""
    x = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 1.8)
    z = rng.uniform(2.0, 6.0)
    return _integrate(x, z, rng.normal(0.0, 0.08), rng.normal(-0.15, 0.25), steps,
                      jitter=0.06, rng=rng), "person", 0.5


def _near_miss_vehicle(rng: np.random.Generator, steps: int):
    """Fast vehicle that passes wide. Must not fire: it is the alarm-fatigue scenario."""
    side = rng.choice([-1.0, 1.0])
    x = side * rng.uniform(5.0, 8.0)
    z = rng.uniform(8.0, 12.0)
    vz = -rng.uniform(4.0, 8.0)
    vx = -side * rng.uniform(0.0, 0.25)
    return _integrate(x, z, vx, vz, steps, jitter=0.05, rng=rng), "car", 1.8


def _cut_in_scooter(rng: np.random.Generator, steps: int):
    """Overtakes on one side, then curves into the corridor. Needs a curvature cue."""
    side = rng.choice([-1.0, 1.0])
    x = side * rng.uniform(1.8, 3.0)
    z = rng.uniform(7.0, 11.0)
    vz = -rng.uniform(2.2, 3.6)
    turn_at = rng.uniform(0.5, 1.2)

    def accel(t, *_):
        return (-np.sign(x) * 1.4, 0.0) if t >= turn_at else (0.0, 0.0)

    return _integrate(x, z, 0.0, vz, steps, accel=accel, jitter=0.05, rng=rng), "scooter", 0.7


SCENARIOS = {
    "crossing_cyclist": _crossing_cyclist,
    "yielding_pedestrian": _yielding_pedestrian,
    "static_obstacle": _static_obstacle,
    "parallel_walker": _parallel_walker,
    "near_miss_vehicle": _near_miss_vehicle,
    "cut_in_scooter": _cut_in_scooter,
}

# Sampling weights: negatives dominate reality, and the head has to learn that. A
# balanced simulator produces a head that fires at everything on a real pavement.
SCENARIO_WEIGHTS = {
    "crossing_cyclist": 0.18,
    "yielding_pedestrian": 0.20,
    "static_obstacle": 0.14,
    "parallel_walker": 0.24,
    "near_miss_vehicle": 0.14,
    "cut_in_scooter": 0.10,
}


# ------------------------------------------------------------------------- labelling


def exact_time_to_breach(states: np.ndarray, radius: float) -> np.ndarray:
    """For each timestep, the exact time until the true trajectory enters the cylinder.

    Computed from the *future of the simulated ground truth*, not from a model. This is
    the label the head regresses its classification boundary against.
    """
    n = states.shape[0]
    dist = np.hypot(states[:, 0], states[:, 1])
    inside = dist <= radius
    ttb = np.full(n, np.inf)
    next_breach = np.inf
    for i in range(n - 1, -1, -1):
        if inside[i]:
            next_breach = 0.0
        ttb[i] = next_breach
        next_breach = next_breach + DT if np.isfinite(next_breach) else np.inf
    return ttb


# ----------------------------------------------------------------------- observation


def _observe(state: np.ndarray, rng: np.random.Generator, cfg: RiskConfig,
             class_name: str, width: float, depth_conf: float) -> tuple[Track, FrameContext]:
    """Corrupt a true state into what the on-device tracker would actually report.

    Depth error on a monocular system is multiplicative in range, not additive -- a 4%
    scale error is 8 cm at 2 m and 40 cm at 10 m. Getting this wrong in simulation is
    the single fastest way to build a head that collapses on real data.
    """
    x, z, vx, vz = state
    z_obs = z * (1.0 + rng.normal(0.0, 0.04)) + rng.normal(0.0, 0.05)
    x_obs = x + rng.normal(0.0, 0.03 * max(z, 1.0))
    v_sigma = 0.12 + 0.02 * max(z, 0.0)
    track = Track(
        track_id=0,
        x_m=float(x_obs),
        z_m=float(max(z_obs, 0.05)),
        vx_mps=float(vx + rng.normal(0.0, v_sigma)),
        vz_mps=float(vz + rng.normal(0.0, v_sigma)),
        width_m=float(width),
        class_name=class_name,
        hits=5,
        covariance_trace=float(v_sigma**2 * 4),
    )
    ctx = FrameContext(
        depth_confidence=float(np.clip(depth_conf + rng.normal(0.0, 0.03), 0.0, 1.0)),
        detector_score=float(np.clip(rng.normal(0.82, 0.09), 0.3, 0.99)),
    )
    return track, ctx


def simulate_clip(clip_id: str, scenario: str, cfg: RiskConfig,
                  rng: np.random.Generator, steps: int = 45) -> Sequence:
    states, class_name, width = SCENARIOS[scenario](rng, steps)
    radius = cfg.ego_radius_m + 0.5 * width
    ttb = exact_time_to_breach(states, radius)
    labels = (ttb <= HORIZON_S).astype(np.float32)

    depth_conf = float(np.clip(rng.normal(0.85, 0.08), 0.4, 0.99))
    feats = np.zeros((steps, len(track_features(
        Track(0, 0.0, 1.0, 0.0, 0.0, 0.5, "person"), cfg))), dtype=np.float32)
    for i in range(steps):
        track, ctx = _observe(states[i], rng, cfg, class_name, width, depth_conf)
        feats[i] = track_features(track, cfg, ctx)

    return Sequence(
        clip_id=clip_id,
        scenario=scenario,
        features=feats,
        labels=labels,
        ttc_true=ttb,
        states=states,
    )


def generate_dataset(n_clips: int = 600, cfg: RiskConfig | None = None,
                     seed: int = 0, steps: int = 45) -> list[Sequence]:
    """Deterministic dataset generation. Same seed, same bytes, forever."""
    cfg = cfg or RiskConfig()
    rng = np.random.default_rng(seed)
    names = list(SCENARIO_WEIGHTS)
    probs = np.array([SCENARIO_WEIGHTS[n] for n in names], dtype=np.float64)
    probs /= probs.sum()

    out: list[Sequence] = []
    for i in range(n_clips):
        scenario = str(rng.choice(names, p=probs))
        out.append(simulate_clip(f"sim-{i:04d}", scenario, cfg, rng, steps))
    return out
