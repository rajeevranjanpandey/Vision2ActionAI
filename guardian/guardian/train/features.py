"""Feature extraction for the learned risk head.

Design rules, in order of importance:

1. **Every feature is causally available at 10 Hz on-device.** No future frames, no
   whole-clip normalisation statistics. A feature that cannot be computed in the fast
   path is not a feature, it is a leak.
2. **Features are ego-frame and metric.** The head must transfer across camera mounts
   and body heights, so nothing is expressed in pixels.
3. **Features are scale-normalised at construction time with fixed constants** rather
   than dataset statistics. Dataset-fit normalisers are the classic silent train/serve
   skew bug: the constants below ship in the weights file and are asserted at load.

The head consumes a window of ``WINDOW`` timesteps (0.8 s at 10 Hz). That length is not
arbitrary: it is the shortest window that contains enough of an acceleration signature
to separate "pedestrian who will yield" from "cyclist who will not", which is the
decision the geometric constant-velocity kernel provably cannot make.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RiskConfig
from ..risk.ttc import closest_approach, closing_speed, cone_half_width_at, time_to_collision
from ..types import Track

WINDOW = 8          # timesteps fed to the head (0.8 s at 10 Hz)
HORIZON_S = 2.0     # label horizon: "will this breach the ego cylinder within 2 s?"

FEATURE_NAMES: tuple[str, ...] = (
    "z_norm",           # forward range / cone_range
    "x_norm",           # lateral offset / cone_range (signed)
    "range_norm",
    "vz_norm",          # closing component of velocity / 3 m/s
    "vx_norm",
    "closing_norm",     # range rate / 3 m/s
    "inv_ttc",          # 1 / (ttc + 0.5); bounded, and linear in urgency
    "cpa_dist_norm",    # closest point of approach distance / cone_range
    "cpa_time_norm",    # time of closest approach / horizon
    "cone_margin",      # (half_width - |x|) / half_width; >0 means inside corridor
    "width_norm",       # object extent / 2 m
    "det_score",        # detector confidence
    "depth_conf",       # ground-plane inlier ratio backing the metric scale
    "class_dynamic",    # prior: can this class self-propel toward the user?
)

D = len(FEATURE_NAMES)

# Classes that can close distance on their own. The prior matters because a parked car
# and a moving car look identical for the first two frames of a track.
DYNAMIC_CLASSES = {
    "person": 0.7,
    "bicycle": 1.0,
    "motorcycle": 1.0,
    "car": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "dog": 0.8,
    "scooter": 1.0,
}

_V_SCALE = 3.0      # m/s, roughly a fast cyclist relative to a walker
_W_SCALE = 2.0      # m


@dataclass(slots=True)
class FrameContext:
    """Per-frame scalars that are not properties of the track itself."""

    depth_confidence: float = 1.0
    detector_score: float = 1.0


def track_features(track: Track, cfg: RiskConfig, ctx: FrameContext | None = None) -> np.ndarray:
    """Map one track at one instant to a ``(D,)`` float32 feature vector."""
    ctx = ctx or FrameContext()
    rng = max(cfg.cone_range_m, 1e-6)

    ttc = time_to_collision(track, cfg)
    cpa_dist, cpa_time = closest_approach(track, cfg)
    half = cone_half_width_at(track.z_m, cfg) + 0.5 * track.width_m

    inv_ttc = 0.0 if not np.isfinite(ttc) else 1.0 / (ttc + 0.5)

    feats = np.array(
        [
            track.z_m / rng,
            track.x_m / rng,
            track.range_m / rng,
            track.vz_mps / _V_SCALE,
            track.vx_mps / _V_SCALE,
            closing_speed(track) / _V_SCALE,
            inv_ttc,
            cpa_dist / rng,
            cpa_time / max(cfg.horizon_s, 1e-6),
            (half - abs(track.x_m)) / max(half, 1e-6),
            track.width_m / _W_SCALE,
            float(ctx.detector_score),
            float(ctx.depth_confidence),
            DYNAMIC_CLASSES.get(track.class_name, 0.2),
        ],
        dtype=np.float32,
    )
    # Hard clip rather than let a depth blow-up produce a 400-sigma activation. A NaN
    # here would silently poison the head; a clipped feature merely saturates it.
    return np.nan_to_num(np.clip(feats, -4.0, 4.0), nan=0.0, posinf=4.0, neginf=-4.0)


class FeatureBuffer:
    """Fixed-length ring buffer of feature vectors for one track id.

    Pre-roll padding repeats the first observation instead of zero-padding: a zero
    vector is a *valid* feature state (object at the origin, stationary), so zero-pad
    would teach the head that new tracks are at your feet.
    """

    __slots__ = ("_buf", "_n")

    def __init__(self, window: int = WINDOW) -> None:
        self._buf = np.zeros((window, D), dtype=np.float32)
        self._n = 0

    def push(self, feats: np.ndarray) -> None:
        if self._n == 0:
            self._buf[:] = feats
        else:
            self._buf[:-1] = self._buf[1:]
            self._buf[-1] = feats
        self._n += 1

    @property
    def ready(self) -> bool:
        """Two real observations is enough to have a velocity estimate worth reading."""
        return self._n >= 2

    def stack(self) -> np.ndarray:
        return self._buf.copy()

    def flat(self) -> np.ndarray:
        return self._buf.reshape(-1)


def window_of(sequence: np.ndarray, t: int, window: int = WINDOW) -> np.ndarray:
    """Causal window ending at index ``t`` from a ``(T, D)`` sequence, edge-padded."""
    start = t - window + 1
    if start >= 0:
        return sequence[start : t + 1]
    pad = np.repeat(sequence[:1], -start, axis=0)
    return np.concatenate([pad, sequence[: t + 1]], axis=0)


def build_windows(
    sequences: list[np.ndarray],
    labels: list[np.ndarray],
    window: int = WINDOW,
    groups: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of ``(T, D)`` sequences into supervised windows.

    Returns ``(X, y, group_ids)`` where ``X`` is ``(N, window * D)``. Group ids carry the
    clip identity so the splitter can hold out *clips*, never frames -- adjacent frames
    of one clip are ~0.1 s apart and are effectively duplicates, so a random frame split
    reports a fantasy AUC.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    gs: list[str] = []
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        gid = groups[i] if groups else str(i)
        for t in range(seq.shape[0]):
            xs.append(window_of(seq, t, window).reshape(-1))
            ys.append(float(lab[t]))
            gs.append(gid)
    if not xs:
        return (
            np.zeros((0, window * D), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
        )
    return (
        np.stack(xs).astype(np.float32),
        np.array(ys, dtype=np.float32),
        np.array(gs, dtype=object),
    )
