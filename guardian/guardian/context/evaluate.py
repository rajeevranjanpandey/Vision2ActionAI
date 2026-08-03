"""Per-context alert scoring.

The roadmap's metric for item 4 is FA/km *broken out by context*, with one explicit
gate: the transit-platform case must be at least as conservative as today. An improved
average is not evidence -- it is the shape of result that hides a regression on the one
context where a regression is a fall onto a track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RiskConfig
from .modes import Mode, apply_profile, profile_for

# Per-context encounter statistics used by the simulator: how often an encounter is a
# genuine hazard, the closing-speed distribution, and lateral spread. These stand in for
# GuardianBench splits that do not exist yet, and the numbers on the site say so.
CONTEXT_STATS: dict[Mode, dict[str, float]] = {
    Mode.SIDEWALK: {"per_km": 46.0, "hazard_frac": 0.13, "v_mu": 1.6, "v_sd": 0.7,
                    "x_sd": 1.1},
    Mode.CROSSING: {"per_km": 62.0, "hazard_frac": 0.22, "v_mu": 6.5, "v_sd": 3.0,
                    "x_sd": 1.6},
    Mode.TRANSIT_PLATFORM: {"per_km": 78.0, "hazard_frac": 0.17, "v_mu": 1.3, "v_sd": 0.9,
                            "x_sd": 0.9},
}


@dataclass(slots=True)
class ContextArmResult:
    mode: Mode
    recall: float
    false_alarms_per_km: float
    n_encounters: int
    n_hazards: int


def _simulate(mode: Mode, cfg: RiskConfig, n_km: float, rng: np.random.Generator,
              hard_floor_m: float | None) -> ContextArmResult:
    stats = CONTEXT_STATS[mode]
    n = int(stats["per_km"] * n_km)
    hazard = rng.random(n) < stats["hazard_frac"]
    v = np.abs(rng.normal(stats["v_mu"], stats["v_sd"], n)) + 0.05
    # Hazards close on the corridor; non-hazards pass wide or slowly.
    x = np.where(
        hazard,
        rng.normal(0.0, 0.35, n),
        rng.normal(0.0, stats["x_sd"], n),
    )
    z = rng.uniform(2.0, 11.0, n)
    ttc = z / v
    half = cfg.cone_half_width_m + cfg.cone_widen_per_m * z

    inside = np.abs(x) <= half
    closing = v >= cfg.min_closing_speed_mps
    fires = inside & closing & (ttc <= cfg.ttc_warn_s)

    if hard_floor_m is not None:
        # Platform edge: anything inside the floor fires on geometry alone.
        edge_dist = np.abs(x) + rng.uniform(0.0, 0.6, n)
        fires = fires | (edge_dist <= hard_floor_m)

    n_haz = int(hazard.sum())
    recall = float(fires[hazard].mean()) if n_haz else 1.0
    fa = float((fires & ~hazard).sum() / n_km)
    return ContextArmResult(mode, recall, fa, n, n_haz)


def evaluate_contexts(
    base: RiskConfig | None = None, n_km: float = 40.0, seed: int = 0
) -> dict[str, object]:
    """Score every context under the sidewalk-only profile and its own profile."""
    base = base or RiskConfig()
    rows: list[dict[str, object]] = []
    for mode in Mode:
        rng_a = np.random.default_rng(seed + hash(mode.value) % 1000)
        rng_b = np.random.default_rng(seed + hash(mode.value) % 1000)  # same draws
        flat = _simulate(mode, base, n_km, rng_a, hard_floor_m=None)
        prof = profile_for(mode)
        aware = _simulate(mode, apply_profile(base, prof), n_km, rng_b, prof.hard_floor_m)
        rows.append(
            {
                "mode": mode.value,
                "flatRecall": round(flat.recall, 4),
                "awareRecall": round(aware.recall, 4),
                "flatFaPerKm": round(flat.false_alarms_per_km, 3),
                "awareFaPerKm": round(aware.false_alarms_per_km, 3),
                "encounters": flat.n_encounters,
                "hazards": flat.n_hazards,
                "hardFloorM": prof.hard_floor_m,
                "rationale": prof.rationale,
            }
        )

    platform = next(r for r in rows if r["mode"] == Mode.TRANSIT_PLATFORM.value)
    return {
        "rows": rows,
        "nKm": n_km,
        # The gate, evaluated rather than asserted: the platform arm must not lose recall.
        "platformNotRegressed": bool(platform["awareRecall"] >= platform["flatRecall"]),
        "meanFaFlat": round(float(np.mean([r["flatFaPerKm"] for r in rows])), 3),
        "meanFaAware": round(float(np.mean([r["awareFaPerKm"] for r in rows])), 3),
    }
