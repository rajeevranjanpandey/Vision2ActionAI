"""Trajectory rollout for visualisation, logging, and the evaluation harness.

The fast path uses the closed-form solutions in ``ttc.py``. This module produces the
explicit rollout used by the recorder overlay and by GuardianBench scoring, where we
need the full predicted path rather than a single scalar.
"""

from __future__ import annotations

import numpy as np

from ..config import RiskConfig
from ..types import Track


def rollout(track: Track, cfg: RiskConfig) -> np.ndarray:
    """Constant-velocity rollout. Returns an (N, 3) array of ``(t, x, z)``."""
    steps = max(1, int(cfg.horizon_s / max(cfg.forecast_dt, 1e-3)))
    ts = np.arange(1, steps + 1, dtype=np.float64) * cfg.forecast_dt
    xs = track.x_m + track.vx_mps * ts
    zs = track.z_m + track.vz_mps * ts
    return np.stack([ts, xs, zs], axis=1)


def rollout_with_uncertainty(
    track: Track,
    cfg: RiskConfig,
    velocity_sigma: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Rollout plus a 1-sigma radius that grows linearly with the horizon.

    Velocity uncertainty integrates into position uncertainty as sigma_v * t, which is
    why long-horizon predictions must widen the corridor rather than assert a point.
    """
    path = rollout(track, cfg)
    ts = path[:, 0]
    sigma = velocity_sigma * ts + 0.05 * np.sqrt(max(track.covariance_trace, 0.0))
    return path, sigma


def min_predicted_distance(track: Track, cfg: RiskConfig) -> tuple[float, float]:
    """Numeric closest approach over the rollout. Cross-checks the analytic solution."""
    path = rollout(track, cfg)
    dists = np.hypot(path[:, 1], path[:, 2])
    i = int(np.argmin(dists))
    return float(dists[i]), float(path[i, 0])


def occupancy_grid(
    tracks: list[Track],
    cfg: RiskConfig,
    horizon_s: float,
    resolution_m: float = 0.25,
) -> np.ndarray:
    """Predicted occupancy of the ground plane at ``horizon_s`` ahead.

    Used by the recorder overlay and as an interpretable artefact in the paper: it shows
    reviewers *what the model thinks the near future looks like*, not just its output.
    """
    extent = cfg.cone_range_m
    size_z = int(extent / resolution_m)
    size_x = int(2 * extent / resolution_m)
    grid = np.zeros((size_z, size_x), dtype=np.float32)

    for track in tracks:
        x = track.x_m + track.vx_mps * horizon_s
        z = track.z_m + track.vz_mps * horizon_s
        if not (0.0 <= z < extent and -extent <= x < extent):
            continue
        iz = int(z / resolution_m)
        ix = int((x + extent) / resolution_m)
        radius = max(1, int((0.5 * track.width_m + 0.35 * horizon_s) / resolution_m))
        z0, z1 = max(0, iz - radius), min(size_z, iz + radius + 1)
        x0, x1 = max(0, ix - radius), min(size_x, ix + radius + 1)
        grid[z0:z1, x0:x1] = np.maximum(grid[z0:z1, x0:x1], 1.0)

    return grid
