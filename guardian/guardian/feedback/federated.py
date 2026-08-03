"""Clipped, noised federated averaging over model deltas.

Three properties are wanted at once and they interact:

* **Utility** -- the aggregate should move the model the way the pooled data would.
* **Privacy** -- no single wearer's session should be recoverable from the update.
* **Robustness** -- one device with a broken sensor, or a malicious one, must not be
  able to drag the ensemble.

Per-client L2 clipping buys the last two simultaneously: it bounds one client's
sensitivity, which is the quantity the Gaussian mechanism's noise scale is calibrated to,
*and* it caps how far a bad client can push. The noise multiplier then trades utility
against the privacy budget, and the honest way to present that trade is a curve, which is
what ``retrain.py`` produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

Delta = Mapping[str, np.ndarray]


@dataclass(slots=True)
class ClientUpdate:
    """One device's contribution: a weight delta and how much data produced it."""

    client_id: str
    delta: dict[str, np.ndarray]
    n_samples: int

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.delta = {k: np.asarray(v, dtype=np.float64) for k, v in self.delta.items()}


@dataclass(slots=True)
class AggregationReport:
    n_clients: int
    total_samples: int
    clip_norm: float
    noise_multiplier: float
    clipped_fraction: float
    mean_delta_norm: float
    aggregate_norm: float
    noise_std: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def privacy_note(self) -> str:
        if self.noise_multiplier <= 0.0:
            return "no formal guarantee: noise_multiplier = 0 (utility ceiling only)"
        return (
            f"Gaussian mechanism at sensitivity {self.clip_norm:g} with sigma = "
            f"{self.noise_std:.4g}; per-round (eps, delta) follows from the standard "
            f"analysis at z = {self.noise_multiplier:g}"
        )


def delta_norm(delta: Delta) -> float:
    """Global L2 norm across all parameter blocks, which is the clipping unit."""
    return float(math.sqrt(sum(float(np.sum(np.square(v))) for v in delta.values())))


def clip_delta(delta: Delta, clip_norm: float) -> tuple[dict[str, np.ndarray], bool]:
    n = delta_norm(delta)
    if clip_norm <= 0 or n <= clip_norm or n == 0.0:
        return {k: np.array(v, dtype=np.float64) for k, v in delta.items()}, False
    scale = clip_norm / n
    return {k: np.asarray(v, dtype=np.float64) * scale for k, v in delta.items()}, True


def aggregate(
    updates: list[ClientUpdate],
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.6,
    rng: np.random.Generator | None = None,
    weight_by_samples: bool = True,
) -> tuple[dict[str, np.ndarray], AggregationReport]:
    """FedAvg with per-client clipping and Gaussian noise on the sum.

    Sample weighting is capped implicitly by clipping: a client with ten times the data
    still contributes at most ``clip_norm`` of direction, so a single heavy user cannot
    become the population.
    """
    if not updates:
        raise ValueError("no client updates to aggregate")
    rng = rng or np.random.default_rng(0)

    keys = list(updates[0].delta)
    for u in updates:
        if list(u.delta) != keys:
            raise ValueError(f"client {u.client_id} has mismatched parameter blocks")

    total = sum(u.n_samples for u in updates)
    acc = {k: np.zeros_like(updates[0].delta[k], dtype=np.float64) for k in keys}
    clipped = 0
    norms: list[float] = []
    contributions: dict[str, float] = {}

    for u in updates:
        d, was_clipped = clip_delta(u.delta, clip_norm)
        clipped += int(was_clipped)
        norms.append(delta_norm(u.delta))
        w = (u.n_samples / total) if weight_by_samples else (1.0 / len(updates))
        contributions[u.client_id] = w
        for k in keys:
            acc[k] += w * d[k]

    noise_std = 0.0
    if noise_multiplier > 0.0:
        # Sensitivity of the weighted mean to one clipped client, conservatively bounded
        # by max weight * clip_norm.
        sensitivity = clip_norm * max(contributions.values())
        noise_std = float(noise_multiplier * sensitivity)
        for k in keys:
            acc[k] = acc[k] + rng.normal(0.0, noise_std, size=acc[k].shape)

    return acc, AggregationReport(
        n_clients=len(updates),
        total_samples=total,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        clipped_fraction=clipped / len(updates),
        mean_delta_norm=float(np.mean(norms)),
        aggregate_norm=delta_norm(acc),
        noise_std=noise_std,
        contributions=contributions,
    )
