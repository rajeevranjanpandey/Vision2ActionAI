"""Runtime for the learned risk head.

This is the only learned component allowed to influence a safety decision, and it is
allowed to do so in exactly one direction: it may raise urgency and it may suppress an
alert *only* within the band the geometric kernel has marked as ambiguous. It can never
veto an urgent geometric warning. That asymmetry is the safety argument -- a
mispredicting head degrades the system to the constant-velocity baseline rather than to
silence.

Runtime cost: 5 members x (112->64->32->1) = ~40 us per track on the Orin CPU, so the
head never competes with the GPU stages for the 100 ms budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RiskConfig
from ..types import Hazard, Track
from .calibration import apply_temperature


@dataclass(slots=True)
class RiskPrediction:
    """Head output for one track."""

    probability: float          # calibrated P(breach within the label horizon)
    epistemic_std: float        # ensemble disagreement
    fired: bool                 # probability >= conformal threshold
    trusted: bool               # disagreement below the abstention band

    @property
    def lower_confidence_bound(self) -> float:
        """Conservative estimate used when *suppressing*. Suppression must be pessimistic
        about its own certainty, so it reasons with mean + spread, not the mean."""
        return float(np.clip(self.probability - 2.0 * self.epistemic_std, 0.0, 1.0))

    @property
    def upper_confidence_bound(self) -> float:
        return float(np.clip(self.probability + 2.0 * self.epistemic_std, 0.0, 1.0))


class LearnedRiskHead:
    """Deep-ensemble MLP with temperature scaling and a conformal firing threshold."""

    def __init__(
        self,
        weights: list[np.ndarray] | None = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        abstain_std: float = 0.18,
    ) -> None:
        self._members = weights or []
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.abstain_std = float(abstain_std)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_npz(cls, path: str | Path, abstain_std: float = 0.18) -> "LearnedRiskHead":
        from ..train.trainer import load_ensemble  # local import keeps import cost off the fast path

        models, meta = load_ensemble(str(path))
        head = cls(
            temperature=float(meta.get("temperature", np.array(1.0))),
            threshold=float(meta.get("threshold", np.array(0.5))),
            abstain_std=abstain_std,
        )
        head._models = models
        return head

    @property
    def available(self) -> bool:
        return bool(getattr(self, "_models", None))

    # --------------------------------------------------------------- inference

    def predict(self, window_flat: np.ndarray) -> RiskPrediction:
        """Score one flattened ``(WINDOW * D,)`` feature window."""
        models = getattr(self, "_models", None)
        if not models:
            return RiskPrediction(0.0, 1.0, fired=False, trusted=False)

        X = np.asarray(window_flat, dtype=np.float64).reshape(1, -1)
        per_member = np.array([float(m.predict(X)[0]) for m in models])
        mean = float(per_member.mean())
        std = float(per_member.std())
        calibrated = float(apply_temperature(np.array([mean]), self.temperature)[0])
        return RiskPrediction(
            probability=calibrated,
            epistemic_std=std,
            fired=calibrated >= self.threshold,
            trusted=std <= self.abstain_std,
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        models = getattr(self, "_models", None)
        if not models:
            return np.zeros(len(X))
        per_member = np.stack([m.predict(np.asarray(X, dtype=np.float64)) for m in models])
        return apply_temperature(per_member.mean(axis=0), self.temperature)


# --------------------------------------------------------------------------- fusion


@dataclass(slots=True)
class FusedRisk:
    severity: float
    source: str      # "geometry", "head-escalate", "head-suppress", "head-abstain"
    probability: float
    epistemic_std: float


def fuse(
    hazard: Hazard,
    prediction: RiskPrediction,
    cfg: RiskConfig,
    suppress_below: float = 0.15,
    escalate_above: float = 0.75,
) -> FusedRisk:
    """Combine the geometric severity with the learned probability, conservatively.

    Three regimes, and the boundaries between them are the safety case:

    * **Urgent geometry wins outright.** If TTC is already inside ``ttc_urgent_s``, the
      head is ignored. A learned model does not get to talk the device out of a
      warning when the physics says contact is 1.2 s away.
    * **Escalation is cheap, so it is permissive.** A confident head can push a WARN
      toward URGENT; the cost is one early alert.
    * **Suppression is expensive, so it is strict.** It requires the head to be
      confident, in agreement with itself (``trusted``), and reasoning about a
      non-imminent hazard -- and it uses the *upper* confidence bound, so ensemble
      disagreement blocks suppression automatically.
    """
    base = hazard.severity

    if hazard.ttc_s <= cfg.ttc_urgent_s:
        return FusedRisk(base, "geometry", prediction.probability, prediction.epistemic_std)

    if not prediction.trusted:
        return FusedRisk(base, "head-abstain", prediction.probability, prediction.epistemic_std)

    if prediction.probability >= escalate_above:
        boosted = float(np.clip(max(base, 0.5 * base + 0.5 * prediction.probability), 0.0, 1.0))
        return FusedRisk(boosted, "head-escalate", prediction.probability,
                         prediction.epistemic_std)

    if prediction.upper_confidence_bound <= suppress_below:
        damped = float(np.clip(base * 0.35, 0.0, 1.0))
        return FusedRisk(damped, "head-suppress", prediction.probability,
                         prediction.epistemic_std)

    return FusedRisk(base, "geometry", prediction.probability, prediction.epistemic_std)


def geometric_baseline_severity(track: Track, cfg: RiskConfig) -> float:
    """Ablation hook: severity with the head removed entirely."""
    from .ttc import closest_approach, severity_of, time_to_collision

    ttc = time_to_collision(track, cfg)
    dist, _ = closest_approach(track, cfg)
    return severity_of(ttc, dist, cfg)
