"""Ablation harness on simulated clips.

The real GuardianBench evaluator lives in ``bench/evaluate.py`` and needs video. This
one runs the identical decision logic over simulated tracks so that every ablation in
the paper's Table 2 is reproducible from a seed in about ten seconds, with exact ground
truth for the quantity being measured (time until the true trajectory breaches the ego
cylinder).

Scored per clip, the way a user experiences it:

* **lead time** -- seconds between the first alert and the true breach. Negative means
  the alert came after contact and is counted as a miss, not as a late success.
* **miss** -- a clip that truly breached and never fired.
* **false alarm** -- a clip that never breached and fired anyway. Reported per clip and
  per simulated kilometre, since alarm fatigue is a rate, not a count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RiskConfig
from ..risk.learned import LearnedRiskHead
from ..train.features import HORIZON_S, WINDOW, window_of
from ..train.simulate import DT, Sequence

WALK_SPEED_MPS = 1.3  # for converting simulated seconds into simulated kilometres


@dataclass
class ArmResult:
    """One ablation arm: a decision rule scored across all clips."""

    name: str
    lead_times_s: list[float] = field(default_factory=list)
    misses: int = 0
    false_alarms: int = 0
    hazard_clips: int = 0
    safe_clips: int = 0
    distance_km: float = 0.0
    late_alerts: int = 0

    @property
    def recall(self) -> float:
        return len(self.lead_times_s) / max(self.hazard_clips, 1)

    @property
    def median_lead_time_s(self) -> float:
        return float(np.median(self.lead_times_s)) if self.lead_times_s else 0.0

    @property
    def p10_lead_time_s(self) -> float:
        return float(np.percentile(self.lead_times_s, 10)) if self.lead_times_s else 0.0

    @property
    def false_alarms_per_km(self) -> float:
        return self.false_alarms / max(self.distance_km, 1e-6)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "recall": round(self.recall, 4),
            "median_lead_time_s": round(self.median_lead_time_s, 3),
            "p10_lead_time_s": round(self.p10_lead_time_s, 3),
            "false_alarms_per_km": round(self.false_alarms_per_km, 3),
            "misses": self.misses,
            "hazard_clips": self.hazard_clips,
            "safe_clips": self.safe_clips,
            "late_alerts": self.late_alerts,
        }


def _first_true_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def score_arm(
    name: str,
    sequences: list[Sequence],
    fire: "callable",
    cfg: RiskConfig | None = None,
) -> ArmResult:
    """Run one decision rule over every clip.

    ``fire(sequence, t) -> bool`` sees only frames up to ``t``; the harness enforces
    causality by construction because it never hands the rule the future.
    """
    cfg = cfg or RiskConfig()
    result = ArmResult(name=name)

    for seq in sequences:
        steps = seq.features.shape[0]
        result.distance_km += steps * DT * WALK_SPEED_MPS / 1000.0

        breach_idx = _first_true_index(np.isfinite(seq.ttc_true) & (seq.ttc_true <= 1e-6))
        alert_idx = None
        for t in range(steps):
            if fire(seq, t):
                alert_idx = t
                break

        if breach_idx is None:
            result.safe_clips += 1
            if alert_idx is not None:
                result.false_alarms += 1
            continue

        result.hazard_clips += 1
        if alert_idx is None or alert_idx > breach_idx:
            result.misses += 1
            continue

        lead = (breach_idx - alert_idx) * DT
        result.lead_times_s.append(lead)
        if lead < 1.0:
            result.late_alerts += 1

    return result


# ------------------------------------------------------------------- decision rules


def geometric_rule(threshold_inv_ttc: float = 1.0 / (2.0 + 0.5)):
    """Constant-velocity baseline: fire when 1/(TTC+0.5) crosses the warn threshold.

    ``inv_ttc`` is feature index 6, so the baseline reads the exact same tensor the head
    does. Comparing against a weaker baseline that saw different inputs would make the
    head's margin meaningless.
    """

    def fire(seq: Sequence, t: int) -> bool:
        return bool(seq.features[t, 6] >= threshold_inv_ttc)

    return fire


def head_rule(head: LearnedRiskHead, threshold: float | None = None):
    """Learned head alone."""
    thr = head.threshold if threshold is None else threshold

    def fire(seq: Sequence, t: int) -> bool:
        window = window_of(seq.features, t, WINDOW).reshape(1, -1)
        return bool(head.predict_batch(window)[0] >= thr)

    return fire


def fused_rule(head: LearnedRiskHead, inv_ttc_urgent: float = 1.0 / (1.5 + 0.5),
               threshold: float | None = None):
    """Shipping rule: geometry fires on imminent contact, head fires earlier when
    confident, and neither can silence the other."""
    thr = head.threshold if threshold is None else threshold

    def fire(seq: Sequence, t: int) -> bool:
        if seq.features[t, 6] >= inv_ttc_urgent:
            return True
        window = window_of(seq.features, t, WINDOW).reshape(1, -1)
        return bool(head.predict_batch(window)[0] >= thr)

    return fire


def oracle_rule():
    """Upper bound: fire exactly when the true breach enters the label horizon.

    Reporting the oracle keeps everyone honest about headroom -- if the fused arm is
    within 0.2 s of it, the remaining error is in perception, not in the decision layer,
    and further work on the head is wasted effort.
    """

    def fire(seq: Sequence, t: int) -> bool:
        return bool(np.isfinite(seq.ttc_true[t]) and seq.ttc_true[t] <= HORIZON_S)

    return fire


def run_ablations(sequences: list[Sequence], head: LearnedRiskHead | None = None,
                  cfg: RiskConfig | None = None) -> list[ArmResult]:
    arms = [score_arm("Constant-velocity TTC (baseline)", sequences, geometric_rule(), cfg)]
    if head is not None and head.available:
        arms.append(score_arm("Learned head only", sequences, head_rule(head), cfg))
        arms.append(score_arm("Geometry + head (shipping)", sequences, fused_rule(head), cfg))
    arms.append(score_arm("Oracle (upper bound)", sequences, oracle_rule(), cfg))
    return arms
