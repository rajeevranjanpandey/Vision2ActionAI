"""Evaluation metrics.

mAP is a component diagnostic, not the headline. What determines whether a blind user
keeps wearing the device is: did it warn me early enough, and did it shut up when there
was nothing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import Clip, HazardLabel
from ..types import Alert, AlertLevel


@dataclass
class ClipScore:
    clip_id: str
    lead_times_s: list[float] = field(default_factory=list)
    missed: int = 0
    false_alarms: int = 0
    distance_m: float = 0.0
    late_alerts: int = 0
    alert_count: int = 0

    @property
    def detected(self) -> int:
        return len(self.lead_times_s)


@dataclass
class BenchmarkReport:
    clips: list[ClipScore] = field(default_factory=list)

    # ------------------------------------------------------------ aggregates

    @property
    def all_lead_times(self) -> np.ndarray:
        return np.array([lt for c in self.clips for lt in c.lead_times_s], dtype=np.float64)

    @property
    def median_lead_time_s(self) -> float:
        lt = self.all_lead_times
        return float(np.median(lt)) if lt.size else 0.0

    @property
    def p10_lead_time_s(self) -> float:
        """The tail is what hurts. A great median with a 0.2 s p10 is an unsafe device."""
        lt = self.all_lead_times
        return float(np.percentile(lt, 10)) if lt.size else 0.0

    @property
    def recall(self) -> float:
        detected = sum(c.detected for c in self.clips)
        total = detected + sum(c.missed for c in self.clips)
        return detected / total if total else 0.0

    @property
    def false_alarms_per_km(self) -> float:
        fa = sum(c.false_alarms for c in self.clips)
        km = sum(c.distance_m for c in self.clips) / 1000.0
        return fa / km if km > 1e-6 else float("inf")

    @property
    def late_rate(self) -> float:
        """Fraction of caught hazards warned about with under 1 s of lead time."""
        total = sum(c.detected for c in self.clips)
        return sum(c.late_alerts for c in self.clips) / total if total else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "median_lead_time_s": self.median_lead_time_s,
            "p10_lead_time_s": self.p10_lead_time_s,
            "recall": self.recall,
            "false_alarms_per_km": self.false_alarms_per_km,
            "late_rate": self.late_rate,
            "clips": float(len(self.clips)),
            "hazards": float(sum(c.detected + c.missed for c in self.clips)),
        }

    def render(self) -> str:
        s = self.summary()
        return (
            "GuardianBench\n"
            f"  clips                {int(s['clips'])}\n"
            f"  hazards              {int(s['hazards'])}\n"
            f"  recall               {s['recall']:.3f}\n"
            f"  lead time (median)   {s['median_lead_time_s']:.2f} s\n"
            f"  lead time (p10)      {s['p10_lead_time_s']:.2f} s\n"
            f"  late alerts (<1 s)   {s['late_rate']:.3f}\n"
            f"  false alarms / km    {s['false_alarms_per_km']:.2f}\n"
        )


def score_clip(
    clip: Clip,
    alerts: list[tuple[int, Alert]],
    match_window_s: float = 4.0,
    late_threshold_s: float = 1.0,
) -> ClipScore:
    """Match emitted alerts to labelled hazards and compute per-clip metrics.

    Matching rule: an alert counts for a hazard if it fires in
    ``[onset - 1 s, contact]`` and points the right way. Firing *before* the O&M-marked
    onset is allowed (early is fine) but only within a second, otherwise a device that
    alarms constantly would score perfect recall.
    """
    fps = clip.fps
    score = ClipScore(clip_id=clip.clip_id, distance_m=clip.distance_m)
    score.alert_count = len(alerts)

    unmatched = list(alerts)
    for hazard in clip.hazards:
        match = _first_match(hazard, unmatched, fps, match_window_s)
        if match is None:
            score.missed += 1
            continue
        unmatched.remove(match)
        frame_idx, _alert = match
        contact = hazard.contact_frame if hazard.contact_frame is not None else hazard.onset_frame
        lead = (contact - frame_idx) / fps
        score.lead_times_s.append(float(lead))
        if lead < late_threshold_s:
            score.late_alerts += 1

    # Anything left over that is not near any hazard is a false alarm. DEGRADED alerts
    # are excluded: they are honest failure reports, not spurious hazard claims.
    for frame_idx, alert in unmatched:
        if alert.level is AlertLevel.DEGRADED:
            continue
        if not _near_any_hazard(frame_idx, clip.hazards, fps, match_window_s):
            score.false_alarms += 1

    return score


def _first_match(
    hazard: HazardLabel,
    alerts: list[tuple[int, Alert]],
    fps: float,
    window_s: float,
) -> tuple[int, Alert] | None:
    lo = hazard.onset_frame - int(1.0 * fps)
    hi = (
        hazard.contact_frame
        if hazard.contact_frame is not None
        else hazard.onset_frame + int(window_s * fps)
    )
    for frame_idx, alert in sorted(alerts, key=lambda pair: pair[0]):
        if not lo <= frame_idx <= hi:
            continue
        if alert.level is AlertLevel.DEGRADED:
            continue
        if hazard.direction != "centre" and alert.direction.value != hazard.direction:
            continue
        return frame_idx, alert
    return None


def _near_any_hazard(
    frame_idx: int, hazards: list[HazardLabel], fps: float, window_s: float
) -> bool:
    span = int(window_s * fps)
    for hazard in hazards:
        end = hazard.contact_frame if hazard.contact_frame is not None else hazard.onset_frame
        if hazard.onset_frame - span <= frame_idx <= end + span:
            return True
    return False


def aggregate(scores: list[ClipScore]) -> BenchmarkReport:
    return BenchmarkReport(clips=scores)
