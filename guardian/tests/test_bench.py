"""Tests for GuardianBench scoring.

If the metric implementation is wrong, every number in the paper is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from guardian.bench.dataset import Clip, HazardLabel
from guardian.bench.metrics import aggregate, score_clip
from guardian.types import Alert, AlertLevel, Direction


def make_clip(hazards, fps=10.0, distance_m=100.0) -> Clip:
    # Constant 1.0 m/s over `distance_m` seconds gives exactly distance_m metres.
    t = np.linspace(0.0, distance_m, int(distance_m) + 1)
    imu = np.stack([t, np.ones_like(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    return Clip(
        clip_id="c0", root=Path("."), fps=fps,
        frame_paths=[Path(f"{i}.jpg") for i in range(int(fps * distance_m))],
        imu=imu, hazards=hazards, meta={"fps": fps},
    )


def alert(level=AlertLevel.WARN, direction=Direction.CENTRE) -> Alert:
    return Alert(level=level, direction=direction, hazard=None, utterance="test")


class TestScoring:
    def test_timely_alert_counts_with_positive_lead(self):
        clip = make_clip([HazardLabel("h0", "vehicle", onset_frame=100,
                                      contact_frame=130, direction="centre", severity="serious")])
        score = score_clip(clip, [(105, alert())])
        assert score.detected == 1
        assert score.missed == 0
        assert score.lead_times_s[0] == (130 - 105) / 10.0

    def test_silence_is_a_miss(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "centre", "serious")])
        score = score_clip(clip, [])
        assert score.missed == 1
        assert score.detected == 0

    def test_alert_after_contact_is_a_miss(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "centre", "serious")])
        score = score_clip(clip, [(200, alert())])
        assert score.missed == 1

    def test_alert_in_empty_stretch_is_a_false_alarm(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "centre", "serious")])
        score = score_clip(clip, [(110, alert()), (800, alert())])
        assert score.detected == 1
        assert score.false_alarms == 1

    def test_degraded_alerts_are_never_false_alarms(self):
        clip = make_clip([])
        score = score_clip(clip, [(500, alert(level=AlertLevel.DEGRADED))])
        assert score.false_alarms == 0

    def test_wrong_direction_does_not_count(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "left", "serious")])
        score = score_clip(clip, [(110, alert(direction=Direction.RIGHT))])
        assert score.missed == 1

    def test_late_alert_is_flagged(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "centre", "serious")])
        score = score_clip(clip, [(125, alert())])   # 0.5 s of lead time
        assert score.detected == 1
        assert score.late_alerts == 1


class TestAggregate:
    def test_false_alarms_per_km(self):
        clip = make_clip([], distance_m=1000.0)
        score = score_clip(clip, [(i * 100, alert()) for i in range(4)])
        report = aggregate([score])
        assert report.false_alarms_per_km == 4.0

    def test_p10_catches_a_bad_tail(self):
        clip = make_clip(
            [HazardLabel(f"h{i}", "vehicle", 100 + i * 200, 140 + i * 200, "centre", "serious")
             for i in range(10)]
        )
        # Nine comfortable warnings, one nearly too late.
        alerts = [(100 + i * 200, alert()) for i in range(9)]
        alerts.append((100 + 9 * 200 + 37, alert()))
        report = aggregate([score_clip(clip, alerts)])
        assert report.median_lead_time_s > report.p10_lead_time_s

    def test_recall_is_zero_without_detections(self):
        clip = make_clip([HazardLabel("h0", "vehicle", 100, 130, "centre", "serious")])
        assert aggregate([score_clip(clip, [])]).recall == 0.0
