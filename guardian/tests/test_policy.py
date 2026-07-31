"""Tests for alert arbitration.

Two failure modes matter, and they pull in opposite directions: chatter (the user takes
the device off) and silence (the user gets hurt). These tests pin both ends.
"""

from __future__ import annotations

from guardian.config import PolicyConfig, RiskConfig
from guardian.risk.policy import AlertPolicy, utterance_for
from guardian.types import AlertLevel, Direction, Hazard, Track


def make_hazard(ttc=1.0, track_id=1, direction=Direction.CENTRE, severity=0.9) -> Hazard:
    track = Track(
        track_id=track_id, x_m=0.0, z_m=ttc * 2.0, vx_mps=0.0, vz_mps=-2.0,
        width_m=0.6, class_name="car", hits=6,
    )
    return Hazard(
        track=track, ttc_s=ttc, closest_approach_m=0.2,
        time_of_closest_approach_s=ttc, direction=direction, severity=severity,
    )


def fresh_policy(**overrides) -> AlertPolicy:
    cfg = PolicyConfig(**overrides)
    return AlertPolicy(cfg, RiskConfig())


class TestHysteresis:
    def test_single_frame_blip_is_suppressed(self):
        # WARN-level: a one-frame flicker must not speak.
        policy = fresh_policy(hysteresis_frames=2)
        assert policy.decide([make_hazard(ttc=2.6, severity=0.4)], now_s=0.0) is None

    def test_persistent_hazard_fires(self):
        policy = fresh_policy(hysteresis_frames=2)
        policy.decide([make_hazard(ttc=2.6, severity=0.4)], now_s=0.0)
        alert = policy.decide([make_hazard(ttc=2.5, severity=0.4)], now_s=0.1)
        assert alert is not None
        assert alert.level is AlertLevel.WARN

    def test_urgent_bypasses_hysteresis(self):
        """Sub-1.5 s TTC cannot afford to wait for confirmation frames."""
        policy = fresh_policy(hysteresis_frames=4)
        alert = policy.decide([make_hazard(ttc=0.8)], now_s=0.0)
        assert alert is not None and alert.level is AlertLevel.URGENT



class TestCooldown:
    def test_no_repeat_within_cooldown(self):
        policy = fresh_policy(hysteresis_frames=1, alert_cooldown_s=2.5, urgent_cooldown_s=2.0)
        first = policy.decide([make_hazard(ttc=2.5)], now_s=0.0)
        assert first is not None
        assert policy.decide([make_hazard(ttc=2.4)], now_s=0.3) is None

    def test_repeats_after_cooldown(self):
        policy = fresh_policy(hysteresis_frames=1, alert_cooldown_s=1.0, urgent_cooldown_s=0.5)
        assert policy.decide([make_hazard(ttc=2.5)], now_s=0.0) is not None
        assert policy.decide([make_hazard(ttc=2.5)], now_s=5.0) is not None

    def test_escalation_bypasses_cooldown(self):
        """A situation getting worse must interrupt. Silence here is the dangerous bug."""
        policy = fresh_policy(hysteresis_frames=1, alert_cooldown_s=5.0, urgent_cooldown_s=0.5)
        warn = policy.decide([make_hazard(ttc=2.8)], now_s=0.0)
        assert warn is not None and warn.level is AlertLevel.WARN
        urgent = policy.decide([make_hazard(ttc=0.8)], now_s=0.2)
        assert urgent is not None and urgent.level is AlertLevel.URGENT


class TestArbitration:
    def test_only_the_worst_hazard_speaks(self):
        policy = fresh_policy(hysteresis_frames=1)
        hazards = [
            make_hazard(ttc=2.9, track_id=1, severity=0.3),
            make_hazard(ttc=0.7, track_id=2, severity=0.95),
        ]
        alert = policy.decide(hazards, now_s=0.0)
        assert alert is not None
        assert alert.hazard is not None
        assert alert.hazard.track.track_id == 2

    def test_no_hazards_means_silence(self):
        assert fresh_policy().decide([], now_s=0.0) is None

    def test_degraded_is_reported_not_hidden(self):
        alert = fresh_policy().decide([], now_s=0.0, degraded=True)
        assert alert is not None
        assert alert.level is AlertLevel.DEGRADED


class TestUtterance:
    def test_urgent_is_short(self):
        text = utterance_for(make_hazard(ttc=0.6), AlertLevel.URGENT)
        assert len(text.split()) <= 5, "urgent phrasing must fit inside the reaction window"

    def test_includes_direction(self):
        text = utterance_for(make_hazard(direction=Direction.LEFT), AlertLevel.WARN)
        assert "left" in text.lower()
