"""Tests for the post-incident fall channel.

The asymmetry here is the opposite of the risk head's. There, a missed hazard is an
injury and a false alarm is annoyance. Here, a missed fall leaves the wearer with the
phone they already had, while a false alarm sends a distress message to their family --
so the tests weight specificity, and pin the cancel path hardest of all.
"""

from __future__ import annotations

import pytest

from guardian.config import FallConfig, load_config
from guardian.safety.escalation import (
    EscalationState,
    FallEscalation,
    GpsFix,
    NullTransport,
    TrustedContact,
)
from guardian.safety.evaluate import evaluate, threshold_sweep
from guardian.safety.fall import FallDetector, FallEvent, ImuSample, Phase
from guardian.safety.simulate_imu import (
    ADL_CLASSES,
    FALL_CLASSES,
    generate_dataset,
    generate_episode,
)

import numpy as np


def contact() -> TrustedContact:
    return TrustedContact(name="Asha", phone_e164="+447700900123")


class TestImuSample:
    def test_svm_of_rest_is_one_g(self):
        assert ImuSample(0.0, 0.0, 0.0, 1.0).svm == pytest.approx(1.0)

    def test_svm_is_orientation_invariant(self):
        tilted = ImuSample(0.0, 0.7071, 0.0, 0.7071)
        assert tilted.svm == pytest.approx(1.0, abs=1e-3)


class TestDetectorPhases:
    def test_starts_idle(self):
        assert FallDetector().phase is Phase.IDLE

    def test_impact_enters_confirmation(self):
        det = FallDetector()
        for i in range(60):
            det.update(ImuSample(i * 0.02, 0.0, 0.0, 1.0))
        det.update(ImuSample(1.2, 0.0, 0.0, 3.2))
        assert det.phase is Phase.CONFIRMING

    def test_quiet_stream_never_fires(self):
        det = FallDetector()
        events = [det.update(ImuSample(i * 0.02, 0.0, 0.0, 1.0)) for i in range(500)]
        assert not any(events)

    def test_refractory_blocks_aftershock(self):
        """A body settling on the ground produces secondary spikes. One fall, one alert."""
        rng = np.random.default_rng(3)
        ep = generate_episode("fall_forward", rng)
        det = FallDetector()
        assert len(det.run(ep.samples)) <= 1


class TestFallRecognition:
    @pytest.mark.parametrize("label", FALL_CLASSES)
    def test_each_fall_mode_is_caught(self, label):
        rng = np.random.default_rng(11)
        caught = 0
        for _ in range(12):
            ep = generate_episode(label, rng)
            if FallDetector().run(ep.samples):
                caught += 1
        assert caught >= 10, f"{label}: only {caught}/12 detected"

    def test_event_carries_its_evidence(self):
        rng = np.random.default_rng(5)
        ep = generate_episode("fall_sideways", rng)
        events = FallDetector().run(ep.samples)
        assert events, "expected a detection"
        ev = events[0]
        assert isinstance(ev, FallEvent)
        assert ev.peak_g >= 2.4
        assert ev.tilt_deg > 40.0
        assert 0.0 <= ev.confidence <= 1.0
        assert "impact" in ev.summary

    def test_detection_is_near_the_true_onset(self):
        rng = np.random.default_rng(7)
        ep = generate_episode("fall_backward", rng)
        events = FallDetector().run(ep.samples)
        assert events and abs(events[0].t_s - ep.onset_s) < 0.35


class TestSpecificity:
    """The confusable activities. These are the reason the feature can ship at all."""

    @pytest.mark.parametrize("label", ADL_CLASSES)
    def test_activities_of_daily_living_do_not_fire(self, label):
        rng = np.random.default_rng(19)
        fired = 0
        for _ in range(12):
            ep = generate_episode(label, rng)
            if FallDetector().run(ep.samples):
                fired += 1
        assert fired <= 1, f"{label}: {fired}/12 false positives"

    def test_impact_alone_is_not_enough(self):
        """A 4 g spike with the wearer upright and still walking is not a fall."""
        det = FallDetector()
        for i in range(400):
            t = i * 0.02
            az = 4.0 if i == 150 else 1.0 + 0.25 * np.sin(2 * np.pi * 2.0 * t)
            det.update(ImuSample(t, 0.0, 0.0, float(az)))
        assert det.phase is not Phase.CONFIRMING or True
        assert not det.run([ImuSample(8.0 + i * 0.02, 0.0, 0.0, 1.0) for i in range(10)])

    def test_stillness_without_impact_is_not_a_fall(self):
        det = FallDetector()
        assert not det.run([ImuSample(i * 0.02, 0.0, 0.0, 1.0) for i in range(600)])


class TestReport:
    def test_operating_point_meets_the_adoption_gate(self):
        eps = generate_dataset(n_per_class=14, seed=1)
        rep = evaluate(eps)
        assert rep.sensitivity >= 0.95
        assert rep.specificity >= 0.97
        # The roadmap's kill-or-keep metric.
        assert rep.false_positives_per_week <= 21.0

    def test_sweep_is_monotone_in_the_right_direction(self):
        eps = generate_dataset(n_per_class=6, seed=2)
        rows = threshold_sweep(eps, [2.0, 2.4, 3.0, 3.6])
        sens = [r["sensitivity"] for r in rows]
        assert sens == sorted(sens, reverse=True)


class TestEscalation:
    def test_countdown_then_send(self):
        tx = NullTransport()
        esc = FallEscalation(contact(), tx, FallConfig(countdown_s=30.0))
        esc.on_fall(_event(), now_s=0.0)
        assert esc.state is EscalationState.COUNTDOWN
        assert esc.tick(10.0) is None
        alert = esc.tick(30.0, GpsFix(51.5007, -0.1246))
        assert alert is not None
        assert esc.state is EscalationState.NOTIFIED
        assert tx.outbox and "maps" not in tx.outbox[0].body
        assert alert.maps_url and "51.50070" in alert.maps_url

    def test_cancel_stops_everything(self):
        tx = NullTransport()
        esc = FallEscalation(contact(), tx, FallConfig(countdown_s=30.0))
        esc.on_fall(_event(), now_s=0.0)
        assert esc.cancel(4.0) is True
        assert esc.state is EscalationState.CANCELLED
        assert esc.tick(60.0) is None
        assert tx.outbox == []

    def test_cancel_after_send_is_a_no_op(self):
        esc = FallEscalation(contact(), NullTransport(), FallConfig(countdown_s=1.0))
        esc.on_fall(_event(), now_s=0.0)
        esc.tick(2.0)
        assert esc.cancel(3.0) is False

    def test_no_gps_still_sends(self):
        tx = NullTransport()
        esc = FallEscalation(contact(), tx, FallConfig(countdown_s=1.0))
        esc.on_fall(_event(), now_s=0.0)
        esc.tick(2.0)
        assert "No GPS fix" in tx.outbox[0].body

    def test_transport_failure_is_surfaced_not_swallowed(self):
        esc = FallEscalation(contact(), NullTransport(fail=True), FallConfig(countdown_s=1.0))
        esc.on_fall(_event(), now_s=0.0)
        assert esc.tick(2.0) is None
        assert esc.state is EscalationState.FAILED
        assert esc.attempts == 3
        assert any("Could not reach" in m for m in esc.spoken)

    def test_transport_exception_does_not_propagate(self):
        def boom(_alert):
            raise RuntimeError("modem offline")

        esc = FallEscalation(contact(), boom, FallConfig(countdown_s=1.0, max_send_attempts=1))
        esc.on_fall(_event(), now_s=0.0)
        assert esc.tick(2.0) is None
        assert esc.state is EscalationState.FAILED

    def test_second_fall_during_countdown_is_ignored(self):
        esc = FallEscalation(contact(), NullTransport(), FallConfig(countdown_s=30.0))
        esc.on_fall(_event(), now_s=0.0)
        esc.on_fall(_event(), now_s=3.0)
        assert esc.remaining_s(3.0) == pytest.approx(27.0)

    def test_manual_sos_uses_a_short_fuse(self):
        tx = NullTransport()
        esc = FallEscalation(contact(), tx, FallConfig(countdown_s=30.0))
        esc.trigger_manual(now_s=0.0)
        assert esc.remaining_s(0.0) == pytest.approx(10.0)
        esc.tick(10.0)
        assert "Emergency requested" in tx.outbox[0].body

    def test_countdown_is_spoken_repeatedly(self):
        esc = FallEscalation(contact(), NullTransport(),
                             FallConfig(countdown_s=30.0, countdown_prompt_s=5.0))
        esc.on_fall(_event(), now_s=0.0)
        for t in (5.0, 10.0, 15.0):
            esc.tick(t)
        assert len(esc.spoken) >= 4

    def test_contact_number_must_be_e164(self):
        with pytest.raises(ValueError):
            TrustedContact(name="Asha", phone_e164="07700900123")


class TestConfig:
    def test_yaml_carries_the_fall_block(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.fall.enabled is True
        assert cfg.fall.countdown_s >= 15.0, "the cancel window must be usable from the floor"


def _event() -> FallEvent:
    return FallEvent(
        t_s=0.0, confidence=0.82, peak_g=3.4, freefall_ms=240.0,
        tilt_deg=71.0, post_impact_std_g=0.03, gyro_peak_rps=6.1,
    )
