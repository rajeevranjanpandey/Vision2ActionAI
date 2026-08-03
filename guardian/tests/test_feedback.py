"""Tests for the federated near-miss loop (roadmap item 5)."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.feedback import (
    ClientUpdate,
    NearMissLog,
    NearMissRecord,
    RedactionError,
    aggregate,
    assert_privacy_safe,
    clip_delta,
    delta_norm,
    simulate_retrain,
)
from guardian.feedback.nearmiss import device_hash
from guardian.feedback.retrain import late_alert_count, privacy_utility_curve
from guardian.train.features import D, WINDOW


def window(seed=0):
    return np.random.default_rng(seed).normal(size=(WINDOW, D)).astype(np.float32)


def record(label="near_miss", seed=0):
    return NearMissRecord(window(seed), label=label)


# ------------------------------------------------------------------------ records


def test_record_requires_the_deployed_window_shape():
    with pytest.raises(ValueError):
        NearMissRecord(np.zeros((3, 3)), label="near_miss")


def test_record_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        NearMissRecord(window(), label="whatever")


def test_record_rejects_an_impossible_hour():
    with pytest.raises(ValueError):
        NearMissRecord(window(), label="near_miss", hour_bucket=47)


def test_payload_only_contains_allowed_fields():
    assert_privacy_safe(record().to_payload())


def test_payload_carries_no_timestamp_finer_than_the_hour():
    p = record().to_payload()
    assert isinstance(p["hour_bucket"], int)
    assert "timestamp" not in p and "t_s" not in p


def test_device_hash_is_pseudonymous_and_stable():
    a, b = device_hash("device-1"), device_hash("device-1")
    assert a == b and "device-1" not in a
    assert device_hash("device-2") != a


def test_device_hash_changes_with_the_salt():
    assert device_hash("d", "salt-a") != device_hash("d", "salt-b")


def test_privacy_check_rejects_smuggled_media():
    bad = record().to_payload()
    bad["thumbnail"] = "base64:AAAA"
    with pytest.raises(RedactionError):
        assert_privacy_safe(bad)


def test_privacy_check_rejects_location():
    bad = record().to_payload()
    bad["lat"] = 51.5
    with pytest.raises(RedactionError):
        assert_privacy_safe(bad)


def test_a_payload_is_small_enough_to_send_on_cellular():
    assert record().payload_bytes < 12_000


# ----------------------------------------------------------------------- the log


def test_log_export_passes_the_privacy_audit():
    log = NearMissLog()
    log.extend([record(seed=i) for i in range(5)])
    assert len(log.export()) == 5


def test_log_is_bounded_and_keeps_the_newest():
    log = NearMissLog(max_records=3)
    log.extend([record(seed=i) for i in range(10)])
    assert len(log.records) == 3
    assert np.allclose(log.records[-1].window, window(9))


def test_labels_map_to_the_right_supervision():
    log = NearMissLog()
    log.flag(record("near_miss", 1))
    log.flag(record("false_alarm", 2))
    log.flag(record("correct_alert", 3))
    _, y = log.as_dataset()
    assert list(y) == [1.0, 0.0, 1.0]


def test_dataset_rows_match_the_head_input_width():
    log = NearMissLog()
    log.extend([record(seed=i) for i in range(4)])
    X, _ = log.as_dataset()
    assert X.shape == (4, WINDOW * D)


def test_empty_log_yields_an_empty_dataset():
    X, y = NearMissLog().as_dataset()
    assert X.shape == (0, WINDOW * D) and y.shape == (0,)


def test_counts_break_down_by_label():
    log = NearMissLog()
    log.flag(record("near_miss", 1))
    log.flag(record("near_miss", 2))
    log.flag(record("false_alarm", 3))
    assert log.counts() == {"near_miss": 2, "false_alarm": 1, "correct_alert": 0}


# ------------------------------------------------------------------ aggregation


def test_delta_norm_is_the_global_l2():
    d = {"w": np.array([3.0, 0.0]), "b": np.array([4.0])}
    assert delta_norm(d) == pytest.approx(5.0)


def test_clipping_leaves_a_small_delta_alone():
    d = {"w": np.array([0.1, 0.1])}
    out, was = clip_delta(d, 1.0)
    assert not was and np.allclose(out["w"], d["w"])


def test_clipping_scales_a_large_delta_to_the_norm():
    d = {"w": np.array([30.0, 40.0])}
    out, was = clip_delta(d, 1.0)
    assert was and delta_norm(out) == pytest.approx(1.0)


def test_clipping_handles_a_zero_delta():
    out, was = clip_delta({"w": np.zeros(4)}, 1.0)
    assert not was and delta_norm(out) == 0.0


def test_aggregate_without_noise_is_the_weighted_mean():
    ups = [
        ClientUpdate("a", {"w": np.array([1.0, 0.0])}, n_samples=1),
        ClientUpdate("b", {"w": np.array([0.0, 1.0])}, n_samples=3),
    ]
    agg, _ = aggregate(ups, clip_norm=10.0, noise_multiplier=0.0)
    assert np.allclose(agg["w"], [0.25, 0.75])


def test_aggregate_reports_the_clipped_fraction():
    ups = [
        ClientUpdate("a", {"w": np.array([100.0, 0.0])}, n_samples=1),
        ClientUpdate("b", {"w": np.array([0.01, 0.0])}, n_samples=1),
    ]
    _, rep = aggregate(ups, clip_norm=1.0, noise_multiplier=0.0)
    assert rep.clipped_fraction == pytest.approx(0.5)


def test_clipping_bounds_a_single_malicious_client():
    honest = [ClientUpdate(f"h{i}", {"w": np.array([0.1, 0.0])}, 10) for i in range(9)]
    attacker = ClientUpdate("bad", {"w": np.array([500.0, 0.0])}, 10)
    agg, _ = aggregate(honest + [attacker], clip_norm=0.5, noise_multiplier=0.0)
    assert agg["w"][0] < 0.15, "one client should not be able to dominate the round"


def test_noise_perturbs_the_aggregate():
    ups = [ClientUpdate("a", {"w": np.zeros(8)}, 5)]
    quiet, _ = aggregate(ups, noise_multiplier=0.0)
    noisy, rep = aggregate(ups, noise_multiplier=1.0, rng=np.random.default_rng(1))
    assert not np.allclose(quiet["w"], noisy["w"])
    assert rep.noise_std > 0.0


def test_privacy_note_reflects_whether_noise_was_added():
    ups = [ClientUpdate("a", {"w": np.zeros(4)}, 5)]
    _, no_dp = aggregate(ups, noise_multiplier=0.0)
    _, dp = aggregate(ups, noise_multiplier=0.8)
    assert "no formal guarantee" in no_dp.privacy_note
    assert "Gaussian mechanism" in dp.privacy_note


def test_aggregate_rejects_mismatched_parameter_blocks():
    ups = [
        ClientUpdate("a", {"w": np.zeros(3)}, 1),
        ClientUpdate("b", {"v": np.zeros(3)}, 1),
    ]
    with pytest.raises(ValueError):
        aggregate(ups)


def test_aggregate_rejects_an_empty_round():
    with pytest.raises(ValueError):
        aggregate([])


def test_client_update_rejects_zero_samples():
    with pytest.raises(ValueError):
        ClientUpdate("a", {"w": np.zeros(2)}, 0)


# -------------------------------------------------------------------- the metric


def test_late_alert_counter_only_counts_late_positives():
    early = np.array([0.1, 0.9, 0.1, 0.1])
    late = np.array([0.9, 0.9, 0.9, 0.1])
    y = np.array([1.0, 1.0, 0.0, 1.0])
    # row 0: fires only from the late view -> late. row 1: fired early -> fine.
    # row 2: not a hazard. row 3: never fires -> a miss, not a late alert.
    assert late_alert_count(early, late, y, threshold=0.5) == 1


def test_retrain_improves_holdout_recall():
    res = simulate_retrain(rounds=6, seed=0)
    rounds = res["rounds"]
    assert rounds[-1]["recall"] > rounds[0]["recall"]


def test_late_alerts_trend_down_across_rounds():
    res = simulate_retrain(rounds=8, seed=0)
    assert res["lateAlertsLast"] <= res["lateAlertsFirst"]


def test_more_noise_costs_utility():
    curve = privacy_utility_curve(noise_multipliers=(0.0, 2.5), rounds=5, seed=0)
    assert curve[0]["recall"] >= curve[-1]["recall"] - 1e-9


def test_privacy_utility_curve_reports_every_setting():
    curve = privacy_utility_curve(noise_multipliers=(0.0, 0.4, 1.0), rounds=3)
    assert [row["noiseMultiplier"] for row in curve] == [0.0, 0.4, 1.0]
