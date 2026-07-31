"""Tests for the learning stack: features, simulator, trainer, calibration, ablations.

These are not smoke tests. Each one pins a property that, if it broke silently, would
produce a model that looks fine on a dashboard and fails on a pavement.
"""

from __future__ import annotations

import numpy as np
import pytest

from guardian.bench.simeval import geometric_rule, oracle_rule, run_ablations, score_arm
from guardian.config import RiskConfig
from guardian.risk.calibration import (
    apply_temperature,
    brier_score,
    calibrate,
    calibrate_alert_threshold,
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from guardian.risk.learned import LearnedRiskHead, RiskPrediction, fuse
from guardian.risk.ttc import assess
from guardian.train.features import D, WINDOW, FrameContext, build_windows, track_features, window_of
from guardian.train.simulate import DT, exact_time_to_breach, generate_dataset, simulate_clip
from guardian.train.trainer import (
    MLP,
    TrainConfig,
    average_precision,
    group_split,
    roc_auc,
    sigmoid,
    train_ensemble,
)
from guardian.types import Track

CFG = RiskConfig()


def _track(**kw) -> Track:
    base = dict(track_id=1, x_m=0.0, z_m=5.0, vx_mps=0.0, vz_mps=-1.0,
                width_m=0.5, class_name="person", hits=5)
    base.update(kw)
    return Track(**base)


# ----------------------------------------------------------------------- features


def test_features_are_finite_and_bounded_for_absurd_inputs():
    """A depth blow-up must saturate, never NaN. A NaN here silently kills the head."""
    for track in [
        _track(z_m=1e6, vz_mps=-1e5),
        _track(z_m=1e-6, x_m=-1e6),
        _track(vx_mps=float("inf")),
        _track(width_m=1e4),
    ]:
        f = track_features(track, CFG)
        assert f.shape == (D,)
        assert np.all(np.isfinite(f))
        assert np.all(np.abs(f) <= 4.0 + 1e-6)


def test_inv_ttc_is_monotone_in_urgency():
    close = track_features(_track(z_m=2.0, vz_mps=-2.0), CFG)[6]
    far = track_features(_track(z_m=10.0, vz_mps=-2.0), CFG)[6]
    assert close > far


def test_cone_margin_sign_matches_geometry():
    inside = track_features(_track(x_m=0.0, z_m=3.0), CFG)[9]
    outside = track_features(_track(x_m=4.0, z_m=3.0), CFG)[9]
    assert inside > 0 > outside


def test_detector_and_depth_context_flow_through():
    f = track_features(_track(), CFG, FrameContext(depth_confidence=0.3, detector_score=0.4))
    assert f[11] == pytest.approx(0.4)
    assert f[12] == pytest.approx(0.3)


def test_window_is_causal_and_edge_padded():
    seq = np.arange(5 * D, dtype=np.float32).reshape(5, D)
    w = window_of(seq, 1, window=4)
    assert w.shape == (4, D)
    # first two rows are the pad (a copy of row 0), then rows 0 and 1
    assert np.allclose(w[0], seq[0]) and np.allclose(w[-1], seq[1])
    # nothing from the future leaked in
    assert not any(np.allclose(row, seq[2]) for row in w)


def test_build_windows_keeps_group_ids_aligned():
    seqs = [np.zeros((3, D), dtype=np.float32), np.ones((4, D), dtype=np.float32)]
    labels = [np.zeros(3), np.ones(4)]
    X, y, g = build_windows(seqs, labels, WINDOW, ["a", "b"])
    assert X.shape == (7, WINDOW * D)
    assert list(g).count("a") == 3 and list(g).count("b") == 4
    assert y.sum() == 4


# ---------------------------------------------------------------------- simulator


def test_exact_time_to_breach_counts_down_at_dt():
    states = np.zeros((10, 4))
    states[:, 1] = np.linspace(3.0, 0.0, 10)  # walking straight in
    ttb = exact_time_to_breach(states, radius=1.0)
    finite = ttb[np.isfinite(ttb)]
    diffs = np.diff(finite)
    assert np.all(diffs <= 1e-9)
    assert np.allclose(np.abs(diffs[diffs != 0]), DT, atol=1e-9)


def test_simulator_is_deterministic_under_seed():
    a = generate_dataset(12, CFG, seed=7)
    b = generate_dataset(12, CFG, seed=7)
    assert all(np.allclose(x.features, y.features) for x, y in zip(a, b))
    assert [x.scenario for x in a] == [y.scenario for y in b]


def test_dataset_has_both_classes_and_realistic_skew():
    data = generate_dataset(200, CFG, seed=1)
    y = np.concatenate([c.labels for c in data])
    assert 0.02 < y.mean() < 0.45, "simulator drifted into a balanced (unrealistic) mix"


def test_yielding_pedestrian_mostly_does_not_breach():
    """If the yielding scenario breached, the hard case would collapse into the easy one."""
    rng = np.random.default_rng(3)
    breaches = 0
    for i in range(30):
        seq = simulate_clip(f"c{i}", "yielding_pedestrian", CFG, rng)
        breaches += int(seq.labels.any())
    assert breaches <= 12


def test_crossing_cyclist_often_breaches():
    rng = np.random.default_rng(4)
    breaches = sum(
        int(simulate_clip(f"c{i}", "crossing_cyclist", CFG, rng).labels.any()) for i in range(30)
    )
    assert breaches >= 8


def test_near_miss_vehicle_rarely_breaches():
    rng = np.random.default_rng(5)
    breaches = sum(
        int(simulate_clip(f"c{i}", "near_miss_vehicle", CFG, rng).labels.any()) for i in range(30)
    )
    assert breaches <= 6


# ------------------------------------------------------------------------ trainer


def test_sigmoid_is_stable_at_extremes():
    z = np.array([-1e4, -50.0, 0.0, 50.0, 1e4])
    p = sigmoid(z)
    assert np.all(np.isfinite(p)) and p[0] == 0.0 and p[-1] == 1.0
    assert p[2] == pytest.approx(0.5)


def test_roc_auc_matches_known_cases():
    assert roc_auc(np.array([0, 1]), np.array([0.1, 0.9])) == pytest.approx(1.0)
    assert roc_auc(np.array([0, 1]), np.array([0.9, 0.1])) == pytest.approx(0.0)
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_average_precision_penalises_skewed_ranking_more_than_auc():
    y = np.array([1] + [0] * 99)
    good = np.concatenate([[0.9], np.linspace(0.0, 0.5, 99)])
    mediocre = np.concatenate([[0.55], np.linspace(0.0, 0.5, 99)])
    assert average_precision(y, good) == pytest.approx(1.0)
    assert average_precision(y, mediocre) == pytest.approx(1.0)
    assert average_precision(y, np.concatenate([[0.1], np.linspace(0.0, 0.5, 99)])) < 0.2


def test_group_split_never_puts_a_clip_in_both_sides():
    groups = np.array([f"clip{i // 10}" for i in range(200)], dtype=object)
    tr, va = group_split(groups, 0.3, seed=0)
    assert not (set(groups[tr]) & set(groups[va]))
    assert tr.sum() + va.sum() == len(groups)


def test_mlp_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64)
    model = MLP(6, (16, 8), seed=0)
    cfg = TrainConfig(lr=5e-3)
    for _ in range(300):
        model.step(X, y, cfg)
    assert roc_auc(y, model.predict(X)) > 0.95


def test_mlp_round_trips_through_state():
    model = MLP(4, (8, 4), seed=1)
    X = np.random.default_rng(0).normal(size=(5, 4))
    clone = MLP.from_state(model.state())
    assert np.allclose(model.predict(X), clone.predict(X), atol=1e-6)


def test_ensemble_beats_chance_on_simulated_hazards():
    data = generate_dataset(220, CFG, seed=2)
    X, y, g = build_windows([c.features for c in data], [c.labels for c in data],
                            WINDOW, [c.clip_id for c in data])
    models, report, holdout = train_ensemble(
        X, y, g, TrainConfig(epochs=45, ensemble=2, seed=0)
    )
    assert len(models) == 2
    assert report.best_auc > 0.85, f"AUC collapsed to {report.best_auc}"
    assert holdout["y"].sum() > 0


# -------------------------------------------------------------------- calibration


def test_temperature_recovers_a_known_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, 4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z))).astype(float)
    p_overconfident = 1 / (1 + np.exp(-z * 2.0))   # logits inflated 2x
    t = fit_temperature(y, p_overconfident)
    assert 1.5 < t < 2.6
    assert expected_calibration_error(y, apply_temperature(p_overconfident, t)) < \
        expected_calibration_error(y, p_overconfident)


def test_temperature_of_one_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-5)


def test_ece_is_zero_for_a_perfectly_calibrated_stream():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02


def test_reliability_bins_are_equal_mass():
    p = np.concatenate([np.zeros(900), np.linspace(0.5, 1.0, 100)])
    y = (p > 0.7).astype(float)
    bins = reliability_curve(y, p, n_bins=10)
    counts = [b.count for b in bins]
    assert max(counts) - min(counts) <= 1


def test_brier_bounds():
    assert brier_score(np.array([1.0]), np.array([1.0])) == 0.0
    assert brier_score(np.array([1.0]), np.array([0.0])) == 1.0


def test_conformal_threshold_respects_the_miss_budget():
    """The core distribution-free guarantee: at most alpha of hazards fall below it."""
    rng = np.random.default_rng(0)
    scores = rng.beta(5, 2, 500)
    thr = conformal_threshold(scores, alpha=0.1)
    fresh = rng.beta(5, 2, 20000)
    assert (fresh < thr).mean() <= 0.13   # finite-sample slack around the 0.10 budget


def test_conformal_threshold_is_monotone_in_alpha():
    scores = np.linspace(0, 1, 200)
    ts = [conformal_threshold(scores, a) for a in (0.02, 0.05, 0.1, 0.2)]
    assert ts == sorted(ts)


def test_conformal_threshold_spends_nothing_on_tiny_calibration_sets():
    """With 3 calibration hazards you cannot certify a 10% miss rate, so fire on everything."""
    thr = conformal_threshold(np.array([0.4, 0.6, 0.8]), alpha=0.1)
    assert thr <= 0.4


def test_calibrate_alert_threshold_reports_both_error_rates():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=2000) < 0.2).astype(float)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)
    out = calibrate_alert_threshold(y, p, alpha=0.1)
    assert out.empirical_miss_rate <= 0.12
    assert 0.0 <= out.empirical_alarm_rate <= 1.0
    assert "P(miss)" in out.guarantee


def test_full_calibration_pipeline_improves_ece():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1.5, 3000)
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-z))).astype(float)
    p = 1 / (1 + np.exp(-z * 2.5))
    art = calibrate(y, p, alpha=0.1)
    assert art.ece_after <= art.ece_before + 1e-6
    assert 0.0 <= art.threshold <= 1.0


# ------------------------------------------------------------------- head + fusion


def test_empty_head_abstains_rather_than_asserting_safety():
    head = LearnedRiskHead()
    pred = head.predict(np.zeros(WINDOW * D))
    assert pred.fired is False and pred.trusted is False


def test_urgent_geometry_ignores_a_confidently_wrong_head():
    hazard = assess([_track(z_m=1.2, vz_mps=-1.5)], CFG)
    assert hazard, "expected the geometric kernel to flag an imminent hazard"
    pred = RiskPrediction(probability=0.0, epistemic_std=0.0, fired=False, trusted=True)
    fused = fuse(hazard[0], pred, CFG)
    assert fused.source == "geometry"
    assert fused.severity == pytest.approx(hazard[0].severity)


def test_head_can_escalate_a_non_urgent_hazard():
    hazard = assess([_track(z_m=8.0, vz_mps=-2.6)], CFG)
    assert hazard
    pred = RiskPrediction(probability=0.95, epistemic_std=0.02, fired=True, trusted=True)
    fused = fuse(hazard[0], pred, CFG)
    assert fused.source == "head-escalate"
    assert fused.severity >= hazard[0].severity


def test_suppression_requires_agreement_and_confidence():
    hazard = assess([_track(z_m=8.0, vz_mps=-2.6)], CFG)
    assert hazard
    confident = RiskPrediction(probability=0.02, epistemic_std=0.01, fired=False, trusted=True)
    assert fuse(hazard[0], confident, CFG).source == "head-suppress"

    disagreeing = RiskPrediction(probability=0.02, epistemic_std=0.4, fired=False, trusted=False)
    assert fuse(hazard[0], disagreeing, CFG).source == "head-abstain"

    # Confident mean, but the ensemble spread pushes the upper bound above the gate.
    borderline = RiskPrediction(probability=0.10, epistemic_std=0.09, fired=False, trusted=True)
    assert fuse(hazard[0], borderline, CFG).source != "head-suppress"


def test_suppression_never_zeroes_severity():
    hazard = assess([_track(z_m=8.0, vz_mps=-2.6)], CFG)
    pred = RiskPrediction(probability=0.0, epistemic_std=0.0, fired=False, trusted=True)
    assert fuse(hazard[0], pred, CFG).severity > 0.0


def test_confidence_bounds_are_clamped():
    pred = RiskPrediction(probability=0.95, epistemic_std=0.5, fired=True, trusted=False)
    assert pred.upper_confidence_bound == 1.0
    assert pred.lower_confidence_bound == 0.0


# --------------------------------------------------------------------- ablations


def test_oracle_dominates_the_geometric_baseline_on_lead_time():
    data = generate_dataset(120, CFG, seed=11)
    baseline = score_arm("baseline", data, geometric_rule(), CFG)
    oracle = score_arm("oracle", data, oracle_rule(), CFG)
    assert oracle.recall >= baseline.recall
    assert oracle.median_lead_time_s >= baseline.median_lead_time_s


def test_run_ablations_without_a_head_still_produces_two_arms():
    data = generate_dataset(40, CFG, seed=12)
    arms = run_ablations(data, head=None, cfg=CFG)
    assert [a.name for a in arms] == [
        "Constant-velocity TTC (baseline)",
        "Oracle (upper bound)",
    ]


def test_arm_accounting_is_consistent():
    data = generate_dataset(80, CFG, seed=13)
    arm = score_arm("baseline", data, geometric_rule(), CFG)
    assert arm.hazard_clips + arm.safe_clips == len(data)
    assert len(arm.lead_times_s) + arm.misses == arm.hazard_clips
    assert arm.false_alarms <= arm.safe_clips
    assert arm.distance_km > 0.0


def test_lead_times_are_never_negative():
    data = generate_dataset(60, CFG, seed=14)
    for arm in run_ablations(data, head=None, cfg=CFG):
        assert all(lt >= 0.0 for lt in arm.lead_times_s)
