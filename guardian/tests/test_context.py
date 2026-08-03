"""Tests for mode-aware context switching (roadmap item 4)."""

from __future__ import annotations

import pytest

from guardian.config import RiskConfig
from guardian.context import ContextClassifier, Mode, conservatism_rank, profile_for
from guardian.context.classifier import score_tags
from guardian.context.evaluate import evaluate_contexts
from guardian.context.modes import PROFILES, apply_profile

SIDEWALK_TAGS = ["a pavement with shopfronts", "pedestrians"]
CROSSING_TAGS = ["a zebra crossing", "traffic light", "road"]
PLATFORM_TAGS = ["a station platform edge", "yellow line", "train"]


# -------------------------------------------------------------------------- profiles


def test_every_mode_has_a_profile():
    assert set(PROFILES) == set(Mode)


def test_conservatism_ordering():
    assert (conservatism_rank(Mode.TRANSIT_PLATFORM)
            > conservatism_rank(Mode.CROSSING)
            > conservatism_rank(Mode.SIDEWALK))


def test_only_the_platform_profile_has_a_hard_floor():
    assert profile_for(Mode.TRANSIT_PLATFORM).hard_floor_m is not None
    assert profile_for(Mode.SIDEWALK).hard_floor_m is None
    assert profile_for(Mode.CROSSING).hard_floor_m is None


def test_crossing_lengthens_the_warning_horizon():
    assert profile_for(Mode.CROSSING).ttc_warn_s > profile_for(Mode.SIDEWALK).ttc_warn_s


def test_apply_profile_never_shrinks_the_validated_corridor():
    base = RiskConfig()
    for mode in Mode:
        cfg = apply_profile(base, profile_for(mode))
        assert cfg.cone_half_width_m >= base.cone_half_width_m
        assert cfg.ttc_warn_s >= base.ttc_warn_s
        assert cfg.ttc_urgent_s >= base.ttc_urgent_s


def test_a_rogue_profile_that_narrows_the_cone_is_clamped():
    from dataclasses import replace

    rogue = replace(profile_for(Mode.CROSSING), cone_half_width_m=0.05, ttc_warn_s=0.5)
    cfg = apply_profile(RiskConfig(), rogue)
    assert cfg.cone_half_width_m == pytest.approx(RiskConfig().cone_half_width_m)
    assert cfg.ttc_warn_s == pytest.approx(RiskConfig().ttc_warn_s)


def test_apply_profile_leaves_unrelated_fields_alone():
    base = RiskConfig()
    cfg = apply_profile(base, profile_for(Mode.CROSSING))
    assert cfg.horizon_s == base.horizon_s
    assert cfg.ego_radius_m == base.ego_radius_m


# ------------------------------------------------------------------------- scoring


def test_scoring_picks_the_right_mode_for_each_scene():
    for tags, expected in (
        (SIDEWALK_TAGS, Mode.SIDEWALK),
        (CROSSING_TAGS, Mode.CROSSING),
        (PLATFORM_TAGS, Mode.TRANSIT_PLATFORM),
    ):
        scores = score_tags(tags)
        assert max(scores, key=lambda m: scores[m]) is expected


def test_scoring_is_case_insensitive():
    assert score_tags(["PLATFORM EDGE"])[Mode.TRANSIT_PLATFORM] > 0


def test_unrelated_tags_score_nothing():
    scores = score_tags(["a bowl of soup"])
    assert max(scores.values()) == 0.0


# ---------------------------------------------------------------------- switching


def test_starts_on_the_sidewalk_profile():
    assert ContextClassifier().mode is Mode.SIDEWALK


def test_escalation_to_platform_is_immediate():
    c = ContextClassifier(dwell_s=10.0)
    c.observe(PLATFORM_TAGS, 0.0)
    assert c.mode is Mode.TRANSIT_PLATFORM
    assert c.state.escalations == 1


def test_escalation_to_crossing_is_immediate():
    c = ContextClassifier(dwell_s=10.0)
    c.observe(CROSSING_TAGS, 0.0)
    assert c.mode is Mode.CROSSING


def test_relaxation_requires_sustained_agreement():
    c = ContextClassifier(dwell_s=6.0)
    c.observe(PLATFORM_TAGS, 0.0)
    c.observe(SIDEWALK_TAGS, 1.0)
    assert c.mode is Mode.TRANSIT_PLATFORM, "one frame must not drop the platform floor"
    c.observe(SIDEWALK_TAGS, 8.0)
    assert c.mode is Mode.SIDEWALK
    assert c.state.relaxations == 1


def test_a_single_contrary_frame_resets_the_relaxation_timer():
    c = ContextClassifier(dwell_s=6.0)
    c.observe(PLATFORM_TAGS, 0.0)
    c.observe(SIDEWALK_TAGS, 1.0)
    c.observe(PLATFORM_TAGS, 3.0)     # still on the platform
    c.observe(SIDEWALK_TAGS, 5.0)     # timer restarts here
    c.observe(SIDEWALK_TAGS, 8.0)
    assert c.mode is Mode.TRANSIT_PLATFORM


def test_unrecognised_scene_holds_the_current_mode():
    c = ContextClassifier(dwell_s=1.0)
    c.observe(PLATFORM_TAGS, 0.0)
    for t in range(1, 10):
        c.observe(["indistinct blur"], float(t))
    assert c.mode is Mode.TRANSIT_PLATFORM


def test_unrecognised_scene_yields_no_proposal():
    obs = ContextClassifier().observe(["indistinct blur"], 0.0)
    assert obs.proposal is None


def test_ties_resolve_to_the_more_conservative_mode():
    c = ContextClassifier()
    obs = c.observe(["pavement platform"], 0.0)  # 3.0 vs 3.0
    assert obs.proposal is Mode.TRANSIT_PLATFORM


def test_history_records_the_reason_for_each_switch():
    c = ContextClassifier(dwell_s=2.0)
    c.observe(PLATFORM_TAGS, 0.0)
    c.observe(SIDEWALK_TAGS, 1.0)
    c.observe(SIDEWALK_TAGS, 5.0)
    reasons = [h[2] for h in c.state.history]
    assert any("escalate" in r for r in reasons)
    assert any("relax" in r for r in reasons)


def test_repeated_observations_of_the_current_mode_do_not_churn():
    c = ContextClassifier()
    for t in range(10):
        c.observe(SIDEWALK_TAGS, float(t))
    assert c.state.switches == 0


# -------------------------------------------------------------------- the metric


def test_context_evaluation_covers_every_mode():
    res = evaluate_contexts(n_km=20.0, seed=0)
    assert {r["mode"] for r in res["rows"]} == {m.value for m in Mode}


def test_platform_recall_does_not_regress_under_context_switching():
    for seed in (0, 1, 2):
        res = evaluate_contexts(n_km=25.0, seed=seed)
        assert res["platformNotRegressed"], f"platform arm regressed at seed {seed}"


def test_crossing_recall_improves_with_its_own_profile():
    res = evaluate_contexts(n_km=40.0, seed=0)
    crossing = next(r for r in res["rows"] if r["mode"] == Mode.CROSSING.value)
    assert crossing["awareRecall"] >= crossing["flatRecall"]


def test_every_row_carries_its_rationale():
    res = evaluate_contexts(n_km=10.0)
    assert all(len(str(r["rationale"])) > 40 for r in res["rows"])
