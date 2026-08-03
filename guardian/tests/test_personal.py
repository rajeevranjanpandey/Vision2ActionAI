"""Tests for per-user conformal recalibration (roadmap item 2)."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.risk.personal import (
    ALPHA_MAX,
    ALPHA_MIN,
    PersonalConformal,
    PersonalProfile,
    alpha_sweep,
    parse_intent,
)


def make_validation(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(float)
    p = np.where(y == 1, rng.beta(6, 2, n), rng.beta(2, 6, n))
    return y, p


def user_hazard_scores(n=120, seed=1, easy=True):
    rng = np.random.default_rng(seed)
    return rng.beta(8, 2, n) if easy else rng.beta(2, 2, n)


# -------------------------------------------------------------------- intent parsing


@pytest.mark.parametrize("phrase", ["fewer alerts please", "too many warnings",
                                    "be quieter", "reduce the warnings"])
def test_fewer_phrases_raise_alpha(phrase):
    intent = parse_intent(phrase)
    assert intent.recognised and intent.direction == 1


@pytest.mark.parametrize("phrase", ["warn me more", "you missed that one",
                                    "I want more warnings"])
def test_more_phrases_lower_alpha(phrase):
    intent = parse_intent(phrase)
    assert intent.recognised and intent.direction == -1


def test_query_phrase_is_recognised_but_neutral():
    intent = parse_intent("how sensitive are you right now")
    assert intent.recognised and intent.direction == 0


def test_empty_and_ambiguous_utterances_are_not_recognised():
    assert not parse_intent("").recognised
    assert not parse_intent("the weather is nice").recognised
    # Both directions present -> refuse rather than guess.
    assert not parse_intent("fewer alerts but warn me more").recognised


def test_unrecognised_intent_leaves_alpha_untouched():
    pc = PersonalConformal(0.5)
    before = pc.profile.alpha
    pc.nudge("open the door")
    assert pc.profile.alpha == before


# ------------------------------------------------------------------------ the dial


def test_alpha_moves_one_step_per_nudge():
    pc = PersonalConformal(0.5, profile=PersonalProfile(alpha=0.10))
    _, a = pc.nudge("fewer alerts")
    assert a > 0.10


def test_alpha_is_bounded_above():
    pc = PersonalConformal(0.5)
    for _ in range(20):
        pc.nudge("fewer alerts")
    assert pc.profile.alpha == pytest.approx(ALPHA_MAX)


def test_alpha_is_bounded_below():
    pc = PersonalConformal(0.5)
    for _ in range(20):
        pc.nudge("warn me more")
    assert pc.profile.alpha == pytest.approx(ALPHA_MIN)


def test_at_limit_reports_the_edge():
    pc = PersonalConformal(0.5)
    assert pc.at_limit() is None
    for _ in range(20):
        pc.nudge("fewer alerts")
    assert pc.at_limit() == "max"
    for _ in range(20):
        pc.nudge("warn me more")
    assert pc.at_limit() == "min"


def test_dial_is_reversible():
    pc = PersonalConformal(0.5, profile=PersonalProfile(alpha=0.10))
    pc.nudge("fewer alerts")
    pc.nudge("warn me more")
    assert pc.profile.alpha == pytest.approx(0.10)


# ------------------------------------------------------------------ recalibration


def test_too_few_user_hazards_falls_back_to_the_global_threshold():
    y, p = make_validation()
    pc = PersonalConformal(0.5, min_user_hazards=40)
    cal = pc.recalibrate(user_hazard_scores(n=5), y, p)
    assert not cal.accepted and cal.clamped
    assert cal.threshold == pytest.approx(0.5)
    assert "logged hazards" in cal.reason


def test_a_clearing_personalisation_is_accepted():
    y, p = make_validation()
    pc = PersonalConformal(0.05, min_recall=0.75)
    cal = pc.recalibrate(user_hazard_scores(n=200), y, p)
    assert cal.accepted and not cal.clamped
    assert cal.validation_recall >= 0.75


def test_recall_floor_rejects_a_threshold_that_would_miss_hazards():
    y, p = make_validation()
    # A user whose own hazards all scored very high produces a high threshold; demanding
    # near-perfect recall on the frozen split must reject it.
    pc = PersonalConformal(0.2, min_recall=0.999, profile=PersonalProfile(alpha=ALPHA_MAX))
    cal = pc.recalibrate(user_hazard_scores(n=200, easy=True), y, p)
    assert not cal.accepted
    assert cal.threshold == pytest.approx(0.2)
    assert "floor" in cal.reason


def test_rejected_updates_are_counted():
    y, p = make_validation()
    pc = PersonalConformal(0.5, min_recall=0.999)
    pc.recalibrate(user_hazard_scores(n=200), y, p)
    assert pc.profile.rejected_updates == 1
    assert pc.profile.accepted_updates == 0


def test_accepted_updates_are_recorded_in_history():
    y, p = make_validation()
    pc = PersonalConformal(0.05, min_recall=0.5)
    pc.recalibrate(user_hazard_scores(n=200), y, p)
    assert pc.profile.accepted_updates == 1
    assert len(pc.profile.history) == 1


def test_alpha_outside_the_band_is_clipped_before_use():
    y, p = make_validation()
    pc = PersonalConformal(0.5, min_recall=0.0, profile=PersonalProfile(alpha=0.9))
    cal = pc.recalibrate(user_hazard_scores(n=200), y, p)
    assert ALPHA_MIN <= cal.alpha <= ALPHA_MAX


def test_higher_alpha_gives_a_higher_threshold():
    y, p = make_validation()
    scores = user_hazard_scores(n=300)
    low = PersonalConformal(0.5, min_recall=0.0, profile=PersonalProfile(alpha=ALPHA_MIN))
    high = PersonalConformal(0.5, min_recall=0.0, profile=PersonalProfile(alpha=ALPHA_MAX))
    assert (high.recalibrate(scores, y, p).threshold
            >= low.recalibrate(scores, y, p).threshold)


def test_calibration_serialises_for_the_web_payload():
    y, p = make_validation()
    pc = PersonalConformal(0.5, min_recall=0.0)
    d = pc.recalibrate(user_hazard_scores(n=200), y, p).to_dict()
    assert set(d) >= {"alpha", "threshold", "validationRecall", "accepted", "reason"}


# ----------------------------------------------------------------------- the sweep


def test_alpha_sweep_covers_the_whole_band():
    y, p = make_validation()
    rows = alpha_sweep(user_hazard_scores(n=200), y, p, 0.5, min_recall=0.0)
    assert [r["alpha"] for r in rows] == [0.05, 0.075, 0.10, 0.125, 0.15]


def test_alpha_sweep_recall_is_monotone_non_increasing():
    y, p = make_validation()
    rows = alpha_sweep(user_hazard_scores(n=300), y, p, 0.5, min_recall=0.0)
    recalls = [r["recall"] for r in rows]
    assert all(a >= b - 1e-9 for a, b in zip(recalls, recalls[1:]))


def test_alpha_sweep_false_alarms_fall_as_alpha_rises():
    y, p = make_validation()
    rows = alpha_sweep(user_hazard_scores(n=300), y, p, 0.5, min_recall=0.0)
    fas = [r["falseAlarmRate"] for r in rows]
    assert fas[0] >= fas[-1]
