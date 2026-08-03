"""Tests for persistent hazard memory (roadmap item 1)."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.memory import (
    HazardMemory,
    HazardObservation,
    RepeatAlertGate,
    haversine_m,
    simulate_repeat_route,
)
from guardian.memory.hazard_map import (
    FINGERPRINT_DIM,
    MemoryParams,
    StaticHazard,
    angular_delta_deg,
    bearing_deg,
    normalise,
)

BASE_LAT, BASE_LON = 51.5074, -0.1278


def fp(seed: int, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=FINGERPRINT_DIM)
    if jitter:
        v = v + np.random.default_rng(seed + 77).normal(0.0, jitter, FINGERPRINT_DIM)
    return normalise(v)


def obs(lat_off_m=0.0, lon_off_m=0.0, seed=1, jitter=0.0, cls="pothole", t=0.0):
    return HazardObservation(
        lat=BASE_LAT + lat_off_m / 111_320.0,
        lon=BASE_LON + lon_off_m / (111_320.0 * np.cos(np.radians(BASE_LAT))),
        class_name=cls,
        fingerprint=fp(seed, jitter),
        t_s=t,
    )


# ------------------------------------------------------------------------ geometry


def test_haversine_matches_known_distance():
    d = haversine_m(BASE_LAT, BASE_LON, BASE_LAT + 1.0 / 111.32e3 * 100, BASE_LON)
    assert 99.0 < d < 101.0


def test_haversine_is_symmetric_and_zero_on_self():
    assert haversine_m(BASE_LAT, BASE_LON, BASE_LAT, BASE_LON) == pytest.approx(0.0)
    a = haversine_m(51.5, -0.1, 51.6, -0.2)
    b = haversine_m(51.6, -0.2, 51.5, -0.1)
    assert a == pytest.approx(b)


def test_bearing_north_is_zero_and_east_is_ninety():
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-6)


def test_angular_delta_wraps():
    assert angular_delta_deg(350.0, 10.0) == pytest.approx(20.0)
    assert angular_delta_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angular_delta_deg(0.0, 180.0) == pytest.approx(180.0)


def test_normalise_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        normalise([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------- association


def test_repeat_sighting_merges_into_one_pin():
    mem = HazardMemory()
    mem.observe(obs(seed=1))
    mem.observe(obs(lat_off_m=6.0, seed=1, jitter=0.1, t=100.0))
    assert len(mem) == 1
    assert mem.hazards[0].confirmations == 2


def test_distant_sighting_creates_a_new_pin():
    mem = HazardMemory()
    mem.observe(obs(seed=1))
    mem.observe(obs(lat_off_m=200.0, seed=1))
    assert len(mem) == 2


def test_different_appearance_does_not_merge_even_when_colocated():
    mem = HazardMemory()
    mem.observe(obs(seed=1))
    mem.observe(obs(lat_off_m=1.0, seed=999))
    assert len(mem) == 2


def test_different_class_does_not_merge():
    mem = HazardMemory()
    mem.observe(obs(seed=1, cls="pothole"))
    mem.observe(obs(lat_off_m=1.0, seed=1, cls="a_board"))
    assert len(mem) == 2


def test_merged_position_moves_toward_the_more_accurate_fix():
    mem = HazardMemory()
    mem.observe(
        HazardObservation(BASE_LAT, BASE_LON, "pothole", fp(1), gps_accuracy_m=20.0)
    )
    far = BASE_LAT + 8.0 / 111_320.0
    mem.observe(
        HazardObservation(far, BASE_LON, "pothole", fp(1, 0.05), gps_accuracy_m=2.0)
    )
    h = mem.hazards[0]
    # The 2 m fix should dominate the 20 m one.
    assert abs(h.lat - far) < abs(h.lat - BASE_LAT)


# ----------------------------------------------------------------------- confidence


def test_single_sighting_is_not_established():
    mem = HazardMemory()
    h = mem.observe(obs(seed=1))
    assert not h.established
    assert h.confidence < 0.6


def test_three_confirmations_establish_a_pin():
    mem = HazardMemory()
    for i in range(3):
        mem.observe(obs(lat_off_m=i * 2.0, seed=1, jitter=0.05, t=i * 100.0))
    assert mem.hazards[0].established


def test_contradiction_lowers_confidence_and_can_unestablish():
    mem = HazardMemory()
    for i in range(3):
        mem.observe(obs(lat_off_m=i, seed=1, jitter=0.05, t=i * 10.0))
    h = mem.hazards[0]
    before = h.confidence
    for _ in range(5):
        mem.contradict(h.id)
    assert h.confidence < before
    assert not h.established


def test_prune_removes_discredited_pins():
    mem = HazardMemory()
    h = mem.observe(obs(seed=1))
    for _ in range(20):
        mem.contradict(h.id)
    assert mem.prune(now_s=1.0) == 1
    assert len(mem) == 0


def test_prune_removes_stale_pins():
    mem = HazardMemory(MemoryParams(forget_after_s=10.0))
    for i in range(3):
        mem.observe(obs(lat_off_m=i, seed=1, jitter=0.05, t=0.0))
    assert mem.prune(now_s=1000.0) == 1


# --------------------------------------------------------------------------- query


def test_query_returns_only_hazards_ahead():
    mem = HazardMemory()
    mem.observe(obs(lat_off_m=10.0, seed=1))       # due north
    ahead = mem.query(BASE_LAT, BASE_LON, heading_deg=0.0)
    behind = mem.query(BASE_LAT, BASE_LON, heading_deg=180.0)
    assert len(ahead) == 1
    assert behind == []


def test_query_respects_lookahead_range():
    mem = HazardMemory(MemoryParams(lookahead_m=15.0))
    mem.observe(obs(lat_off_m=40.0, seed=1))
    assert mem.query(BASE_LAT, BASE_LON, 0.0) == []


def test_prior_grows_as_the_walker_closes_in():
    mem = HazardMemory()
    for i in range(4):
        mem.observe(obs(lat_off_m=20.0, seed=1, jitter=0.02, t=i * 100.0))
    far = mem.query(BASE_LAT, BASE_LON, 0.0)[0].prior
    near_lat = BASE_LAT + 15.0 / 111_320.0
    near = mem.query(near_lat, BASE_LON, 0.0)[0].prior
    assert near > far


def test_query_is_sorted_nearest_first():
    mem = HazardMemory()
    mem.observe(obs(lat_off_m=18.0, seed=2, cls="bollard"))
    mem.observe(obs(lat_off_m=6.0, seed=3, cls="a_board"))
    hits = mem.query(BASE_LAT, BASE_LON, 0.0)
    assert [round(h.distance_m) for h in hits] == sorted(
        round(h.distance_m) for h in hits
    )


# --------------------------------------------------------------------------- cache


def test_round_trips_through_the_local_cache():
    mem = HazardMemory()
    for i in range(3):
        mem.observe(obs(lat_off_m=i * 3.0, seed=1, jitter=0.05, t=i * 50.0))
    mem.observe(obs(lat_off_m=60.0, seed=5, cls="kerb_drop"))

    restored = HazardMemory.from_dict(mem.to_dict())
    assert len(restored) == len(mem)
    a, b = mem.hazards[0], restored.hazards[0]
    assert a.id == b.id and a.confirmations == b.confirmations
    assert b.lat == pytest.approx(a.lat)
    assert np.allclose(a.fingerprint, b.fingerprint, atol=1e-5)


def test_restored_cache_keeps_allocating_fresh_ids():
    mem = HazardMemory()
    mem.observe(obs(seed=1))
    restored = HazardMemory.from_dict(mem.to_dict())
    new = restored.observe(obs(lat_off_m=300.0, seed=7))
    assert new.id != restored.hazards[0].id


def test_serialised_pin_contains_no_imagery():
    mem = HazardMemory()
    mem.observe(obs(seed=1))
    blob = repr(mem.to_dict()).lower()
    assert "image" not in blob and "jpeg" not in blob and "base64" not in blob


def test_from_dict_handles_an_empty_cache():
    assert len(HazardMemory.from_dict({})) == 0


# ---------------------------------------------------------------------- the gate


def test_gate_never_touches_urgent_geometry():
    gate = RepeatAlertGate()
    d = gate.decide("h0001", prior=0.99, ttc_s=1.0, established=True, t_s=0.0)
    assert d.announce and d.style == "full"


def test_gate_announces_unknown_hazards_in_full():
    gate = RepeatAlertGate()
    d = gate.decide(None, prior=0.0, ttc_s=3.0, established=False, t_s=0.0)
    assert d.announce and d.style == "full"


def test_gate_announces_the_first_encounter_then_downgrades():
    gate = RepeatAlertGate()
    first = gate.decide("h0001", 0.8, 3.0, True, 0.0)
    second = gate.decide("h0001", 0.8, 2.6, True, 0.7)
    assert first.announce
    assert not second.announce and second.style == "known"


def test_gate_re_announces_after_the_timeout():
    gate = RepeatAlertGate(re_announce_after_s=10.0)
    gate.decide("h0001", 0.8, 3.0, True, 0.0)
    later = gate.decide("h0001", 0.8, 3.0, True, 30.0)
    assert later.announce


def test_gate_ignores_unestablished_pins():
    gate = RepeatAlertGate()
    gate.decide("h0001", 0.8, 3.0, False, 0.0)
    again = gate.decide("h0001", 0.8, 2.8, False, 0.5)
    assert again.announce


def test_gate_reset_restores_full_announcement():
    gate = RepeatAlertGate()
    gate.decide("h0001", 0.8, 3.0, True, 0.0)
    gate.reset_pass()
    assert gate.decide("h0001", 0.8, 3.0, True, 0.5).announce


# ------------------------------------------------------------------- route metric


def test_repeat_route_reduces_alerts_after_the_first_pass():
    res = simulate_repeat_route(seed=3)
    assert res["repeat_pass_per_km"] < res["first_pass_per_km"]
    assert res["reduction_vs_first_pass"] > 0.05


def test_repeat_route_never_drops_an_urgent_alert():
    for seed in (0, 1, 2, 3, 4):
        res = simulate_repeat_route(seed=seed)
        assert res["urgent_preserved"], f"urgent alert suppressed at seed {seed}"


def test_pin_count_saturates_at_the_number_of_real_hazards():
    res = simulate_repeat_route(n_static=7, n_passes=6, seed=1)
    rows = res["rows"]
    assert rows[-1]["pins"] <= 7 * 3  # some duplication under GPS noise, but bounded
    assert rows[-1]["established_pins"] >= 1


def test_static_hazard_from_dict_defaults():
    h = StaticHazard.from_dict(
        {"id": "h0009", "lat": 1.0, "lon": 2.0, "class_name": "kerb_drop",
         "fingerprint": list(fp(4))}
    )
    assert h.confirmations == 1 and h.contradictions == 0
