"""Tests for the risk kernel.

These are the numbers a reviewer will interrogate. A tracking bug degrades the demo; a
TTC bug injures someone.
"""

from __future__ import annotations

import math

import pytest

from guardian.config import RiskConfig
from guardian.risk import ttc as risk
from guardian.types import Direction, Track


def make_track(x=0.0, z=6.0, vx=0.0, vz=-2.0, width=0.6, cls="person", hits=5) -> Track:
    return Track(
        track_id=1, x_m=x, z_m=z, vx_mps=vx, vz_mps=vz,
        width_m=width, class_name=cls, hits=hits,
    )


class TestCone:
    def test_widens_with_distance(self):
        cfg = RiskConfig()
        assert risk.cone_half_width_at(10.0, cfg) > risk.cone_half_width_at(2.0, cfg)

    def test_object_dead_ahead_is_inside(self):
        assert risk.in_risk_cone(make_track(x=0.0, z=5.0), RiskConfig())

    def test_far_lateral_object_is_outside(self):
        assert not risk.in_risk_cone(make_track(x=6.0, z=5.0), RiskConfig())

    def test_object_behind_is_outside(self):
        assert not risk.in_risk_cone(make_track(x=0.0, z=-3.0), RiskConfig())


class TestTimeToCollision:
    def test_head_on_closing(self):
        # 6 m out, closing at 2 m/s, ego radius 1 m -> breach at (6-1)/2 = 2.5 s
        t = risk.time_to_collision(make_track(z=6.0, vz=-2.0), RiskConfig())
        assert t == pytest.approx(2.5, abs=0.2)

    def test_receding_object_never_collides(self):
        assert math.isinf(risk.time_to_collision(make_track(z=6.0, vz=+2.0), RiskConfig()))

    def test_stationary_object_while_ego_still(self):
        assert math.isinf(risk.time_to_collision(make_track(z=6.0, vz=0.0), RiskConfig()))

    def test_faster_closing_gives_shorter_ttc(self):
        cfg = RiskConfig()
        slow = risk.time_to_collision(make_track(z=8.0, vz=-1.0), cfg)
        fast = risk.time_to_collision(make_track(z=8.0, vz=-4.0), cfg)
        assert fast < slow

    def test_crossing_object_that_misses_is_not_a_collision(self):
        # Moving hard right, never enters the 1 m cylinder.
        t = risk.time_to_collision(make_track(x=3.0, z=5.0, vx=4.0, vz=-0.5), RiskConfig())
        assert math.isinf(t)


class TestClosestApproach:
    def test_zero_for_direct_hit(self):
        d, _ = risk.closest_approach(make_track(x=0.0, z=6.0, vz=-2.0), RiskConfig())
        assert d < 0.3

    def test_lateral_pass_keeps_distance(self):
        d, t = risk.closest_approach(make_track(x=2.5, z=6.0, vx=0.0, vz=-2.0), RiskConfig())
        assert d == pytest.approx(2.5, abs=0.3)
        assert t > 0.0


class TestDirection:
    @pytest.mark.parametrize(
        "x,expected",
        [(-2.5, Direction.LEFT), (0.0, Direction.CENTRE), (2.5, Direction.RIGHT)],
    )
    def test_lateral_offset_maps_to_direction(self, x, expected):
        assert risk.direction_of(make_track(x=x, z=5.0), RiskConfig()) is expected


class TestAssess:
    def test_returns_hazard_for_closing_track(self):
        hazards = risk.assess([make_track(z=5.0, vz=-2.5)], RiskConfig())
        assert len(hazards) == 1
        assert hazards[0].ttc_s < 3.0

    def test_ignores_unconfirmed_tracks(self):
        assert risk.assess([make_track(hits=1)], RiskConfig()) == []

    def test_sorted_by_severity(self):
        near = make_track(z=2.5, vz=-3.0)
        near.track_id = 2
        far = make_track(z=7.0, vz=-2.5)   # inside the warn horizon, but less urgent
        hazards = risk.assess([far, near], RiskConfig())
        assert len(hazards) >= 2
        assert hazards[0].severity >= hazards[1].severity
        assert hazards[0].track.track_id == 2


    def test_receding_crowd_produces_no_hazards(self):
        tracks = [make_track(x=i - 1.0, z=4.0 + i, vz=+1.5) for i in range(3)]
        for i, t in enumerate(tracks):
            t.track_id = i
        assert risk.assess(tracks, RiskConfig()) == []
