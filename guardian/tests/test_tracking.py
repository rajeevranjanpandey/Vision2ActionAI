"""Tests for tracking and metric back-projection."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.config import CameraConfig, TrackingConfig
from guardian.perception.backprojection import backproject
from guardian.tracking.kalman import KalmanCV
from guardian.tracking.tracker import Tracker
from guardian.types import Detection, Observation


def make_obs(x, z, name="person") -> Observation:
    det = Detection(xyxy=(100.0, 100.0, 180.0, 400.0), score=0.9, class_id=0, class_name=name)
    return Observation(x_m=x, z_m=z, width_m=0.6, class_name=name, score=0.9, detection=det)


class TestKalman:
    def test_converges_to_constant_velocity(self):
        kf = KalmanCV(x=0.0, z=10.0, process_noise=0.5, measurement_noise=0.1)
        for step in range(1, 25):
            kf.predict(0.1)
            kf.update(0.0, 10.0 - 0.2 * step)   # 2 m/s approach
        _, vz = kf.velocity
        assert vz == pytest.approx(-2.0, abs=0.35)

    def test_ego_compensation_shifts_forward_objects(self):
        kf = KalmanCV(x=0.0, z=10.0, process_noise=0.5, measurement_noise=0.1)
        kf.ego_compensate(ego_speed_mps=1.5, yaw_rate_rps=0.0, dt=1.0)
        _, z = kf.position
        assert z < 10.0, "walking forward must reduce range to a static object"

    def test_covariance_grows_without_updates(self):
        kf = KalmanCV(x=0.0, z=5.0, process_noise=0.5, measurement_noise=0.1)
        before = kf.trace
        for _ in range(5):
            kf.predict(0.1)
        assert kf.trace > before


class TestTracker:
    def test_identity_is_stable_across_frames(self):
        tracker = Tracker(TrackingConfig(min_hits=1))
        ids = []
        for step in range(6):
            tracks = tracker.step([make_obs(0.0, 8.0 - 0.4 * step)], dt=0.1)
            ids.append(tracks[0].track_id)
        assert len(set(ids)) == 1

    def test_two_objects_get_two_ids(self):
        tracker = Tracker(TrackingConfig(min_hits=1))
        tracks = tracker.step([make_obs(-2.0, 6.0), make_obs(2.0, 6.0)], dt=0.1)
        assert len({t.track_id for t in tracks}) == 2

    def test_lost_track_is_dropped(self):
        tracker = Tracker(TrackingConfig(min_hits=1, max_age_frames=3))
        tracker.step([make_obs(0.0, 6.0)], dt=0.1)
        for _ in range(6):
            tracks = tracker.step([], dt=0.1)
        assert tracks == []

    def test_velocity_sign_is_negative_when_approaching(self):
        tracker = Tracker(TrackingConfig(min_hits=1))
        for step in range(12):
            tracks = tracker.step([make_obs(0.0, 9.0 - 0.25 * step)], dt=0.1)
        assert tracks[0].vz_mps < 0.0


class TestBackprojection:
    def test_centred_detection_has_zero_lateral_offset(self):
        cam = CameraConfig()
        det = Detection(
            xyxy=(cam.cx - 40, 300.0, cam.cx + 40, 500.0),
            score=0.9, class_id=0, class_name="person",
        )
        obs = backproject(det, range_m=5.0, camera=cam)
        assert obs.x_m == pytest.approx(0.0, abs=0.05)
        assert obs.z_m == pytest.approx(5.0, abs=0.05)

    def test_right_of_centre_is_positive_x(self):
        cam = CameraConfig()
        det = Detection(
            xyxy=(cam.cx + 200, 300.0, cam.cx + 300, 500.0),
            score=0.9, class_id=0, class_name="person",
        )
        assert backproject(det, range_m=5.0, camera=cam).x_m > 0.0

    def test_physical_width_scales_with_range(self):
        cam = CameraConfig()
        det = Detection(xyxy=(600.0, 300.0, 700.0, 500.0), score=0.9, class_id=0, class_name="car")
        near = backproject(det, range_m=3.0, camera=cam).width_m
        far = backproject(det, range_m=9.0, camera=cam).width_m
        assert far == pytest.approx(3.0 * near, rel=0.05)
