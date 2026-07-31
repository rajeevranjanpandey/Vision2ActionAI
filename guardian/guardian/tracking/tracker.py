"""3D multi-object tracker over ego-frame observations.

Association is nearest-neighbour in metres with a gate, solved optimally by the Hungarian
algorithm. In the ground plane with a 1.2 m gate this is effectively unambiguous, and it
costs microseconds -- unlike appearance re-ID, which we cannot afford in the fast path.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackingConfig
from ..types import Observation, Track
from .kalman import KalmanCV


class _TrackState:
    __slots__ = ("track_id", "kf", "class_name", "width_m", "hits", "age", "misses")

    def __init__(self, track_id: int, obs: Observation, cfg: TrackingConfig) -> None:
        self.track_id = track_id
        self.kf = KalmanCV(obs.x_m, obs.z_m, cfg.process_noise, cfg.measurement_noise)
        self.class_name = obs.class_name
        self.width_m = obs.width_m
        self.hits = 1
        self.age = 0
        self.misses = 0

    def to_track(self) -> Track:
        x, z = self.kf.position
        vx, vz = self.kf.velocity
        return Track(
            track_id=self.track_id,
            x_m=x,
            z_m=z,
            vx_mps=vx,
            vz_mps=vz,
            width_m=self.width_m,
            class_name=self.class_name,
            hits=self.hits,
            age=self.age,
            time_since_update=self.misses,
            covariance_trace=self.kf.trace,
        )


class Tracker:
    """Lifecycle-managed 3D tracker producing velocity estimates for the forecaster."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def step(
        self,
        observations: list[Observation],
        dt: float,
        ego_speed_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
    ) -> list[Track]:
        dt = max(dt, 1e-3)

        for t in self._tracks:
            t.kf.predict(dt)
            t.kf.ego_compensate(ego_speed_mps, yaw_rate_rps, dt)
            t.age += 1

        matches, unmatched_obs, unmatched_tracks = self._associate(observations)

        for track_idx, obs_idx in matches:
            t = self._tracks[track_idx]
            obs = observations[obs_idx]
            t.kf.update(obs.x_m, obs.z_m)
            # Slew the width estimate; single-frame box widths are noisy.
            t.width_m = 0.8 * t.width_m + 0.2 * obs.width_m
            t.class_name = obs.class_name
            t.hits += 1
            t.misses = 0

        for idx in unmatched_tracks:
            self._tracks[idx].misses += 1

        for idx in unmatched_obs:
            self._tracks.append(_TrackState(self._next_id, observations[idx], self.cfg))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.cfg.max_age_frames]
        return [t.to_track() for t in self._tracks if t.hits >= self.cfg.min_hits or t.misses == 0]

    # ----------------------------------------------------------------- private

    def _associate(
        self, observations: list[Observation]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not observations:
            return [], list(range(len(observations))), list(range(len(self._tracks)))

        cost = np.zeros((len(self._tracks), len(observations)), dtype=np.float64)
        for i, t in enumerate(self._tracks):
            tx, tz = t.kf.position
            for j, obs in enumerate(observations):
                d = float(np.hypot(tx - obs.x_m, tz - obs.z_m))
                # Class mismatch is a soft penalty, not a hard block: detectors flip
                # labels (truck/bus) far more often than objects teleport.
                penalty = 0.0 if t.class_name == obs.class_name else 0.4
                cost[i, j] = d + penalty

        gate = self.cfg.association_radius_m
        cost[cost > gate] = 1e6
        rows, cols = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e6:
                continue
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_obs.add(int(c))

        unmatched_obs = [j for j in range(len(observations)) if j not in matched_obs]
        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in matched_tracks]
        return matches, unmatched_obs, unmatched_tracks
