"""Constant-velocity Kalman filter in the 2D ground plane (x, z).

State: [x, z, vx, vz]. Height is discarded deliberately -- for pedestrian navigation the
hazard question is entirely a ground-plane question, and dropping y halves the state.
"""

from __future__ import annotations

import numpy as np


class KalmanCV:
    """Minimal constant-velocity filter. No dependencies beyond numpy for edge determinism."""

    def __init__(
        self,
        x: float,
        z: float,
        process_noise: float = 0.6,
        measurement_noise: float = 0.25,
    ) -> None:
        self.state = np.array([x, z, 0.0, 0.0], dtype=np.float64)
        # Initial velocity is unknown -> large variance on the velocity block.
        self.P = np.diag([0.5, 0.5, 4.0, 4.0])
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def predict(self, dt: float) -> None:
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Discrete white-noise acceleration model.
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        q = self.q
        Q = np.array(
            [
                [dt4 / 4, 0.0, dt3 / 2, 0.0],
                [0.0, dt4 / 4, 0.0, dt3 / 2],
                [dt3 / 2, 0.0, dt2, 0.0],
                [0.0, dt3 / 2, 0.0, dt2],
            ]
        ) * q
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, x: float, z: float) -> None:
        measurement = np.array([x, z], dtype=np.float64)
        R = np.eye(2) * self.r
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        identity = np.eye(4)
        self.P = (identity - K @ self.H) @ self.P

    def ego_compensate(self, ego_speed_mps: float, yaw_rate_rps: float, dt: float) -> None:
        """Remove the user's own motion so velocities are world-relative.

        Without this every static lamp post appears to close at walking pace and the
        false-alarm rate is unusable.
        """
        x, z = self.state[0], self.state[1]
        theta = yaw_rate_rps * dt
        c, s = np.cos(-theta), np.sin(-theta)
        self.state[0] = c * x - s * z
        self.state[1] = s * x + c * z - ego_speed_mps * dt

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    @property
    def trace(self) -> float:
        return float(np.trace(self.P))
