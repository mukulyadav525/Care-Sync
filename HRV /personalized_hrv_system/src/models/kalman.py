"""1D Kalman filter used as (a) a cold-start HR baseline tracker and (b) an
online tracker of residual drift for anomaly detection (DESIGN.md sections 5, 7.4)."""
from __future__ import annotations

import numpy as np


class KalmanBaselineTracker:
    """Tracks a slowly-varying scalar (e.g. resting HR, or forecast residual) with
    a constant-velocity model: state = [level, trend].

    x_k = F x_{k-1} + w,  w ~ N(0, Q)
    z_k = H x_k + v,      v ~ N(0, R)
    """

    def __init__(self, process_var: float = 1e-4, measurement_var: float = 1.0, dt: float = 1.0):
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = process_var * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.R = np.array([[measurement_var]])

        self.x = np.array([[0.0], [0.0]])  # state: [level, trend]
        self.P = np.eye(2) * 1e3
        self.initialized = False

    def update(self, z: float) -> tuple[float, float, float]:
        """Process one measurement `z`. Returns (level_estimate, predicted_next, innovation_z)."""
        if not self.initialized:
            self.x[0, 0] = z
            self.initialized = True

        # predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # innovation
        z_arr = np.array([[z]])
        y = z_arr - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        innovation_z = float(y[0, 0] / np.sqrt(S[0, 0]))

        # update
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        level = float(self.x[0, 0])
        predicted_next = float((self.F @ self.x)[0, 0])
        return level, predicted_next, innovation_z
