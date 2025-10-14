
# ekf.py
# Extended Kalman Filter for rigid-body + reaction wheels (3-axis), diagonal inertia.
# Key points (as requested):
#   1) 6D measurement: z = [omega; rw_speed]  (spacecraft angular rates and wheel speeds)
#   2) Joseph-form covariance update for numerical stability
# The rest is intentionally simple and minimal.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class EKFConfig:
    dt: float
    I_rw_diag: np.ndarray            # (3,) wheel inertias along axes
    Qc_diag: np.ndarray              # (9,) continuous-time process noise diagonal for [omega(3), I(3), rw(3)]
    R_diag: np.ndarray               # (6,) measurement noise diagonal for [omega(3), rw(3)]
    x0: np.ndarray                   # (9,) initial state [omega(3), I(3), rw_speed(3)]
    P0: np.ndarray                   # (9,9) initial covariance

class EKFInertiaRW:
    """
    State: x = [omega(3), I(3), rw_speed(3)]
    Input: u = wheel accelerations (3) [rad/s^2]
    Measurement: z = [omega_meas(3), rw_speed_meas(3)]

    Dynamics (internal torque only, idealized):
      h_rw      = I_rw * rw_speed
      h_rw_dot  = I_rw * u
      tau_rw    = - h_rw_dot                          (equal & opposite on body)
      omega_dot = I^{-1} ( tau_ext + tau_rw - omega x (I*omega + h_rw) )
      I_dot     = 0                                   (modeled as random walk via Q)
      rw_dot    = u
    """
    def __init__(self, cfg: EKFConfig):
        self.dt = float(cfg.dt)
        self.I_rw = np.asarray(cfg.I_rw_diag, dtype=np.float64).reshape(3,)
        self.x = np.asarray(cfg.x0, dtype=np.float64).reshape(9,)
        self.P = np.array(cfg.P0, dtype=np.float64, copy=True)
        self.Qc = np.asarray(cfg.Qc_diag, dtype=np.float64).reshape(9,)
        self.R = np.diag(np.asarray(cfg.R_diag, dtype=np.float64).reshape(6,))

        # Precompute measurement matrix (linear)
        self.H = np.zeros((6, 9), dtype=np.float64)
        self.H[0:3, 0:3] = np.eye(3)   # omega
        self.H[3:6, 6:9] = np.eye(3)   # rw speeds

        # Small floors to keep physical parameters in range
        self._I_floor = 1e-6

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[0, -z,  y],
                         [z,  0, -x],
                         [-y, x,  0]], dtype=np.float64)

    def f(self, x: np.ndarray, u: np.ndarray, tau_ext: np.ndarray | None = None) -> np.ndarray:
        """ Continuous-time dynamics xdot = f(x,u) """
        omega = x[0:3]
        I     = x[3:6]
        rw    = x[6:9]

        I = np.maximum(I, self._I_floor)  # keep positive
        Is = I
        Irw = self.I_rw

        if tau_ext is None:
            tau_ext = np.zeros(3, dtype=np.float64)

        h_rw     = Irw * rw              # wheel angular momentum in body axes
        h_rw_dot = Irw * u               # time derivative
        tau_rw   = - h_rw_dot            # reaction torque on body

        M = Is * omega + h_rw            # total angular momentum in body frame
        omega_dot = (tau_ext + tau_rw - np.cross(omega, M)) / Is

        I_dot  = np.zeros(3, dtype=np.float64)  # modeled as constant (random walk via Q)
        rw_dot = u

        xdot = np.zeros_like(x)
        xdot[0:3] = omega_dot
        xdot[3:6] = I_dot
        xdot[6:9] = rw_dot
        return xdot

    def _numeric_F(self, x: np.ndarray, u: np.ndarray, tau_ext: np.ndarray | None) -> np.ndarray:
        """ Discrete-time Jacobian via first-order Euler: x_{k+1} = x_k + dt * f(x,u) """
        n = x.size
        F = np.eye(n, dtype=np.float64)
        # State-wise relative eps to handle scaling (omega ~1e-2, I ~ 1-10, rw ~ up to 1e3)
        eps = 1e-6
        for i in range(n):
            dx = np.zeros(n, dtype=np.float64)
            scale = max(abs(x[i]), 1.0)
            dx[i] = eps * scale
            f_plus  = self.f(x + dx, u, tau_ext)
            f_minus = self.f(x - dx, u, tau_ext)
            dfdxi = (f_plus - f_minus) / (2.0 * dx[i])
            F[:, i] += self.dt * dfdxi
        return F

    def predict(self, u: np.ndarray, tau_ext: np.ndarray | None = None):
        """ Discrete-time EKF prediction with Euler integration and Qd = Qc * dt """
        x = self.x
        # State propagation
        xdot = self.f(x, u, tau_ext)
        x_pred = x + self.dt * xdot

        # Jacobian and covariance propagation
        F = self._numeric_F(x, u, tau_ext)
        Qd = np.diag(self.Qc * self.dt)
        P_pred = F @ self.P @ F.T + Qd
        # Enforce symmetry
        self.P = 0.5 * (P_pred + P_pred.T)
        self.x = x_pred
        # Keep physical
        self.x[3:6] = np.maximum(self.x[3:6], self._I_floor)

    def update(self, z: np.ndarray):
        """ Measurement update with Joseph-form covariance protection.
            z = [omega_meas(3), rw_speed_meas(3)]
        """
        H = self.H
        R = self.R
        I9 = np.eye(9, dtype=np.float64)

        z_pred = H @ self.x
        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[3:6] = np.maximum(self.x[3:6], self._I_floor)

        # Joseph form for numerical stability
        I_KH = I9 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

    def step(self, u: np.ndarray, z: np.ndarray, tau_ext: np.ndarray | None = None):
        self.predict(u, tau_ext)
        self.update(z)
        return self.x.copy(), self.P.copy()
