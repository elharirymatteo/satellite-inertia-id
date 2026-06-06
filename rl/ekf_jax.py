"""JAX EKF, augmented to jointly estimate inertia + external body torque.

State: x = [omega(3), I_diag(3), I_offdiag(3), rw(3), tau_ext(3)]   (15-dim)
  x[0:3]   = omega                       (angular velocity)
  x[3:6]   = (Ixx, Iyy, Izz)             (inertia diagonal)
  x[6:9]   = (Ixy, Ixz, Iyz)             (inertia off-diagonal)
  x[9:12]  = Omega_rw                    (reaction wheel speeds)
  x[12:15] = tau_ext                     (body-frame disturbance, near-constant)
Measurement: z = [omega_meas(3), rw_meas(3)]            (6-dim, H linear)

The tau_ext augmentation lets the EKF attribute persistent omega innovations
to an unmodeled torque rather than to inertia error. Without it, the two
are observationally indistinguishable and the inertia estimate diverges
under any disturbance.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


I_DIAG_FLOOR = 1e-6
STATE_DIM = 15


def _skew(v):
    return jnp.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def inertia_from_state(x):
    """Build the 3x3 symmetric I from EKF state x[3:9]."""
    Ixx, Iyy, Izz = x[3], x[4], x[5]
    Ixy, Ixz, Iyz = x[6], x[7], x[8]
    return jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])


_inertia_from_state = inertia_from_state


def tau_ext_from_state(x):
    """Read the EKF's estimate of the body-torque disturbance from x[12:15]."""
    return x[12:15]


def _floor_I_diag(x):
    x = x.at[3].set(jnp.maximum(x[3], I_DIAG_FLOOR))
    x = x.at[4].set(jnp.maximum(x[4], I_DIAG_FLOOR))
    x = x.at[5].set(jnp.maximum(x[5], I_DIAG_FLOOR))
    return x


def _project_psd_if_needed(x):
    """Cauchy-Schwarz off-diag clip + eigval floor on I. stop_gradient on
    eigh (NaN grads at degenerate eigs; this is a forward-only correction)."""
    MAX_OFFDIAG_RATIO = 0.4
    MIN_COND_RATIO = 0.02

    I_diag = jnp.maximum(x[3:6], I_DIAG_FLOOR)
    Ixx, Iyy, Izz = I_diag[0], I_diag[1], I_diag[2]
    x = x.at[6].set(jnp.clip(x[6], -MAX_OFFDIAG_RATIO * jnp.sqrt(Ixx * Iyy),
                                   MAX_OFFDIAG_RATIO * jnp.sqrt(Ixx * Iyy)))
    x = x.at[7].set(jnp.clip(x[7], -MAX_OFFDIAG_RATIO * jnp.sqrt(Ixx * Izz),
                                   MAX_OFFDIAG_RATIO * jnp.sqrt(Ixx * Izz)))
    x = x.at[8].set(jnp.clip(x[8], -MAX_OFFDIAG_RATIO * jnp.sqrt(Iyy * Izz),
                                   MAX_OFFDIAG_RATIO * jnp.sqrt(Iyy * Izz)))

    I = _inertia_from_state(x)
    eigs, V = jnp.linalg.eigh(jax.lax.stop_gradient(I))
    floor = MIN_COND_RATIO * jnp.maximum(eigs.max(), I_DIAG_FLOOR)
    needs_clip = eigs.min() < floor
    I_safe = V @ jnp.diag(jnp.maximum(eigs, floor)) @ V.T

    def _do_project(_):
        x2 = x.at[3].set(I_safe[0, 0])
        x2 = x2.at[4].set(I_safe[1, 1])
        x2 = x2.at[5].set(I_safe[2, 2])
        x2 = x2.at[6].set(I_safe[0, 1])
        x2 = x2.at[7].set(I_safe[0, 2])
        x2 = x2.at[8].set(I_safe[1, 2])
        return x2

    return jax.lax.cond(needs_clip, _do_project, lambda _: x, operand=None)


class EKFParams(NamedTuple):
    dt: float
    I_rw: jnp.ndarray         # (3,) wheel inertia per axis
    Qc: jnp.ndarray           # (15,) continuous-time process-noise diagonal
    R: jnp.ndarray            # (6,) measurement-noise diagonal


class EKFState(NamedTuple):
    x: jnp.ndarray   # (15,)
    P: jnp.ndarray   # (15, 15)


def _H_matrix():
    """Linear measurement matrix: z = H x where z = [omega, rw_speed]."""
    H = jnp.zeros((6, STATE_DIM))
    H = H.at[0:3, 0:3].set(jnp.eye(3))     # omega
    H = H.at[3:6, 9:12].set(jnp.eye(3))    # rw_speed
    return H


_H = _H_matrix()


def f(x, u, params: EKFParams):
    """xdot = f(x, u). tau_ext is read from the augmented state x[12:15]."""
    omega = x[0:3]
    I = _inertia_from_state(x)
    rw = x[9:12]
    tau_ext = tau_ext_from_state(x)
    Irw = params.I_rw
    M = I @ omega + Irw * rw
    h_rw_dot = Irw * u
    tau_rw = -h_rw_dot
    rhs = tau_ext + tau_rw - jnp.cross(omega, M)
    omega_dot = jnp.linalg.solve(I, rhs)
    return jnp.concatenate([
        omega_dot,
        jnp.zeros(6),      # I_diag, I_offdiag derivatives are zero
        u,                  # rw_dot
        jnp.zeros(3),      # tau_ext_dot = 0 (constant bias prior)
    ])


# Forward-Euler substeps within one dt; true dynamics use 10 RK4 substeps.
EKF_PREDICT_SUBSTEPS = 5


def _predict_mean(x, u, params: EKFParams, n_substeps: int):
    h = params.dt / n_substeps
    def body(xx, _):
        return xx + h * f(xx, u, params), None
    new_x, _ = jax.lax.scan(body, x, None, length=n_substeps)
    return new_x


def analytic_F(x, u, params: EKFParams):
    return jax.jacobian(
        lambda xx: _predict_mean(xx, u, params, EKF_PREDICT_SUBSTEPS)
    )(x)


def predict(state: EKFState, u, params: EKFParams) -> EKFState:
    x_pred = _predict_mean(state.x, u, params, EKF_PREDICT_SUBSTEPS)
    F = analytic_F(state.x, u, params)
    Qd = jnp.diag(params.Qc * params.dt)
    P_pred = F @ state.P @ F.T + Qd
    P_pred = 0.5 * (P_pred + P_pred.T)
    x_pred = _floor_I_diag(x_pred)
    x_pred = _project_psd_if_needed(x_pred)
    return EKFState(x=x_pred, P=P_pred)


def update(state: EKFState, z, params: EKFParams) -> EKFState:
    R = jnp.diag(params.R)
    z_pred = _H @ state.x
    S = _H @ state.P @ _H.T + R
    K = jnp.linalg.solve(S.T, _H @ state.P.T).T  # (15, 6)
    x_new = state.x + K @ (z - z_pred)
    x_new = _floor_I_diag(x_new)
    x_new = _project_psd_if_needed(x_new)
    I_KH = jnp.eye(STATE_DIM) - K @ _H
    P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return EKFState(x=x_new, P=P_new)


def step(state: EKFState, u, z, params: EKFParams):
    """One predict + update. Returns (new_state, inertia_info_gain).

    inertia_info_gain = log-det(P_pred[3:9, 3:9]) - log-det(P_post[3:9, 3:9]).
    """
    state_pred = predict(state, u, params)
    state_post = update(state_pred, z, params)
    eps_I = 1e-12 * jnp.eye(6)
    L_pred = jnp.linalg.cholesky(state_pred.P[3:9, 3:9] + eps_I)
    L_post = jnp.linalg.cholesky(state_post.P[3:9, 3:9] + eps_I)
    ld_pred = 2.0 * jnp.sum(jnp.log(jnp.diag(L_pred)))
    ld_post = 2.0 * jnp.sum(jnp.log(jnp.diag(L_post)))
    info_gain = ld_pred - ld_post
    return state_post, info_gain
