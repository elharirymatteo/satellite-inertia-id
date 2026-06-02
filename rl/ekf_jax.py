"""JAX port of the EKF, generalized to the full 6-parameter inertia tensor.

State: x = [omega(3), I_diag(3), I_offdiag(3), rw(3)]   (12-dim)
  x[0:3]   = omega                       (angular velocity)
  x[3:6]   = (Ixx, Iyy, Izz)             (inertia diagonal)
  x[6:9]   = (Ixy, Ixz, Iyz)             (inertia off-diagonal)
  x[9:12]  = Omega_rw                    (reaction wheel speeds)
Measurement: z = [omega_meas(3), rw_meas(3)]            (6-dim, H linear)

Pure-functional, jittable, vmappable. The Jacobian is taken via `jax.jacobian`
on the discrete update map (closed-form via AD); the update uses Joseph form
and the inertia mean is guarded against drifting non-PSD by a lazy eigval-clip
projection.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


I_DIAG_FLOOR = 1e-6


def _skew(v):
    return jnp.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _inertia_from_state(x):
    """Build the 3x3 symmetric I from x[3:9]."""
    Ixx, Iyy, Izz = x[3], x[4], x[5]
    Ixy, Ixz, Iyz = x[6], x[7], x[8]
    return jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])


def _floor_I_diag(x):
    x = x.at[3].set(jnp.maximum(x[3], I_DIAG_FLOOR))
    x = x.at[4].set(jnp.maximum(x[4], I_DIAG_FLOOR))
    x = x.at[5].set(jnp.maximum(x[5], I_DIAG_FLOOR))
    return x


def _project_psd_if_needed(x):
    """Lazy guard: if min_eig(I) < threshold, eigendecompose, clip, recompose."""
    I = _inertia_from_state(x)
    eigs, V = jnp.linalg.eigh(I)
    needs_clip = eigs.min() < 1e-6
    eigs_safe = jnp.maximum(eigs, 1e-4 * jnp.maximum(eigs.max(), 1.0))
    I_safe = V @ jnp.diag(eigs_safe) @ V.T

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
    Qc: jnp.ndarray           # (12,) continuous-time process-noise diagonal
    R: jnp.ndarray            # (6,) measurement-noise diagonal


class EKFState(NamedTuple):
    x: jnp.ndarray   # (12,)
    P: jnp.ndarray   # (12, 12)


def _H_matrix():
    """Linear measurement matrix: z = H x where z = [omega, rw_speed]."""
    H = jnp.zeros((6, 12))
    H = H.at[0:3, 0:3].set(jnp.eye(3))    # omega measurement
    H = H.at[3:6, 9:12].set(jnp.eye(3))   # rw_speed measurement
    return H


_H = _H_matrix()


def f(x, u, params: EKFParams, tau_ext=None):
    """Continuous-time state derivative xdot = f(x, u).

    omega_dot = I^{-1} ( tau_ext - I_rw * u - omega x (I omega + I_rw * rw) )
    I_dot     = 0    (inertia is a constant unknown — random walk via Q)
    rw_dot    = u    (control is wheel acceleration command)
    """
    omega = x[0:3]
    I = _inertia_from_state(x)
    rw = x[9:12]
    Irw = params.I_rw
    if tau_ext is None:
        tau_ext = jnp.zeros(3)
    M = I @ omega + Irw * rw
    h_rw_dot = Irw * u
    tau_rw = -h_rw_dot
    rhs = tau_ext + tau_rw - jnp.cross(omega, M)
    omega_dot = jnp.linalg.solve(I, rhs)
    return jnp.concatenate([
        omega_dot,
        jnp.zeros(6),   # I_diag, I_offdiag derivatives are zero
        u,
    ])


def analytic_F(x, u, params: EKFParams, tau_ext=None):
    """Discrete-time Jacobian F = d(x + dt * f(x, u)) / dx via jax.jacobian.

    Closed-form (auto-derived). Acceptable per the F1 task spec; can be
    replaced by a hand-derived expression later for speed.
    """
    return jax.jacobian(lambda xx: xx + params.dt * f(xx, u, params, tau_ext))(x)


def predict(state: EKFState, u, params: EKFParams, tau_ext=None) -> EKFState:
    x_pred = state.x + params.dt * f(state.x, u, params, tau_ext)
    F = analytic_F(state.x, u, params, tau_ext)
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
    # Solve K = P H^T S^{-1} via linear solve for numerical stability.
    K = jnp.linalg.solve(S.T, _H @ state.P.T).T  # (12, 6)
    x_new = state.x + K @ (z - z_pred)
    x_new = _floor_I_diag(x_new)
    x_new = _project_psd_if_needed(x_new)
    # Joseph form keeps P PSD even with non-optimal K.
    I_KH = jnp.eye(12) - K @ _H
    P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return EKFState(x=x_new, P=P_new)


def step(state: EKFState, u, z, params: EKFParams, tau_ext=None):
    """One predict + update step. Returns (new_state, inertia_info_gain).

    inertia_info_gain = log-det(P_pred[3:9, 3:9]) - log-det(P_post[3:9, 3:9])
    Aggregated over measurement updates, this is non-negative in expectation.
    """
    state_pred = predict(state, u, params, tau_ext)
    state_post = update(state_pred, z, params)

    # Log-det of the inertia 6x6 block via Cholesky (stable backward).
    eps_I = 1e-12 * jnp.eye(6)
    L_pred = jnp.linalg.cholesky(state_pred.P[3:9, 3:9] + eps_I)
    L_post = jnp.linalg.cholesky(state_post.P[3:9, 3:9] + eps_I)
    ld_pred = 2.0 * jnp.sum(jnp.log(jnp.diag(L_pred)))
    ld_post = 2.0 * jnp.sum(jnp.log(jnp.diag(L_post)))
    info_gain = ld_pred - ld_post
    return state_post, info_gain
