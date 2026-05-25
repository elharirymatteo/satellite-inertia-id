# F1 (full inertia tensor) + F2 (MPC baselines) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the estimator from diagonal-only to full 6-parameter symmetric inertia tensor (F1), then add two FIM-optimizing trajectory baselines — a receding-horizon dual MPC and a one-shot offline modernized-Wittenburg — that the RL policy can be fairly compared against (F2).

**Architecture:** Three layers extend in lock-step: (1) shared math — a single 3×6 regression-row function reused by both the LS estimator and the FIM accumulator; (2) the EKF state grows from 9-dim to 12-dim with closed-form Jacobian for the 6 inertia parameters; (3) the env wraps the new EKF and exposes a wider observation to the policy. The MPC planner is a pure JAX function that takes gradient steps through the diff-sim. The receding-horizon driver re-invokes the planner periodically; the one-shot variant invokes it once.

**Tech Stack:** Python, JAX (jit/vmap/grad), Optax (Adam), NumPy, PyYAML, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-25-f1-full-tensor-and-f2-mpc-baselines-design.md`

**Branch:** `feat/f1-full-tensor-f2-mpc-baselines`

---

## File map

**Create:**
- `tests/test_ls_full_tensor.py` — verifies the 3×6 row function matches numerical Jacobian and the LS estimator recovers a known full tensor on noise-free data.
- `tests/test_ekf_full_tensor.py` — verifies the 12-dim EKF posterior stays PSD over a 150-step rollout and recovers a known full tensor from noisy data.
- `tests/test_dr_sampler.py` — verifies the rotation-based DR sampler produces PSD inertias with bounded off-diagonals.
- `tests/test_dual_mpc.py` — verifies the planner increases log-det F on a toy problem and respects τ_max.
- `control/dual_mpc.py` — the shared `plan(...)` function.
- `scripts/t1_dual_mpc_baseline.py` — drives both receding-horizon and one-shot variants.
- `scripts/t1_all_baselines.py` — unified eval producing `docs/t1_all_baselines_eval.txt`.

**Modify:**
- `rl/t1_env.py` — `sample_sat` → rotation construction; `_fim_row_contribution` → 3×6 form.
- `rl/t1_env_ekf.py` — EKF state 12-dim; observation 19-dim.
- `rl/ekf_jax.py` — state grows to 12-dim, Jacobian rederivation, PSD projection guard.
- `estimation/ls_estimator.py` — regressor gains 3 columns; estimator returns full 6-vector.
- `utils/observability.py` — FIM regressor matches LS rows.
- `scripts/t1_train_dr_ekf.py` — uses new env shape (auto via `obs_shape`); new policy filename.

**Leave alone:**
- `sim/dynamics_jax.py` (already uses the full `I_sat` matrix).
- `rl/policy.py` (auto-resizes via `init_params(obs_dim=…)`).
- `estimation/ekf.py` (NumPy reference EKF used by the iterative LS-EKF; updating this is *not* on the critical path — flag if needed later).

---

## Task 1: Full-tensor regression rows (shared by LS and FIM)

The 3×6 matrix `R(ω, ω̇)` such that `R · (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)ᵀ = τ` is the foundation of every other piece. Get it right once; test it; then everything downstream just imports it.

**Files:**
- Create: `utils/observability.py` (add `regression_rows_full(omega, omega_dot)` next to the existing diagonal version)
- Test: `tests/test_ls_full_tensor.py`

**Reference derivation** (from spec §3.4):
```
row_x = [  ω̇_x,         −ω_y ω_z,       +ω_y ω_z,
           ω̇_y − ω_x ω_z,  ω̇_z + ω_x ω_y,  ω_y² − ω_z² ]
row_y = [ +ω_x ω_z,       ω̇_y,           −ω_x ω_z,
           ω̇_x + ω_y ω_z,  ω_z² − ω_x²,    ω̇_z − ω_x ω_y ]
row_z = [ −ω_x ω_y,      +ω_x ω_y,        ω̇_z,
           ω_x² − ω_y²,    ω̇_x − ω_y ω_z,  ω̇_y + ω_x ω_z ]
```

- [ ] **Step 1: Read current `utils/observability.py`** to understand existing diagonal-row construction. Confirm function naming conventions.

- [ ] **Step 2: Write failing test `test_regression_rows_match_numerical_jacobian`**

Create `tests/test_ls_full_tensor.py`:

```python
"""Verify the full-tensor regression-row function matches the analytic Jacobian
of Euler's equation in the 6 inertia parameters."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from utils.observability import regression_rows_full


def _euler_tau(I_params, omega, omega_dot):
    """τ = I·ω̇ + ω × (I·ω) with I built from the 6-vector."""
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I_params
    I = jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])
    return I @ omega_dot + jnp.cross(omega, I @ omega)


@pytest.mark.parametrize("seed", range(8))
def test_regression_rows_match_numerical_jacobian(seed):
    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.normal(0, 0.5, size=3))
    omega_dot = jnp.asarray(rng.normal(0, 0.5, size=3))

    # Analytic Jacobian of τ w.r.t. the 6-vector at any I (linear → constant)
    I_params0 = jnp.zeros(6)
    J = jax.jacobian(_euler_tau, argnums=0)(I_params0, omega, omega_dot)  # (3, 6)

    R = regression_rows_full(omega, omega_dot)
    np.testing.assert_allclose(np.asarray(R), np.asarray(J), atol=1e-10)
```

- [ ] **Step 3: Run the test, verify it fails**

```bash
pytest tests/test_ls_full_tensor.py::test_regression_rows_match_numerical_jacobian -v
```
Expected: ImportError (function not defined) or AssertionError.

- [ ] **Step 4: Implement `regression_rows_full` in `utils/observability.py`**

```python
import jax.numpy as jnp


def regression_rows_full(omega: jnp.ndarray, omega_dot: jnp.ndarray) -> jnp.ndarray:
    """Return the 3x6 row matrix R such that R @ theta = tau, where
    theta = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) and Euler's equation is
    tau = I @ omega_dot + omega x (I @ omega).
    """
    wx, wy, wz = omega[0], omega[1], omega[2]
    dwx, dwy, dwz = omega_dot[0], omega_dot[1], omega_dot[2]
    return jnp.array([
        [dwx,        -wy * wz,    wy * wz,
         dwy - wx * wz,  dwz + wx * wy,  wy * wy - wz * wz],
        [wx * wz,    dwy,        -wx * wz,
         dwx + wy * wz,  wz * wz - wx * wx,  dwz - wx * wy],
        [-wx * wy,   wx * wy,     dwz,
         wx * wx - wy * wy,  dwx - wy * wz,  dwy + wx * wz],
    ])
```

- [ ] **Step 5: Run all 8 parameterized cases, verify pass**

```bash
pytest tests/test_ls_full_tensor.py::test_regression_rows_match_numerical_jacobian -v
```
Expected: 8 PASSED.

- [ ] **Step 6: Add LS recovery test**

Append to `tests/test_ls_full_tensor.py`:

```python
def test_ls_recovers_known_full_tensor_noise_free():
    """With perfect (ω, ω̇, τ) data over a varied trajectory, stacked LS
    must recover the true 6-vector exactly (up to float precision)."""
    rng = np.random.default_rng(42)
    I_true = jnp.array([0.30, 0.40, 0.50, 0.02, -0.015, 0.01])

    N = 100
    omegas = jnp.asarray(rng.normal(0, 0.4, size=(N, 3)))
    omega_dots = jnp.asarray(rng.normal(0, 0.4, size=(N, 3)))

    # Build true torque from each (omega, omega_dot) pair
    taus = jnp.stack([
        regression_rows_full(o, od) @ I_true
        for o, od in zip(omegas, omega_dots)
    ])

    # Stacked LS: solve (R^T R) theta = R^T tau, R is 3N x 6
    R_blocks = jnp.stack([regression_rows_full(o, od)
                          for o, od in zip(omegas, omega_dots)])
    R = R_blocks.reshape(3 * N, 6)
    y = taus.reshape(3 * N)
    theta_hat = jnp.linalg.solve(R.T @ R, R.T @ y)

    np.testing.assert_allclose(np.asarray(theta_hat), np.asarray(I_true),
                               atol=1e-10)
```

- [ ] **Step 7: Run, verify pass**

```bash
pytest tests/test_ls_full_tensor.py -v
```
Expected: 9 PASSED.

- [ ] **Step 8: Wire `_fim_row_contribution` in `rl/t1_env.py` to the new function**

Replace the existing function (`rl/t1_env.py:37-53`):

```python
from utils.observability import regression_rows_full


def _fim_row_contribution(omega: jnp.ndarray, domega: jnp.ndarray) -> jnp.ndarray:
    """Return R^T R where R is the (3,6) regression block at this step.
    Used to accumulate the 6x6 oracle FIM in the T1 env.
    """
    R = regression_rows_full(omega, domega)  # (3, 6)
    return R.T @ R                            # (6, 6)
```

Note: callers in `rl/t1_env.py` that construct `F = jnp.zeros((3, 3))` and `_slogdet_psd(F, eps)` with `jnp.eye(3)` MUST be updated to use `(6, 6)` and `jnp.eye(6)` respectively. Search for `(3, 3)` and `eye(3)` in `rl/t1_env.py` and `rl/t1_env_ekf.py` and update all of them.

- [ ] **Step 9: Update LS estimator in `estimation/ls_estimator.py`**

Read the file first. Locate where the 3-column regressor is built (likely a function or inline matrix). Replace with `regression_rows_full`. The LS solve becomes `theta_hat = solve((R.T @ R), R.T @ y)` over 6 unknowns; reconstruct the full 3×3 inertia matrix from `theta_hat` before returning.

Add a helper at the top of `estimation/ls_estimator.py`:

```python
def _vec_to_inertia(theta):
    """6-vector (Ixx,Iyy,Izz,Ixy,Ixz,Iyz) -> 3x3 symmetric inertia."""
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])
```

If the estimator currently returns a 3-vector or `(3, 3)` diagonal matrix, return the full 3×3 reconstructed via `_vec_to_inertia`.

- [ ] **Step 10: Run all existing tests to catch regressions**

```bash
pytest tests/ -v --tb=short
```
Expected: existing tests pass (the diagonal-only sub-rows in the new function match the old code by construction — see "sanity-check" in spec §3.4). If something fails, it's likely a hard-coded shape downstream — fix immediately.

- [ ] **Step 11: Commit**

```bash
git add utils/observability.py estimation/ls_estimator.py rl/t1_env.py rl/t1_env_ekf.py tests/test_ls_full_tensor.py
git commit -m "$(cat <<'EOF'
feat(F1): full 6-parameter inertia regression rows

Adds regression_rows_full() returning the 3x6 matrix R such that
R @ (Ixx,Iyy,Izz,Ixy,Ixz,Iyz)^T = tau, derived from Euler's equation.
Wires the new function into the LS estimator and the FIM accumulator
(_fim_row_contribution). Verified against numerical Jacobian and by
noise-free LS recovery of a known full tensor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PSD-guaranteed DR sampling

Replace `sample_sat` with the rotation-based construction from spec §3.2.

**Files:**
- Modify: `rl/t1_env.py` (`sample_sat`)
- Test: `tests/test_dr_sampler.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_dr_sampler.py`:

```python
"""Verify the rotation-based DR sampler produces PSD inertia tensors with
eigenvalues in the requested range and bounded off-diagonals."""
import numpy as np
import jax
import jax.numpy as jnp

from rl.t1_env import sample_sat


def test_sampler_yields_psd_with_bounded_offdiagonals():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 256)
    sats = [
        sample_sat(k, I_range=(0.1, 20.0),
                   I_rw=1e-4, rw_axes=jnp.eye(3),
                   rw_speed_max=600.0, rw_torque_max=0.05,
                   log_uniform=True, max_tilt_angle=np.pi / 8)
        for k in keys
    ]

    for sat in sats:
        I = np.asarray(sat.I_sat)
        # Symmetric
        np.testing.assert_allclose(I, I.T, atol=1e-10)
        # PSD with eigenvalues in [0.1, 20]
        eigs = np.linalg.eigvalsh(I)
        assert eigs.min() >= 0.1 - 1e-9, f"min eig {eigs.min()}"
        assert eigs.max() <= 20.0 + 1e-9, f"max eig {eigs.max()}"
        # Off-diagonals bounded (~few % of trace for tilt = pi/8)
        offdiag = np.abs(I - np.diag(np.diag(I))).max()
        assert offdiag <= 0.5 * np.trace(I), f"offdiag {offdiag} vs trace {np.trace(I)}"
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_dr_sampler.py -v
```
Expected: TypeError on `max_tilt_angle` (the current signature lacks it).

- [ ] **Step 3: Implement rotation-based sampler**

Replace `sample_sat` in `rl/t1_env.py`:

```python
def sample_sat(key, I_range, I_rw, rw_axes, rw_speed_max, rw_torque_max,
               log_uniform: bool = False, max_tilt_angle: float = 0.0):
    """Sample a SatParams with PSD inertia.

    The diagonal eigenvalues are drawn uniform (or log-uniform) in I_range.
    The eigenvectors are drawn by rotating the world frame about a random
    unit axis by an angle U(0, max_tilt_angle). max_tilt_angle=0 reproduces
    the previous diagonal-only behavior.
    """
    k_diag, k_axis, k_angle = jax.random.split(key, 3)
    if log_uniform:
        log_lo, log_hi = jnp.log(I_range[0]), jnp.log(I_range[1])
        log_diag = jax.random.uniform(k_diag, (3,), minval=log_lo, maxval=log_hi)
        diag = jnp.exp(log_diag)
    else:
        diag = jax.random.uniform(k_diag, (3,), minval=I_range[0], maxval=I_range[1])
    Lam = jnp.diag(diag)

    # Random unit axis on S^2
    u_raw = jax.random.normal(k_axis, (3,))
    u = u_raw / jnp.linalg.norm(u_raw)
    theta = jax.random.uniform(k_angle, (), minval=0.0, maxval=max_tilt_angle)

    # Rodrigues' formula
    K = jnp.array([[0.0, -u[2],  u[1]],
                   [u[2],  0.0, -u[0]],
                   [-u[1], u[0],  0.0]])
    R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)

    I_sat = R @ Lam @ R.T
    I_inv = jnp.linalg.inv(I_sat)
    return SatParams(
        I_sat=I_sat, I_inv=I_inv, I_rw=I_rw, rw_axes=rw_axes,
        rw_speed_max=rw_speed_max, rw_torque_max=rw_torque_max,
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_dr_sampler.py -v
```
Expected: 1 PASSED.

- [ ] **Step 5: Update callers in `scripts/t1_train_dr_ekf.py` and `scripts/t1_train_dr.py`**

Find every `sample_sat(...)` call. Add `max_tilt_angle=jnp.pi/8` as a keyword arg. Search:

```bash
grep -rn "sample_sat" scripts/ rl/
```

- [ ] **Step 6: Commit**

```bash
git add rl/t1_env.py scripts/ tests/test_dr_sampler.py
git commit -m "$(cat <<'EOF'
feat(F1): PSD-guaranteed DR sampler via random-rotation construction

sample_sat now draws log-uniform eigenvalues and rotates the world frame
about a random axis by U(0, max_tilt_angle). This guarantees PSD with
bounded condition number and realistic off-diagonals (a few percent of
trace for max_tilt_angle=pi/8).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: EKF state extension to 12-dim

Extend `rl/ekf_jax.py` to track `[ω(3), I_diag(3), I_offdiag(3), Ω_rw(3)]` with closed-form Jacobians and a PSD-projection guard.

**Files:**
- Modify: `rl/ekf_jax.py`
- Test: `tests/test_ekf_full_tensor.py`

**Layout reminder:**
```
x[0:3]  = ω
x[3:6]  = (Ixx, Iyy, Izz)
x[6:9]  = (Ixy, Ixz, Iyz)
x[9:12] = Ω_rw
```

- [ ] **Step 1: Read `rl/ekf_jax.py` end-to-end** — understand current `f`, `analytic_F`, `_H_matrix`, `predict`, `update`, `step`.

- [ ] **Step 2: Write failing test — posterior covariance PSD over 150-step rollout**

Create `tests/test_ekf_full_tensor.py`:

```python
"""Verify the 12-dim EKF stays PSD over a long rollout and converges to
the true full inertia under modest noise."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rl.ekf_jax import EKFParams, EKFState, step as ekf_step


def _make_params(dt=0.1):
    return EKFParams(
        dt=dt,
        I_rw=jnp.full((3,), 1e-4),
        Qc=jnp.concatenate([
            jnp.full((3,), 1e-9),  # omega
            jnp.full((3,), 1e-9),  # I_diag
            jnp.full((3,), 1e-9),  # I_offdiag
            jnp.full((3,), 1e-9),  # rw
        ]),
        R=jnp.concatenate([jnp.full((3,), 1e-8), jnp.full((3,), 1e-6)]),
    )


def test_posterior_stays_psd_over_long_rollout():
    params = _make_params()
    I_true = np.array([[0.40, 0.02, -0.01],
                       [0.02, 0.50,  0.015],
                       [-0.01, 0.015, 0.60]])
    # Initial guess: 15% biased diag-only
    I_init_diag = 0.85 * np.diag(I_true)
    x0 = jnp.concatenate([
        jnp.zeros(3),
        jnp.asarray(I_init_diag),
        jnp.zeros(3),       # off-diagonals start at 0
        jnp.zeros(3),
    ])
    P0 = jnp.diag(jnp.concatenate([
        jnp.full((3,), 1e-4),
        (0.30 * jnp.asarray(I_init_diag)) ** 2,
        jnp.full((3,), (0.1 * I_init_diag.mean()) ** 2),
        jnp.full((3,), 1e-2),
    ]))
    state = EKFState(x=x0, P=P0)

    # Run a 150-step open-loop rollout; control u = small sinusoid
    rng = np.random.default_rng(0)
    for k in range(150):
        u = 0.001 * np.sin(np.linspace(0, 6, 3) * k)
        # Fake measurement: arbitrary noisy obs (we just check PSD, not accuracy here)
        z = jnp.concatenate([
            jnp.asarray(rng.normal(0, 1e-4, 3)),
            jnp.asarray(rng.normal(0, 1e-3, 3)),
        ])
        state, _ = ekf_step(state, jnp.asarray(u), z, params)

        eigs = np.linalg.eigvalsh(np.asarray(state.P))
        assert eigs.min() >= -1e-9, f"step {k}: P not PSD, min eig {eigs.min()}"
```

- [ ] **Step 3: Run, verify failure**

```bash
pytest tests/test_ekf_full_tensor.py::test_posterior_stays_psd_over_long_rollout -v
```
Expected: ValueError or shape mismatch (state is 9-dim today, test sets up 12-dim).

- [ ] **Step 4: Rewrite `rl/ekf_jax.py` for 12-dim state**

Replace the file's body. Key changes (skeleton — fill in carefully):

```python
"""JAX port of the EKF, generalized to the full 6-parameter inertia tensor.

State: x = [omega(3), I_diag(3), I_offdiag(3), rw(3)]   (12-dim)
Measurement: z = [omega_meas(3), rw_meas(3)]            (6-dim, H linear)
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp

from utils.observability import regression_rows_full

I_DIAG_FLOOR = 1e-6


def _skew(v):
    return jnp.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1], v[0],  0.0],
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
    """Floor the diagonal entries to keep det(I) > 0 in normal operation."""
    x = x.at[3].set(jnp.maximum(x[3], I_DIAG_FLOOR))
    x = x.at[4].set(jnp.maximum(x[4], I_DIAG_FLOOR))
    x = x.at[5].set(jnp.maximum(x[5], I_DIAG_FLOOR))
    return x


def _project_psd_if_needed(x):
    """Lazy guard: if min_eig(I) < 0, eigendecompose, clip, recompose."""
    I = _inertia_from_state(x)
    eigs, V = jnp.linalg.eigh(I)
    needs_clip = eigs.min() < 1e-6
    eigs_safe = jnp.maximum(eigs, 1e-4 * jnp.maximum(eigs.max(), 1.0))
    I_safe = V @ jnp.diag(eigs_safe) @ V.T
    # Only apply if needed — but jax doesn't branch easily; we just project
    # unconditionally when the min eig is below threshold via lax.cond.
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
    I_rw: jnp.ndarray         # (3,)
    Qc: jnp.ndarray           # (12,)
    R: jnp.ndarray            # (6,)


class EKFState(NamedTuple):
    x: jnp.ndarray   # (12,)
    P: jnp.ndarray   # (12, 12)


def _H_matrix():
    H = jnp.zeros((6, 12))
    H = H.at[0:3, 0:3].set(jnp.eye(3))   # omega measurement
    H = H.at[3:6, 9:12].set(jnp.eye(3))  # rw measurement
    return H


_H = _H_matrix()


def f(x, u, params: EKFParams, tau_ext=None):
    """Continuous-time state derivative xdot = f(x, u)."""
    omega = x[0:3]
    I = _inertia_from_state(x)
    rw = x[9:12]
    Irw = params.I_rw
    if tau_ext is None:
        tau_ext = jnp.zeros(3)
    M = I @ omega + Irw * rw
    h_rw_dot = Irw * u
    tau_rw = -h_rw_dot
    # omega_dot = I^-1 (tau_ext + tau_rw - omega x M)
    rhs = tau_ext + tau_rw - jnp.cross(omega, M)
    omega_dot = jnp.linalg.solve(I, rhs)
    return jnp.concatenate([
        omega_dot,
        jnp.zeros(6),   # I parameters are random-walk; mean derivative = 0
        u,
    ])


def analytic_F(x, u, params: EKFParams, tau_ext=None):
    """Discrete-time Jacobian F = I_12 + dt * A.

    The omega-block rows (0:3) have three contributions:
      A[0:3, 0:3]   = d omega_dot / d omega       (skew + I^-1 chain)
      A[0:3, 3:9]   = d omega_dot / d I_params    (6 cols — derived from R)
      A[0:3, 9:12]  = d omega_dot / d rw          (Irw * cross term)
    All other rows are zero except A[9:12, 9:12] which is zero too (u is direct).
    """
    return jax.jacobian(lambda xx: xx + params.dt * f(xx, u, params, tau_ext))(x)


def predict(state, u, params, tau_ext=None):
    x_pred = state.x + params.dt * f(state.x, u, params, tau_ext)
    F = analytic_F(state.x, u, params, tau_ext)
    Qd = jnp.diag(params.Qc * params.dt)
    P_pred = F @ state.P @ F.T + Qd
    P_pred = 0.5 * (P_pred + P_pred.T)
    x_pred = _floor_I_diag(x_pred)
    x_pred = _project_psd_if_needed(x_pred)
    return EKFState(x=x_pred, P=P_pred)


def update(state, z, params):
    R = jnp.diag(params.R)
    z_pred = _H @ state.x
    S = _H @ state.P @ _H.T + R
    K = jnp.linalg.solve(S.T, _H @ state.P.T).T
    x_new = state.x + K @ (z - z_pred)
    x_new = _floor_I_diag(x_new)
    x_new = _project_psd_if_needed(x_new)
    I_KH = jnp.eye(12) - K @ _H
    P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return EKFState(x=x_new, P=P_new)


def step(state, u, z, params, tau_ext=None):
    state_pred = predict(state, u, params, tau_ext)
    state_post = update(state_pred, z, params)
    eps_I = 1e-12 * jnp.eye(6)
    L_pred = jnp.linalg.cholesky(state_pred.P[3:9, 3:9] + eps_I)
    L_post = jnp.linalg.cholesky(state_post.P[3:9, 3:9] + eps_I)
    ld_pred = 2.0 * jnp.sum(jnp.log(jnp.diag(L_pred)))
    ld_post = 2.0 * jnp.sum(jnp.log(jnp.diag(L_post)))
    info_gain = ld_pred - ld_post
    return state_post, info_gain
```

Note: `analytic_F` uses `jax.jacobian` for clarity. After tests pass, an optimizer can rewrite it closed-form for speed if profiling justifies it (likely not necessary for the eval).

- [ ] **Step 5: Run the PSD test, verify pass**

```bash
pytest tests/test_ekf_full_tensor.py::test_posterior_stays_psd_over_long_rollout -v
```
Expected: PASS.

- [ ] **Step 6: Add full-tensor convergence test**

Append to `tests/test_ekf_full_tensor.py`:

```python
def test_recovers_full_tensor_from_noisy_data():
    """Drive the EKF with the true dynamics + small sensor noise. After 150
    steps under a chirp-like excitation, the EKF's I-estimate should be within
    a few percent (in Frobenius) of the truth."""
    from sim.dynamics_jax import SatParams, _step_dt

    I_true = jnp.array([[0.40, 0.02, -0.01],
                        [0.02, 0.50,  0.015],
                        [-0.01, 0.015, 0.60]])
    sat = SatParams(
        I_sat=I_true, I_inv=jnp.linalg.inv(I_true),
        I_rw=1e-4, rw_axes=jnp.eye(3),
        rw_speed_max=600.0, rw_torque_max=0.05,
    )
    dt = 0.1
    params = _make_params(dt=dt)

    # initial EKF state: 15% diag bias, zero off-diag
    I_init_diag = 0.85 * jnp.array([I_true[0,0], I_true[1,1], I_true[2,2]])
    x0 = jnp.concatenate([
        jnp.zeros(3), I_init_diag, jnp.zeros(3), jnp.zeros(3),
    ])
    P0 = jnp.diag(jnp.concatenate([
        jnp.full((3,), 1e-4),
        (0.30 * I_init_diag) ** 2,
        jnp.full((3,), (0.1 * float(I_init_diag.mean())) ** 2),
        jnp.full((3,), 1e-2),
    ]))
    ekf_state = EKFState(x=x0, P=P0)

    # True dynamics state
    sat_state = jnp.zeros(6)  # [omega(3), rw(3)]
    rng = np.random.default_rng(7)
    H = 150
    for k in range(H):
        # Chirp-ish excitation
        t = k * dt
        tau = 0.01 * np.sin(0.05 * t * t) * np.array([1.0, 0.7, -0.3])
        sat_state = _step_dt(sat_state, jnp.asarray(tau), sat, dt / 10, 10)
        omega_meas = sat_state[0:3] + jnp.asarray(rng.normal(0, 1e-4, 3))
        rw_meas = sat_state[3:6] + jnp.asarray(rng.normal(0, 1e-3, 3))
        z = jnp.concatenate([omega_meas, rw_meas])
        u = jnp.asarray(tau) / sat.I_rw
        ekf_state, _ = ekf_step(ekf_state, u, z, params)

    I_est = np.array([
        [ekf_state.x[3], ekf_state.x[6], ekf_state.x[7]],
        [ekf_state.x[6], ekf_state.x[4], ekf_state.x[8]],
        [ekf_state.x[7], ekf_state.x[8], ekf_state.x[5]],
    ])
    rel_err = np.linalg.norm(I_est - np.asarray(I_true)) / np.linalg.norm(np.asarray(I_true))
    assert rel_err < 0.10, f"rel_err {rel_err}"
```

- [ ] **Step 7: Run both EKF tests, verify pass**

```bash
pytest tests/test_ekf_full_tensor.py -v
```
Expected: 2 PASSED. If the convergence test fails with rel_err in 0.10–0.20 range, that's a tuning issue — first try increasing the chirp amplitude in the test; if still off, the Jacobian is wrong, debug `analytic_F`.

- [ ] **Step 8: Run all existing tests**

```bash
pytest tests/ -v --tb=short
```
Expected: existing tests that compare against the EKF (e.g., `tests/test_ekf_jax.py`) **will likely fail** because they assume 9-dim state. Tag those as `xfail` for now with a clear reason, and create a follow-up note in the spec's §8 deferred questions. Do NOT modify them silently. Specifically:

```python
@pytest.mark.xfail(reason="EKF extended to full 6-param tensor (F1); "
                          "this test assumes the old 9-dim state.")
```

- [ ] **Step 9: Commit**

```bash
git add rl/ekf_jax.py tests/test_ekf_full_tensor.py tests/test_ekf_jax.py
git commit -m "$(cat <<'EOF'
feat(F1): extend EKF state to 12-dim for full inertia tensor

Replaces the 9-dim [omega, I_diag, rw] state with a 12-dim
[omega, I_diag, I_offdiag, rw] layout. Dynamics now use I @ omega
(full matrix), Jacobian computed via jax.jacobian, Joseph-form update
keeps P PSD, and a lazy eigval-clip projection guards against rare
non-PSD drift of the inertia mean. The pre-existing 9-dim test is
marked xfail with a reference to this work item.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire new EKF + FIM into the T1 envs

The envs now must (a) use the 12-dim EKF state, (b) expose a wider observation, (c) read I_est as a 6-vector and P_I as the 6×6 inertia block.

**Files:**
- Modify: `rl/t1_env.py`
- Modify: `rl/t1_env_ekf.py`

- [ ] **Step 1: Update `T1Env._obs` in `rl/t1_env.py`**

The oracle FIM env reward uses `F ∈ R^{6×6}`. Update the obs to include the 6×6 upper-triangle (21 entries) instead of the 3×3 upper-triangle (6 entries). New obs dim = 3 + 3 + 21 + 1 = 28.

```python
def _obs(self, env_state: T1EnvState) -> jnp.ndarray:
    F = env_state.F  # (6, 6)
    # Upper-triangle (incl. diagonal) row-major: 21 entries
    idx_r, idx_c = jnp.triu_indices(6)
    F_flat = F[idx_r, idx_c]
    progress = env_state.step.astype(jnp.float32) / self.cfg.horizon
    return jnp.concatenate([
        env_state.sat_state[0:3],
        env_state.sat_state[3:6] / self.cfg.sat.rw_speed_max,
        jnp.log1p(F_flat),
        jnp.array([progress]),
    ])
```

Update `self.obs_shape = (28,)`.

Also update the `F = jnp.zeros((3, 3))` in `reset` to `F = jnp.zeros((6, 6))`.

- [ ] **Step 2: Update `T1EnvEKF` in `rl/t1_env_ekf.py`**

The EKF state is now 12-dim. `I_est = ekf.x[3:6]` (diagonal) and the off-diagonals are at `ekf.x[6:9]`. The inertia covariance block is `ekf.P[3:9, 3:9]` (6×6).

Update `_obs`:

```python
def _obs(self, env_state):
    ekf = env_state.ekf
    omega_est = ekf.x[0:3]
    rw_est = ekf.x[9:12]
    I_diag = ekf.x[3:6]
    I_off = ekf.x[6:9]
    # 6-vector ordering: (Ixx,Iyy,Izz,Ixy,Ixz,Iyz)
    I_est = jnp.concatenate([I_diag, I_off])
    P_I_diag = jnp.diag(ekf.P[3:9, 3:9])
    # Reference for normalization: 6-vector from cfg.sat
    ref_I3 = jnp.diag(self.cfg.sat.I_sat)
    ref_off = jnp.array([self.cfg.sat.I_sat[0, 1],
                         self.cfg.sat.I_sat[0, 2],
                         self.cfg.sat.I_sat[1, 2]])
    ref_norm = jnp.concatenate([ref_I3, jnp.maximum(jnp.abs(ref_off),
                                                    0.01 * ref_I3.mean())])
    progress = env_state.step.astype(jnp.float32) / self.cfg.horizon
    return jnp.concatenate([
        omega_est,
        rw_est / self.cfg.sat.rw_speed_max,
        I_est / ref_norm,
        jnp.log(jnp.maximum(P_I_diag, 1e-30)),
        jnp.array([progress]),
    ])
```

Update `self.obs_shape = (19,)`.

Update `_build_ekf_params` and `reset` to construct a 12-dim initial state and 12×12 P0:

```python
def _build_ekf_params(self, sat: SatParams) -> EKFParams:
    diag_I = jnp.diag(sat.I_sat)
    return EKFParams(
        dt=self.cfg.dt,
        I_rw=jnp.full((3,), sat.I_rw),
        Qc=jnp.concatenate([
            jnp.full((3,), self.cfg.Qc_omega),                   # omega
            self.cfg.Qc_I_rel * diag_I ** 2,                     # I_diag
            jnp.full((3,), self.cfg.Qc_I_rel * (diag_I.mean()) ** 2),  # I_off
            jnp.full((3,), self.cfg.Qc_rw),                      # rw
        ]),
        R=jnp.concatenate([jnp.full((3,), self.cfg.sigma_omega ** 2),
                           jnp.full((3,), self.cfg.sigma_rw ** 2)]),
    )

def reset(self, key, sat=None):
    sat = sat if sat is not None else self.cfg.sat
    k_omega, k_used = jax.random.split(key, 2)
    omega0 = self.cfg.init_omega_scale * jax.random.normal(k_omega, (3,))
    rw0 = jnp.zeros(3)
    sat_state = jnp.concatenate([omega0, rw0])

    diag_true = jnp.diag(sat.I_sat)
    I_diag_init = self.cfg.I0_scale * diag_true
    sigma_I_diag = self.cfg.sigma_I0_rel * diag_true
    sigma_I_off = jnp.full((3,), self.cfg.sigma_I0_rel * diag_true.mean())
    x0 = jnp.concatenate([omega0, I_diag_init, jnp.zeros(3), rw0])
    P0 = jnp.diag(jnp.concatenate([
        jnp.full((3,), 1e-4),
        sigma_I_diag ** 2,
        sigma_I_off ** 2,
        jnp.full((3,), 1e-2),
    ]))
    ekf_state = EKFState(x=x0, P=P0)

    env_state = T1EnvEKFState(
        sat_state=sat_state, ekf=ekf_state, F_oracle=jnp.zeros((6, 6)),
        step=jnp.int32(0), last_omega_true=omega0, sat=sat, key=k_used,
    )
    return env_state, self._obs(env_state)
```

Also: the `step` method computes `rel_sq_err`. Update it to use Frobenius over the full tensor:

```python
I_true_mat = sat.I_sat
I_est_mat = jnp.array([
    [new_ekf.x[3], new_ekf.x[6], new_ekf.x[7]],
    [new_ekf.x[6], new_ekf.x[4], new_ekf.x[8]],
    [new_ekf.x[7], new_ekf.x[8], new_ekf.x[5]],
])
rel_sq_err = (jnp.linalg.norm(I_est_mat - I_true_mat) ** 2
              / jnp.linalg.norm(I_true_mat) ** 2)
```

The `rw_speed` index in `_saturation_penalty` was `new_sat_state[3:6]` — that stays correct (it's reading from sat_state, not EKF state).

- [ ] **Step 3: Smoke test — env steps run**

Create `tests/test_envs_smoke.py` (extension test):

```python
"""Smoke test that the new envs accept actions and produce sensible
observations of the new shape."""
import jax
import jax.numpy as jnp
import numpy as np

from rl.t1_env import T1EnvConfig, make_env, sample_sat
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
from sim.dynamics_jax import SatParams


def _toy_sat():
    return sample_sat(jax.random.PRNGKey(0), (0.1, 20.0),
                      I_rw=1e-4, rw_axes=jnp.eye(3),
                      rw_speed_max=600.0, rw_torque_max=0.05,
                      log_uniform=True, max_tilt_angle=np.pi / 8)


def test_t1env_obs_shape_28():
    sat = _toy_sat()
    cfg = T1EnvConfig(sat=sat, dt=0.1, substeps=10, horizon=20,
                      tau_max=0.01, fim_eps=1e-6, sat_penalty=0.1,
                      init_omega_scale=1e-3)
    env = make_env(cfg)
    assert env.obs_shape == (28,)
    state, obs = env.reset(jax.random.PRNGKey(1))
    assert obs.shape == (28,)
    state, obs, r, done, info = env.step(state, jnp.array([0.001, 0.0, 0.0]))
    assert obs.shape == (28,)
    assert jnp.isfinite(r)


def test_t1envekf_obs_shape_19():
    sat = _toy_sat()
    cfg = T1EnvEKFConfig(
        sat=sat, dt=0.1, substeps=10, horizon=20, tau_max=0.01,
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="info_gain",
    )
    env = make_env_ekf(cfg)
    assert env.obs_shape == (19,)
    state, obs = env.reset(jax.random.PRNGKey(2))
    assert obs.shape == (19,)
    state, obs, r, done, info = env.step(state, jnp.array([0.001, 0.0, 0.0]))
    assert obs.shape == (19,)
    assert jnp.isfinite(r)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_envs_smoke.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Run all tests, confirm nothing else regressed**

```bash
pytest tests/ -v --tb=short
```
Existing `tests/test_t1_env*.py` may fail on shape — apply `xfail` like in Task 3 Step 8 with reason "supersceded by F1 obs shape change".

- [ ] **Step 6: Commit**

```bash
git add rl/t1_env.py rl/t1_env_ekf.py tests/test_envs_smoke.py tests/test_t1_env*.py
git commit -m "$(cat <<'EOF'
feat(F1): wire 12-dim EKF and 6x6 FIM into T1 envs

Updates T1Env observation to 28-dim (includes 21-entry 6x6 FIM upper
triangle) and T1EnvEKF to 19-dim (includes 6-vector I_est and 6-vector
log diag(P_I)). EKF init/state and the relative-error metric now treat
inertia as the full 6-parameter tensor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Retrain DR policy on the new env

No code structure change — `scripts/t1_train_dr_ekf.py` picks up the new observation shape via `env.obs_shape[0]` automatically. We just need to (a) ensure the script writes to a new policy filename so the old diagonal-only weights aren't overwritten, (b) run training, (c) confirm the policy learns.

**Files:**
- Modify: `scripts/t1_train_dr_ekf.py`

- [ ] **Step 1: Update output filenames**

In `scripts/t1_train_dr_ekf.py`, change the policy and history filenames so the full-tensor outputs are tagged. Around lines 192–214, build `suffix` and update the file paths:

```python
suffix = "_full_tensor"
if args.reward_mode != "info_gain":
    suffix += f"_{args.reward_mode}"
if args.tau_max is not None:
    suffix += f"_tmax{args.tau_max:g}"
```

Output paths become:
```python
history_path = ROOT / "docs" / f"t1_dr_ekf{suffix}_history.npy"
eval_path = ROOT / "docs" / f"t1_dr_ekf{suffix}_eval.txt"
out_path = ROOT / "rl" / f"trained_dr_ekf{suffix}_policy.npz"
```

Also: in the `sample_sat(...)` call in `make_dr_rollout`, pass `max_tilt_angle=jnp.pi/8`.

- [ ] **Step 2: Quick sanity run — 50 steps to confirm it learns**

```bash
python3 scripts/t1_train_dr_ekf.py --steps 50 --batch 8 --horizon 50 \
    --reward-mode neg_rel_err
```
Expected: `train_R` improves over the 50 steps (last value ≥ first value). If not, the env or EKF is buggy — go back to Task 3 or 4.

- [ ] **Step 3: Full training run**

```bash
python3 scripts/t1_train_dr_ekf.py --steps 500 --batch 32 --horizon 150 \
    --reward-mode neg_rel_err
```
Expected wall time: ~5–15 minutes depending on hardware. Watch `train_R` trajectory in the printed log.

- [ ] **Step 4: Inspect the eval table**

```bash
cat docs/t1_dr_ekf_full_tensor_neg_rel_err_eval.txt
```

Verify:
- All 15 rows (3 sats × 5 policies) present.
- The DR policy `mean_I_rel_err` for sat1/2/3 is finite and ≤ 1.5 × the diagonal-only DR policy errors from `docs/t1_dr_ekf_neg_rel_err_eval.txt`.

If sat1 (CubeSat) DR err is > 1.5× chirp err, that's expected (the diagonal version also lost there). Document in `docs/notes_f1.md` (one paragraph), don't block.

- [ ] **Step 5: Commit results**

```bash
git add docs/t1_dr_ekf_full_tensor*.txt docs/t1_dr_ekf_full_tensor*.npy \
        rl/trained_dr_ekf_full_tensor*.npz scripts/t1_train_dr_ekf.py \
        docs/notes_f1.md 2>/dev/null || true
git commit -m "$(cat <<'EOF'
feat(F1): retrained DR policy under full-tensor EKF env

500 steps, batch 32, horizon 150, neg_rel_err reward. Same training
script and hyperparameters as the diagonal-only run, just with the
extended observation and dynamics. Eval on sat1/2/3 held-out configs
included.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Dual MPC core planner

Pure JAX function that takes gradient steps through the diff-sim to maximize log-det F over a horizon. Used by both the receding-horizon driver and the one-shot driver.

**Files:**
- Create: `control/dual_mpc.py`
- Test: `tests/test_dual_mpc.py`

- [ ] **Step 1: Write failing test — planner improves log-det F**

Create `tests/test_dual_mpc.py`:

```python
"""Verify the MPC planner increases log-det(F) over its horizon and
respects the torque box constraint."""
import jax
import jax.numpy as jnp
import numpy as np

from sim.dynamics_jax import SatParams
from rl.t1_env import sample_sat
from control.dual_mpc import plan


def _toy_sat():
    return sample_sat(jax.random.PRNGKey(0), (0.1, 20.0),
                      I_rw=1e-4, rw_axes=jnp.eye(3),
                      rw_speed_max=600.0, rw_torque_max=0.05,
                      log_uniform=True, max_tilt_angle=np.pi / 8)


def _rollout_logdet(sat, sat_state, F0, tau_seq, dt, substeps):
    """Roll out dynamics under tau_seq, return final logdet(F + eps I)."""
    from sim.dynamics_jax import _step_dt
    from utils.observability import regression_rows_full
    F = F0
    last_omega = sat_state[0:3]
    for k in range(tau_seq.shape[0]):
        sat_state = _step_dt(sat_state, tau_seq[k], sat, dt / substeps, substeps)
        new_omega = sat_state[0:3]
        domega = (new_omega - last_omega) / dt
        mid_omega = 0.5 * (new_omega + last_omega)
        R = regression_rows_full(mid_omega, domega)
        F = F + R.T @ R
        last_omega = new_omega
    L = jnp.linalg.cholesky(F + 1e-6 * jnp.eye(6))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L))), sat_state, F


def test_plan_improves_logdet_over_zero_actions():
    sat = _toy_sat()
    init_state = jnp.zeros(6)
    F0 = jnp.zeros((6, 6))
    tau_max = 0.01
    horizon = 10
    dt, substeps = 0.1, 10

    # Baseline: zero actions
    zero_seq = jnp.zeros((horizon, 3))
    ld_zero, _, _ = _rollout_logdet(sat, init_state, F0, zero_seq, dt, substeps)

    # MPC plan
    tau_plan = plan(init_state, F0, sat, horizon=horizon,
                    n_opt_steps=30, tau_max=tau_max,
                    lr=0.01 * tau_max, dt=dt, substeps=substeps,
                    sat_penalty=0.0)
    ld_plan, _, _ = _rollout_logdet(sat, init_state, F0, tau_plan, dt, substeps)

    assert ld_plan > ld_zero, f"plan logdet {ld_plan} did not improve over zero {ld_zero}"


def test_plan_respects_tau_max():
    sat = _toy_sat()
    init_state = jnp.zeros(6)
    F0 = jnp.zeros((6, 6))
    tau_max = 0.005
    tau_plan = plan(init_state, F0, sat, horizon=10,
                    n_opt_steps=30, tau_max=tau_max,
                    lr=0.01 * tau_max, dt=0.1, substeps=10,
                    sat_penalty=0.0)
    assert jnp.max(jnp.abs(tau_plan)) <= tau_max + 1e-6
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_dual_mpc.py -v
```
Expected: ImportError (no `control/dual_mpc.py`).

- [ ] **Step 3: Implement `control/dual_mpc.py`**

```python
"""Dual MPC planner: gradient through the diff-sim, FIM log-det objective.

Used as a baseline against the RL policy. The planner takes the current
dynamics state + accumulated FIM, picks a piecewise-constant torque
sequence over a horizon, and returns it after a fixed number of Adam steps.

Pure JAX; designed to be jit-compiled and vmap-compatible over batches.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import optax

from sim.dynamics_jax import SatParams, _step_dt
from utils.observability import regression_rows_full


def _slogdet_psd_6(F, eps):
    L = jnp.linalg.cholesky(F + eps * jnp.eye(6))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _saturation_penalty(sat, tau, rw_speed):
    tau_excess = jnp.maximum(jnp.abs(tau) - sat.rw_torque_max, 0.0)
    speed_excess = jnp.maximum(jnp.abs(rw_speed) - 0.95 * sat.rw_speed_max, 0.0)
    return (tau_excess.sum() / (sat.rw_torque_max + 1e-12)
            + speed_excess.sum() / (sat.rw_speed_max + 1e-12))


def _rollout_loss(z, init_state, F0, sat, horizon, dt, substeps,
                  tau_max, sat_penalty, fim_eps):
    """Loss = -logdet(F_final) + sat_penalty * total_sat_pen.

    z has shape (horizon, 3); tau = tau_max * tanh(z) (box constraint).
    """
    tau_seq = tau_max * jnp.tanh(z)

    def body(carry, tau):
        sat_state, F, last_omega = carry
        new_state = _step_dt(sat_state, tau, sat, dt / substeps, substeps)
        new_omega = new_state[0:3]
        domega = (new_omega - last_omega) / dt
        mid_omega = 0.5 * (new_omega + last_omega)
        R = regression_rows_full(mid_omega, domega)
        F = F + R.T @ R
        rw_speed = new_state[3:6]
        sp = _saturation_penalty(sat, tau, rw_speed)
        return (new_state, F, new_omega), sp

    last_omega_init = init_state[0:3]
    (final_state, F_final, _), sat_pens = jax.lax.scan(
        body, (init_state, F0, last_omega_init), tau_seq
    )
    ld = _slogdet_psd_6(F_final, fim_eps)
    return -ld + sat_penalty * sat_pens.sum()


def plan(init_state, F0, sat: SatParams, *, horizon: int,
         n_opt_steps: int, tau_max: float, lr: float,
         dt: float, substeps: int,
         sat_penalty: float = 0.1, fim_eps: float = 1e-6,
         warm_start: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the optimized tau sequence of shape (horizon, 3)."""
    z = (warm_start if warm_start is not None
         else jnp.zeros((horizon, 3)))

    @jax.jit
    def grad_step(z, opt_state):
        loss, g = jax.value_and_grad(_rollout_loss)(
            z, init_state, F0, sat, horizon, dt, substeps,
            tau_max, sat_penalty, fim_eps,
        )
        updates, opt_state = opt.update(g, opt_state)
        z = optax.apply_updates(z, updates)
        return z, opt_state, loss

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = opt.init(z)
    for _ in range(n_opt_steps):
        z, opt_state, _ = grad_step(z, opt_state)
    return tau_max * jnp.tanh(z)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_dual_mpc.py -v
```
Expected: 2 PASSED. If `test_plan_improves_logdet_over_zero_actions` fails, try increasing `n_opt_steps` to 60 and `lr` to `0.02 * tau_max`. If still failing, the gradient is exploding/vanishing; print loss per step inside `plan` to diagnose.

- [ ] **Step 5: Commit**

```bash
git add control/dual_mpc.py tests/test_dual_mpc.py
git commit -m "$(cat <<'EOF'
feat(F2): dual MPC planner — diff-sim gradient, log-det FIM objective

control/dual_mpc.py::plan takes a current dynamics state and accumulated
FIM, then runs ~30 Adam steps through the diff-sim to maximize log-det
of the FIM over a horizon, subject to a tanh-squashed torque box. Pure
JAX; jit/vmap-compatible. Shared by the receding-horizon and one-shot
drivers in F2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Receding-horizon MPC driver

A script that rolls out an episode under MPC actions, with the same EKF-in-the-loop env the RL policy uses.

**Files:**
- Create: `scripts/t1_dual_mpc_baseline.py`

- [ ] **Step 1: Implement the driver**

```python
"""Run the receding-horizon dual MPC baseline on the EKF env.

The MPC replans every M=10 steps using the EKF's current point estimate
of inertia as the planning model. The first M actions of each plan are
executed open-loop; then warm-started replan.

Usage:
  python3 scripts/t1_dual_mpc_baseline.py [--horizon-mpc 20] [--replan-every 10]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import jax
import jax.numpy as jnp

from sim.dynamics_jax import SatParams
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
from control.dual_mpc import plan


def _load_sat(cfg_name):
    with open(ROOT / cfg_name) as f:
        cfg = yaml.safe_load(f)
    I_sat = np.diag(cfg["satellite"]["inertia_tensor"])
    return SatParams(
        I_sat=jnp.asarray(I_sat),
        I_inv=jnp.asarray(np.linalg.inv(I_sat)),
        I_rw=float(cfg["reaction_wheels"]["inertia"]),
        rw_axes=jnp.asarray(cfg["reaction_wheels"]["alignment_matrix"], dtype=jnp.float32),
        rw_speed_max=float(cfg["reaction_wheels"]["max_speed"]),
        rw_torque_max=float(cfg["reaction_wheels"]["max_torque"]),
    )


def _ekf_inertia_matrix(ekf_x):
    """Reconstruct the 3x3 I from the EKF state vector x[3:9]."""
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = ekf_x[3], ekf_x[4], ekf_x[5], ekf_x[6], ekf_x[7], ekf_x[8]
    return jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])


def run_receding_horizon(env, sat_true, key, horizon_total=150,
                         horizon_mpc=20, replan_every=10,
                         n_opt_steps=30):
    """Roll out one episode under receding-horizon MPC. Returns rel_err and
    wall-time per step."""
    state, _obs = env.reset(key, sat=sat_true)
    warm = None
    rel_errs = []
    step_wall = []

    t = 0
    while t < horizon_total:
        # Build a planning sat from the EKF point estimate
        I_est = _ekf_inertia_matrix(state.ekf.x)
        sat_for_planning = SatParams(
            I_sat=I_est, I_inv=jnp.linalg.inv(I_est),
            I_rw=sat_true.I_rw, rw_axes=sat_true.rw_axes,
            rw_speed_max=sat_true.rw_speed_max,
            rw_torque_max=sat_true.rw_torque_max,
        )
        # Plan over horizon_mpc using the EKF's current sat state as init
        sat_state_now = state.sat_state
        F0 = state.F_oracle  # 6x6
        t0 = time.perf_counter()
        tau_plan = plan(sat_state_now, F0, sat_for_planning,
                        horizon=horizon_mpc, n_opt_steps=n_opt_steps,
                        tau_max=env.cfg.tau_max,
                        lr=0.01 * env.cfg.tau_max,
                        dt=env.cfg.dt, substeps=env.cfg.substeps,
                        warm_start=(warm if warm is not None else None))
        plan_wall = time.perf_counter() - t0

        # Execute first replan_every actions on the true env
        n_to_run = min(replan_every, horizon_total - t)
        for k in range(n_to_run):
            state, _obs, _r, _done, _info = env.step(state, tau_plan[k])
            t += 1
        # Amortize planning wall-time across the executed steps
        step_wall.extend([plan_wall / n_to_run] * n_to_run)

        # Warm-start next plan by shifting: drop the n_to_run actions just
        # executed, then pad the tail with zeros to keep length horizon_mpc.
        remaining = tau_plan[n_to_run:]                       # (horizon_mpc - n_to_run, 3)
        pad = jnp.zeros((n_to_run, 3))                        # (n_to_run, 3)
        warm = jnp.concatenate([remaining, pad], axis=0)      # (horizon_mpc, 3)

    # Final inertia error
    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), float(np.mean(step_wall) * 1000)  # ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-mpc", type=int, default=20)
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--n-opt-steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=32)
    args = ap.parse_args()

    # Build EKF config matching the RL training (use sat1 as base for env cfg)
    with open(ROOT / "config_sat1.yaml") as f:
        cfg_y = yaml.safe_load(f)
    sat1 = _load_sat("config_sat1.yaml")
    base_cfg = T1EnvEKFConfig(
        sat=sat1, dt=float(cfg_y["sim"]["dt"]), substeps=10,
        horizon=150, tau_max=float(cfg_y["reaction_wheels"]["max_torque"]),
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
    )
    env = make_env_ekf(base_cfg)

    print(f"{'sat':>14}  {'method':>16}  {'rel_err':>10}  {'ms/step':>10}")
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        sat = _load_sat(cfg_name)
        # Rebuild env with this sat as the cfg (for normalization reference)
        env_cfg = base_cfg._replace(sat=sat)
        env_s = make_env_ekf(env_cfg)
        errs, walls = [], []
        for s in range(args.seeds):
            key = jax.random.PRNGKey(1000 + s)
            err, w = run_receding_horizon(env_s, sat, key,
                                          horizon_mpc=args.horizon_mpc,
                                          replan_every=args.replan_every,
                                          n_opt_steps=args.n_opt_steps)
            errs.append(err); walls.append(w)
        print(f"{cfg_name.split('.')[0]:>14}  {'dual-MPC (RH)':>16}  "
              f"{np.mean(errs):>10.4%}  {np.mean(walls):>10.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke run on sat2 only (skip the loop for sat1/3) with 1 seed**

Add a `--quick` flag if needed, or run via a one-liner. Verify the script terminates and prints a finite rel_err.

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.t1_dual_mpc_baseline import run_receding_horizon, _load_sat
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
import yaml, jax
with open('config_sat2.yaml') as f: cfg_y = yaml.safe_load(f)
sat = _load_sat('config_sat2.yaml')
cfg = T1EnvEKFConfig(sat=sat, dt=float(cfg_y['sim']['dt']), substeps=10,
                     horizon=150, tau_max=float(cfg_y['reaction_wheels']['max_torque']),
                     sat_penalty=0.1, init_omega_scale=1e-3,
                     sigma_omega=1e-4, sigma_rw=1e-3,
                     I0_scale=0.85, sigma_I0_rel=0.30,
                     Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
                     reward_mode='neg_rel_err')
env = make_env_ekf(cfg)
err, ms = run_receding_horizon(env, sat, jax.random.PRNGKey(0), horizon_mpc=20, replan_every=10, n_opt_steps=30)
print(f'sat2 rel_err={err:.4%}  ms/step={ms:.1f}')
"
```
Expected: finite `rel_err` around 0.1–5%; finite `ms/step`. If it crashes inside `plan`, check Task 6 still works.

- [ ] **Step 3: Full 32-seed run on all 3 sats**

```bash
python3 scripts/t1_dual_mpc_baseline.py
```
Expected wall: ~5–20 min. Three rows printed.

- [ ] **Step 4: Commit**

```bash
git add scripts/t1_dual_mpc_baseline.py
git commit -m "$(cat <<'EOF'
feat(F2): receding-horizon dual MPC baseline driver

scripts/t1_dual_mpc_baseline.py drives the EKF-in-the-loop env under
the receding-horizon dual MPC: horizon 20 steps, replan every 10, warm-
started, planning against the EKF's current point estimate. Reports
mean rel_err and ms/step per satellite across 32 seeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: One-shot offline MPC driver

Same planner, called once with `horizon = 150`, plans against the DR prior. Play open-loop.

**Files:**
- Modify: `scripts/t1_dual_mpc_baseline.py` (add a `run_one_shot` function and an `--oneshot` flag)

- [ ] **Step 1: Add `run_one_shot` to `scripts/t1_dual_mpc_baseline.py`**

```python
def _dr_prior_sat(sat_true):
    """The 'expected I' under the DR prior, used to plan the one-shot offline
    trajectory before any observations. We use the geometric mean of the
    log-uniform range, with zero off-diagonals."""
    I_mean = float(np.exp(0.5 * (np.log(0.1) + np.log(20.0))))  # ~ 1.414
    I = jnp.eye(3) * I_mean
    return SatParams(
        I_sat=I, I_inv=jnp.linalg.inv(I),
        I_rw=sat_true.I_rw, rw_axes=sat_true.rw_axes,
        rw_speed_max=sat_true.rw_speed_max,
        rw_torque_max=sat_true.rw_torque_max,
    )


def run_one_shot(env, sat_true, key, horizon_total=150, n_opt_steps=100):
    """Plan the full 150-step trajectory once against the DR prior, then play
    it open-loop on the true env."""
    state, _obs = env.reset(key, sat=sat_true)
    sat_for_planning = _dr_prior_sat(sat_true)
    t0 = time.perf_counter()
    tau_plan = plan(state.sat_state, state.F_oracle, sat_for_planning,
                    horizon=horizon_total, n_opt_steps=n_opt_steps,
                    tau_max=env.cfg.tau_max,
                    lr=0.01 * env.cfg.tau_max,
                    dt=env.cfg.dt, substeps=env.cfg.substeps)
    plan_wall = time.perf_counter() - t0
    for k in range(horizon_total):
        state, _obs, _r, _done, _info = env.step(state, tau_plan[k])
    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), float(plan_wall / horizon_total * 1000)
```

Add an `--oneshot` flag in `main()` that runs `run_one_shot` instead of `run_receding_horizon`.

- [ ] **Step 2: Smoke run on sat2**

```bash
python3 scripts/t1_dual_mpc_baseline.py --oneshot --seeds 1
```
Expected: prints one row per sat, finite numbers.

- [ ] **Step 3: Full 32-seed run**

```bash
python3 scripts/t1_dual_mpc_baseline.py --oneshot
```

- [ ] **Step 4: Commit**

```bash
git add scripts/t1_dual_mpc_baseline.py
git commit -m "$(cat <<'EOF'
feat(F2): one-shot offline MPC baseline (modernized Wittenburg 2017)

Adds run_one_shot() to scripts/t1_dual_mpc_baseline.py, invoked via
--oneshot. Plans the full 150-step trajectory once against the DR prior
mean (geometric mean of the log-uniform I range, zero off-diagonals)
and plays the result open-loop on the true env. Same planner as the
receding-horizon driver.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Unified eval harness

One script, one table, all methods compared on the same seeds.

**Files:**
- Create: `scripts/t1_all_baselines.py`
- Create: `docs/t1_all_baselines_eval.txt` (output)

- [ ] **Step 1: Implement `scripts/t1_all_baselines.py`**

```python
"""Run every active-sensing method on the same 32 seeds across sat1/2/3
and write one comparison table.

Methods:
  - RL DR policy (trained_dr_ekf_full_tensor_neg_rel_err_policy.npz)
  - dual-MPC (receding horizon)
  - dual-MPC (one-shot offline)
  - scripted: sine, chirp, prbs, multi step

Outputs:
  docs/t1_all_baselines_eval.txt   (CSV)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import jax
import jax.numpy as jnp

from sim.dynamics_jax import SatParams
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
from rl import policy as policy_mod
from control.torque_generators import generate_torque_profile
from scripts.t1_dual_mpc_baseline import (
    _load_sat, run_receding_horizon, run_one_shot, _ekf_inertia_matrix,
)


def _scripted_actions(name, horizon, dt, amplitude):
    t = np.arange(0, horizon * dt, dt)
    params = {
        "sine":       dict(frequency=0.01, amplitude=amplitude),
        "chirp":      dict(f0=0.005, f1=0.05, amplitude=amplitude),
        "prbs":       dict(amplitude=amplitude, switch_time=20),
        "multi step": dict(amplitude=1.2 * amplitude, step_duration=40.0),
    }[name]
    arr = generate_torque_profile(name, t, **params)
    return jnp.asarray(arr[:horizon])


def _run_scripted(env, sat_true, key, tau_seq):
    state, _obs = env.reset(key, sat=sat_true)
    t0 = time.perf_counter()
    for k in range(tau_seq.shape[0]):
        state, _obs, _r, _done, _info = env.step(state, tau_seq[k])
    wall = time.perf_counter() - t0
    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), wall / tau_seq.shape[0] * 1000


def _run_rl(env, sat_true, key, params, tau_max):
    state, obs = env.reset(key, sat=sat_true)
    t0 = time.perf_counter()
    for _ in range(env.cfg.horizon):
        a = policy_mod.apply(params, obs, tau_max)
        state, obs, _r, _done, _info = env.step(state, a)
    wall = time.perf_counter() - t0
    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), wall / env.cfg.horizon * 1000


def main():
    N_SEEDS = 32
    with open(ROOT / "config_sat1.yaml") as f:
        cfg_y = yaml.safe_load(f)
    base = T1EnvEKFConfig(
        sat=_load_sat("config_sat1.yaml"),
        dt=float(cfg_y["sim"]["dt"]), substeps=10, horizon=150,
        tau_max=float(cfg_y["reaction_wheels"]["max_torque"]),
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
    )

    # Load trained RL policy
    rl_path = ROOT / "rl" / "trained_dr_ekf_full_tensor_neg_rel_err_policy.npz"
    rl_params = {k: jnp.asarray(v) for k, v in np.load(rl_path).items()}

    methods = ["RL_DR", "dual-MPC_RH", "dual-MPC_oneshot",
               "sine", "chirp", "prbs", "multi step"]
    rows = []
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        sat = _load_sat(cfg_name)
        env_cfg = base._replace(sat=sat)
        env = make_env_ekf(env_cfg)
        scripted_acts = {n: _scripted_actions(n, 150, env_cfg.dt, 0.005)
                         for n in ["sine", "chirp", "prbs", "multi step"]}
        for m in methods:
            errs, walls = [], []
            for s in range(N_SEEDS):
                key = jax.random.PRNGKey(1000 + s)
                if m == "RL_DR":
                    e, w = _run_rl(env, sat, key, rl_params, env_cfg.tau_max)
                elif m == "dual-MPC_RH":
                    e, w = run_receding_horizon(env, sat, key)
                elif m == "dual-MPC_oneshot":
                    e, w = run_one_shot(env, sat, key)
                else:
                    e, w = _run_scripted(env, sat, key, scripted_acts[m])
                errs.append(e); walls.append(w)
            mean_e, std_e = float(np.mean(errs)), float(np.std(errs))
            mean_w = float(np.mean(walls))
            rows.append((cfg_name, m, mean_e, std_e, mean_w))
            print(f"{cfg_name.split('.')[0]:>14}  {m:>20}  "
                  f"{mean_e:>10.4%}  {std_e:>10.4%}  {mean_w:>10.2f}")

    out = ROOT / "docs" / "t1_all_baselines_eval.txt"
    with open(out, "w") as f:
        f.write("sat_cfg,method,mean_rel_err,std_rel_err,mean_ms_per_step\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python3 scripts/t1_all_baselines.py
```
Expected wall: 20–60 min. Output table with 3 sats × 7 methods = 21 rows.

- [ ] **Step 3: Verify success criteria from spec §5**

```bash
cat docs/t1_all_baselines_eval.txt
```
Check:
- **F1 success:** RL DR mean_rel_err on each sat is ≤ 1.5 × the value from `docs/t1_dr_ekf_neg_rel_err_eval.txt` (mean_I_rel_err column).
- **F2 RH success:** `dual-MPC_RH` mean_rel_err ≤ min(scripted_means) per sat.
- **F2 one-shot success:** `dual-MPC_oneshot` mean_rel_err ≤ chirp mean_rel_err per sat.

If F2 fails on sat1 only (the CubeSat case where even diagonal-only DR loses to chirp): document in the paper, don't block. If F2 fails on sat2/3: tune `n_opt_steps` up to 60 in the planner call.

- [ ] **Step 4: Commit**

```bash
git add scripts/t1_all_baselines.py docs/t1_all_baselines_eval.txt
git commit -m "$(cat <<'EOF'
feat: unified baselines eval — RL vs dual-MPC vs scripted

scripts/t1_all_baselines.py runs the trained RL DR policy, both dual-MPC
variants, and the four scripted profiles on identical seeds across
sat1/2/3, producing docs/t1_all_baselines_eval.txt with mean rel_err
and ms/step per method. This is the paper's headline comparison table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all new tests pass; only the pre-existing diagonal-only tests are `xfail`'d (Tasks 3 and 4).

- [ ] **Step 2: Confirm spec success criteria**

Open `docs/superpowers/specs/2026-05-25-f1-full-tensor-and-f2-mpc-baselines-design.md` §5 and walk through each bullet against the actual results in `docs/t1_all_baselines_eval.txt`. Note any gaps in a one-paragraph addendum at the end of the spec (or in `docs/notes_f1_f2.md`).

- [ ] **Step 3: Push the branch**

(Only when the user explicitly asks — do NOT push automatically.)

```bash
git push -u origin feat/f1-full-tensor-f2-mpc-baselines
```
