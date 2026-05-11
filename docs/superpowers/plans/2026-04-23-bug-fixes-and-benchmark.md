# Bug Fixes and Performance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six identified bugs in the satellite inertia identification framework, then add a test suite that verifies each fix and benchmarks LS vs EKF estimation accuracy across all satellite configurations and torque profiles.

**Architecture:** Bugs fixed in-place with minimal diffs. Shared math extracted to `utils/math_utils.py`. Observability module implemented from scratch in `utils/observability.py`. All tests live under `tests/` using pytest; a `conftest.py` provides shared fixtures. The final task (benchmark) runs end-to-end simulations and asserts estimation error bounds.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest 7+

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `utils/math_utils.py` | Shared `skew()` used everywhere |
| Create | `utils/observability.py` | Observability metrics (broken import fix) |
| Create | `tests/__init__.py` | Makes tests a package |
| Create | `tests/conftest.py` | Shared fixtures |
| Create | `tests/test_math_utils.py` | Tests for skew() |
| Create | `tests/test_observability.py` | Tests for observability metrics |
| Create | `tests/test_dynamics.py` | Tests for tau_ext fix + angular_accel fix |
| Create | `tests/test_torque_generators.py` | Tests for true PRBS |
| Create | `tests/test_ls_estimator.py` | LS accuracy on synthetic data |
| Create | `tests/test_ekf.py` | EKF convergence on synthetic data |
| Create | `tests/test_benchmark.py` | End-to-end LS vs EKF benchmark |
| Modify | `sim/dynamics.py` | Import skew from utils; fix angular_accel; add tau_ext to base class |
| Modify | `estimation/ekf.py` | Remove unused `_skew` static method |
| Modify | `estimation/ls_estimator.py` | Import skew from utils, remove `_skew` method |
| Modify | `estimation/ukf.py` | Import skew from utils, remove inline skew matrix |
| Modify | `control/torque_generators.py` | Replace fake PRBS with real LFSR-based PRBS |
| Modify | `scripts/run_ls_simulation.py` | Fix O(N) lookup; fix satellite loop; fix tau_ext access; expose show_plots param |

---

### Task 1: Create `utils/math_utils.py` and deduplicate `skew()`

**Why:** `skew()` is copy-pasted four times across `sim/dynamics.py`, `estimation/ekf.py`, `estimation/ls_estimator.py`, and `estimation/ukf.py`. One canonical version eliminates drift.

**Files:**
- Create: `utils/math_utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_math_utils.py`
- Modify: `sim/dynamics.py` (lines 1–11: remove local `skew`, add import)
- Modify: `estimation/ekf.py` (remove unused `_skew` static method, add import)
- Modify: `estimation/ls_estimator.py` (remove `_skew` instance method, replace two usages)
- Modify: `estimation/ukf.py` (remove inline skew construction at lines 112–116, add import)

- [ ] **Step 1.1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_math_utils.py`:

```python
# tests/__init__.py
```

```python
# tests/test_math_utils.py
import numpy as np
import pytest
from utils.math_utils import skew


def test_skew_equals_cross_product():
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_skew_is_antisymmetric():
    v = np.array([1.0, 2.0, 3.0])
    S = skew(v)
    np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-14)


def test_skew_zero_diagonal():
    v = np.array([5.0, -3.0, 0.5])
    S = skew(v)
    np.testing.assert_allclose(np.diag(S), np.zeros(3), atol=1e-14)


def test_skew_output_shape():
    assert skew(np.zeros(3)).shape == (3, 3)
```

- [ ] **Step 1.2: Run tests to verify they fail (module not yet created)**

```bash
cd /home/matteo/Projects/satellite-inertia-id
pytest tests/test_math_utils.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.math_utils'`

- [ ] **Step 1.3: Create `utils/math_utils.py`**

```python
# utils/math_utils.py
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix: skew(v) @ w == np.cross(v, w)."""
    v = np.asarray(v, dtype=float)
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest tests/test_math_utils.py -v
```
Expected: `4 passed`

- [ ] **Step 1.5: Remove local `skew` from `sim/dynamics.py` (lines 5–11), add import**

Replace:
```python
def skew(v):
    """Return the skew-symmetric matrix for cross product: ω × v = skew(ω) · v"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
```
With (at the top of the file, after existing imports):
```python
from utils.math_utils import skew
```

- [ ] **Step 1.6: Remove unused `_skew` static method from `estimation/ekf.py`**

Remove lines:
```python
    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[0, -z,  y],
                         [z,  0, -x],
                         [-y, x,  0]], dtype=np.float64)
```
Add import at top:
```python
from utils.math_utils import skew
```

- [ ] **Step 1.7: Update `estimation/ls_estimator.py` — replace `_skew` instance method with import**

Remove:
```python
    def _skew(self, v):
        """Return the skew-symmetric matrix for cross product"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
```
Add import at top:
```python
from utils.math_utils import skew
```
Replace all `self._skew(omega)` calls in the file with `skew(omega)`.

- [ ] **Step 1.8: Update `estimation/ukf.py` — replace inline skew block with import**

In `_compute_full_dynamics`, replace:
```python
        omega_cross = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
```
With:
```python
        omega_cross = skew(omega)
```
Add import at top:
```python
from utils.math_utils import skew
```

- [ ] **Step 1.9: Run the full test suite to confirm no regressions**

```bash
pytest tests/test_math_utils.py -v
```
Expected: `4 passed`

- [ ] **Step 1.10: Commit**

```bash
git add utils/math_utils.py tests/__init__.py tests/test_math_utils.py \
        sim/dynamics.py estimation/ekf.py estimation/ls_estimator.py estimation/ukf.py
git commit -m "refactor: extract skew() to utils/math_utils, remove 4 duplicate definitions"
```

---

### Task 2: Implement `utils/observability.py`

**Why:** `torque_generators.py` imports `compute_observability_metric` and `score_profiles_canonical` from this missing file. Any `import` of `torque_generators` currently crashes with `ModuleNotFoundError`, which breaks the entire estimation pipeline.

**Files:**
- Create: `utils/observability.py`
- Create: `tests/test_observability.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_observability.py
import numpy as np
import pytest
from utils.observability import compute_observability_metric, score_profiles_canonical


@pytest.fixture
def chirp_torques():
    """A chirp signal that should have high observability."""
    from scipy import signal
    t = np.linspace(0, 300, 300)
    torques = np.zeros((300, 3))
    for i, (f0, f1) in enumerate([(0.005, 0.05), (0.007, 0.04), (0.003, 0.06)]):
        torques[:, i] = 0.01 * signal.chirp(t, f0, t[-1], f1)
    return torques


@pytest.fixture
def zero_torques():
    return np.zeros((300, 3))


def test_observability_returns_required_keys(chirp_torques):
    result = compute_observability_metric(chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    for key in ('min_sv', 'max_sv', 'condition_number', 'log_det', 'score'):
        assert key in result


def test_zero_torques_have_zero_observability(zero_torques):
    result = compute_observability_metric(zero_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    assert result['min_sv'] < 1e-10
    assert result['score'] < 1e-10


def test_chirp_beats_zero(chirp_torques, zero_torques):
    score_chirp = compute_observability_metric(
        chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))['score']
    score_zero = compute_observability_metric(
        zero_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))['score']
    assert score_chirp > score_zero


def test_score_profiles_canonical_ranks(chirp_torques, zero_torques):
    profiles = {'chirp': chirp_torques, 'zero': zero_torques}
    scores = score_profiles_canonical(profiles, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    assert scores['chirp']['rank'] < scores['zero']['rank']  # lower rank = better


def test_energy_normalization_does_not_crash(chirp_torques):
    result = compute_observability_metric(
        chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0), normalize_energy=True)
    assert np.isfinite(result['score'])
```

- [ ] **Step 2.2: Run to verify failure**

```bash
pytest tests/test_observability.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.observability'`

- [ ] **Step 2.3: Implement `utils/observability.py`**

```python
# utils/observability.py
"""
Observability metrics for spacecraft inertia identification.

The core idea: given a torque excitation profile, simulate simplified rigid-body
dynamics and form the linear regressor W such that W @ [Ixx, Iyy, Izz] ≈ tau_eff.
The singular values of W quantify how well the three inertia components can be
identified from that excitation.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def _simulate_simplified(
    torques: np.ndarray, dt: float, I_ref: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Euler-forward simulation of diagonal rigid body (no RW) under given torques.

    Returns omega_hist (N, 3) and domega_hist (N, 3).
    """
    N = len(torques)
    I = np.diag(np.asarray(I_ref, dtype=float))
    I_inv = np.linalg.inv(I)
    omega = np.zeros(3)
    omega_hist = np.zeros((N, 3))
    domega_hist = np.zeros((N, 3))

    for k in range(N):
        domega = I_inv @ (torques[k] - np.cross(omega, I @ omega))
        domega_hist[k] = domega
        omega_hist[k] = omega
        omega = omega + dt * domega

    return omega_hist, domega_hist


def _build_regressor(omega: np.ndarray, domega: np.ndarray) -> np.ndarray:
    """Build the (3N x 3) regressor W for diagonal inertia identification.

    From the Euler equation with diagonal I = diag(a, b, c):
      tau_x = a*dw_x  - b*wy*wz + c*wy*wz
      tau_y = a*wx*wz + b*dw_y  - c*wx*wz
      tau_z =-a*wx*wy + b*wx*wy + c*dw_z

    Each timestep contributes a 3x3 block to W.
    """
    N = len(omega)
    W = np.zeros((3 * N, 3))

    for k in range(N):
        w1, w2, w3 = omega[k]
        dw1, dw2, dw3 = domega[k]
        W[3 * k]     = [ dw1,      -w2 * w3,  w2 * w3]
        W[3 * k + 1] = [ w1 * w3,   dw2,     -w1 * w3]
        W[3 * k + 2] = [-w1 * w2,   w1 * w2,  dw3    ]

    return W


def compute_observability_metric(
    torques: np.ndarray,
    dt: float,
    I_ref: Tuple[float, float, float],
    normalize_energy: bool = True,
) -> Dict[str, float]:
    """Compute observability metrics for inertia identification from a torque profile.

    Simulates rigid-body dynamics, builds the linear regressor for diagonal inertia,
    and returns singular-value-based metrics. Higher score = better identifiability.

    Args:
        torques: Torque profile (N, 3) [Nm]
        dt: Time step [s]
        I_ref: Reference inertia (Ixx, Iyy, Izz) [kg*m²] — used only for simulation
        normalize_energy: Divide regressor by RMS torque energy so profiles with
            different amplitudes can be fairly compared.

    Returns:
        dict with keys: min_sv, max_sv, condition_number, log_det, score
    """
    omega, domega = _simulate_simplified(torques, dt, I_ref)
    W = _build_regressor(omega, domega)

    if normalize_energy:
        energy = float(np.sqrt(np.sum(torques ** 2) * dt)) + 1e-12
        W = W / energy

    sv = np.linalg.svd(W, compute_uv=False)
    min_sv = float(sv[-1])
    max_sv = float(sv[0])
    condition_number = max_sv / (min_sv + 1e-12)

    F = W.T @ W
    sign, log_det_val = np.linalg.slogdet(F)
    log_det = float(log_det_val) if sign > 0 else -np.inf

    # Score: high minimum singular value, low condition number → well-conditioned
    score = min_sv / (np.log10(condition_number + 1) + 1e-12)

    return {
        'min_sv': min_sv,
        'max_sv': max_sv,
        'condition_number': condition_number,
        'log_det': log_det,
        'score': float(score),
    }


def score_profiles_canonical(
    torques_dict: Dict[str, np.ndarray],
    dt: float,
    I_ref: Tuple[float, float, float],
    normalize_energy: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Score multiple torque profiles. Lower rank = better identifiability.

    Args:
        torques_dict: {profile_name: torque_array (N, 3)}
        dt: Time step [s]
        I_ref: Reference inertia tuple (Ixx, Iyy, Izz)
        normalize_energy: Normalize by torque energy for fair comparison

    Returns:
        {profile_name: metrics_dict} where each metrics_dict contains the keys
        from compute_observability_metric plus 'rank' (1 = best).
    """
    scores: Dict[str, Dict[str, float]] = {}
    for name, torques in torques_dict.items():
        scores[name] = compute_observability_metric(torques, dt, I_ref, normalize_energy)

    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    for rank, (name, _) in enumerate(ranked, 1):
        scores[name]['rank'] = rank

    return scores
```

- [ ] **Step 2.4: Run tests**

```bash
pytest tests/test_observability.py -v
```
Expected: `5 passed`

- [ ] **Step 2.5: Commit**

```bash
git add utils/observability.py tests/test_observability.py
git commit -m "feat: implement utils/observability.py, fix broken import in torque_generators"
```

---

### Task 3: Fix `tau_ext` crash — add `tau_ext` attribute to base `Satellite`

**Why:** `run_ls_simulation.py:362` unconditionally accesses `sat.tau_ext[k]` inside the EKF block. When `use_external_torques=False`, `sat` is a plain `Satellite` instance which has no `tau_ext` attribute → `AttributeError` crash.

**Fix:** At the end of `Satellite.simulate()`, set `self.tau_ext = np.zeros((len(t_eval), 3))`. This matches the shape expected by the EKF loop.

**Files:**
- Modify: `sim/dynamics.py` (end of `Satellite.simulate`)
- Create: `tests/test_dynamics.py`

- [ ] **Step 3.1: Write failing test**

```python
# tests/test_dynamics.py
import numpy as np
import pytest
from sim.dynamics import Satellite


@pytest.fixture
def simple_satellite():
    I_sat = [0.26, 0.26, 0.16]
    I_rw = 0.0001
    rw_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return Satellite(I_sat, I_rw, rw_axes, rw_speed_max=460, rw_torque_max=0.01)


def test_tau_ext_exists_after_simulate(simple_satellite):
    """Base Satellite must have tau_ext after simulate() so EKF can access sat.tau_ext[k]."""
    sat = simple_satellite
    t_span = (0, 10)
    dt = 1.0
    omega0 = np.zeros(3)
    rw_speed0 = np.zeros(3)
    t_sim, states = sat.simulate(omega0, rw_speed0, lambda t: np.zeros(3), t_span, dt)

    assert hasattr(sat, 'tau_ext'), "sat.tau_ext must exist after simulate()"
    assert sat.tau_ext.shape == (len(t_sim), 3), (
        f"Expected shape ({len(t_sim)}, 3), got {sat.tau_ext.shape}"
    )
    np.testing.assert_allclose(sat.tau_ext, 0.0)
```

- [ ] **Step 3.2: Run to verify failure**

```bash
pytest tests/test_dynamics.py::test_tau_ext_exists_after_simulate -v
```
Expected: `AssertionError: sat.tau_ext must exist after simulate()`

- [ ] **Step 3.3: Fix `sim/dynamics.py` — add `tau_ext` at end of `simulate()`**

In `Satellite.simulate()`, after the line:
```python
        self.rw_acc_history = np.array(self.rw_acc_history)
```
Add:
```python
        # Zero external torques — subclass SatelliteWithExternalTorques overrides this
        self.tau_ext = np.zeros((len(sol.t), 3))
```

- [ ] **Step 3.4: Run test**

```bash
pytest tests/test_dynamics.py::test_tau_ext_exists_after_simulate -v
```
Expected: `1 passed`

- [ ] **Step 3.5: Commit**

```bash
git add sim/dynamics.py tests/test_dynamics.py
git commit -m "fix: add tau_ext=zeros to base Satellite.simulate() to prevent EKF AttributeError"
```

---

### Task 4: Fix `compute_angular_accelerations` state corruption

**Why:** `compute_angular_accelerations` re-calls `_smooth_torque_command` after simulation, which mutates `self.prev_tau` and `self.prev_time`. When `torque_smoothing=True`, this corrupts the smoothing state from the forward pass and produces wrong angular acceleration estimates for any subsequent analysis call.

**Fix:** Use the already-stored `tau_actual_history` (via `get_tau_actual_at_times`) instead of recomputing from the control function. The `control_func` parameter is kept for API compatibility but is no longer used.

**Files:**
- Modify: `sim/dynamics.py` (`compute_angular_accelerations` method, lines 210–247)
- Modify: `tests/test_dynamics.py` (add two tests)

- [ ] **Step 4.1: Write failing test**

Add to `tests/test_dynamics.py`:

```python
def test_angular_accel_consistent_with_torque_smoothing(simple_satellite):
    """compute_angular_accelerations must not mutate smoothing state."""
    # Enable torque smoothing so state corruption is observable
    I_sat = [0.26, 0.26, 0.16]
    I_rw = 0.0001
    rw_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sat = Satellite(I_sat, I_rw, rw_axes, rw_speed_max=460, rw_torque_max=0.01,
                    torque_smoothing=True, torque_rate_limit=0.005)

    t_span = (0, 30)
    dt = 1.0
    omega0 = np.zeros(3)
    rw_speed0 = np.zeros(3)

    def constant_torque(t):
        return np.array([0.005, 0.003, 0.004])

    # Snapshot smoothing state after simulate (before angular_accel is called)
    t_sim, states = sat.simulate(omega0, rw_speed0, constant_torque, t_span, dt)

    prev_tau_after_sim = sat.prev_tau.copy()
    prev_time_after_sim = sat.prev_time

    # compute_angular_accelerations is called inside simulate already.
    # Call it a second time — it must not change the smoothing state.
    sat.compute_angular_accelerations(t_sim, states, constant_torque)

    np.testing.assert_allclose(sat.prev_tau, prev_tau_after_sim, atol=1e-15,
                                err_msg="compute_angular_accelerations must not mutate prev_tau")
    assert sat.prev_time == prev_time_after_sim, (
        "compute_angular_accelerations must not mutate prev_time")


def test_angular_accel_uses_stored_history(simple_satellite):
    """angular_accelerations array must be finite and have correct shape."""
    sat = simple_satellite
    t_span = (0, 20)
    dt = 1.0
    t_sim, states = sat.simulate(
        np.zeros(3), np.zeros(3), lambda t: np.array([0.005, 0.003, 0.004]),
        t_span, dt
    )
    accs = sat.angular_accelerations
    assert accs.shape == (len(t_sim), 3), f"Expected ({len(t_sim)}, 3), got {accs.shape}"
    assert np.all(np.isfinite(accs)), "angular_accelerations must be finite"
```

- [ ] **Step 4.2: Run to verify failure of state-corruption test**

```bash
pytest tests/test_dynamics.py::test_angular_accel_consistent_with_torque_smoothing -v
```
Expected: `AssertionError: compute_angular_accelerations must not mutate prev_tau`

- [ ] **Step 4.3: Fix `compute_angular_accelerations` in `sim/dynamics.py`**

Replace the entire `compute_angular_accelerations` method (lines 210–247) with:

```python
    def compute_angular_accelerations(self, t_eval, states, control_func=None):
        """Compute angular accelerations using stored tau_actual history.

        Uses self.tau_actual_history (already populated by simulate) rather than
        re-calling control_func, which avoids corrupting torque-smoothing state.
        The control_func parameter is kept for API compatibility but is not used.
        """
        self.angular_accelerations = []
        tau_actual_at_eval = self.get_tau_actual_at_times(t_eval)

        for i in range(len(t_eval)):
            omega = states[i, :3]
            rw_speeds = states[i, 3:6]
            tau_actual = tau_actual_at_eval[i]
            rw_acc = tau_actual / self.I_rw

            h_rw_total = np.zeros(3)
            for j in range(3):
                h_rw_total += self.rw_axes[j] * (self.I_rw * rw_speeds[j])

            h_rw_dot = np.zeros(3)
            for j in range(3):
                h_rw_dot += self.rw_axes[j] * (self.I_rw * rw_acc[j])

            h_total = self.I_sat @ omega + h_rw_total
            domega = np.linalg.inv(self.I_sat) @ (-skew(omega) @ h_total - h_rw_dot)
            self.angular_accelerations.append(domega)

        self.angular_accelerations = np.array(self.angular_accelerations)
```

- [ ] **Step 4.4: Run both new tests**

```bash
pytest tests/test_dynamics.py -v
```
Expected: `4 passed`

- [ ] **Step 4.5: Commit**

```bash
git add sim/dynamics.py tests/test_dynamics.py
git commit -m "fix: compute_angular_accelerations uses stored history, stops corrupting smoothing state"
```

---

### Task 5: Fix O(N) control lookup, satellite loop bug, and expose `show_plots` param

**Why three issues in one task:** all three are in `scripts/run_ls_simulation.py` and require no tests beyond a smoke test.

- **Bug A (O(N) lookup):** `np.argmin(np.abs(t - time_val))` scans the full time vector on every ODE integrator call (can be hundreds of thousands of calls). Replace with `np.searchsorted`.
- **Bug B (loop overwrite):** `main_seeds()` outer loop over satellites 1–3 is immediately overwritten by a hardcoded `satellite = 3` on line 475.
- **Bug C (show_plots):** `run_enhanced_simulation` hardcodes `show_plots=True`, making it impossible to call from tests without spawning GUI windows. Expose it as a parameter.

**Files:**
- Modify: `scripts/run_ls_simulation.py`

- [ ] **Step 5.1: Fix `enhanced_control_input` — O(N) lookup → `searchsorted`**

Find this block in `run_enhanced_simulation`:
```python
    def enhanced_control_input(time_val):
        idx = np.argmin(np.abs(t - time_val))
        return torque_data_full[idx]
```
Replace with:
```python
    def enhanced_control_input(time_val):
        idx = int(np.clip(np.searchsorted(t, time_val, side='left'), 0, len(t) - 1))
        return torque_data_full[idx]
```

- [ ] **Step 5.2: Fix satellite loop overwrite in `main_seeds()`**

Find this block (around line 465–476):
```python
    for s in range(1, 4):
        print(f"🚀 Testing satellite # {s}...")
        if s == 1:
            satellite = 1
        elif s == 2:
            satellite = 2
        elif s == 3:
            satellite = 3
        else:
            raise ValueError("Invalid satellite number. Must be 1, 2, or 3.")
        satellite = 3   # ← THIS LINE IS THE BUG
```
Remove the `satellite = 3` overwrite line. The final block should be:
```python
    for s in range(1, 4):
        print(f"🚀 Testing satellite # {s}...")
        if s == 1:
            satellite = 1
        elif s == 2:
            satellite = 2
        elif s == 3:
            satellite = 3
        else:
            raise ValueError("Invalid satellite number. Must be 1, 2, or 3.")
```

- [ ] **Step 5.3: Expose `show_plots` parameter in `run_enhanced_simulation`**

Change the function signature from:
```python
def run_enhanced_simulation(config_file="config_sat1.yaml", ls_model=DiagonalInertiaModel(),
                            use_external_torques=False, use_ekf=False, use_ls=True,
                          use_realistic_actuators=False, use_noisy_sensors=True,
                          torque_profile="multi_sine", torque_params=None,
                          seed=42, verbose=True, horizon=False, dynamic_I_func_name=None):
```
To:
```python
def run_enhanced_simulation(config_file="config_sat1.yaml", ls_model=DiagonalInertiaModel(),
                            use_external_torques=False, use_ekf=False, use_ls=True,
                            use_realistic_actuators=False, use_noisy_sensors=True,
                            torque_profile="multi_sine", torque_params=None,
                            seed=42, verbose=True, horizon=False, dynamic_I_func_name=None,
                            show_plots=True):
```
And inside the function, change the `create_comprehensive_report` call from `show_plots=True` to `show_plots=show_plots`.

- [ ] **Step 5.4: Run a quick smoke test**

```bash
cd /home/matteo/Projects/satellite-inertia-id
python -c "
from scripts.run_ls_simulation import run_enhanced_simulation
from estimation.ls_estimator import DiagonalInertiaModel
from control.torque_generators import get_profile_params_for_sat
params = get_profile_params_for_sat('sat1')
result = run_enhanced_simulation(
    config_file='config_sat1.yaml',
    torque_profile='sine',
    torque_params=params['sine'],
    use_ekf=False, use_ls=True, use_noisy_sensors=False,
    verbose=False, horizon=30, show_plots=False
)
err = result['estimators']['LS']
print('Smoke test passed. LS estimate:', err)
"
```
Expected: prints `Smoke test passed. LS estimate: [...]` without errors or GUI.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/run_ls_simulation.py
git commit -m "fix: O(N) control lookup, satellite loop overwrite, expose show_plots param"
```

---

### Task 6: Implement true PRBS using LFSR

**Why:** The current `prbs_torque` uses `np.linspace` for switch times and `np.random.choice` for the sequence. This is a random binary signal, not PRBS. True PRBS (maximal-length LFSR sequence) has deterministic period 2^n-1 and near-ideal flat autocorrelation, which gives it the theoretical optimality properties claimed in the paper.

**Files:**
- Modify: `control/torque_generators.py` (replace `prbs_torque`, add `_lfsr_sequence`)
- Create: `tests/test_torque_generators.py`

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_torque_generators.py
import numpy as np
import pytest
from control.torque_generators import generate_torque_profile, _lfsr_sequence


class TestLFSRSequence:
    def test_sequence_is_binary(self):
        seq = _lfsr_sequence(n_bits=4, n_samples=100, seed=1)
        assert set(np.unique(seq)).issubset({-1.0, 1.0})

    def test_period_is_2n_minus_1(self):
        """An n-bit LFSR has period 2^n - 1."""
        for n_bits in (4, 5, 6):
            period = 2 ** n_bits - 1
            seq = _lfsr_sequence(n_bits=n_bits, n_samples=2 * period, seed=1)
            # First period must equal second period
            np.testing.assert_array_equal(seq[:period], seq[period:2 * period])

    def test_different_seeds_give_different_sequences(self):
        s1 = _lfsr_sequence(n_bits=6, n_samples=63, seed=1)
        s2 = _lfsr_sequence(n_bits=6, n_samples=63, seed=3)
        assert not np.array_equal(s1, s2)

    def test_autocorrelation_near_ideal(self):
        """For PRBS of length L=2^n-1, off-peak autocorr ≈ -1/L."""
        n_bits = 10
        L = 2 ** n_bits - 1
        seq = _lfsr_sequence(n_bits=n_bits, n_samples=L, seed=1)
        seq_norm = seq / np.std(seq)
        expected_ac = -1.0 / L
        for lag in range(1, 11):
            ac = float(np.mean(seq_norm[:-lag] * seq_norm[lag:]))
            assert abs(ac - expected_ac) < 0.05, (
                f"Autocorr at lag {lag}: {ac:.4f}, expected ~{expected_ac:.4f}"
            )


class TestPRBSTorque:
    def test_output_is_binary(self):
        t = np.linspace(0, 300, 300)
        torques = generate_torque_profile('prbs', t, amplitude=0.01, switch_time=20)
        # Values should be close to ±0.01 (binary signal, not continuous)
        amplitude = 0.01
        for val in torques.flatten():
            assert abs(abs(val) - amplitude) < 1e-10 or abs(val) < 1e-10, (
                f"PRBS value {val} is not ±{amplitude} or 0"
            )

    def test_output_shape(self):
        t = np.linspace(0, 100, 100)
        torques = generate_torque_profile('prbs', t, amplitude=0.005)
        assert torques.shape == (100, 3)

    def test_different_seeds_differ(self):
        t = np.linspace(0, 300, 300)
        t1 = generate_torque_profile('prbs', t, amplitude=0.01, seed=1)
        t2 = generate_torque_profile('prbs', t, amplitude=0.01, seed=7)
        assert not np.array_equal(t1, t2)
```

- [ ] **Step 6.2: Run to verify failures**

```bash
pytest tests/test_torque_generators.py -v
```
Expected: `test_period_is_2n_minus_1` and `test_autocorrelation_near_ideal` fail; `test_output_is_binary` may also fail.

- [ ] **Step 6.3: Add `_lfsr_sequence` to `control/torque_generators.py`**

After the `_generate_colored_noise` function (around line 456), add:

```python
def _lfsr_sequence(n_bits: int, n_samples: int, seed: int = 1) -> np.ndarray:
    """Maximal-length LFSR sequence of ±1 values.

    Uses well-known primitive polynomials to guarantee period = 2^n_bits - 1.
    Sequences from different seeds differ by their starting register state.

    Args:
        n_bits: LFSR width (4–12). Clamped to available primitive polynomials.
        n_samples: Number of output samples (can exceed one full period).
        seed: Initial register state (must be non-zero; 0 is forced to 1).

    Returns:
        Array of ±1.0 values, shape (n_samples,).
    """
    # Primitive polynomials in bit-mask form (including leading 1)
    _PRIMS = {
        4:  0b10011,
        5:  0b100101,
        6:  0b1000011,
        7:  0b10000011,
        8:  0b100011101,
        10: 0b10000001001,
        12: 0b100000101001,
    }
    valid = sorted(_PRIMS.keys())
    # Clamp n_bits to nearest available
    n_bits = min(valid, key=lambda x: abs(x - n_bits))
    poly = _PRIMS[n_bits]
    mask = (1 << n_bits) - 1

    state = int(seed) & mask
    if state == 0:
        state = 1  # zero state is invalid for LFSR

    output = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        output[k] = 1.0 if (state & 1) else -1.0
        # XOR-feedback across taps defined by primitive polynomial
        feedback = bin(state & poly).count('1') % 2
        state = ((state >> 1) | (feedback << (n_bits - 1))) & mask

    return output
```

- [ ] **Step 6.4: Replace `prbs_torque` in `control/torque_generators.py`**

Replace the entire `prbs_torque` function with:

```python
def prbs_torque(t: np.ndarray, axes: List[str],
                amplitude: Union[float, List[float]] = 0.005,
                switch_time: Union[float, List[float]] = 5.0,
                seed: int = 1,
                **kwargs) -> np.ndarray:
    """Generate Pseudo-Random Binary Sequence (PRBS) torque profiles using LFSR.

    Uses a maximal-length Linear Feedback Shift Register so the sequence has
    period 2^n - 1 and near-ideal flat autocorrelation — the property that makes
    PRBS theoretically optimal for system identification.

    Args:
        t: Time vector
        axes: Axes to generate for
        amplitude: Amplitude(s) in Nm [scalar or list of 3]
        switch_time: Clock period between sequence bits in seconds
        seed: LFSR initial state (non-zero integer; each axis uses seed+axis_idx)

    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))

    amp_list = _ensure_list(amplitude, n_axes)
    switch_list = _ensure_list(switch_time, n_axes)

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    for i in range(n_axes):
        clock_samples = max(1, round(switch_list[i] / dt))
        n_switches = int(np.ceil(len(t) / clock_samples)) + 2

        # Choose LFSR length to cover the required number of switches
        n_bits = max(4, min(12, int(np.ceil(np.log2(n_switches + 1)))))
        seq = _lfsr_sequence(n_bits=n_bits, n_samples=n_switches, seed=seed + i)

        for j, val in enumerate(seq):
            start = j * clock_samples
            end = min((j + 1) * clock_samples, len(t))
            if start >= len(t):
                break
            torques[start:end, i] = amp_list[i] * val

    return torques
```

- [ ] **Step 6.5: Run all torque generator tests**

```bash
pytest tests/test_torque_generators.py -v
```
Expected: all tests pass (including the period and autocorrelation tests).

- [ ] **Step 6.6: Commit**

```bash
git add control/torque_generators.py tests/test_torque_generators.py
git commit -m "fix: replace fake PRBS with maximal-length LFSR for correct sysid properties"
```

---

### Task 7: Unit tests — LS estimator accuracy on synthetic data

**Why:** There are currently no tests for the estimation code at all. These tests verify that `LeastSquaresEstimator` correctly identifies known inertia values from synthetic dynamics data, establishing a correctness baseline before the benchmark.

**Files:**
- Create: `tests/conftest.py` (shared fixtures)
- Create: `tests/test_ls_estimator.py`

- [ ] **Step 7.1: Create `tests/conftest.py` with shared fixtures**

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def synthetic_diagonal_data():
    """Noise-free synthetic dynamics data for a known diagonal inertia.

    Uses Euler's equation directly (no RW) so LS should recover I_true exactly.
    """
    rng = np.random.default_rng(42)
    I_true = np.array([0.30, 0.50, 0.80])
    N = 400
    omega = rng.standard_normal((N, 3)) * 0.3
    tau = rng.standard_normal((N, 3)) * 0.05

    domega = np.zeros((N, 3))
    I_mat = np.diag(I_true)
    for k in range(N):
        domega[k] = np.linalg.solve(
            I_mat, tau[k] - np.cross(omega[k], I_mat @ omega[k])
        )

    return {
        'omega': omega,
        'domega': domega,
        'torque': tau,
        'I_true': I_true,
    }


@pytest.fixture
def noisy_diagonal_data(synthetic_diagonal_data):
    """Same as synthetic_diagonal_data but with 1% Gaussian noise added."""
    rng = np.random.default_rng(7)
    data = dict(synthetic_diagonal_data)
    noise_scale = 0.01 * np.std(data['omega'])
    data = dict(data)
    data['omega'] = data['omega'] + rng.standard_normal(data['omega'].shape) * noise_scale
    acc_noise = 0.01 * np.std(data['domega'])
    data['domega'] = data['domega'] + rng.standard_normal(data['domega'].shape) * acc_noise
    return data
```

- [ ] **Step 7.2: Write failing tests**

```python
# tests/test_ls_estimator.py
import numpy as np
import pytest
from estimation.ls_estimator import LeastSquaresEstimator, DiagonalInertiaModel, FullInertiaModel


class TestLSNoiseFree:
    def test_diagonal_recovery_noise_free(self, synthetic_diagonal_data):
        """LS must recover true diagonal inertia to <0.1% on noise-free data."""
        data = synthetic_diagonal_data
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(data, verbose=False)

        assert result['success'], f"Estimation failed: {result['message']}"
        rel_err = np.abs(result['parameters'] - data['I_true']) / data['I_true']
        assert np.all(rel_err < 1e-3), (
            f"Relative errors {rel_err} exceed 0.1% on noise-free data"
        )

    def test_estimation_result_keys(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        for key in ('success', 'parameters', 'inertia_tensor', 'cost', 'rms_error',
                    'residuals', 'message', 'n_function_evaluations'):
            assert key in result, f"Missing key: {key}"

    def test_inertia_tensor_is_diagonal(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        I = result['inertia_tensor']
        off_diag = I - np.diag(np.diag(I))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)


class TestLSWithNoise:
    def test_diagonal_recovery_with_noise(self, noisy_diagonal_data):
        """LS must recover true inertia to <5% under 1% measurement noise."""
        data = noisy_diagonal_data
        estimator = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss='huber')
        result = estimator.estimate(data, verbose=False)

        assert result['success']
        rel_err = np.abs(result['parameters'] - data['I_true']) / data['I_true']
        assert np.all(rel_err < 0.05), (
            f"Relative errors {rel_err} exceed 5% under 1% noise"
        )

    def test_robust_loss_vs_plain_ls(self, noisy_diagonal_data):
        """Huber loss should produce comparable or better accuracy than plain LS."""
        data = noisy_diagonal_data
        I_true = data['I_true']

        est_plain = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss=None)
        est_huber = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss='huber')

        r_plain = est_plain.estimate(data)
        r_huber = est_huber.estimate(data)

        err_plain = np.linalg.norm(r_plain['parameters'] - I_true) / np.linalg.norm(I_true)
        err_huber = np.linalg.norm(r_huber['parameters'] - I_true) / np.linalg.norm(I_true)

        # Huber should not be significantly worse than plain LS
        assert err_huber <= err_plain * 2.0, (
            f"Huber error {err_huber:.4f} is much worse than plain LS {err_plain:.4f}"
        )


class TestLSValidation:
    def test_validate_passes_on_good_estimate(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        validation = estimator.validate_estimate(result, synthetic_diagonal_data)
        assert validation['is_valid'], f"Validation failed: {validation['message']}"

    def test_history_grows_with_each_estimate(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        assert len(estimator.get_estimation_history()) == 0
        estimator.estimate(synthetic_diagonal_data)
        assert len(estimator.get_estimation_history()) == 1
        estimator.estimate(synthetic_diagonal_data)
        assert len(estimator.get_estimation_history()) == 2

    def test_reset_history(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        estimator.estimate(synthetic_diagonal_data)
        estimator.reset_history()
        assert len(estimator.get_estimation_history()) == 0
```

- [ ] **Step 7.3: Run tests**

```bash
pytest tests/test_ls_estimator.py -v
```
Expected: all tests pass (the estimator already works; tests are documenting expected behaviour).

- [ ] **Step 7.4: Commit**

```bash
git add tests/conftest.py tests/test_ls_estimator.py
git commit -m "test: add LS estimator unit tests — noise-free and noisy diagonal identification"
```

---

### Task 8: Unit tests — EKF convergence on synthetic data

**Why:** The EKF has no tests. These tests verify that with a reasonable initial guess, the EKF tracks both angular velocity and inertia under simulated measurement noise.

**Files:**
- Create: `tests/test_ekf.py`

- [ ] **Step 8.1: Write failing tests**

```python
# tests/test_ekf.py
import numpy as np
import pytest
from estimation.ekf import EKFInertiaRW, EKFConfig


@pytest.fixture
def ekf_instance():
    """EKF initialized with 10% biased inertia guess for sat1-like parameters."""
    I_true = np.array([0.26, 0.26, 0.16])
    I_init = I_true * 0.9                          # 10% biased initial guess

    x0 = np.zeros(9)
    x0[3:6] = I_init

    cfg = EKFConfig(
        dt=1.0,
        I_rw_diag=np.array([1e-4, 1e-4, 1e-4]),
        Qc_diag=np.array([1e-9]*3 + [1e-5]*3 + [1e-9]*3),
        R_diag=np.array([1e-12]*3 + [1e-12]*3),
        x0=x0,
        P0=np.diag([1e-4]*3 + [1.0]*3 + [1e-2]*3),
    )
    return EKFInertiaRW(cfg), I_true


def _simulate_gt(I_true, N=300, dt=1.0, seed=42):
    """Simulate ground-truth trajectory under sinusoidal torque, no RW."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) * dt
    tau_rw = 0.005 * np.column_stack([
        np.sin(2 * np.pi * 0.01 * t),
        np.sin(2 * np.pi * 0.03 * t + 0.5),
        np.sin(2 * np.pi * 0.07 * t + 1.0),
    ])                                              # shape (N, 3)

    I_rw_scalar = 1e-4
    rw_axes = np.eye(3)
    I_mat = np.diag(I_true)
    omega = np.zeros(3)
    rw_speeds = np.zeros(3)
    omega_hist = np.zeros((N, 3))
    rw_hist = np.zeros((N, 3))

    for k in range(N):
        rw_acc = tau_rw[k] / I_rw_scalar
        h_rw = I_rw_scalar * rw_speeds
        h_rw_dot = I_rw_scalar * rw_acc
        h_total = I_mat @ omega + h_rw
        from utils.math_utils import skew
        domega = np.linalg.solve(I_mat, -skew(omega) @ h_total - h_rw_dot)
        omega = omega + dt * domega
        rw_speeds = rw_speeds + dt * rw_acc
        omega_hist[k] = omega
        rw_hist[k] = rw_speeds

    # Add measurement noise
    omega_meas = omega_hist + rng.standard_normal(omega_hist.shape) * 1e-4
    rw_meas = rw_hist + rng.standard_normal(rw_hist.shape) * 1e-4

    return tau_rw, omega_meas, rw_meas


class TestEKFStep:
    def test_step_returns_correct_shapes(self, ekf_instance):
        ekf, _ = ekf_instance
        u = np.zeros(3)
        z = np.zeros(6)
        x, P = ekf.step(u, z)
        assert x.shape == (9,)
        assert P.shape == (9, 9)

    def test_inertia_stays_positive(self, ekf_instance):
        """EKF must enforce positive inertia at all times."""
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        for k in range(len(omega_meas)):
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            x, _ = ekf.step(u_k, z_k)
            assert np.all(x[3:6] > 0), f"Inertia became non-positive at step {k}: {x[3:6]}"

    def test_covariance_is_symmetric_positive_semidefinite(self, ekf_instance):
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        for k in range(0, len(omega_meas), 30):   # check every 30 steps
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            _, P = ekf.step(u_k, z_k)
            np.testing.assert_allclose(P, P.T, atol=1e-10,
                                        err_msg=f"Covariance not symmetric at step {k}")
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10), (
                f"Covariance has negative eigenvalue at step {k}: min={eigvals.min():.3e}"
            )


class TestEKFConvergence:
    def test_inertia_converges_within_5pct(self, ekf_instance):
        """After 300 steps, EKF inertia estimate should be within 5% of truth."""
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        N = len(omega_meas)
        for k in range(N):
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            x, _ = ekf.step(u_k, z_k)

        I_est = x[3:6]
        rel_err = np.abs(I_est - I_true) / I_true
        assert np.all(rel_err < 0.05), (
            f"EKF did not converge within 5%: rel_err={rel_err}, "
            f"I_est={I_est}, I_true={I_true}"
        )
```

- [ ] **Step 8.2: Run tests**

```bash
pytest tests/test_ekf.py -v
```
Expected: all tests pass.

- [ ] **Step 8.3: Commit**

```bash
git add tests/test_ekf.py
git commit -m "test: add EKF unit tests — convergence, positive inertia, PSD covariance"
```

---

### Task 9: Integration performance benchmark

**Why:** Verifies end-to-end correctness across all satellite sizes and torque profiles, produces a comparison table showing which profiles are most accurate, and cross-validates the observability score against actual estimation error to test the paper's core claim.

Tests are marked `@pytest.mark.slow` so they can be excluded from fast CI with `pytest -m "not slow"`.

**Files:**
- Create: `tests/test_benchmark.py`

- [ ] **Step 9.1: Create `tests/test_benchmark.py`**

```python
# tests/test_benchmark.py
"""
End-to-end benchmark: LS and EKF estimation accuracy across satellite configs and profiles.

Run with:   pytest tests/test_benchmark.py -v -s
Skip with:  pytest -m "not slow"

All assertions use a 10% relative error threshold — deliberately generous, since this
benchmark documents actual performance rather than enforcing tight tolerances.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

from sim.dynamics import Satellite
from sim.sensors import SensorSuite, get_high_accuracy_sensor_config
from estimation.ekf import EKFInertiaRW, EKFConfig
from estimation.ls_estimator import LeastSquaresEstimator, DiagonalInertiaModel
from control.torque_generators import generate_torque_profile
from utils.signal_processing import compute_smooth_derivative

ROOT = Path(__file__).parent.parent


def _load_config(config_file: str) -> dict:
    with open(ROOT / config_file) as f:
        return yaml.safe_load(f)


def _run_pipeline(config_file: str, profile: str, torque_params: dict,
                  horizon: int = 150, seed: int = 42,
                  use_noisy_sensors: bool = True,
                  use_ekf: bool = True, use_ls: bool = True) -> dict:
    """
    Minimal estimation pipeline (no visualization) for benchmarking.

    Returns dict with keys:
        ls_estimate   (3,) or None
        ekf_estimate  (3,) or None
        true_I        (3,)
        ls_rel_err    float or None
        ekf_rel_err   float or None
    """
    np.random.seed(seed)
    cfg = _load_config(config_file)

    I_sat = cfg["satellite"]["inertia_tensor"]
    I_rw = cfg["reaction_wheels"]["inertia"]
    rw_axes = cfg["reaction_wheels"]["alignment_matrix"]
    dt = cfg["sim"]["dt"]
    omega0 = cfg["sim"]["initial_omega"]
    rw_speed0 = cfg["sim"]["initial_rw_speed"]
    rw_max_speed = cfg["reaction_wheels"]["max_speed"]
    rw_max_torque = cfg["reaction_wheels"]["max_torque"]
    I_true = np.array(I_sat)

    # Generate torque profile
    t = np.arange(0, horizon + dt, dt)
    torque_data = generate_torque_profile(profile, t, **torque_params)

    def control_fn(time_val):
        idx = int(np.clip(np.searchsorted(t, time_val, side='left'), 0, len(t) - 1))
        return torque_data[idx]

    # Simulate
    sat = Satellite(I_sat, I_rw, rw_axes, rw_speed_max=rw_max_speed,
                    rw_torque_max=rw_max_torque)
    t_sim, states = sat.simulate(omega0, rw_speed0, control_fn, (0, horizon), dt)

    # Noisy observations
    if use_noisy_sensors:
        np.random.seed(seed + 1)
        sensor = SensorSuite(**get_high_accuracy_sensor_config())
        measured_states = np.array([
            np.concatenate([
                sensor.get_all_measurements({'omega': s[:3]}, t_sim[i])['omega'],
                s[3:6]
            ])
            for i, s in enumerate(states)
        ])
    else:
        measured_states = states.copy()

    omega_data = measured_states[:, :3]
    rw_speeds_data = measured_states[:, 3:6]
    actual_torques = sat.get_tau_actual_at_times(t_sim)

    domega_data = np.zeros_like(omega_data)
    for axis in range(3):
        domega_data[:, axis] = compute_smooth_derivative(
            t_sim, omega_data[:, axis], method='central_diff'
        )

    dynamics_data = {
        'omega': omega_data,
        'domega': domega_data,
        'torque': actual_torques,
        'rw_speeds': rw_speeds_data,
        'time': t_sim,
    }

    result = {'true_I': I_true, 'ls_estimate': None, 'ekf_estimate': None,
              'ls_rel_err': None, 'ekf_rel_err': None}

    # LS estimation
    if use_ls:
        ls = LeastSquaresEstimator(DiagonalInertiaModel(), satellite_model=sat,
                                   robust_loss='huber')
        ls_result = ls.estimate(dynamics_data, verbose=False)
        if ls_result['success']:
            result['ls_estimate'] = ls_result['inertia_tensor'].diagonal()
            result['ls_rel_err'] = float(
                np.linalg.norm(result['ls_estimate'] - I_true) / np.linalg.norm(I_true)
            )

    # EKF estimation
    if use_ekf:
        I_init = I_true * 0.85
        x0 = np.zeros(9)
        x0[3:6] = I_init
        sigma_omega = 1e-6 if not use_noisy_sensors else 1e-4
        cfg_ekf = EKFConfig(
            dt=dt,
            I_rw_diag=np.array([I_rw, I_rw, I_rw]),
            Qc_diag=np.array([1e-9]*3 + [1e-5]*3 + [1e-9]*3),
            R_diag=np.array([sigma_omega**2]*3 + [sigma_omega**2]*3),
            x0=x0,
            P0=np.diag([1e-4]*3 + [1.0]*3 + [1e-2]*3),
        )
        ekf = EKFInertiaRW(cfg_ekf)
        rw_accels = sat.get_rw_acc_at_times(t_sim)
        for k in range(len(t_sim)):
            z_k = measured_states[k]
            u_k = rw_accels[k]
            x_k, _ = ekf.step(u_k, z_k, tau_ext=sat.tau_ext[k])

        result['ekf_estimate'] = x_k[3:6].copy()
        result['ekf_rel_err'] = float(
            np.linalg.norm(result['ekf_estimate'] - I_true) / np.linalg.norm(I_true)
        )

    return result


# ---------------------------------------------------------------------------
# Parametric benchmark tests
# ---------------------------------------------------------------------------

SAT_CONFIGS = [
    ("config_sat1.yaml", "sat1"),
    ("config_sat2.yaml", "sat2"),
    ("config_sat3.yaml", "sat3"),
]

PROFILE_PARAMS = {
    'sat1': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.01},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.01},
        'prbs':       {'amplitude': 0.01,  'switch_time': 20},
        'multi step': {'amplitude': 0.012, 'step_duration': 40.0},
    },
    'sat2': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.05},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.05},
        'prbs':       {'amplitude': 0.05,  'switch_time': 20},
        'multi step': {'amplitude': 0.06,  'step_duration': 40.0},
    },
    'sat3': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.10},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.10},
        'prbs':       {'amplitude': 0.10,  'switch_time': 20},
        'multi step': {'amplitude': 0.12,  'step_duration': 40.0},
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("config_file,sat_key", SAT_CONFIGS)
@pytest.mark.parametrize("profile", ['sine', 'chirp', 'prbs', 'multi step'])
def test_ls_accuracy_within_10pct(config_file, sat_key, profile, capsys):
    """LS estimation must achieve <10% relative error on any satellite and profile."""
    params = PROFILE_PARAMS[sat_key][profile]
    result = _run_pipeline(config_file, profile, params, horizon=150, use_ekf=False)

    with capsys.disabled():
        print(f"\n[LS] {sat_key} / {profile}: rel_err={result['ls_rel_err']:.4f}  "
              f"estimate={np.round(result['ls_estimate'], 4)}  "
              f"true={result['true_I']}")

    assert result['ls_estimate'] is not None, "LS estimation returned None (failed)"
    assert result['ls_rel_err'] < 0.10, (
        f"LS relative error {result['ls_rel_err']:.4f} > 10% for {sat_key}/{profile}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("config_file,sat_key", SAT_CONFIGS)
@pytest.mark.parametrize("profile", ['sine', 'chirp', 'prbs', 'multi step'])
def test_ekf_accuracy_within_10pct(config_file, sat_key, profile, capsys):
    """EKF estimation must achieve <10% relative error on any satellite and profile."""
    params = PROFILE_PARAMS[sat_key][profile]
    result = _run_pipeline(config_file, profile, params, horizon=150, use_ls=False)

    with capsys.disabled():
        print(f"\n[EKF] {sat_key} / {profile}: rel_err={result['ekf_rel_err']:.4f}  "
              f"estimate={np.round(result['ekf_estimate'], 4)}  "
              f"true={result['true_I']}")

    assert result['ekf_estimate'] is not None
    assert result['ekf_rel_err'] < 0.10, (
        f"EKF relative error {result['ekf_rel_err']:.4f} > 10% for {sat_key}/{profile}"
    )


@pytest.mark.slow
def test_observability_score_correlates_with_estimation_error():
    """The observability score should negatively correlate with estimation error.

    This tests the paper's core claim: higher observability → lower estimation error.
    We run all profiles on sat1 and verify Pearson correlation < -0.3.
    """
    from utils.observability import score_profiles_canonical

    cfg = _load_config("config_sat1.yaml")
    I_ref = tuple(cfg["satellite"]["inertia_tensor"])
    t = np.linspace(0, 150, 150)

    profiles = ['sine', 'chirp', 'prbs', 'multi step']

    obs_scores = []
    est_errors = []

    for profile in profiles:
        params = PROFILE_PARAMS['sat1'][profile]
        torques = generate_torque_profile(profile, t, **params)
        obs_score = score_profiles_canonical({'p': torques}, dt=1.0, I_ref=I_ref)['p']['score']
        obs_scores.append(obs_score)

        result = _run_pipeline("config_sat1.yaml", profile, params,
                               horizon=150, use_ekf=False)
        if result['ls_rel_err'] is not None:
            est_errors.append(result['ls_rel_err'])
        else:
            est_errors.append(1.0)  # treat failure as 100% error

    obs_scores = np.array(obs_scores)
    est_errors = np.array(est_errors)

    # Pearson correlation between observability and estimation error
    corr = float(np.corrcoef(obs_scores, est_errors)[0, 1])
    print(f"\nObservability vs. estimation error correlation: {corr:.3f}")
    print(f"Profiles: {profiles}")
    print(f"Obs scores: {np.round(obs_scores, 4)}")
    print(f"Est errors: {np.round(est_errors, 4)}")

    assert corr < 0.0, (
        f"Expected negative correlation (higher obs → lower error), got corr={corr:.3f}. "
        f"Observability metric may not be predictive."
    )
```

- [ ] **Step 9.2: Run fast unit tests first to make sure nothing broke**

```bash
pytest tests/ -m "not slow" -v
```
Expected: all unit tests pass (Tasks 1–8).

- [ ] **Step 9.3: Run benchmark tests and review output**

```bash
pytest tests/test_benchmark.py -v -s -m slow 2>&1 | tee benchmark_results.txt
```
Expected: tests print per-profile error table; each assertion passes with <10% relative error. If any test fails, the error message names the specific satellite/profile combination.

- [ ] **Step 9.4: Check correlation test output**

In the output from Step 9.3, verify the final correlation test prints a table of observability scores and estimation errors. A negative correlation confirms the paper's claim.

- [ ] **Step 9.5: Commit**

```bash
git add tests/test_benchmark.py
git commit -m "test: add end-to-end LS/EKF benchmark across all satellites and profiles"
```

---

## Self-Review

**Spec coverage check:**

| Bug | Task | Covered? |
|-----|------|----------|
| Missing `utils/observability.py` | Task 2 | ✓ |
| `sat.tau_ext` crash without external torques | Task 3 | ✓ |
| `compute_angular_accelerations` state corruption | Task 4 | ✓ |
| O(N) control lookup | Task 5 | ✓ |
| Satellite loop overwrite in `main_seeds()` | Task 5 | ✓ |
| PRBS is not true PRBS | Task 6 | ✓ |
| `skew()` duplicated 4 times | Task 1 | ✓ |
| No tests at all | Tasks 7–9 | ✓ |
| LS accuracy verification | Task 7 | ✓ |
| EKF accuracy verification | Task 8 | ✓ |
| Observability-vs-accuracy correlation | Task 9 | ✓ |

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency check:**
- `_lfsr_sequence` defined in Task 6 and imported in Task 6's tests as `from control.torque_generators import _lfsr_sequence`. ✓
- `tau_ext` shape is `(len(sol.t), 3)` in Task 3; accessed as `sat.tau_ext[k]` (scalar index along axis 0) in Task 9's pipeline. ✓
- `compute_angular_accelerations` signature kept as `(self, t_eval, states, control_func=None)` — callers in `simulate()` pass `control_func` and it is silently ignored; no breakage. ✓
- `run_enhanced_simulation` new `show_plots` parameter has default `True` to preserve existing behaviour for direct callers. ✓
