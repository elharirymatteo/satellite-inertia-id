# F1 — known issues and gaps from the retrained full-tensor DR policy

## 2026-06-05 update — partial sat1 fix attempt

Three numerical guards were added to `rl/ekf_jax.py` and `rl/t1_env_ekf.py`:

1. **Multi-substep EKF predict** (`EKF_PREDICT_SUBSTEPS = 5`): replace the
   single-step forward Euler in `predict` with 5 Euler substeps over
   `params.dt`. The reviewer's recommendation.
2. **Cauchy-Schwarz off-diagonal clip** in `_project_psd_if_needed`: bound
   each `|I_ij|` to `0.4 * sqrt(I_ii * I_jj)` so the Kalman update can't
   manufacture huge off-diagonals to "explain" omega innovations.
3. **Tighter off-diagonal prior** in `T1EnvEKF.reset`: initial cov on
   `(Ixy, Ixz, Iyz)` uses `0.05 * mean(I_diag)` instead of the diagonal's
   `0.30 * mean(I_diag)` — a 6× tighter prior matching the spec.

**Result:** sat2/sat3 rel_err is roughly unchanged (~1.5–1.7%). sat1 is
still unreliable. The deeper issue is that `config_sat1.yaml` uses
`dt = 1.0 s`, so even with 5 EKF substeps each predict spans 0.2 s of
real dynamics — coarse for the small-inertia / high-relative-torque
CubeSat regime where omega grows fast. The Euler-based EKF linearization
fundamentally cannot keep up. Real options remaining:

- Use a UKF (sigma-point Kalman filter) on the inertia block — handles
  the nonlinear coupling much better at the cost of ~2-3× compute.
- Drop `dt` to 0.1 s for sat1 (config edit) so the EKF predict has the
  same temporal resolution as the true dynamics. Will require retraining.
- Restrict the DR sampling lower bound from 0.1 to ~0.5 so the policy is
  never trained against the extreme small-inertia regime.

None of these are a one-line fix; sat1 is left in the "known limitation"
column for now.



After F1 (full inertia tensor + 12-dim EKF + rotation-based DR sampler), the retrained DR policy on the held-out CubeSat/MicroSat/SmallSat configs (`docs/t1_dr_ekf_full_tensor_neg_rel_err_eval.txt`):

| Sat | DR policy rel_err | diagonal-only baseline | Verdict |
|---|---|---|---|
| sat1 (CubeSat) | NaN | 5.90% | env eval blow-up — see below |
| sat2 (MicroSat) | 0.71% | 0.09% | within ~10×; beats sine/chirp/multi-step |
| sat3 (SmallSat) | 2.03% | 0.13% | within ~15×; beats sine/chirp/multi-step |

## sat1 NaN gap

Every policy on `config_sat1.yaml` (including scripted sine/chirp/PRBS/multi-step) returns NaN in the full-tensor eval. The diagonal-only version of the same eval was fine. This is therefore an EKF/env runtime issue specific to F1 changes, not a policy training failure.

Likely root cause: the 12-dim EKF's combination of `jnp.linalg.solve(I, ...)` inside `f()` and second-order autodiff through `jax.jacobian(analytic_F)` becomes unstable for the smallest-inertia / largest-relative-torque regime. The CubeSat has `tau_max/||I|| ≈ 0.05/0.4 ≈ 0.13` Nm·kg^-1·m^-2 — about 5× the ratio for sat2 and 20× for sat3 — so the rotational acceleration regime is far more extreme. Over 150 steps the EKF state likely accumulates non-finite values.

Fixes deferred to a follow-up patch (not blocking F2):
1. Soft-clip `omega_dot` inside the EKF's `f()`.
2. Use a smaller integration step in the EKF (split the predict into substeps, like the true dynamics).
3. Tighten the DR sampling lower bound (e.g., `I_range = (0.3, 20.0)` instead of `(0.1, 20.0)`) so the policy is never trained on the most extreme regime.

## Training-time safeguards added

To make the diff-sim policy gradient survive rare divergent batch seeds, two guards were added to `scripts/t1_train_dr_ekf.py`:

```python
Rs_safe = jnp.where(jnp.isnan(Rs), -1e4, Rs)
Rs_safe = jnp.clip(Rs_safe, -1e4, 1e4)
```

Without these, even one NaN among 32 vmapped seeds would zero the entire batch's gradient via `optax.zero_nans`, halting learning. With them, training is stable at ~-2000 train_R baseline and produces the table above.

## PSD-projection autodiff fix

`rl/ekf_jax.py::_project_psd_if_needed` was originally written to differentiate through `jnp.linalg.eigh`, which has NaN gradients near degenerate eigenvalues — and second-order autodiff via the policy-gradient path through `analytic_F = jax.jacobian(...)` made this surface frequently. Wrapped the `eigh` input in `jax.lax.stop_gradient` so the projection corrects the state forward but doesn't contribute to gradients. This is a sensible numerical guard; the projection is a sparse correction, not a learnable component.
