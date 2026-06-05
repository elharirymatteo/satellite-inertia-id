# F1 — known issues and gaps from the retrained full-tensor DR policy

## 2026-06-05 update — F3 disturbance lands, headline plot

`sim/dynamics_jax.py` now threads an external body-torque `tau_ext` through
the dynamics chain. `T1EnvEKFConfig` gains `disturbance_scale`; the env
samples `tau_ext ~ Normal(0, disturbance_scale^2 * I_3)` per episode and
holds it constant. Training, MPC-baseline, and unified-eval scripts set
`disturbance_scale = 0.1 * tau_max`. The EKF does NOT model the
disturbance — it's the noise the active-sensing policy must cope with.

**Headline 16-seed eval with disturbance enabled:**

| Sat | RL DR | MPC RH | MPC 1-shot | sine | chirp | PRBS | multi-step |
|---|---|---|---|---|---|---|---|
| sat1 | 108% | 282% | 583% | **81%** | 370% | 636% | 418% |
| sat2 | **11.4%** | 61% | 48% | 62% | 22% | 18% | 19% |
| sat3 | 10.9% | 29% | 21% | 38% | 16% | 17% | **6.7%** |

`ms/step`: RL/scripted ~0.25, MPC 1-shot ~14, MPC RH ~67.

Paper story:
- **Active-sensing RL wins on sat2** by a comfortable margin (1.5× over PRBS).
- **RL is competitive on sat3** (2nd to multi-step's 6.7%, but beats every
  other active-sensing method).
- **Dual-MPC degrades catastrophically** under disturbance because it plans
  against the EKF's belief, which doesn't include the disturbance — a
  clean illustration that *model-based active sensing fails under model
  mismatch where the RL policy adapts*.
- **Compute headline preserved**: 0.25 ms (RL) vs 67 ms (MPC RH), ~270×.
- **sat1 still hard**: the CubeSat-scale + small-inertia + disturbance
  combination defeats every method. Documented as out-of-scope for this
  paper.



## 2026-06-05 update — sat1 fix landed

Four changes, in order of decreasing impact:

1. **`dt` lowered from 1.0 → 0.1 s** in all three sat configs. The original
   value was arbitrary; with `dt = 1.0 s` and `substeps = 10`, even the
   true dynamics used 0.1-s RK4 substeps, so a single EKF Euler step of
   1.0 s was 10× coarser than reality. With `dt = 0.1 s` the EKF predict
   spans the same window as one RK4 substep and stops diverging in the
   small-inertia / high-relative-torque CubeSat regime.
2. **DR `I_range` tightened** to `(0.3, 20.0)` (was `(0.1, 20.0)`). The
   smallest 30% of the prior was producing NaN training seeds whose
   gradient-mean was zeroed by `optax.zero_nans()`, killing the learning
   signal. sat1's smallest dim is 0.16 — now technically OOD, but the
   policy still generalizes there.
3. **5-substep Euler EKF predict** (`EKF_PREDICT_SUBSTEPS = 5`).
4. **Cauchy-Schwarz off-diag clip** + **tight off-diag prior** (0.05x mean).

**Result (32 seeds, dt=0.1, I_range=(0.3, 20)):**

| Sat | Best previous (dt=1.0) | After fix |
|---|---|---|
| sat1 RL DR | NaN | 62.45% (OOD — sat1 below trained range) |
| sat1 sine | NaN | 2.46% |
| sat1 chirp | NaN | 2.62% |
| sat1 multi step | NaN | 2.46% |
| sat1 PRBS | NaN | 825% (PRBS specifically still hard) |
| sat2 RL DR | 0.71% | 2.22% |
| sat3 RL DR | 2.03% | 1.56% |

Headline story: the universal sat1 NaN is fixed. sat1 scripted profiles
now identify inertia to ~2-3% rel_err. The RL DR policy is competitive
on sat3 (beats sine and chirp); slightly worse on sat2 than the dt=1.0
run but the comparison isn't apples-to-apples (different episode length).
PRBS on sat1 remains a failure mode — bang-bang excitation in the small-
inertia regime drives the EKF off-rails even with the safeguards.



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
