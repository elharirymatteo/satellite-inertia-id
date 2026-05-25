# F1 + F2 design — full inertia tensor and MPC baselines

**Status:** approved 2026-05-25
**Scope:** items F1 and F2 from `docs/explainer.html` §6 — the prerequisite work for the RAL-tier follow-up paper to the active-sensing inertia-ID study.
**Out of scope:** F3 (disturbances), F4 (hardware demo), F5 (PPO ablation), F6 (compute paragraph).

## 1. Motivation

The current codebase identifies a **diagonal-only** 3-parameter inertia tensor with a closed-loop RL policy whose reward is the log-det drop of the EKF posterior covariance. Two reviewer objections will sink a RAL submission of that exact result:

1. Candan & Servadio (arXiv:2603.27361, March 2026) already handle the full 6-parameter tensor — the diagonal-only restriction makes the present work strictly less general.
2. The only active-excitation baselines in the current comparison are hand-tuned scripted profiles (sine, chirp, PRBS, multi-step). A reviewer will ask for a competitive *active-sensing* baseline. Without one, "RL beats chirp" is a thin claim.

F1 extends the estimator and FIM machinery to the full 6-parameter symmetric inertia tensor. F2 implements two FIM-optimizing trajectory baselines (receding-horizon dual MPC, and a one-shot offline modernized Wittenburg 2017) against which the RL policy can be fairly compared.

## 2. Order of work

F1 first, then F2 on top of it. F2's planning model and FIM accumulator must match the estimator the RL policy is competing against. If F2 ships against the diagonal-only model, the comparison must be redone after F1 lands.

## 3. F1 — full inertia tensor

### 3.1 Data model

No new fields on `SatParams`. `SatParams.I_sat` is already `(3,3)` and the JAX dynamics in `sim/dynamics_jax.py::_step_dt` already uses the full matrix via `I_inv` — the simulator already supports the full tensor; we simply stopped exploiting it. The 6-parameter view (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) is purely an *estimator* concept and lives inside the EKF state and the LS regressor.

Convention for the 6-vector ordering: `θ = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`. This is the same order used in the FIM regressor rows in §3.4.

### 3.2 PSD-guaranteed DR sampling

Replace the current `rl/t1_env.py::sample_sat`, which draws a diagonal log-uniform. New procedure:

1. Draw diagonal `Λ ∈ R^3` log-uniform in `[0.1, 20]` per axis (unchanged).
2. Draw a random unit axis `u ∈ S^2` and tilt angle `θ ∈ U(0, max_tilt_angle)`; build rotation `R = exp([u]_× θ)`.
3. Return `I_sat = R Λ R^T`.

This guarantees PSD with eigenvalues exactly in `[0.1, 20]` and off-diagonals bounded by `~sin(θ/2) · trace(Λ)`. With `max_tilt_angle = π/8 ≈ 22°`, off-diagonals stay at a few percent of trace — matching the regime real spacecraft live in.

One new config knob: `max_tilt_angle` (default `π/8`).

### 3.3 EKF state extension

State layout grows from 9-dim `[ω, I_diag, Ω_rw]` to 12-dim `[ω(3), I_diag(3), I_offdiag(3), Ω_rw(3)]`, with the same axis ordering as θ in §3.1 — i.e. `I_offdiag = (Ixy, Ixz, Iyz)`.

EKF initialization:

- Initial mean for off-diag: 0 (we expect small).
- Initial cov for off-diag: `(σ_off_rel · mean(I_diag))²` per axis, default `σ_off_rel = 0.1`.
- Process noise on off-diag: relative random walk, same shape as on the diagonal (driven by `Qc_I_rel`).

EKF predict step: the dynamics Jacobian gains 6 new columns for `(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`, derived from the analytic linearization of Euler's equation. Closed form; ~30 lines.

EKF update step (Joseph form): unchanged — keeps `P` PSD regardless of state dim.

**PSD projection guard** (lazy, applied after the Joseph-form update): reconstruct `I_est` from the 6 params; if `min_eig(I_est) < 0`, do one eigendecompose, clip eigenvalues at `1e-4 · max_eig`, recompose into the 6 params. Expected to fire near-never given the regime (low sensor noise, diagonal-dominant truth, 30%-std prior). If telemetry shows it firing on more than 5% of training episodes, escalate to a soft penalty `λ · ReLU(-det(I_est))` in the diff-sim loss; that escalation is not on the critical path.

### 3.4 LS regressor — full 6-parameter rows

Euler's equation `τ = I·ω̇ + ω × (I·ω)` is linear in the 6 unique entries of I. Grouping each component of τ by the parameter ordering `θ = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`:

```
row_x = [  ω̇_x,         −ω_y ω_z,       +ω_y ω_z,
           ω̇_y − ω_x ω_z,  ω̇_z + ω_x ω_y,  ω_y² − ω_z² ]

row_y = [ +ω_x ω_z,      ω̇_y,           −ω_x ω_z,
           ω̇_x + ω_y ω_z,  ω_z² − ω_x²,    ω̇_z − ω_x ω_y ]

row_z = [ −ω_x ω_y,     +ω_x ω_y,       ω̇_z,
           ω_x² − ω_y²,    ω̇_x − ω_y ω_z,  ω̇_y + ω_x ω_z ]
```

Sanity-check: the leading 3 columns collapse exactly to the diagonal-only rows already in `rl/t1_env.py::_fim_row_contribution` (lines 48–51) when off-diagonals are zero.

`tests/test_ls_full_tensor.py` MUST verify these against numerical Jacobians of Euler's equation by finite differences in `(Ixx,…,Iyz)`, evaluated at a handful of random (ω, ω̇) points. Tolerance `1e-8` on absolute coefficient match.

The LS regressor in `estimation/ls_estimator.py` simply gets 3 more columns. Normal-equations form is unchanged; iterative LS-EKF loop in `scripts/run_ls_simulation.py` is unchanged.

### 3.5 FIM regressor

`rl/t1_env.py::_fim_row_contribution` currently returns a 3×3 `R^T R` block (because the regression matrix R is 3×3 in diagonal-only). It becomes 6×6, since R is now 3×6. The reward `log det F` is taken over a 6×6 matrix. `_slogdet_psd` works unchanged; the eps-regularizer still PSD-stabilizes it.

### 3.6 RL policy retraining

The policy observation shape changes:

- `T1Env` (oracle FIM env): 13 → 28 (`omega(3) + rw(3)/rw_max + log1p(F_upper_tri = 21) + progress(1)`).
- `T1EnvEKF` (EKF-in-loop env): 13 → 19 (`omega(3) + rw(3)/rw_max + I_est(6)/I_ref + log diag(P_I)(6) + progress(1)`).

Policy MLP input layer is resized via `policy_mod.init_params(obs_dim=env.obs_shape[0])`. Existing checkpoints are not compatible — retrain from scratch with the same training script (`scripts/t1_train_dr_ekf.py`) modified to use the new env.

## 4. F2 — MPC baselines

### 4.1 Shared planner

New module `control/dual_mpc.py` (~120 lines) implements one core function:

```python
def plan(init_dyn_state, init_F, sat_for_planning, horizon, n_opt_steps, tau_max,
         lr, sat_penalty) -> jnp.ndarray:  # returns tau seq, shape (horizon, 3)
```

Internals:

1. Decision variable: a free `(horizon, 3)` array `z`, initialized from a warm-start (zeros for the first call).
2. Action: `tau = tau_max * tanh(z)` — bound constraints handled by squashing.
3. Loss: roll out the diff-sim dynamics for `horizon` steps starting from `init_dyn_state`, accumulating FIM rows via the same `_fim_row_contribution` used in training; return `-slogdet_psd(F_final + ε I) + λ * sat_pen_total`. The planner uses `sat_for_planning`, which is the EKF's current point estimate of inertia — *not* ground truth. The MPC has no privileged information.
4. Optimizer: ~30 Adam steps, `lr ≈ 0.01 · tau_max`, global-norm clip 1.0.
5. Return: `tau_max * tanh(z)`.

`plan` is JIT-compiled and vmappable over eval seeds.

### 4.2 Receding-horizon dual MPC driver

`scripts/t1_dual_mpc_baseline.py` (~80 lines). Driver loop:

```
horizon = 20             # 2 s at dt=0.1
replan_every M = 10      # the rest of the plan plays open-loop
for t in 0..150 by M:
    plan_tau = plan(init_dyn_state=ekf_mean_at_t,
                    init_F=accumulated_F_so_far,
                    sat_for_planning=SatParams(I_est=ekf_mean.I, ...),
                    horizon=20, ...)
    execute plan_tau[:M] on the true env (which keeps its own EKF)
    warm-start next call: shift plan_tau by M, pad zeros
```

The dual-MPC has no access to ground-truth I; it plans against the EKF's current point estimate. This is *certainty-equivalence dual control* — the simpler of two formulations. A full dual control formulation (planning over the posterior distribution of I) is out of scope for this spec.

### 4.3 One-shot offline driver (modernized Wittenburg 2017)

Same `plan` function called once with `horizon = 150` (the full episode), `n_opt_steps = 100`. The full trajectory is played open-loop with no replanning. The planner uses `sat_for_planning = SatParams(I_est = E[I under DR prior], ...)` — i.e., the prior mean of the DR distribution, since no observations are available before t=0.

This is a strict Wittenburg-class baseline: pre-computed open-loop, FIM-optimized.

### 4.4 Eval harness

New script `scripts/t1_all_baselines.py` runs on identical seeds and writes one results table at `docs/t1_all_baselines_eval.txt`:

| sat_cfg | policy | mean_I_rel_err | std_I_rel_err | mean_logdet_F | wall_ms_per_step |
|---|---|---|---|---|---|
| sat1/2/3 | DR policy (RL) | … | … | … | … |
| sat1/2/3 | dual-MPC (RH) | … | … | … | … |
| sat1/2/3 | one-shot MPC | … | … | … | … |
| sat1/2/3 | sine / chirp / PRBS / multi step | … | … | … | … |

32 eval seeds per row. Metric `‖I_est − I_true‖_F / ‖I_true‖_F` (Frobenius — full tensor).

Wall-clock-per-step is logged for all methods, giving the "RL matches MPC at 100× lower cost" line for the paper without extra work.

## 5. Success criteria

**F1 done when:**

- `pytest tests/test_ls_full_tensor.py tests/test_ekf_full_tensor.py` pass. The LS test verifies recovery of a known full-tensor on noise-free data within `1e-8`. The EKF test verifies posterior cov stays PSD over a 150-step rollout under the new DR sampler.
- A retrained DR policy (with the new full-tensor env and 19-dim obs) reports `mean_I_rel_err ≤ 1.5 × diag_only_DR_err` on sat1, sat2, sat3 — i.e., generality is not bought at more than a 1.5× accuracy penalty.

**F2 done when:**

- The receding-horizon dual MPC achieves `mean_I_rel_err ≤ best_scripted_profile_err` on sat1/2/3, with all methods running under the same `τ_max` enforced by the env. (Sanity: an FIM-optimal planner should at least match a chirp at the same actuator limit.)
- The one-shot MPC achieves `mean_I_rel_err ≤ chirp_err` on sat1/2/3. (Sanity: this is what Wittenburg-class methods do.)
- Both MPC variants run under `scripts/t1_all_baselines.py` and write to one results table.
- `wall_ms_per_step` is logged for each method.

## 6. Risks and guards

| Risk | Likelihood | Guard |
|---|---|---|
| EKF mean wanders non-PSD on hard episodes | low | Eigval-clip projection (§3.3). Telemetry log of clip events. |
| Diff-sim gradient through MPC explodes at N=20 | low–mid | Global-norm clip 1.0; tanh-squash bounds actions; fallback drop N to 10. |
| MPC inference time is high relative to RL | high — this is the *point* | Just measure and report; it's the "RL matches MPC at 100× lower cost" line. |
| Retrained DR policy underperforms its diagonal-only predecessor | mid | Document honestly. The point is full-tensor coverage, not better diag-only numbers. |
| Off-diagonal info is small at the sampled tilts → FIM ill-conditioned | mid | The ε regularizer in `_slogdet_psd` already handles this. If logdet is dominated by diag block, mention it explicitly in the paper. |

## 7. What this spec does NOT modify

- `sim/dynamics_jax.py` — already supports full I.
- `rl/policy.py` — only the input layer is resized via the existing `init_params(obs_dim=…)` call.
- `tests/test_benchmark.py`, `tests/test_ekf_jax.py` — left alone; new tests added in `tests/test_ls_full_tensor.py` and `tests/test_ekf_full_tensor.py`.
- `scripts/run_ls_simulation.py` and its iterative LS-EKF loop — only the LS regressor it depends on is extended (column count).

## 8. Open questions deferred to plan time

These are intentionally left for the implementation plan, not pinned here:

- Exact training-hyperparameter tuning for the retrained DR policy (steps, lr, batch).
- Whether to land F1 in two PRs (LS first, then EKF + retrain) or one.
- Whether the eval table reports D-opt log-det or trace of `P_I` (depends on what's most-cited in the active-SysID baselines we're comparing against).

## 9. Files touched (anticipated)

New:
- `control/dual_mpc.py`
- `scripts/t1_dual_mpc_baseline.py`
- `scripts/t1_all_baselines.py`
- `tests/test_ls_full_tensor.py`
- `tests/test_ekf_full_tensor.py`

Modified:
- `rl/t1_env.py` (`sample_sat` → rotation construction; `_fim_row_contribution` → 6 columns)
- `rl/t1_env_ekf.py` (EKF state grows to 12-dim; obs grows to 19-dim)
- `rl/ekf_jax.py` (state/Jacobian extension for full I)
- `estimation/ls_estimator.py` (regressor gains 3 columns)
- `utils/observability.py` (FIM regressor matches LS)
- `scripts/t1_train_dr_ekf.py` (uses new env; resizes policy)
