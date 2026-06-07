# Active-sensing inertia ID system (F1–F6 + augmented EKF)

## Summary

Builds the full active-sensing inertia identification system for the RAL submission. Closes F1–F6 from the explainer roadmap plus an augmented-EKF extension that emerged during sat1 polish.

## Headline result (16-seed eval under disturbance, augmented EKF)

|     | sat1 | sat2 | sat3 | ms/step |
|---|---|---|---|---|
| **multi-step + augmented EKF** | **1.7%** | **2.4%** | **2.7%** | 0 |
| chirp | 2.1% | 2.8% | 2.9% | 0 |
| RL DR (diff-sim) | 53% (OOD) | 6.4% | 6.5% | 0.006 |
| PPO | 1342% (OOD) | 6.5% | 7.6% | 0.007 |
| dual-MPC RH | 7.2% | 10.9% | 10.7% | 67 |
| sine | 2.8% | 3.1% | 4.4% | 0 |

The augmented EKF closed the disturbance-induced sat1 blow-up. After augmentation, scripted multi-step + EKF is the winning practical combination across all three satellite configs; learned policies are competitive but no longer dominate.

## What's in the branch

- **F1** — full 6-parameter inertia tensor: regression rows, PSD-guaranteed DR sampler (rotation-based), 12-dim EKF with Cauchy-Schwarz off-diag clip + eigval floor + multi-substep Euler predict.
- **F2** — dual-MPC baselines: shared planner (diff-sim gradient through FIM log-det) + receding-horizon driver + one-shot offline (modernized Wittenburg) driver.
- **F3** — per-episode body-torque disturbance threading through dynamics_jax. Active-sensing story landed here (scripted profiles degraded 10-50× under disturbance).
- **F5** — minimal PPO ablation in pure JAX. Matches diff-sim policy gradient (8.4% vs 11.4% on sat2 under disturbance, no augmented EKF). Rules out "diff-sim is the secret sauce."
- **Augmented EKF** — state grows 12→15 dims (adds tau_ext). Closes the model-mismatch gap that broke sat1; rewrites the paper conclusion to "augmented EKF + multi-step is the winning practical combination."
- **F4** — sim-to-sim: sinusoidal disturbance mode tests constant-trained policies/EKF on time-varying disturbance. All methods degrade 2-6× but stay finite — the pipeline generalizes.
- **F6** — careful per-method microbenchmark and a drop-in compute paragraph mapping to Φ-sat-2 / Cortex-M7 / LEON3.

## What's out

- Hardware demo
- Paper writing itself

## Test plan

- [x] 27 passed, 1 xfailed in F1+F2 test subset
- [x] All headline numbers in the eval CSV reproducible via `scripts/t1_all_baselines.py`
- [x] Augmented EKF tests pass individually (XLA memory pressure when running together is a known engineering issue)
- [x] Inference microbenchmark reproducible via `scripts/bench_inference.py`
- [x] Sim-to-sim test reproducible via `scripts/t1_all_baselines.py --disturbance-mode sinusoidal`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
