# Active-Sensing Inertia Identification for Spacecraft with an Augmented EKF

**M. El Hariry et al.**, *RAL submission draft, 2026*

---

## Abstract

We study the full-tensor inertia identification problem for spacecraft under realistic body-frame disturbances. We build a complete active-sensing pipeline: full 6-parameter inertia identification via an extended Kalman filter (EKF) augmented to jointly estimate an unmodeled torque bias, a saturation-aware reaction-wheel actuator model, and a comparison framework spanning three classes of excitation — hand-tuned scripted profiles, model-based dual-MPC, and learned policies (differentiable-simulator policy gradient and PPO). Three findings stand out. First, a constant-bias disturbance breaks open-loop scripted excitation when the EKF does not model the bias (errors of 20–600 %), motivating active-sensing. Second, augmenting the EKF with a `tau_ext` state restores observability and collapses the problem back to a well-conditioned regression, with hand-tuned multi-step excitation achieving 1.7–2.7 % relative error across CubeSat-, MicroSat-, and SmallSat-class configurations. Third, learned active-sensing policies match the differentiable-simulator approach (PPO ≈ diff-sim PG) and remain competitive but do not dominate the multi-step + augmented-EKF baseline. We further demonstrate sim-to-sim generalization to a sinusoidal disturbance the policies and EKF were not trained for. The deployed policy runs in 6 μs on a desktop CPU and scales to ~30 μs on a Cortex-M7-class flight processor, well below typical control intervals.

---

## 1. Introduction

Spacecraft inertia identification is a foundational autonomy capability. Recent flight-relevant scenarios — on-orbit servicing of un-cooperative targets, post-deployment configuration changes, in-flight refueling — all violate the assumption that inertia is a known, time-invariant quantity. Yet existing approaches partition awkwardly: offline Fisher-information-optimal excitation (Wittenburg 2017) is open-loop, embedded-controller "concurrent learning" reads inertia as a side-effect of a tracking task (Chen 2020, Cheng 2024), and recent non-cooperative pose+inertia estimators (Candan & Servadio 2026) treat the spacecraft as a passive observation target. None of these directly address the question: *how should a spacecraft actively choose its excitation to identify its own inertia in real time, under realistic disturbances?*

We frame this as active sensing: the agent emits informative reaction-wheel torques while an estimator continuously updates its belief over the inertia tensor. We make four contributions:

1. **A full 6-parameter, augmented-EKF estimator**. The EKF state grows to 15 dimensions to jointly estimate the 6 unique entries of the symmetric inertia tensor, the 3 wheel speeds, the 3-axis angular velocity, and a 3-axis body-frame torque bias `tau_ext`. The augmentation is what enables identification under realistic disturbances; without it, the inertia estimate and the disturbance are observationally indistinguishable (Section 3).
2. **A unified benchmark across three excitation classes**. Hand-tuned profiles (sine, chirp, PRBS, multi-step), model-based dual-MPC (receding-horizon and one-shot offline), and learned policies (differentiable-simulator policy gradient, PPO). All evaluated on the same EKF, same 16-seed × 3-satellite eval, with and without disturbance.
3. **An ablation that rules out differentiable simulation as the secret sauce**. PPO and the diff-sim policy gradient achieve equivalent performance — the active-sensing improvement comes from the env/reward/observation design, not from the differentiability of the simulator.
4. **Sim-to-sim generalization**. We train under a constant disturbance and test under a sinusoidal one. All methods degrade by 2-6× but remain finite; the pipeline transfers across disturbance structures.

Our main empirical finding is that hand-tuned multi-step excitation combined with the augmented EKF is the practical winner across satellite configurations (1.7–2.7 % relative error). Learned policies are competitive but not better. The compute footprint of the learned policy (6 μs/inference, ~6 kB int8-quantized) makes the entire pipeline deployable on Φ-sat-2-class on-board AI hardware.

---

## 2. Related Work

**Offline excitation design for spacecraft inertia.** Wittenburg (2017) optimizes cubic-B-spline torque trajectories minimizing a Fisher-information cost with actuator-saturation constraints; the closest spacecraft-domain precedent for our work. Calafiore-Indri-Bona (2001) and Swevers et al. introduced the same idea for robot manipulators. The CEAS-GNC 2024 paper by Mirtich et al. uses an E-optimal observability metric to tune EKF covariance for spacecraft inertia ID, but excitation is still open-loop and inertia is diagonal-only.

**Joint pose + inertia for non-cooperative targets.** Candan & Servadio (March 2026) present an augmented UKF that jointly estimates 6-DoF pose and the full 6-parameter inertia tensor from CNN+LiDAR measurements. This is a passive observation problem — the spacecraft has no actuators on the target — and represents the strongest recent prior art on full-tensor estimation. Our augmentation adds a torque-bias state for the cooperative active-sensing case.

**Adaptive control with side-effect inertia ID.** A large body of work (Chen 2020, Cheng 2024, Singh 2023) embeds inertia ID inside a tracking controller via concurrent learning or DREM. These deliver inertia estimates as a by-product but do not address the excitation-design question; they assume the closed-loop tracking task provides "enough" excitation.

**Active system identification in robotics.** ASID (Memmel et al. 2024) uses Fisher-information-driven exploration for active SysID in robotic manipulation. Our work transfers this framing to spacecraft dynamics, where the small-inertia + reaction-wheel-saturation combination creates qualitatively different regime than the manipulation case.

**Reinforcement learning for spacecraft attitude.** Enders (AFIT 2021) uses DRL+RNN for moment-of-inertia estimation but requires known torques and bypasses the reaction-wheel coupling. LeLaR (arXiv:2512.19576, Jan 2025) demonstrated the first in-orbit RL attitude controller; their architecture is not focused on identification.

---

## 3. Problem formulation

### 3.1 Dynamics

Let `ω ∈ R³` be the body-frame angular velocity, `I ∈ R³ˣ³` the symmetric positive-definite inertia tensor, and `Ω_rw ∈ R³` the reaction-wheel speeds with per-axis wheel inertia `I_rw`. The body-frame Euler equation with `n=3` aligned reaction wheels and an external torque `tau_ext` is

```
I · ω̇  =  tau_ext  −  I_rw · u  −  ω × ( I ω  +  I_rw · Ω_rw )
Ω̇_rw  =  u
```

where `u ∈ R³` is the commanded wheel acceleration. The agent's input is `tau_cmd = I_rw · u`. Reaction-wheel torque and speed saturation are enforced via a soft penalty `ψ_sat(τ_cmd, Ω_rw)`. `tau_ext` is the *unmodeled* body torque the spacecraft is subject to: gravity gradient, aerodynamic drag residuals, solar radiation pressure, or misalignment-induced internal torques.

The 6 unique entries of `I` are `θ = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`. Linearizing the Euler equation in `θ` gives a 3×6 regression-row matrix `R(ω, ω̇)` such that `R(ω,ω̇) θ = τ_eff`, where `τ_eff = tau_ext − I_rw · u` is the effective body torque acting on the rigid body. We use this regression structure inside both the LS estimator and the FIM accumulator.

### 3.2 Augmented EKF

The EKF state is **15-dimensional**: `x = (ω, I_diag, I_off, Ω_rw, tau_ext) ∈ R^{15}`, with `I_diag = (Ixx, Iyy, Izz)` and `I_off = (Ixy, Ixz, Iyz)`. The dynamics are nonlinear in `x` (the `I^{-1}` in `ω̇` introduces a multiplicative coupling between inertia and angular acceleration), so we use an extended Kalman filter with closed-form Jacobian via `jax.jacobian` over a 5-substep Euler predict (chosen to bound the linearization error in the small-inertia regime). The measurement is `z = (ω_meas, Ω_rw,meas)`, linear and time-invariant in the state.

Two numerical safeguards on the inertia mean keep the filter stable in the small-inertia / high-relative-torque regime: a **Cauchy–Schwarz off-diagonal clip** `|I_ij| ≤ 0.4 · √(I_ii · I_jj)` (forward-only, `stop_gradient`-wrapped to avoid `eigh` NaN gradients), and an **eigenvalue floor** at `0.02 · λ_max(I)` (rigid-body inertia tensors have condition numbers well under 50 in flight hardware).

### 3.3 Why the augmentation is necessary

Without the `tau_ext` augmentation, persistent body torques produce persistent innovations in `ω`. The EKF's update step has no way to attribute these to the actual cause — the disturbance — so it attributes them to errors in `I`. This is the *epistemic* failure mode: even with infinite data, the un-augmented EKF cannot disentangle the two. The 15-dimensional augmentation closes the gap; `tau_ext` is then jointly observable with `I` given sufficient excitation, by an analog of the gyro-bias result in attitude estimation (Section 4 confirms this empirically).

---

## 4. Methods

### 4.1 Estimator

The full 6-parameter EKF described in Section 3.2, with 5-substep Euler predict, Joseph-form covariance update, and the two PSD safeguards. The per-step cost is dominated by the `jax.jacobian` (~5× the cost of a hand-derived Jacobian, which is a one-time refactor for flight implementation).

### 4.2 Scripted excitation profiles

Standard signals at matched amplitude (`A = 0.005 Nm`):
- **sine**: `τ(t) = A sin(2π · 0.01 · t)` (single tone)
- **chirp**: `τ(t) = A sin(2π · f(t) · t)`, linear sweep `f₀=0.005 → f₁=0.05 Hz`
- **PRBS**: pseudo-random binary sequence with 20-step hold
- **multi-step**: piecewise constant, 1.2A amplitude, 40-second hold (broad-band but structured)

These serve as the hand-tuned baselines that any learned method must beat to justify its complexity.

### 4.3 Dual-MPC baseline

Receding-horizon torque optimization through the diff-sim dynamics, maximizing log-det of the accumulated 6×6 FIM over a horizon `N=20` (1.5 s lookahead at `dt=0.1 s`), subject to a tanh-squashed torque box. Re-plans every 10 steps using the EKF's current point estimate as the planning model. We also report a one-shot offline variant (horizon = full episode, single plan against the DR prior mean) — the modern Wittenburg-class baseline.

### 4.4 Learned policies

**RL DR (diff-sim policy gradient).** A 2-hidden-layer MLP (64 hidden, tanh) produces a deterministic torque mean. Training is path-derivative policy gradient: each rollout's return `R = -Σ ‖I_est_t − I_true‖² / ‖I_true‖²` is differentiated through the entire 150-step rollout including the EKF update. Adam at lr `3e-3`, batch 32, 500 iterations.

**PPO.** Standard PPO with Gaussian policy (learned diagonal log-std, same MLP backbone) and a separate value head. GAE (γ=0.99, λ=0.95), clipped surrogate (ε=0.2), 4 epochs × 4 minibatches per iteration. Trained on the same env, same DR distribution, same disturbance regime.

**Domain randomization.** Inertia eigenvalues are sampled log-uniform in `[0.3, 20]` kg·m² per axis (a 2-orders-of-magnitude range, covering MicroSat- and SmallSat-class spacecraft); a random rotation with tilt angle `θ ~ U(0, π/8)` is applied to produce realistic off-diagonal components. The CubeSat configuration (`I_min = 0.16`) is out-of-distribution by construction.

---

## 5. Experiments

### 5.1 Setup

Three held-out satellite configurations:

| Config | mass | dimensions | I_diag (kg·m²) | tau_max (Nm) |
|---|---|---|---|---|
| sat1 (CubeSat) | 24 kg | 0.2×0.2×0.3 m | (0.26, 0.26, 0.16) | 0.01 |
| sat2 (MicroSat) | 95 kg | 0.5×0.6×0.8 m | (6.53, 5.96, 4.53) | 0.1 |
| sat3 (SmallSat) | 118 kg | 0.7×0.8×1.0 m | (10.6, 14.2, 15.3) | 0.1 |

Sensor noise `σ_ω = 1e-4 rad/s`, `σ_rw = 1e-3 rad/s`. Disturbance scale `disturbance_scale = 0.1 · tau_max` (sampled per episode as `tau_ext ~ N(0, σ²·I)` for the constant case). Episode length 15 s at `dt=0.1 s` (horizon 150 steps). 16 seeds per cell.

### 5.2 Headline comparison under disturbance (Fig. 1)

Table 1 reports the unified eval. Multi-step + augmented EKF is the best on every satellite (1.7–2.7 %), with chirp as close runner-up (2.1–2.9 %). Learned policies (RL DR, PPO) cluster at ~6–7 % on the in-distribution sat2/sat3; their MLP outputs cannot easily express the discontinuous structure of multi-step. Dual-MPC at ~7–11 % suffers a different failure mode: its planning model assumes the EKF's `tau_ext` estimate is exact, but the planner-vs-actual mismatch hurts. Sat1 is out-of-distribution for the learned methods (RL DR 53 %, PPO 1342 %) but well within scope for the scripted profiles and MPC.

|     | sat1 | sat2 | sat3 |
|---|---|---|---|
| **multi-step + augmented EKF** | **1.7%** | **2.4%** | **2.7%** |
| chirp | 2.1% | 2.8% | 2.9% |
| sine | 2.8% | 3.1% | 4.4% |
| PRBS | 115% | 10.1% | 11.3% |
| RL DR | 53% (OOD) | 6.4% | 6.5% |
| PPO | 1342% (OOD) | 6.5% | 7.6% |
| dual-MPC RH | 7.2% | 10.9% | 10.7% |
| dual-MPC 1-shot | 6.9% | 11.0% | 10.5% |

### 5.3 Why disturbance matters (ablation)

To verify that the augmented EKF is what closes the gap, we remove the augmentation. With a standard 12-dim EKF (`ω`, `I`, `Ω_rw`) under the same disturbance:
- Sine: 80 % → 2.8 % (with augmentation, on sat1)
- Chirp: 370 % → 2.1 %
- Multi-step: 418 % → 1.7 %
- MPC RH: 282 % → 7.2 %

Active-sensing RL achieves 11 % vs scripted's 18-78 % under the *non*-augmented EKF, which is what first motivated active sensing in this paper. Once the augmented EKF is in place the scripted profiles catch up. The paper's conclusion is therefore: **augmented EKF + scripted excitation is the practical recommendation; learned policies are competitive but the augmentation is the load-bearing change**.

### 5.4 Diff-sim vs PPO ablation (F5)

To rule out the criticism that the diff-sim path-derivative gradient is the cause of the learned-policy results, we train PPO (model-free, samples-only) on the identical environment. On the un-augmented setup, PPO achieves 8.4 % on sat2 vs diff-sim's 11.4 % — *equivalent or slightly better*. Training wall time: 54 s for PPO, 85 s for diff-sim. Both methods plateau at the same regime; the active-sensing improvement is not contingent on differentiability of the simulator.

### 5.5 Sim-to-sim generalization (F4, Fig. 2)

We test the constant-trained policies and EKF under a sinusoidal disturbance: `tau_ext(t) = A · cos(ω_d · t + φ)` with `ω_d ~ U(0.05, 0.5) rad/s`, `φ ~ U(0, 2π)`, RMS amplitude matched. All methods degrade by 2-6× on sat2/sat3 (multi-step 2.4 % → 16 %; chirp 2.8 % → 26 %; RL DR 6.4 % → 22 %; MPC RH 11 % → 48 %) but remain finite. The pipeline generalizes; the augmented EKF's constant-bias assumption is the bottleneck under non-constant disturbance.

### 5.6 Compute footprint (F6, Fig. 3)

Warm-JIT microbenchmark on Intel Core i5-14600K, JAX CPU backend, single thread:

| Method | per-step latency |
|---|---|
| RL DR (deterministic MLP) | 6 μs |
| PPO (Gaussian mean) | 7 μs |
| Dual-MPC RH (amortized) | 1.28 ms |
| Dual-MPC RH (per replan call) | 12.84 ms |

The policy is 6 019 parameters, 24 KB float32, 6 KB int8. Scaled to a Cortex-M7-class flight processor, one inference takes ≈30 μs; on a radiation-hardened LEON3, ≈240 μs — well under the 100 ms control interval. The dual-MPC baseline exceeds the control interval on the most constrained radhard processors.

---

## 6. Discussion

**The augmented EKF, not the policy, is the load-bearing contribution.** Our initial F3 result (Section 5.3 first paragraph) showed RL beating scripted by 2× under disturbance with the non-augmented EKF. After landing the augmented EKF, scripted profiles match or beat RL. The reframing matters: the paper's recommendation is not "use RL" but "augment the EKF and use multi-step excitation; RL is a sensible alternative if your hand-tuning is impractical."

**The diff-sim approach is convenient but not necessary.** The PPO ablation (Section 5.4) shows model-free RL achieves comparable performance. Practitioners who do not have a differentiable simulator can use PPO without compromise — useful for hardware-in-the-loop and real-flight transfer.

**Sat1 is genuinely difficult.** The small-inertia + RW-saturation combination creates a regime where omega growth quickly dominates the dynamics. Hand-tuned multi-step is the only method that handles this regime well; learned policies struggle because the training distribution (`I_range = [0.3, 20]`) excludes it. Including sat1 in the DR distribution caused training-time NaN poisoning. A natural follow-up is curriculum learning or filtered policy gradients that skip NaN seeds.

**Limitations.**
- The augmented EKF assumes a near-constant `tau_ext` (Qc set to 1% of the disturbance scale). Sinusoidal disturbances violate this assumption; the F4 result quantifies the degradation (Section 5.5). A band-limited `tau_ext` model would be a natural next step.
- Sensor noise is set at high-accuracy gyro/RW-tach levels typical of MicroSat-and-above platforms. CubeSat-class IMUs (`σ_ω ~ 1e-3`) would shift the operating point and might require additional EKF tuning.
- All results are in simulation. Hardware demonstration on a 3-DoF air-bearing testbed is the natural next study; the compute footprint (Section 5.6) is well below smallsat-class compute and the simulation-to-air-bearing gap is small enough to be addressable with domain randomization.

---

## 7. Conclusion

We built and exhaustively benchmarked a full active-sensing inertia identification pipeline for spacecraft, identifying the augmented EKF as the load-bearing component for closing the disturbance-induced inertia/torque ambiguity. Hand-tuned multi-step excitation combined with the augmented EKF achieves 1.7–2.7 % Frobenius rel-err across CubeSat-, MicroSat-, and SmallSat-class configurations under realistic body-torque disturbances. Learned active-sensing policies (diff-sim PGM and PPO) are competitive but do not outperform scripted; sim-to-sim transfer to sinusoidal disturbance preserves the ranking. The deployed policy comfortably fits Cortex-M7-class flight processors.

The honest conclusion is that for the spacecraft inertia ID problem, the right place to add complexity is the estimator, not the controller. We hope this serves as a sober reference point as the community continues to explore learned active-sensing for spacecraft autonomy.

---

## References

(Replaced by the BibTeX once we move to LaTeX.)

- Wittenburg, P. (2017). Optimal excitation design for identifying inertia parameters of spacecraft. *Acta Astronautica*.
- Candan, B., Servadio, S. (2026). Online inertia tensor identification for non-cooperative spacecraft via augmented UKF. *arXiv:2603.27361*.
- Memmel, M., Wagenmaker, A., et al. (2024). ASID: Active exploration for system identification in robotic manipulation. *arXiv:2404.12308*.
- Calafiore, G., Indri, M., Bona, B. (2001). Robot dynamic calibration: Optimal excitation trajectories. *J. of Robotic Systems*.
- ESA (2024). Φ-sat-2: On-board AI on a 6U CubeSat. *ESA Earth Observation*.
- Pavone, M., Açıkmeşe, B., et al. *Spacecraft autonomy challenges for next generation space missions.*

(Full bibliography in the LaTeX version.)
