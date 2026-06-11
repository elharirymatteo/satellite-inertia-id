# Does observability predict inertia-ID accuracy? A first-principles ladder

A controlled, incremental study of the question underlying this project:
*can the richness of a reaction-wheel torque signal be modulated, using
observability, to improve spacecraft inertia identification?*

Each rung adds **one** realism increment to a clean baseline and measures
the rank correlation between an observability metric (Fisher-information
`log-det F`) and the achieved inertia estimation error, across a fixed
family of torque profiles. Scripts: `scripts/rung{0..6}_*.py`; figures:
`docs/figs/rung*.png`.

## The ladder

| rung | increment | Spearman(log-det F, rel-err) | finding |
|---|---|---|---|
| 0 | clean linear-Gaussian LS | **−0.93** | observability = accuracy is a *theorem* (Cramér–Rao, A-opt identity ρ=+1.00) |
| 1 | gyro noise + finite-diff ω̇ | **−0.09** | link collapses; error becomes **bias**-dominated (bias/std ≈ 10×) |
| 2 | integral (momentum) regression | **−0.20** | reformulating the algebra does **not** help — breakdown is intrinsic to noisy-regressor LS |
| 3 | noise-aware estimator (IV / smoothing) | **−0.87** | link **restored**; IV is consistent (bias 0.43% → 0.02%) |
| 4 | reaction-wheel momentum budget | **−0.82** | excitation design becomes a real problem (error spread 0.06%→182%); ranking inverts vs rung 0 |
| 5 | unmodeled disturbance + augmented estimator | **−0.81** | augmenting `[R \| −I₃]` absorbs `τ_ext` (3% → 0.10%); joint observability needs rich excitation |
| 6 | optimize excitation (hard regime) | — | FIM-optimal beats best-fixed by **+1%** rel-err — good broadband excitation is already near-optimal |
| 7 | time-varying inertia (windowed tracker) | **−0.88** | link holds for tracking; *appeared* to show regime-dependent optimum (chirp > PRBS) — but see rung 8 |
| 8 | recursive RLS tracker (corrects rung 7) | — | chirp ≈ PRBS (3.2–3.3%); rung-7 gap was a batch-tracker artifact; recursive estimator improves tracking 5–50× — estimator is the lever again |
| 9 | real EKF (`rl/ekf_jax.py`) + saturation diagnostic | −0.42 (in-envelope) | EKF nails it in-envelope (0.3–0.8%); open-loop FIM mispredicts when excitation saturates the wheels (model mismatch) → link inverts to +0.34, returns to −0.52 once saturation removed |
| 10 | closed-loop in-envelope excitation (real EKF) | — | greedy info-max s.t. momentum envelope. Loose budget: ties best-fixed. **Tight budget: 0.30% (0% sat) vs fixed-broadband 3.0% (46% sat) vs no-envelope 7.6% (91% sat)** — closing the loop wins by respecting the envelope |

## What it establishes

1. **Observability is a valid excitation-design criterion** — but only conditional
   on a **consistent, noise-aware** estimator. With naive differentiate-then-OLS,
   errors-in-variables bias swamps everything and observability is uncorrelated
   with accuracy (rungs 1–2).

2. **The estimator is the dominant lever.** A consistent estimator (rung 3) and,
   under disturbance, an *augmented* estimator (rung 5) are prerequisites; without
   them, excitation choice is irrelevant. This is why augmenting the EKF was the
   load-bearing change in the prior benchmark — that project varied excitation
   *before* fixing the estimator.

3. **Excitation richness matters, but only under constraints.** Unconstrained, the
   optimal excitation is trivial ("spin up ω"); a reaction-wheel momentum budget
   makes it a genuine problem where multi-axis broadband signals win and single-tone
   / single-axis profiles fail catastrophically (rung 4).

4. **The adaptive/optimal-excitation layer adds little — for *static* inertia.** Once
   the estimator is correct and the excitation is decent broadband (PRBS/chirp), the
   observability→accuracy curve flattens at the measurement-noise floor. Optimizing
   the signal raises the FIM but not the accuracy (rung 6) — explaining, from first
   principles, why hand-tuned excitation matched learned policies in the prior work.

5. **Time-varying inertia does NOT change the verdict (rungs 7–8).** When inertia ramps,
   the problem becomes *tracking*. A crude windowed-batch tracker (rung 7) suggested the
   optimal excitation was regime-dependent (chirp > PRBS) — apparent headroom for adaptive
   excitation. But a proper recursive tracker (RLS with forgetting, rung 8) closes that gap
   (chirp ≈ PRBS, 3.2–3.3%) and improves tracking 5–50× — the rung-7 gap was an estimator
   artifact. Once again the estimator is the dominant lever; decent broadband excitation is
   near-optimal even under time-varying inertia. Excitation richness still matters modestly
   (single-tone tracks ~2× worse than broadband).

6. **The deployed nonlinear EKF adds a model-validity constraint (rung 9).** Validated
   against the project's real `rl/ekf_jax.py`: in-envelope it reproduces the story
   (0.3–0.8% from a biased 85%-of-true init under disturbance). But the open-loop FIM
   *mispredicts* the EKF's accuracy when the excitation saturates the reaction wheels —
   the filter models `Ω̇ = u` with no saturation, so FIM-greedy spin-up profiles break it
   (correlation inverts to +0.34, returns to −0.52 once saturation is removed). The
   practical objective is therefore *observability subject to staying inside the filter's
   model-validity envelope* (no wheel saturation, bounded ω). This is the precise reason
   moderate broadband excitation beats FIM-greedy and learned policies on the real system —
   it stays in-envelope by construction.

7. **Closed-loop excitation pays off only under tight actuator constraints (rung 10).** A greedy
   controller that maximizes information in the EKF's least-observed inertia direction *subject to a
   hard wheel-momentum envelope* ties the best fixed profile in a loose budget, but under a tight
   momentum budget beats it 10× (0.30% vs 3.0%) by avoiding the wheel saturation the fixed profile
   incurs (46% of the time). Without the envelope constraint the same controller saturates 91% and
   fails (7.6%), reproducing the rung-9 pathology. This delimits where active sensing is worth its
   complexity: when fixed open-loop profiles cannot respect the actuator envelope.

## Reproduce

```
python3 scripts/rung0_observability_vs_accuracy.py
python3 scripts/rung1_measurement_noise.py
python3 scripts/rung2_consistent_estimator.py
python3 scripts/rung3_noise_modeling_estimator.py
python3 scripts/rung4_actuator_constraints.py
python3 scripts/rung5_disturbance.py
python3 scripts/rung6_adaptive_excitation.py
python3 scripts/rung7_time_varying_inertia.py
python3 scripts/rung8_recursive_tracker.py
python3 scripts/rung9_real_ekf.py
python3 scripts/rung9b_no_saturation.py
python3 scripts/rung10_closed_loop.py
python3 scripts/ladder_summary_figure.py
```

The LaTeX paper (IEEEtran conference, for i-SAIRAS/iSpaRo 2026) is in `docs/paper/paper.tex` + `docs/paper/biblio.bib`.

## Open directions (not yet on the ladder)

- closed-loop (online) excitation that maximizes observability *subject to the EKF's model-validity envelope* (no wheel saturation, bounded ω) — the constraint rung 9 showed is missing from open-loop FIM-greedy design.
- harder time-varying inertia (step changes / multi-parameter / faster ramps) where the recursive tracker's forgetting-lag might create real adaptive headroom.
- non-constant disturbance (band-limited `τ_ext` model) and a matching augmented EKF state.
- write up the ladder as the paper (lead with the decomposition; demote RL to a marginal-returns + model-validity-envelope result).
