# When Does Excitation Design Help Spacecraft Inertia Identification? A First-Principles Study of the Observability–Accuracy Link

**M. El Hariry et al.** — *draft for Joint i-SAIRAS & iSpaRo 2026, Cologne (Nov 3–6, 2026)*

> **Note:** the current submission version is the LaTeX draft `docs/paper/paper.tex` (IEEEtran conference, + `docs/paper/biblio.bib`), which adds rung 10 (closed-loop in-envelope excitation) and the nuanced conclusion. This markdown is the prose source; some content here predates rung 10.

Target length ≈ 6 pages (confirm page limit on the submission portal). Figures: `docs/figs/rung*.png`.

---

## Abstract

A recurring premise in spacecraft self-characterization is that an agent can *shape* its reaction-wheel torque to enrich the observability of its inertia tensor, and that an observability functional of the excitation (e.g. the Fisher-information log-determinant) is therefore the right thing to optimize. We test this premise from the ground up. Starting from a clean linear-Gaussian least-squares baseline — where maximizing observability is provably identical to minimizing estimation variance — we add one realism increment at a time (measurement noise, actuator constraints, unmodeled disturbance, time-varying inertia, and finally the deployed nonlinear EKF) and measure, at each step, the rank correlation between observability and achieved inertia-estimation error. The link is a theorem at the bottom (ρ = −0.93), **collapses** at the first realistic increment because differentiating noisy rate measurements makes least squares biased (ρ = −0.09), and is **restored** only by a consistent, noise-aware estimator (ρ = −0.87). Once restored, excitation design becomes a genuine problem only under actuator constraints, where any decent broadband multi-axis profile is near-optimal and explicit optimization buys ≈1% — even under a 43% time-varying inertia change. Validating against a production augmented EKF, we find observability is *additionally* gated by the filter's model-validity envelope: open-loop-information-greedy excitation saturates the reaction wheels, which the filter does not model, inverting the link. The synthesis — *estimator first, observability second, excitation third, all subject to the filter's model-validity envelope* — explains from first principles why hand-tuned broadband excitation is hard to beat and bounds the gain available to adaptive/learned excitation, resolving the apparent tension between the excitation-richness premise and the empirical observation that hand-tuned profiles match learned policies. The errors-in-variables mechanism is classical in system identification; our contribution is to show that it specifically invalidates the Fisher-information excitation-design criterion used in the spacecraft literature unless the estimator is consistent, and that a deployed nonlinear filter imposes a further model-validity constraint that open-loop information-greedy design ignores.

---

## 1. Introduction

Spacecraft inertia identification underpins attitude control, on-orbit servicing, and post-deployment reconfiguration. A natural and recurring idea is *active sensing*: the spacecraft emits informative reaction-wheel (RW) torques while an estimator updates its belief over the inertia tensor, and the excitation is chosen — possibly adaptively, e.g. by reinforcement learning — to maximize an observability functional of the resulting motion. The implicit assumptions are (A1) that observability predicts estimation accuracy, and (A2) that shaping the signal to maximize it yields a worthwhile accuracy gain.

These assumptions are rarely tested in isolation. Prior work either (i) optimizes an open-loop excitation against a Fisher-information cost (Wittenburg 2017; Calafiore et al. 2001), (ii) tunes an estimator with an observability metric (CEAS-GNC 2024), or (iii) learns or derives a policy and reports end-to-end error. End-to-end results entangle the excitation, the estimator, the actuator constraints, and the disturbance model, so they cannot say *which* assumption holds or fails.

This paper refines our own prior study [25], which proposed actively shaping reaction-wheel torque to enrich inertia observability; here we step back and ask whether that premise actually holds, isolating the conditions under which it does. We isolate the assumptions with a **ladder of controlled experiments**. Each rung adds exactly one realism increment to a baseline in which the observability–accuracy relationship is analytically known, and measures how the relationship changes. Our contributions:

1. **A provable anchor.** In the linear-Gaussian least-squares regime, observability *is* accuracy (Cramér–Rao with equality); we confirm it numerically and use it as ground truth.
2. **A precise failure boundary.** The observability–accuracy link collapses at the first realistic increment — noisy rate measurements make the regression an errors-in-variables (EIV) problem, and least squares becomes bias-dominated, which the Fisher information cannot see. Reformulating the regression does not help; only a consistent (noise-aware) estimator restores the link. While EIV bias is classical, the consequence we draw is specific: the FIM-based excitation-design criterion adopted from the optimal-experiment-design literature is *not* a valid accuracy surrogate for spacecraft inertia ID unless paired with a consistent estimator.
3. **The locus of the design problem.** Excitation choice is irrelevant unconstrained (just spin up) and only becomes consequential under RW momentum/torque limits — where any decent broadband multi-axis profile is near-optimal: explicit optimization adds ≈1% in the static case, and the adaptive headroom that *appears* under time-varying inertia vanishes once a proper recursive estimator is used.
4. **A model-validity caveat for the deployed filter.** Validating against a production augmented EKF, observability is additionally gated by the filter's model-validity envelope: information-greedy excitation saturates the wheels, which the filter does not model, and the link inverts until the envelope is respected.

The synthesis is a sober design rule and a first-principles explanation of why hand-tuned broadband excitation is hard to beat.

---

## 2. Related work

**Classical inertia identification for spacecraft.** Estimating a spacecraft's inertia from rate/attitude telemetry is a mature problem solved predominantly by (recursive) least squares on Euler's equation. Tanygin and Williams [1] estimate inertia and center of mass from coasting maneuvers; Bordany et al. [2] run a recursive least-squares (RLS) scheme in orbit on UoSAT-12, exciting one axis with a wheel while regulating the others; Psiaki [3] integrates Euler's equation and solves nested least-squares for inertia, wheel alignments, scale factors, and biases from flight data; and Keim, Açıkmeşe and Shields [4] pose inertia estimation as a *constrained* least-squares (LMI/SDP) problem that enforces physical realizability, improving on unconstrained LS. Norman, Peck and O'Shaughnessy [5] jointly estimate inertia and momentum-actuator alignment in orbit. These works establish least squares as the workhorse estimator — and, crucially for us, they obtain `ω̇` from differentiated or integrated rate data, the very operation our rung 1 shows turns the regression into an errors-in-variables problem.

**Adaptive control with embedded identification.** A parallel line embeds identification in the controller. Ahmed, Coppola and Bernstein [6] achieve adaptive asymptotic attitude tracking with simultaneous inertia-matrix identification, using *periodic commands* to render the inertia identifiable — an early statement that excitation richness governs identifiability. Modern concurrent-learning / DREM controllers [7] deliver finite-time inertia estimates as a by-product of tracking under interval (rather than persistent) excitation. These methods assume the closed-loop task provides "enough" excitation; none isolates the observability–accuracy relationship.

**Optimal excitation and experiment design.** Designing an input to minimize parameter uncertainty is the subject of optimal experiment design (OED): Fedorov [8], Pukelsheim [9], and, for dynamic systems, Goodwin and Payne [10] formalize A-/D-/E-optimality on the Fisher information matrix. The robotics community applies this to manipulator calibration — Swevers et al. [11] use finite Fourier-series excitation to minimize estimation covariance, and Calafiore, Indri and Bona [12] optimize trajectories against the log-det FIM. Zhai et al. [13] transfer this to spacecraft, designing optimal excitation for least-squares inertia ID via an information-based index beyond the usual condition number; the CEAS-GNC observability-metric work [14] uses an E-optimal criterion to tune the EKF. All of these adopt the FIM as an accuracy surrogate — precisely the assumption our ladder isolates and bounds.

**System identification and errors-in-variables.** That noisy regressors bias least squares is foundational system-identification theory: Ljung [15] and Söderström and Stoica [16] treat the bias and the instrumental-variable / consistent estimators that remove it, and Söderström's survey [17] formalizes the errors-in-variables (EIV) problem we encounter when `ω̇` is obtained from noisy rates. The EIV *mechanism* is therefore classical; our contribution is its specific consequence for FIM-based spacecraft excitation design.

**Estimation and attitude dynamics.** Our estimators draw on standard tools: the Kalman filter [18], its nonlinear/optimal-estimation treatment by Simon [19] and Crassidis and Junkins [20], and the spacecraft-specific attitude-estimation and reaction-wheel/momentum-management framework of Markley and Crassidis [21]. The augmented-state idea we use to absorb a body-torque disturbance is the inertia-ID analogue of gyro-bias augmentation, standard in that literature.

**Active system identification and learning.** Recent work makes excitation *adaptive*. ASID [22] drives exploration with Fisher information for manipulators; reinforcement-learning controllers handle unknown/varying spacecraft inertia [23]; and Candan and Servadio [24] jointly estimate 6-DoF pose and the full inertia tensor of a non-cooperative target with an augmented UKF (a passive-observation problem). These works motivate the active-sensing premise; we test whether it pays off for cooperative spacecraft inertia ID and find it does so only conditionally and marginally.

---

## 3. Problem formulation

**Dynamics.** With body rate `ω ∈ ℝ³`, symmetric inertia `I`, `n=3` aligned RWs of scalar inertia `I_rw` and speeds `Ω`, commanded wheel acceleration `u`, and an unmodeled body torque `τ_ext`,
```
I ω̇ = τ_ext − ω × (I ω + I_rw Ω) − I_rw u,    Ω̇ = u,
```
with RW torque/speed saturation. The 6 unique inertia entries are `θ = (Ixx,Iyy,Izz,Ixy,Ixz,Iyz)`. Linearizing Euler's equation in `θ` gives a 3×6 regressor `R(ω,ω̇)` with `R(ω,ω̇) θ = τ_eff`, the effective body torque.

**Observability.** The Fisher information for `θ` from a trajectory is `F = Σ_k R_kᵀ R_k / σ²`; we use D-optimality (`log-det F`) and report A-optimality (`tr F⁻¹`) where the theory is exact. Under disturbance we augment to estimate `τ_ext` jointly via `[R | −I₃]`.

**Estimators (the ladder's controlled variable).** (i) ordinary least squares on `R`; (ii) instrumental variables (IV) with a time-lagged instrument, consistent under errors-in-variables; (iii) Savitzky–Golay-smoothed LS; (iv) recursive least squares with forgetting (for time-varying θ); (v) a 15-state augmented EKF (`[ω, I, Ω, τ_ext]`) representative of a deployed flight estimator.

**Metric.** Frobenius relative error `‖Î−I‖/‖I‖`, averaged over Monte-Carlo measurement-noise draws, and its Spearman rank correlation with `log-det F` across a fixed family of 14 torque profiles (single-tone, multi-tone, chirp, PRBS, random, constant), all energy-matched.

---

## 4. Method: the realism ladder

| rung | increment | estimator | question |
|---|---|---|---|
| 0 | none (clean LS) | LS | does observability = accuracy hold? |
| 1 | gyro noise, ω̇ by finite difference | LS | does it survive measurement realism? |
| 2 | integral (momentum) regression | LS | is the breakdown an algebra artifact? |
| 3 | — | IV / smoothed | does a consistent estimator restore it? |
| 4 | RW momentum budget | IV | is excitation design now non-trivial? |
| 5 | constant `τ_ext` | IV vs augmented IV | does augmentation absorb disturbance? |
| 6 | optimize excitation | augmented IV | does adaptation beat best-fixed? |
| 7–8 | time-varying inertia | windowed / recursive | does tracking change the verdict? |
| 9 | deployed nonlinear filter | augmented EKF | does it hold for the real estimator? |

All rungs share one satellite (MicroSat-class, asymmetric `I`), `dt = 0.1 s`, 60 s episodes, gyro `σ_ω = 10⁻⁴ rad/s`.

---

## 5. Results

**Figure 1 (`docs/figs/ladder_summary.png`)** is the paper's summary: the Spearman correlation ρ between observability and inertia accuracy across the ladder — a theorem at rung 0, broken by measurement noise at rungs 1–2, restored by a consistent estimator at rung 3, sustained through constraints/disturbance/time-variation (rungs 4–7), and validated on the real EKF in-envelope (rung 9) but inverted when information-greedy excitation saturates the wheels.

**Table 1. The ladder at a glance.** Spearman rank correlation ρ between observability (`log-det F`) and inertia relative error across the 14-profile family, with the headline number at each rung. All at gyro `σ_ω = 10⁻⁴ rad/s`, MicroSat-class `I`, 60 s episodes.

| rung | realism increment | estimator | ρ(obs, err) | headline |
|---|---|---|---|---|
| 0 | clean LS (anchor) | LS | **−0.93** | A-optimality identity exact (emp/pred MSE = 0.99) |
| 1 | gyro noise, `ω̇` by finite difference | LS | **−0.09** | bias-dominated (bias/std ≈ 10×) |
| 2 | integral (momentum) regression | LS | **−0.20** | 70× better noise-free, still uncorrelated |
| 3 | (estimator change) | IV / smoothed | **−0.87 / −0.86** | bias 0.43% → 0.02% (IV) |
| 4 | RW momentum budget | IV | **−0.82** | error spread 0.06% → 182% across profiles |
| 5 | constant disturbance `τ_ext` | IV → augmented IV | **−0.81** | un-aug 3% → augmented 0.10%; `τ_ext` to 0.3–1.5% |
| 6 | optimize excitation (hard regime) | augmented IV | — | search-opt vs best-fixed: **+1%** (log-det F 5.1→7.4) |
| 7 | time-varying inertia (43% ramp) | windowed IV | **−0.88** | chirp 14% / PRBS 63% (tracker artifact) |
| 8 | time-varying inertia (recursive) | RLS-forgetting | — | chirp ≈ PRBS ≈ 3.2%; 5–50× better than rung 7 |
| 9 | deployed nonlinear filter | augmented EKF | **−0.42** (no-sat) | 0.3–0.8% in-envelope; **+0.34** when wheels saturate |

**Rung 0 — the anchor (Fig. rung0).** In clean LS, A-optimality is exact: empirical/predicted MSE = 0.994, Spearman(`tr F⁻¹`, MSE) = +1.00. D-optimality ranks accuracy at ρ = **−0.93**. Observability = accuracy is confirmed as a theorem. A side observation: under matched torque-*energy*, the most observable profiles are those that build the most angular velocity (constant/low-frequency), because the off-diagonal terms enter only through the gyroscopic `ω×(Iω)` coupling — so the unconstrained excitation problem is trivial.

**Rung 1 — collapse (Fig. rung1).** Adding gyro noise and obtaining `ω̇` by differentiation makes `R` noisy: an errors-in-variables problem. Least squares becomes **bias-dominated** (bias/standard-deviation ≈ 10×), and Spearman drops to **−0.09**. The Fisher information models only the variance of a clean-regressor estimator and is structurally blind to the bias.

**Rung 2 — intrinsic (Fig. rung2).** An integral (momentum-form) regression that never differentiates is 70× more accurate noise-free, yet *more* noise-sensitive and still uncorrelated with observability (ρ = −0.20). The breakdown is intrinsic to regressing on noisy states, not an artifact of differentiation.

**Rung 3 — restoration (Fig. rung3).** A consistent instrumental-variables estimator restores the link to ρ = **−0.87** and cuts the bias 20× (0.43% → 0.02%); Savitzky–Golay smoothing reaches ρ = −0.86. Observability is a valid criterion **iff** the estimator is noise-aware.

**Rung 4 — where design matters (Fig. rung4).** Under a RW momentum budget, the trivial "spin-up" optimum is infeasible, and the across-profile error spread explodes from <2.6% (unconstrained) to **0.06%–182%**. The ranking inverts versus rung 0: broadband multi-axis profiles win; single-tone/single-axis fail. The link holds (ρ = −0.82).

**Rung 5 — disturbance and augmentation (Fig. rung5).** A constant `τ_ext` inflates un-augmented error 10–100× (median 3%); augmenting the estimator with `[R|−I₃]` absorbs it (median **0.10%**, `τ_ext` recovered to 0.3–1.5%). Joint observability of `(θ,τ_ext)` requires *rich* excitation — constant excitation cannot separate a constant disturbance from inertia. ρ = −0.81.

**Rung 6 — adaptation adds little (Fig. rung6).** In a hard (tight-budget) regime, a 500-trial random search for the most-observable multi-sine (a search-optimized, not provably-optimal, excitation) raises observability (log-det F 5.1 → 7.4) but improves accuracy by **+1%** over the best fixed PRBS/chirp (0.11% → 0.11%): the observability–accuracy curve flattens at the measurement-noise floor. Since any learned policy is also a means of finding good excitation, this bounds the achievable benefit of adaptive excitation in this regime.

**Rungs 7–8 — time-varying inertia (Figs. rung7, rung8).** Under a 43% Izz ramp, a windowed-batch tracker *appeared* to make the optimum regime-dependent (chirp 14% vs PRBS 63%); a proper recursive (forgetting-RLS) tracker closes the gap (chirp ≈ PRBS ≈ 3.2%) and improves tracking 5–50×. The apparent adaptive headroom was an estimator artifact; the estimator is again the lever.

**Rung 9 — the deployed EKF (Fig. rung9).** The production augmented EKF, from a biased 85%-of-true initial inertia under disturbance, reproduces the story *in-envelope* (0.3–0.8%). But the open-loop FIM *mispredicts* accuracy when excitation saturates the wheels — the filter models `Ω̇ = u` with no saturation — and the correlation inverts to **+0.34**. Removing saturation restores it to **−0.42** (−0.52 among non-saturating profiles), confirming a filter-model-mismatch mechanism rather than a failure of observability.

---

## 6. Discussion

**Estimator first.** Across every regime, switching to a consistent/augmented/recursive estimator improved accuracy by 1–2 orders of magnitude — far more than any excitation difference. The estimator is the dominant lever; observability becomes meaningful only once it is correct.

**Observability is valid but bounded.** Given a noise-aware estimator, `log-det F` ranks excitation accuracy (ρ ≈ −0.8 to −0.9). But for the *deployed* nonlinear filter it must be maximized **subject to the filter's model-validity envelope** — no wheel saturation, bounded ω for linearization validity — because information-greedy excitation otherwise leaves that envelope and breaks the filter.

**Why adaptation rarely pays.** Among in-envelope broadband profiles the FIM optimum is flat and pinned at the noise floor; explicit excitation optimization yields marginal returns (rung 6). Because a learned policy is itself only a means of finding good excitation, this bounds — but does not directly measure — the benefit of learned active-sensing here; it is consistent with the common empirical finding that hand-tuned broadband excitation matches learned policies. We do not re-run learned policies in the ladder; that bridge is by extension, and a direct in-ladder policy comparison is left to future work.

**Limitations.** Single satellite and disturbance class; offline (not closed-loop) excitation optimization; constant-`τ_ext` augmentation; simulation only. The model-validity-envelope-constrained, closed-loop excitation problem is the natural next study.

---

## 7. Conclusion

Excitation design for spacecraft inertia identification helps — but second-order, and conditionally. The estimator (consistent, augmented, recursive) is the load-bearing element; observability is a valid excitation criterion only once the estimator is noise-aware and only within the deployed filter's model-validity envelope; and decent broadband excitation is near-optimal, leaving little for adaptive policies. We offer the ladder as a reproducible template for separating estimator, excitation, and observability effects that end-to-end studies conflate.

---

## References

*Classical spacecraft inertia identification*
1. S. Tanygin and T. Williams, "Mass property estimation using coasting maneuvers," *J. Guidance, Control, and Dynamics*, 20(4):625–632, 1997.
2. R. Bordany, W. H. Steyn, and M. Crawford, "In-orbit estimation of the inertia matrix and thruster parameters of UoSAT-12," *14th AIAA/USU Conf. on Small Satellites*, 2000.
3. M. L. Psiaki, "Estimation of a spacecraft's attitude dynamics parameters by using flight data," *J. Guidance, Control, and Dynamics*, 28(4):594–603, 2005. doi:10.2514/1.7362.
4. D. Keim, B. Açıkmeşe, and J. F. Shields, "Spacecraft inertia estimation via constrained least squares," *IEEE Aerospace Conf.*, 2006. doi:10.1109/AERO.2006.1655995.
5. M. C. Norman, M. A. Peck, and D. J. O'Shaughnessy, "In-orbit estimation of inertia and momentum-actuator alignment parameters," *J. Guidance, Control, and Dynamics*, 34(6):1798–1814, 2011 (AAS 11-164). doi:10.2514/1.53692.

*Adaptive control with embedded identification*
6. J. Ahmed, V. T. Coppola, and D. S. Bernstein, "Adaptive asymptotic tracking of spacecraft attitude motion with inertia matrix identification," *J. Guidance, Control, and Dynamics*, 21(5):684–691, 1998. doi:10.2514/2.4310.
7. Q. Zhao and G.-R. Duan, "Finite-time concurrent learning adaptive control for spacecraft with inertia parameter identification," *J. Guidance, Control, and Dynamics*, 43(3):574–584, 2020. doi:10.2514/1.G004803.

*Optimal excitation and experiment design*
8. V. V. Fedorov, *Theory of Optimal Experiments*. Academic Press, 1972.
9. F. Pukelsheim, *Optimal Design of Experiments*. SIAM (Classics in Applied Mathematics), 2006 (orig. 1993).
10. G. C. Goodwin and R. L. Payne, *Dynamic System Identification: Experiment Design and Data Analysis*. Academic Press, 1977.
11. J. Swevers, C. Ganseman, D. B. Tükel, J. De Schutter, and H. Van Brussel, "Optimal robot excitation and identification," *IEEE Trans. Robotics and Automation*, 13(5):730–740, 1997.
12. G. Calafiore, M. Indri, and B. Bona, "Robot dynamic calibration: optimal excitation trajectories and experimental parameter estimation," *J. Robotic Systems*, 18(2):55–68, 2001.
13. K. Zhai, T. Wang, and D. Meng, "Optimal excitation design for identifying inertia parameters of spacecraft," *Acta Astronautica*, vol. 140, 2017. doi:10.1016/j.actaastro.2017.08.002.
14. H. Evain and S. Delavault, "Improving satellite inertia identification with an observability metric," CEAS-GNC-2024-027, *CEAS EuroGNC*, Bristol, UK, 2024.

*System identification and errors-in-variables*
15. L. Ljung, *System Identification: Theory for the User*, 2nd ed. Prentice Hall, 1999.
16. T. Söderström and P. Stoica, *System Identification*. Prentice Hall, 1989.
17. T. Söderström, "Errors-in-variables methods in system identification," *Automatica*, 43(6):939–958, 2007.

*Estimation and attitude dynamics*
18. R. E. Kalman, "A new approach to linear filtering and prediction problems," *J. Basic Engineering*, 82(1):35–45, 1960.
19. D. Simon, *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*. Wiley, 2006.
20. J. L. Crassidis and J. L. Junkins, *Optimal Estimation of Dynamic Systems*, 2nd ed. CRC Press, 2011.
21. F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*. Springer, 2014.

*Active system identification and learning*
22. M. Memmel, A. Wagenmaker, et al., "ASID: Active exploration for system identification in robotic manipulation," *ICLR / arXiv:2404.12308*, 2024.
23. J. Enders, "Deep reinforcement learning for spacecraft attitude control and moment-of-inertia estimation," M.S. thesis, AFIT, 2021.
24. B. Candan and S. Servadio, "Online inertia tensor identification for non-cooperative spacecraft via an augmented UKF," *arXiv:2603.27361*, 2026.

*Immediate predecessor (this work)*
25. M. El Hariry et al., "Towards active excitation-based dynamic inertia identification in satellites," *arXiv:2510.16738*, 2025.

*(25 references; author/title/venue/year verified against primary sources. To be exported to BibTeX for the LaTeX submission.)*
