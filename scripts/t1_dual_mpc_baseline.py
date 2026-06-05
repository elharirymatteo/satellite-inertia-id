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
from rl.ekf_jax import inertia_from_state
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


# Backwards-compatible alias — the canonical helper lives in rl/ekf_jax.py.
# Kept under this name because scripts/t1_all_baselines.py imports it.
_ekf_inertia_matrix = inertia_from_state


def run_receding_horizon(env, sat_true, key, horizon_total=150,
                         horizon_mpc=20, replan_every=10,
                         n_opt_steps=30):
    """Roll out one episode under receding-horizon MPC. Returns (rel_err, ms/step)."""
    state, _obs = env.reset(key, sat=sat_true)
    warm = None
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
        sat_state_now = state.sat_state
        F0 = state.F_oracle  # 6x6
        t0 = time.perf_counter()
        tau_plan = plan(sat_state_now, F0, sat_for_planning,
                        horizon=horizon_mpc, n_opt_steps=n_opt_steps,
                        tau_max=env.cfg.tau_max,
                        lr=0.01 * env.cfg.tau_max,
                        dt=env.cfg.dt, substeps=env.cfg.substeps,
                        warm_start=warm)
        # Materialize the plan, then drop JAX's compilation cache: plan() builds
        # a fresh @jit'd grad_step closure on every call (sat is a concrete arg
        # baked into the closure), so without clearing, LLVM-cached traces
        # accumulate across the ~1500 plan calls and exhaust process memory.
        tau_plan = jnp.asarray(np.asarray(tau_plan))
        jax.clear_caches()
        plan_wall = time.perf_counter() - t0

        # Execute first replan_every actions on the true env
        n_to_run = min(replan_every, horizon_total - t)
        for k in range(n_to_run):
            state, _obs, _r, _done, _info = env.step(state, tau_plan[k])
            t += 1
        step_wall.extend([plan_wall / n_to_run] * n_to_run)

        # Warm-start next plan: drop the executed actions, pad tail with zeros.
        remaining = tau_plan[n_to_run:]                     # (horizon_mpc - n_to_run, 3)
        pad = jnp.zeros((n_to_run, 3))
        warm = jnp.concatenate([remaining, pad], axis=0)    # (horizon_mpc, 3)

    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), float(np.mean(step_wall) * 1000)


def _dr_prior_sat(sat_true):
    """The 'expected I' under the DR prior: geometric mean of the log-uniform
    range, zero off-diagonals. The one-shot MPC plans against this since no
    observations are available before t=0."""
    I_mean = float(np.exp(0.5 * (np.log(0.1) + np.log(20.0))))  # sqrt(0.1*20) ~ 1.414
    I = jnp.eye(3) * I_mean
    return SatParams(
        I_sat=I, I_inv=jnp.linalg.inv(I),
        I_rw=sat_true.I_rw, rw_axes=sat_true.rw_axes,
        rw_speed_max=sat_true.rw_speed_max,
        rw_torque_max=sat_true.rw_torque_max,
    )


def run_one_shot(env, sat_true, key, horizon_total=150, n_opt_steps=100):
    """Plan the full 150-step trajectory once against the DR prior, then play
    it open-loop on the true env. Returns (rel_err, ms/step)."""
    state, _obs = env.reset(key, sat=sat_true)
    sat_for_planning = _dr_prior_sat(sat_true)
    t0 = time.perf_counter()
    tau_plan = plan(state.sat_state, state.F_oracle, sat_for_planning,
                    horizon=horizon_total, n_opt_steps=n_opt_steps,
                    tau_max=env.cfg.tau_max,
                    lr=0.01 * env.cfg.tau_max,
                    dt=env.cfg.dt, substeps=env.cfg.substeps)
    # Match run_receding_horizon: drop JAX's compilation cache after planning
    # so the next seed/sat doesn't accumulate stale @jit closures.
    tau_plan = jnp.asarray(np.asarray(tau_plan))
    jax.clear_caches()
    plan_wall = time.perf_counter() - t0
    for k in range(horizon_total):
        state, _obs, _r, _done, _info = env.step(state, tau_plan[k])
    I_est_final = _ekf_inertia_matrix(state.ekf.x)
    rel_err = (jnp.linalg.norm(I_est_final - sat_true.I_sat)
               / jnp.linalg.norm(sat_true.I_sat))
    return float(rel_err), float(plan_wall / horizon_total * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-mpc", type=int, default=20)
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--n-opt-steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--oneshot", action="store_true",
                    help="Use one-shot offline MPC instead of receding-horizon")
    args = ap.parse_args()

    with open(ROOT / "config_sat1.yaml") as f:
        cfg_y = yaml.safe_load(f)
    sat1 = _load_sat("config_sat1.yaml")
    tau_max = float(cfg_y["reaction_wheels"]["max_torque"])
    base_cfg = T1EnvEKFConfig(
        sat=sat1, dt=float(cfg_y["sim"]["dt"]), substeps=10,
        horizon=150, tau_max=tau_max,
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
        disturbance_scale=0.1 * tau_max,
    )

    print(f"{'sat':>14}  {'method':>16}  {'rel_err':>10}  {'ms/step':>10}")
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        sat = _load_sat(cfg_name)
        env_cfg = base_cfg._replace(sat=sat)
        env_s = make_env_ekf(env_cfg)
        errs, walls = [], []
        for s in range(args.seeds):
            key = jax.random.PRNGKey(1000 + s)
            if args.oneshot:
                err, w = run_one_shot(env_s, sat, key)
                method_label = "dual-MPC (oneshot)"
            else:
                err, w = run_receding_horizon(env_s, sat, key,
                                              horizon_mpc=args.horizon_mpc,
                                              replan_every=args.replan_every,
                                              n_opt_steps=args.n_opt_steps)
                method_label = "dual-MPC (RH)"
            errs.append(err); walls.append(w)
        print(f"{cfg_name.split('.')[0]:>14}  {method_label:>18}  "
              f"{np.mean(errs):>10.4%}  {np.mean(walls):>10.2f}")


if __name__ == "__main__":
    main()
