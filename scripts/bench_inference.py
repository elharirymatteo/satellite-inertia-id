"""Tight microbenchmark for the per-step inference cost of each active-sensing
method. Distinct from scripts/t1_all_baselines.py — that one amortizes the
JIT compile across the rollout. Here we measure the per-step cost of the
*compiled* policy alone, after warmup.

Output: docs/t1_inference_bench.txt
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
from rl.ekf_jax import inertia_from_state
from rl import policy as policy_mod
from rl import ppo
from control.dual_mpc import plan as mpc_plan


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


def main():
    with open(ROOT / "config_sat1.yaml") as f:
        cfg_y = yaml.safe_load(f)
    tau_max = float(cfg_y["reaction_wheels"]["max_torque"])
    sat = _load_sat("config_sat1.yaml")
    base = T1EnvEKFConfig(
        sat=sat, dt=float(cfg_y["sim"]["dt"]), substeps=10, horizon=150,
        tau_max=tau_max,
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
        disturbance_scale=0.1 * tau_max,
    )
    env = make_env_ekf(base)
    state, obs = env.reset(jax.random.PRNGKey(0))

    N_WARMUP = 20
    N_TIMING = 200

    # ---- RL DR policy
    rl = {k: jnp.asarray(v) for k, v in np.load(
        ROOT / "rl" / "trained_dr_ekf_full_tensor_neg_rel_err_policy.npz"
    ).items()}

    @jax.jit
    def rl_apply(o):
        return policy_mod.apply(rl, o, tau_max)

    rl_apply(obs).block_until_ready()  # JIT compile
    for _ in range(N_WARMUP):
        rl_apply(obs).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        rl_apply(obs).block_until_ready()
    rl_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    # ---- PPO (Gaussian-mean) policy
    loaded = dict(np.load(ROOT / "rl" / "trained_ppo_policy.npz"))
    ppo_params = {"actor": {}, "critic": {}, "log_std": jnp.asarray(loaded["log_std"])}
    for k, v in loaded.items():
        if k.startswith("actor_"):
            ppo_params["actor"][k[len("actor_"):]] = jnp.asarray(v)
        elif k.startswith("critic_"):
            ppo_params["critic"][k[len("critic_"):]] = jnp.asarray(v)

    @jax.jit
    def ppo_apply(o):
        return jnp.clip(ppo.actor_mean(ppo_params, o), -tau_max, tau_max)

    ppo_apply(obs).block_until_ready()
    for _ in range(N_WARMUP):
        ppo_apply(obs).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        ppo_apply(obs).block_until_ready()
    ppo_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    # ---- Dual-MPC plan() per call (RH)
    # Use a small horizon=20 (matching the receding-horizon driver) and 30 opt steps.
    F0 = jnp.zeros((6, 6))
    init_state = state.sat_state
    # Warm + time
    tau_plan = mpc_plan(init_state, F0, sat,
                       horizon=20, n_opt_steps=30, tau_max=tau_max,
                       lr=0.01 * tau_max, dt=base.dt, substeps=base.substeps)
    tau_plan.block_until_ready()
    for _ in range(3):
        tau_plan = mpc_plan(init_state, F0, sat,
                           horizon=20, n_opt_steps=30, tau_max=tau_max,
                           lr=0.01 * tau_max, dt=base.dt, substeps=base.substeps)
        tau_plan.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):  # MPC is slow — only need a few samples
        tau_plan = mpc_plan(init_state, F0, sat,
                           horizon=20, n_opt_steps=30, tau_max=tau_max,
                           lr=0.01 * tau_max, dt=base.dt, substeps=base.substeps)
        tau_plan.block_until_ready()
    mpc_rh_call_ms = (time.perf_counter() - t0) / 10 * 1000
    # The receding-horizon driver re-plans every 10 steps, so per-step amortized.
    mpc_rh_ms = mpc_rh_call_ms / 10

    # ---- Scripted (no policy work, just env.step) — purely for reference
    # (the scripted method has 0 inference cost — the action is precomputed)
    scripted_ms = 0.0

    rows = [
        ("RL DR (deterministic MLP)", rl_apply, rl_ms),
        ("PPO (Gaussian mean)", ppo_apply, ppo_ms),
        ("Dual-MPC RH (per-step amortized)", None, mpc_rh_ms),
        ("Dual-MPC RH (per-plan call, every 10 steps)", None, mpc_rh_call_ms),
        ("Scripted excitation", None, scripted_ms),
    ]
    print(f"\nCPU: Intel Core i5-14600K @ up to 5.3 GHz "
          f"(20 threads, JAX CPU backend)\n")
    print(f"{'method':>48}  {'ms/step':>10}")
    for name, _, ms in rows:
        print(f"{name:>48}  {ms:>10.3f}")

    out = ROOT / "docs" / "t1_inference_bench.txt"
    with open(out, "w") as f:
        f.write("method,ms_per_step\n")
        for name, _, ms in rows:
            f.write(f"{name},{ms}\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
