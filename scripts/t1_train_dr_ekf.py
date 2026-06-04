"""Train T1 policy with EKF-in-the-loop reward, continuous domain randomization.

The reward is the per-step inertia-uncertainty reduction in the EKF posterior
covariance — not the oracle ω̇ regression FIM. This removes the privileged
information leak in the oracle reward.

Saves:
  rl/trained_dr_ekf_policy.npz
  docs/t1_dr_ekf_history.npy
  docs/t1_dr_ekf_eval.txt

Usage:  python3 scripts/t1_train_dr_ekf.py [--steps N] [--batch B]
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
import optax

from sim.dynamics_jax import SatParams
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
from rl.t1_env import sample_sat
from rl.ekf_jax import inertia_from_state
from rl import policy as policy_mod
from control.torque_generators import generate_torque_profile


def _build_cfg(horizon: int = 150, reward_mode: str = "info_gain",
               tau_max_override: float | None = None) -> T1EnvEKFConfig:
    with open(ROOT / "config_sat1.yaml") as f:
        cfg = yaml.safe_load(f)
    I_sat = np.diag(cfg["satellite"]["inertia_tensor"])
    sat = SatParams(
        I_sat=jnp.asarray(I_sat),
        I_inv=jnp.asarray(np.linalg.inv(I_sat)),
        I_rw=float(cfg["reaction_wheels"]["inertia"]),
        rw_axes=jnp.asarray(cfg["reaction_wheels"]["alignment_matrix"], dtype=jnp.float32),
        rw_speed_max=float(cfg["reaction_wheels"]["max_speed"]),
        rw_torque_max=float(cfg["reaction_wheels"]["max_torque"]),
    )
    tau_max = (tau_max_override if tau_max_override is not None
               else float(cfg["reaction_wheels"]["max_torque"]))
    return T1EnvEKFConfig(
        sat=sat, dt=float(cfg["sim"]["dt"]), substeps=10,
        horizon=horizon, tau_max=tau_max,
        sat_penalty=0.1, init_omega_scale=1e-3,
        # High-accuracy sensors (match tests/test_benchmark.py's defaults).
        sigma_omega=1e-4, sigma_rw=1e-3,
        # EKF priors: 15% bias on initial I, 30% std
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode=reward_mode,
    )


def _load_sat(cfg_name: str, base_cfg) -> SatParams:
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


def make_dr_rollout(env, horizon, tau_max, sat_template, I_range):
    def rollout(params, key):
        k_sat, k_reset = jax.random.split(key)
        sat = sample_sat(k_sat, I_range,
                         I_rw=sat_template.I_rw, rw_axes=sat_template.rw_axes,
                         rw_speed_max=sat_template.rw_speed_max,
                         rw_torque_max=sat_template.rw_torque_max,
                         log_uniform=True,
                         max_tilt_angle=jnp.pi / 8)
        state, obs = env.reset(k_reset, sat=sat)
        def body(carry, _):
            state, obs = carry
            a = policy_mod.apply(params, obs, tau_max)
            state, obs, r, _done, info = env.step(state, a)
            return (state, obs), (r, info["info_gain"], info["logdet_F_oracle"])
        (final, _), (rewards, ig, ld_orc) = jax.lax.scan(
            body, (state, obs), None, length=horizon
        )
        # Final inertia estimation error for reporting
        # Full-tensor Frobenius rel_err — matches the metric used by the env's
        # neg_rel_err reward and the unified baselines eval table.
        I_est_mat = inertia_from_state(final.ekf.x)
        I_true_mat = sat.I_sat
        rel_err = (jnp.linalg.norm(I_est_mat - I_true_mat)
                   / jnp.linalg.norm(I_true_mat))
        return rewards.sum(), ld_orc[-1], rel_err
    return rollout


def make_eval_rollout(env, horizon, tau_max):
    def rollout(params, key, sat):
        state, obs = env.reset(key, sat=sat)
        def body(carry, _):
            state, obs = carry
            a = policy_mod.apply(params, obs, tau_max)
            state, obs, r, _done, info = env.step(state, a)
            return (state, obs), (r, info["info_gain"], info["logdet_F_oracle"])
        (final, _), (rewards, ig, ld_orc) = jax.lax.scan(
            body, (state, obs), None, length=horizon
        )
        # Full-tensor Frobenius rel_err — matches the metric used by the env's
        # neg_rel_err reward and the unified baselines eval table.
        I_est_mat = inertia_from_state(final.ekf.x)
        I_true_mat = sat.I_sat
        rel_err = (jnp.linalg.norm(I_est_mat - I_true_mat)
                   / jnp.linalg.norm(I_true_mat))
        return rewards.sum(), ld_orc[-1], rel_err
    return rollout


def scripted_rollout(env, horizon, tau_max):
    def rollout(actions_seq, key, sat):
        state, obs = env.reset(key, sat=sat)
        def body(carry, a):
            state, _obs = carry
            state, obs, r, _done, info = env.step(state, a)
            return (state, obs), (r, info["info_gain"], info["logdet_F_oracle"])
        (final, _), (rewards, ig, ld_orc) = jax.lax.scan(
            body, (state, obs), actions_seq
        )
        # Full-tensor Frobenius rel_err — matches the metric used by the env's
        # neg_rel_err reward and the unified baselines eval table.
        I_est_mat = inertia_from_state(final.ekf.x)
        I_true_mat = sat.I_sat
        rel_err = (jnp.linalg.norm(I_est_mat - I_true_mat)
                   / jnp.linalg.norm(I_true_mat))
        return rewards.sum(), ld_orc[-1], rel_err
    return rollout


def _scripted_actions(name, horizon, dt, amplitude):
    t = np.arange(0, horizon * dt, dt)
    params = {
        "sine":       dict(frequency=0.01, amplitude=amplitude),
        "chirp":      dict(f0=0.005, f1=0.05, amplitude=amplitude),
        "prbs":       dict(amplitude=amplitude, switch_time=20),
        "multi step": dict(amplitude=1.2 * amplitude, step_duration=40.0),
    }[name]
    arr = generate_torque_profile(name, t, **params)
    return jnp.asarray(arr[:horizon])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-mode", choices=["info_gain", "neg_rel_err"],
                    default="info_gain",
                    help="Reward signal: EKF info gain (D-optimal) or "
                         "negative squared relative I-estimation error")
    ap.add_argument("--tau-max", type=float, default=None,
                    help="Override RW torque limit (per axis). Default uses config.")
    args = ap.parse_args()

    base_cfg = _build_cfg(horizon=args.horizon, reward_mode=args.reward_mode,
                          tau_max_override=args.tau_max)
    env = make_env_ekf(base_cfg)
    I_range = (0.1, 20.0)
    rollout_dr = make_dr_rollout(env, args.horizon, base_cfg.tau_max,
                                 base_cfg.sat, I_range)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = policy_mod.init_params(init_key, obs_dim=env.obs_shape[0])

    def loss_fn(params, keys):
        Rs, _, _ = jax.vmap(rollout_dr, in_axes=(None, 0))(params, keys)
        # NaN-safe + clipped reduction: replace NaN returns with a large
        # negative penalty (so a single divergent seed doesn't zero out the
        # batch gradient via optax.zero_nans) AND clip into a sane range so
        # very-bad EKF blow-ups in small-inertia tail seeds don't dominate
        # the gradient. Bound chosen well above typical good returns (~100s).
        Rs_safe = jnp.where(jnp.isnan(Rs), -1e4, Rs)
        Rs_safe = jnp.clip(Rs_safe, -1e4, 1e4)
        return -Rs_safe.mean(), Rs_safe.mean()

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(args.lr),
    )
    opt_state = opt.init(params)

    print(f"DR training under EKF env (reward_mode={args.reward_mode}, "
          f"tau_max={base_cfg.tau_max}): I in {I_range} log-uniform, "
          f"steps={args.steps}, batch={args.batch}, horizon={args.horizon}, "
          f"lr={args.lr}")
    suffix = "_full_tensor"
    if args.reward_mode != "info_gain":
        suffix += f"_{args.reward_mode}"
    if args.tau_max is not None:
        suffix += f"_tmax{args.tau_max:g}"
    print(f"JAX device: {jax.devices()[0]}\n")
    print(f"{'step':>5}  {'train_R':>9}  {'wall_s':>7}")
    t0 = time.perf_counter()
    history = []
    for step in range(args.steps):
        key, *subkeys = jax.random.split(key, args.batch + 1)
        keys = jnp.stack(subkeys)
        (_, mean_r), g = grad_fn(params, keys)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(mean_r))
        if step % 25 == 0 or step == args.steps - 1:
            print(f"{step:>5}  {float(mean_r):>9.3f}  {time.perf_counter()-t0:>7.1f}")
    train_wall = time.perf_counter() - t0
    print(f"\nTraining done in {train_wall:.1f}s")

    history_path = ROOT / "docs" / f"t1_dr_ekf{suffix}_history.npy"
    history_path.parent.mkdir(exist_ok=True)
    np.save(history_path, np.array(history))
    print(f"History saved to {history_path}")

    # ---- Eval on held-out config_sat{1,2,3}
    print("\n=== Eval on held-out configs (32 seeds each, DR policy vs scripted) ===")
    eval_rollout = jax.jit(make_eval_rollout(env, args.horizon, base_cfg.tau_max))
    scripted = jax.jit(scripted_rollout(env, args.horizon, base_cfg.tau_max))

    scripted_actions = {
        name: _scripted_actions(name, args.horizon, base_cfg.dt, 0.005)
        for name in ["sine", "chirp", "prbs", "multi step"]
    }
    eval_keys = jax.random.split(jax.random.PRNGKey(42), 32)

    eval_rows = []
    print(f"\n{'sat':>14}  {'policy':>14}  {'EKF-R':>8}  {'oracle ld':>10}  "
          f"{'I rel err':>10}")
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        sat = _load_sat(cfg_name, base_cfg)
        diag = np.asarray(jnp.diag(sat.I_sat))
        print(f"--- {cfg_name} (diag I = {diag}) ---")

        Rs, lds, errs = jax.vmap(eval_rollout, in_axes=(None, 0, None))(
            params, eval_keys, sat
        )
        print(f"{cfg_name.split('.')[0]:>14}  {'DR policy':>14}  "
              f"{float(Rs.mean()):>8.2f}  {float(lds.mean()):>10.2f}  "
              f"{float(errs.mean()):>9.2%}")
        eval_rows.append((cfg_name, "DR policy",
                          float(Rs.mean()), float(Rs.std()),
                          float(lds.mean()), float(errs.mean())))

        for name, a in scripted_actions.items():
            Rs, lds, errs = jax.vmap(scripted, in_axes=(None, 0, None))(
                a, eval_keys, sat
            )
            print(f"{cfg_name.split('.')[0]:>14}  {name:>14}  "
                  f"{float(Rs.mean()):>8.2f}  {float(lds.mean()):>10.2f}  "
                  f"{float(errs.mean()):>9.2%}")
            eval_rows.append((cfg_name, name,
                              float(Rs.mean()), float(Rs.std()),
                              float(lds.mean()), float(errs.mean())))

    eval_path = ROOT / "docs" / f"t1_dr_ekf{suffix}_eval.txt"
    with open(eval_path, "w") as f:
        f.write("sat_cfg,policy,mean_EKF_reward,std_EKF_reward,"
                "mean_logdet_F_oracle,mean_I_rel_err\n")
        for row in eval_rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"\nEval saved to {eval_path}")

    out_path = ROOT / "rl" / f"trained_dr_ekf{suffix}_policy.npz"
    np.savez(out_path, **{k: np.asarray(v) for k, v in params.items()})
    print(f"Params saved to {out_path}")


if __name__ == "__main__":
    main()
