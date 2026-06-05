"""Train a PPO policy on the same env+DR+disturbance setup as the diff-sim
baseline. F5 ablation: shows what the model-free RL story looks like
without the differentiable-simulator advantage.

Usage:
  python3 scripts/t1_train_ppo.py [--iters 200] [--rollout-len 64] [--n-envs 32]
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
from rl import ppo
from control.torque_generators import generate_torque_profile


def _build_cfg(horizon: int = 150):
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
    tau_max = float(cfg["reaction_wheels"]["max_torque"])
    return T1EnvEKFConfig(
        sat=sat, dt=float(cfg["sim"]["dt"]), substeps=10,
        horizon=horizon, tau_max=tau_max,
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
        disturbance_scale=0.1 * tau_max,
    )


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


def make_rollout_fn(env, horizon, tau_max, sat_template, I_range):
    """Roll out one episode under the current PPO policy.

    Returns (obs (T+1, obs_dim), actions (T, 3), log_probs (T,),
             rewards (T,), final_value scalar, rel_err scalar)."""
    def rollout(params, key):
        k_sat, k_reset, k_act = jax.random.split(key, 3)
        sat = sample_sat(k_sat, I_range,
                         I_rw=sat_template.I_rw, rw_axes=sat_template.rw_axes,
                         rw_speed_max=sat_template.rw_speed_max,
                         rw_torque_max=sat_template.rw_torque_max,
                         log_uniform=True, max_tilt_angle=jnp.pi / 8)
        state, obs0 = env.reset(k_reset, sat=sat)

        def body(carry, step_key):
            state, obs = carry
            a, lp = ppo.sample_action(params, obs, step_key, tau_max)
            state, next_obs, r, _done, _info = env.step(state, a)
            return (state, next_obs), (obs, a, lp, r)

        step_keys = jax.random.split(k_act, horizon)
        (final, final_obs), (obs_t, act_t, lp_t, rew_t) = jax.lax.scan(
            body, (state, obs0), step_keys,
        )
        final_value = ppo.critic_value(params, final_obs)
        I_est_mat = inertia_from_state(final.ekf.x)
        rel_err = (jnp.linalg.norm(I_est_mat - sat.I_sat)
                   / jnp.linalg.norm(sat.I_sat))
        return obs_t, act_t, lp_t, rew_t, final_value, rel_err
    return rollout


def make_eval_fn(env, horizon, tau_max):
    """Deterministic eval (use mean, no noise)."""
    def rollout(params, key, sat):
        state, obs = env.reset(key, sat=sat)
        def body(carry, _):
            state, obs = carry
            a = jnp.clip(ppo.actor_mean(params, obs), -tau_max, tau_max)
            state, obs, _r, _done, _info = env.step(state, a)
            return (state, obs), None
        (final, _), _ = jax.lax.scan(body, (state, obs), None, length=horizon)
        I_est_mat = inertia_from_state(final.ekf.x)
        rel_err = (jnp.linalg.norm(I_est_mat - sat.I_sat)
                   / jnp.linalg.norm(sat.I_sat))
        return rel_err
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
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--rollout-len", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = _build_cfg(horizon=args.rollout_len)
    env = make_env_ekf(cfg)
    I_range = (0.3, 20.0)

    rollout = jax.jit(make_rollout_fn(env, args.rollout_len, cfg.tau_max,
                                       cfg.sat, I_range))
    eval_rollout = jax.jit(make_eval_fn(env, args.rollout_len, cfg.tau_max))

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = ppo.init_ppo_params(init_key, obs_dim=env.obs_shape[0])

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(0.5),
        optax.adam(args.lr),
    )
    opt_state = opt.init(params)

    @jax.jit
    def ppo_step(params, opt_state, batch):
        grad = jax.grad(ppo.ppo_loss)(
            params, batch, args.clip_eps, args.value_coef, args.ent_coef,
        )
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    print(f"PPO ablation (disturbance={cfg.disturbance_scale:g}): "
          f"I in {I_range} log-uniform, iters={args.iters}, "
          f"n_envs={args.n_envs}, rollout_len={args.rollout_len}")
    print(f"JAX device: {jax.devices()[0]}\n")
    print(f"{'iter':>5}  {'train_R':>9}  {'avg_err':>9}  {'wall_s':>7}")

    history = []
    t0 = time.perf_counter()
    for it in range(args.iters):
        key, *subkeys = jax.random.split(key, args.n_envs + 1)
        keys = jnp.stack(subkeys)
        obs_b, act_b, lp_b, rew_b, last_v_b, rel_b = jax.vmap(
            rollout, in_axes=(None, 0))(params, keys)
        # NaN-safe rewards (same trick as diff-sim training)
        rew_b = jnp.where(jnp.isnan(rew_b), -1e4, jnp.clip(rew_b, -1e4, 1e4))
        last_v_b = jnp.where(jnp.isnan(last_v_b), 0.0, last_v_b)

        # GAE per env, then flatten
        values_b = jax.vmap(lambda p, o: jax.vmap(ppo.critic_value, in_axes=(None, 0))(p, o),
                            in_axes=(None, 0))(params, obs_b)
        adv_b, ret_b = jax.vmap(ppo.compute_gae, in_axes=(0, 0, 0, None, None))(
            rew_b, values_b, last_v_b, args.gamma, args.lam,
        )
        # Flatten (n_envs, T, ...) -> (n_envs*T, ...)
        flat_obs = obs_b.reshape((-1,) + obs_b.shape[2:])
        flat_act = act_b.reshape((-1,) + act_b.shape[2:])
        flat_lp = lp_b.reshape((-1,))
        flat_adv = adv_b.reshape((-1,))
        flat_ret = ret_b.reshape((-1,))
        # Normalize advantages
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        n_samples = flat_obs.shape[0]
        minibatch_size = n_samples // args.minibatches
        for _ in range(args.epochs):
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n_samples)
            for mb in range(args.minibatches):
                idx = perm[mb * minibatch_size:(mb + 1) * minibatch_size]
                batch = (flat_obs[idx], flat_act[idx], flat_lp[idx],
                         flat_adv[idx], flat_ret[idx])
                params, opt_state = ppo_step(params, opt_state, batch)

        mean_R = float(jnp.where(jnp.isnan(rew_b), 0.0, rew_b).sum(axis=1).mean())
        mean_err = float(jnp.where(jnp.isnan(rel_b), 0.0, rel_b).mean())
        history.append((mean_R, mean_err))
        if it % 10 == 0 or it == args.iters - 1:
            print(f"{it:>5}  {mean_R:>9.2f}  {mean_err:>8.2%}  "
                  f"{time.perf_counter()-t0:>7.1f}")
    train_wall = time.perf_counter() - t0
    print(f"\nPPO training done in {train_wall:.1f}s "
          f"({train_wall / args.iters * 1000:.1f} ms/iter)")

    history_path = ROOT / "docs" / "t1_ppo_full_tensor_history.npy"
    history_path.parent.mkdir(exist_ok=True)
    np.save(history_path, np.array(history))
    print(f"History saved to {history_path}")

    out_path = ROOT / "rl" / "trained_ppo_policy.npz"
    flat_params = {f"actor_{k}": v for k, v in params["actor"].items()}
    flat_params.update({f"critic_{k}": v for k, v in params["critic"].items()})
    flat_params["log_std"] = params["log_std"]
    np.savez(out_path, **{k: np.asarray(v) for k, v in flat_params.items()})
    print(f"Params saved to {out_path}")

    # Eval on held-out sat1/2/3
    print(f"\n=== Eval on held-out configs (32 seeds each, PPO deterministic mean) ===")
    eval_keys = jax.random.split(jax.random.PRNGKey(42), 32)
    eval_rows = []
    print(f"\n{'sat':>14}  {'method':>14}  {'I rel err':>10}")
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        sat = _load_sat(cfg_name)
        errs = jax.vmap(eval_rollout, in_axes=(None, 0, None))(
            params, eval_keys, sat,
        )
        errs_np = np.asarray(errs)
        finite_mask = np.isfinite(errs_np)
        mean_err = float(errs_np[finite_mask].mean()) if finite_mask.any() else float("nan")
        std_err = float(errs_np[finite_mask].std()) if finite_mask.any() else float("nan")
        print(f"{cfg_name.split('.')[0]:>14}  {'PPO':>14}  {mean_err:>9.2%}")
        eval_rows.append((cfg_name, "PPO", mean_err, std_err))

    eval_path = ROOT / "docs" / "t1_ppo_eval.txt"
    with open(eval_path, "w") as f:
        f.write("sat_cfg,policy,mean_I_rel_err,std_I_rel_err\n")
        for row in eval_rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"\nEval saved to {eval_path}")


if __name__ == "__main__":
    main()
