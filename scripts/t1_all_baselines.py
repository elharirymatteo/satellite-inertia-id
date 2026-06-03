"""Run every active-sensing method on the same N_SEEDS across sat1/2/3 and
write one CSV comparison table.

Methods: RL DR policy, dual-MPC (RH), dual-MPC (one-shot), and four scripted
profiles (sine, chirp, prbs, multi step). Metric: ||I_est - I_true||_F /
||I_true||_F over 32 seeds.

Output: docs/t1_all_baselines_eval.txt (CSV with sat_cfg, method,
mean_rel_err, std_rel_err, mean_ms_per_step).

Implementation note: this script orchestrates one *subprocess per (sat,
method) cell*. The MPC variants build ~thousands of @jit'd grad-step
closures across seeds and the LLVM JIT allocator does not return memory
to the OS even after jax.clear_caches(). Subprocess isolation ensures
each cell starts with a fresh allocator and exits cleanly, so the
21-cell sweep completes without exhausting the LLVM page pool. RL and
scripted cells are also subprocessed for uniformity.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_cell(cfg_name: str, method: str, seeds: int) -> dict:
    """Spawn a fresh Python subprocess to run one (sat, method) cell.

    The subprocess invokes this same script with --cell mode; results are
    parsed from a single JSON line printed at the end of its stdout.
    """
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--cell", "--cfg-name", cfg_name, "--method", method,
        "--seeds", str(seeds),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"\n[cell {cfg_name} / {method}] subprocess failed "
            f"(returncode={proc.returncode}):\n{proc.stderr[-2000:]}\n"
        )
        return {"errs": [float("nan")] * seeds, "walls": [float("nan")] * seeds}
    # The cell prints exactly one JSON object on its last non-empty line.
    last = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last = line
    if not last:
        sys.stderr.write(
            f"\n[cell {cfg_name} / {method}] no JSON in stdout:\n"
            f"{proc.stdout[-2000:]}\n"
        )
        return {"errs": [float("nan")] * seeds, "walls": [float("nan")] * seeds}
    return json.loads(last)


def _cell_main(cfg_name: str, method: str, seeds: int) -> None:
    """Body of a single cell — runs N_SEEDS for one (sat, method) and prints
    a JSON {'errs': [...], 'walls': [...]} line on stdout for the parent."""
    # JAX imports go *inside* so the parent process never imports JAX.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import yaml
    import jax
    import jax.numpy as jnp

    from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf
    from rl import policy as policy_mod
    from control.torque_generators import generate_torque_profile
    from scripts.t1_dual_mpc_baseline import (
        _load_sat, run_receding_horizon, run_one_shot, _ekf_inertia_matrix,
    )

    with open(ROOT / "config_sat1.yaml") as f:
        cfg_y = yaml.safe_load(f)
    base = T1EnvEKFConfig(
        sat=_load_sat("config_sat1.yaml"),
        dt=float(cfg_y["sim"]["dt"]), substeps=10, horizon=150,
        tau_max=float(cfg_y["reaction_wheels"]["max_torque"]),
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="neg_rel_err",
    )

    sat = _load_sat(cfg_name)
    env_cfg = base._replace(sat=sat)
    env = make_env_ekf(env_cfg)

    def _rel_err(I_est):
        return float(jnp.linalg.norm(I_est - sat.I_sat)
                     / jnp.linalg.norm(sat.I_sat))

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

    if method == "RL_DR":
        rl_path = ROOT / "rl" / "trained_dr_ekf_full_tensor_neg_rel_err_policy.npz"
        rl_params = {k: jnp.asarray(v) for k, v in np.load(rl_path).items()}
        tau_max = float(env_cfg.tau_max)
        horizon = int(env_cfg.horizon)

        def rl_rollout(key):
            state, obs = env.reset(key, sat=sat)
            def body(carry, _):
                st, o = carry
                a = policy_mod.apply(rl_params, o, tau_max)
                st2, o2, _r, _d, _i = env.step(st, a)
                return (st2, o2), None
            (final_state, _), _ = jax.lax.scan(
                body, (state, obs), None, length=horizon)
            return final_state
        rl_rollout = jax.jit(rl_rollout)

        errs, walls = [], []
        for s in range(seeds):
            key = jax.random.PRNGKey(1000 + s)
            t0 = time.perf_counter()
            st = rl_rollout(key)
            st.ekf.x.block_until_ready()
            wall = time.perf_counter() - t0
            errs.append(_rel_err(_ekf_inertia_matrix(st.ekf.x)))
            walls.append(wall / horizon * 1000)
    elif method == "dual-MPC_RH":
        errs, walls = [], []
        for s in range(seeds):
            key = jax.random.PRNGKey(1000 + s)
            e, w = run_receding_horizon(env, sat, key)
            errs.append(e); walls.append(w)
    elif method == "dual-MPC_oneshot":
        errs, walls = [], []
        for s in range(seeds):
            key = jax.random.PRNGKey(1000 + s)
            e, w = run_one_shot(env, sat, key)
            errs.append(e); walls.append(w)
    else:  # scripted
        tau_seq = _scripted_actions(method, 150, env_cfg.dt, 0.005)

        def scripted_rollout(key):
            state, _ = env.reset(key, sat=sat)
            def body(st, tau):
                st2, _o, _r, _d, _i = env.step(st, tau)
                return st2, None
            final_state, _ = jax.lax.scan(body, state, tau_seq)
            return final_state
        scripted_rollout = jax.jit(scripted_rollout)

        errs, walls = [], []
        for s in range(seeds):
            key = jax.random.PRNGKey(1000 + s)
            t0 = time.perf_counter()
            st = scripted_rollout(key)
            st.ekf.x.block_until_ready()
            wall = time.perf_counter() - t0
            errs.append(_rel_err(_ekf_inertia_matrix(st.ekf.x)))
            walls.append(wall / tau_seq.shape[0] * 1000)

    print(json.dumps({"errs": errs, "walls": walls}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--cell", action="store_true",
                    help="Internal: run a single (sat, method) cell and "
                         "print JSON on stdout for the parent.")
    ap.add_argument("--cfg-name", type=str, default=None)
    ap.add_argument("--method", type=str, default=None)
    args = ap.parse_args()

    if args.cell:
        _cell_main(args.cfg_name, args.method, args.seeds)
        return

    # Lazy import numpy in the orchestrator only after avoiding JAX.
    import numpy as np

    methods = ["RL_DR", "dual-MPC_RH", "dual-MPC_oneshot",
               "sine", "chirp", "prbs", "multi step"]
    rows = []
    print(f"{'sat':>14}  {'method':>20}  {'mean_rel_err':>14}  "
          f"{'std':>10}  {'ms/step':>10}", flush=True)
    for cfg_name in ["config_sat1.yaml", "config_sat2.yaml", "config_sat3.yaml"]:
        for m in methods:
            result = _run_cell(cfg_name, m, args.seeds)
            errs_arr = np.asarray(result["errs"], dtype=float)
            walls_arr = np.asarray(result["walls"], dtype=float)
            if np.all(np.isnan(errs_arr)):
                mean_e = float("nan"); std_e = float("nan")
            else:
                mean_e = float(np.nanmean(errs_arr))
                std_e = float(np.nanstd(errs_arr))
            mean_w = float(np.nanmean(walls_arr)) if walls_arr.size else float("nan")
            rows.append((cfg_name, m, mean_e, std_e, mean_w))
            print(f"{cfg_name.split('.')[0]:>14}  {m:>20}  "
                  f"{mean_e:>13.4%}   {std_e:>9.4%}  {mean_w:>10.2f}",
                  flush=True)

    out = ROOT / "docs" / "t1_all_baselines_eval.txt"
    with open(out, "w") as f:
        f.write("sat_cfg,method,mean_rel_err,std_rel_err,mean_ms_per_step\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
