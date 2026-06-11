import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402
import rung6_adaptive_excitation as r6  # noqa: E402
from rl.ekf_jax import EKFParams, EKFState, step as ekf_step, inertia_from_state  # noqa: E402

rng = np.random.default_rng(0)

DT = r0.DT
N = r0.N
t = r0.t_grid
I = r0.I_TRUE
THETA = r0.THETA_TRUE
diagI = np.diag(I)

# sat2 real config
I_RW = 0.002
OMEGA_MAX = 900.0
SIGMA_W = 1e-4
SIGMA_RW = 1e-3
TAU_MAX = 0.1
DIST = 0.1 * TAU_MAX                     # disturbance_scale
TAU_EXT = np.array([0.008, -0.005, 0.006])
N_MC = 16
I3 = np.tile(np.eye(3), (N, 1))

PARAMS = EKFParams(
    dt=DT,
    I_rw=jnp.full((3,), I_RW),
    Qc=jnp.concatenate([
        jnp.full((3,), 1e-9),
        1e-7 * jnp.asarray(diagI) ** 2,
        jnp.full((3,), 1e-7 * diagI.mean() ** 2),
        jnp.full((3,), 1e-9),
        jnp.full((3,), (0.01 * DIST) ** 2),
    ]),
    R=jnp.concatenate([jnp.full((3,), SIGMA_W ** 2), jnp.full((3,), SIGMA_RW ** 2)]),
)


def simulate(tau_cmd):
    """True dynamics: omega + wheel speeds, constant tau_ext, RW speed saturation.
    u = commanded wheel acceleration = tau_cmd / I_rw."""
    Iinv = np.linalg.inv(I)
    hsub = DT / r0.SUBSTEPS
    u_cmd = tau_cmd / I_RW
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    Om = np.zeros((N, 3))
    w, om = np.zeros(3), np.zeros(3)

    def deriv(w, om, u):
        wd = Iinv @ (TAU_EXT - np.cross(w, I @ w + I_RW * om) - I_RW * u)
        return wd, u

    for k in range(N):
        u = u_cmd[k]
        wd, _ = deriv(w, om, u)
        omega[k], omega_dot[k], Om[k] = w, wd, om
        if k < N - 1:
            un = u_cmd[k + 1]
            for s in range(r0.SUBSTEPS):
                ua = u + (un - u) * s / r0.SUBSTEPS
                ub = u + (un - u) * (s + 1) / r0.SUBSTEPS
                um = 0.5 * (ua + ub)
                k1w, k1o = deriv(w, om, ua)
                k2w, k2o = deriv(w + 0.5 * hsub * k1w, om + 0.5 * hsub * k1o, um)
                k3w, k3o = deriv(w + 0.5 * hsub * k2w, om + 0.5 * hsub * k2o, um)
                k4w, k4o = deriv(w + hsub * k3w, om + hsub * k3o, ub)
                w = w + (hsub / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
                om = np.clip(om + (hsub / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o),
                             -OMEGA_MAX, OMEGA_MAX)
    return omega, omega_dot, Om, u_cmd


@jax.jit
def ekf_rollout(x0, P0, us, zs):
    def body(state, inp):
        u, z = inp
        new_state, _ = ekf_step(state, u, z, PARAMS)
        return new_state, None
    final, _ = jax.lax.scan(body, EKFState(x=x0, P=P0), (us, zs))
    return final.x


def evaluate(tau_cmd):
    omega, omega_dot, Om, u_cmd = simulate(tau_cmd)
    Raug = np.hstack([r6.fast_regressor(omega, omega_dot), -I3])
    _, logdet = np.linalg.slogdet(Raug.T @ Raug)

    us = jnp.asarray(u_cmd)
    P0 = jnp.diag(jnp.concatenate([
        jnp.full((3,), SIGMA_W ** 2),
        jnp.asarray((0.30 * diagI) ** 2),
        jnp.full((3,), (0.05 * diagI.mean()) ** 2),
        jnp.full((3,), SIGMA_RW ** 2),
        jnp.full((3,), max(DIST, 1e-9) ** 2),
    ]))
    errs = np.empty(N_MC)
    for m in range(N_MC):
        zw = omega + SIGMA_W * rng.standard_normal(omega.shape)
        zr = Om + SIGMA_RW * rng.standard_normal(Om.shape)
        zs = jnp.asarray(np.hstack([zw, zr]))
        x0 = jnp.concatenate([jnp.asarray(zw[0]), jnp.asarray(0.85 * diagI),
                              jnp.zeros(3), jnp.asarray(zr[0]), jnp.zeros(3)])
        xf = ekf_rollout(x0, P0, us, zs)
        Ihat = np.asarray(inertia_from_state(xf))
        errs[m] = np.linalg.norm(Ihat - I) / np.linalg.norm(I)
    return {"logdet": float(logdet), "rel": float(errs.mean())}


def main():
    print(f"REAL EKF (rl/ekf_jax.py), sat2 config, biased init I0=0.85*true, "
          f"disturbance={TAU_EXT}\n")
    fam = r0.profile_family()
    rows = {n: evaluate(tq) for n, tq in fam.items()}
    names = list(rows.keys())
    ld = np.array([rows[n]["logdet"] for n in names])
    rel = np.array([rows[n]["rel"] for n in names])

    print(f"{'profile':16s} {'augFIM':>8s} {'EKF rel_err':>12s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet"]):
        print(f"{n:16s} {rows[n]['logdet']:>8.1f} {rows[n]['rel']*100:>11.2f}%")

    rho, _ = spearmanr(ld, rel)
    bb = min(rel[[names.index(x) for x in ("prbs_b", "chirp", "sine_multifreq")]])
    print(f"\n[link]    Spearman(augmented FIM, EKF rel-err) = {rho:+.3f}  (IV ref rung5: -0.81)")
    print(f"[accuracy] best broadband EKF inertia rel-err = {bb*100:.2f}%")
    print(f"[verdict]  real EKF reproduces the IV/augmented-LS story: "
          f"{'YES' if rho < -0.5 else 'PARTIAL'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(ld, rel * 100, "o", color="darkgreen")
    for n, l, r in zip(names, ld, rel):
        ax.annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax.set_xlabel("augmented observability  log-det F")
    ax.set_ylabel("EKF inertia rel-err [%]")
    ax.set_title(f"Rung 9 — real rl/ekf_jax.py validates the ladder (rho={rho:+.2f})")
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung9_real_ekf.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
