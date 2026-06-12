import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402

rng = np.random.default_rng(0)

SIGMA_W = 1e-4
N_MC = 200
LAG = 3
TAU_MAX = 0.5
H_MAX = 0.5
TAU_EXT = np.array([0.008, -0.005, 0.006])   # constant unmodeled body torque [Nm]
DT = r0.DT
N = r0.N
I = r0.I_TRUE
IINV = np.linalg.inv(I)
THETA = r0.THETA_TRUE
I3 = np.tile(np.eye(3), (N, 1))               # stacked identity, (3N, 3)


def simulate_rw(tau_cmd):
    h = DT / r0.SUBSTEPS
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    h_rw = np.zeros((N, 3))
    w = np.zeros(3)
    hr = np.zeros(3)

    def deriv(w, hr, tau):
        td = np.clip(tau, -TAU_MAX, TAU_MAX)
        hdot = np.where((np.abs(hr) >= H_MAX) & (np.sign(hr) == np.sign(td)), 0.0, td)
        wd = IINV @ (TAU_EXT - np.cross(w, I @ w + hr) - hdot)
        return wd, hdot

    for k in range(N):
        tau = np.clip(tau_cmd[k], -TAU_MAX, TAU_MAX)
        wd, hdot = deriv(w, hr, tau)
        omega[k], omega_dot[k], h_rw[k] = w, wd, hr
        if k < N - 1:
            tn = np.clip(tau_cmd[k + 1], -TAU_MAX, TAU_MAX)
            for s in range(r0.SUBSTEPS):
                ta = tau + (tn - tau) * s / r0.SUBSTEPS
                tb = tau + (tn - tau) * (s + 1) / r0.SUBSTEPS
                tm = 0.5 * (ta + tb)
                k1w, k1h = deriv(w, hr, ta)
                k2w, k2h = deriv(w + 0.5 * h * k1w, hr + 0.5 * h * k1h, tm)
                k3w, k3h = deriv(w + 0.5 * h * k2w, hr + 0.5 * h * k2h, tm)
                k4w, k4h = deriv(w + h * k3w, hr + h * k3h, tb)
                w = w + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
                hr = np.clip(hr + (h / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h), -H_MAX, H_MAX)
    return omega, omega_dot, h_rw


def iv(R, y, n_inertia_cols):
    """IV with lagged instrument on the (noisy) inertia columns; the trailing
    constant columns are noise-free and instrument themselves."""
    d = 3 * LAG
    Z = R.copy()
    Z[:, :n_inertia_cols] = np.vstack([R[:d, :n_inertia_cols], R[:-d, :n_inertia_cols]])
    return np.linalg.solve(Z.T @ R, Z.T @ y)


def evaluate(tau_cmd):
    omega, omega_dot, h_rw = simulate_rw(tau_cmd)
    hdot = np.gradient(h_rw, DT, axis=0)
    R0 = r0.stacked_regressor(omega, omega_dot)
    Raug0 = np.hstack([R0, -I3])
    _, ld6 = np.linalg.slogdet(R0.T @ R0)
    _, ld9 = np.linalg.slogdet(Raug0.T @ Raug0)

    un, au, te = np.empty((N_MC, 6)), np.empty((N_MC, 6)), np.empty((N_MC, 3))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        y = (-np.cross(w_meas, h_rw) - hdot).reshape(-1)
        R = r0.stacked_regressor(w_meas, w_meas_dot := np.gradient(w_meas, DT, axis=0))
        un[m] = iv(R, y, 6)
        aug = iv(np.hstack([R, -I3]), y, 6)
        au[m], te[m] = aug[:6], aug[6:]
    nrm = np.linalg.norm(THETA)
    return {
        "ld6": float(ld6), "ld9": float(ld9),
        "rel_unaug": float(np.mean(np.linalg.norm(un - THETA, axis=1)) / nrm),
        "rel_aug": float(np.mean(np.linalg.norm(au - THETA, axis=1)) / nrm),
        "tau_ext_err": float(np.mean(np.linalg.norm(te - TAU_EXT, axis=1)) / np.linalg.norm(TAU_EXT)),
    }


def main():
    fam = r0.profile_family()
    rows = {n: evaluate(tq) for n, tq in fam.items()}
    names = list(rows.keys())
    ld9 = np.array([rows[n]["ld9"] for n in names])
    ru = np.array([rows[n]["rel_unaug"] for n in names])
    ra = np.array([rows[n]["rel_aug"] for n in names])

    print(f"constant disturbance tau_ext = {TAU_EXT} Nm.  estimator = IV\n")
    print(f"{'profile':16s} {'augFIM':>8s} | {'unaug rel':>10s} {'aug rel':>9s} {'tau_ext err':>12s}")
    for n in sorted(names, key=lambda k: rows[k]["ld9"]):
        r = rows[n]
        print(f"{n:16s} {r['ld9']:>8.1f} | {r['rel_unaug']*100:>9.1f}% {r['rel_aug']*100:>8.2f}% "
              f"{r['tau_ext_err']*100:>11.1f}%")

    rho, _ = spearmanr(ld9, ra)
    print(f"\n[break]   un-augmented IV under disturbance: median rel-err = {np.median(ru)*100:.0f}%")
    print(f"[restore] augmented   IV under disturbance: median rel-err = {np.median(ra)*100:.2f}%")
    print(f"[rank]    Spearman(augmented FIM, aug rel-err) = {rho:+.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    order = sorted(range(len(names)), key=lambda i: ra[i])
    ax[0].bar(x - 0.2, ru[order] * 100, 0.4, label="un-augmented IV", color="crimson")
    ax[0].bar(x + 0.2, ra[order] * 100, 0.4, label="augmented IV", color="navy")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([names[i] for i in order], rotation=90, fontsize=6)
    ax[0].set_ylabel("inertia rel-err [%]")
    ax[0].legend(fontsize=8)

    ax[1].semilogy(ld9, ra * 100, "s", color="navy")
    for n, l, r in zip(names, ld9, ra):
        ax[1].annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax[1].set_xlabel("augmented (9-param) observability   log-det $F$")
    ax[1].set_ylabel("augmented-IV rel-err [%]")

    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung5_disturbance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
