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
DT = r0.DT
N = r0.N
THETA = r0.THETA_TRUE


def A_mat(v):
    vx, vy, vz = v
    return np.array([[vx, 0, 0, vy, vz, 0],
                     [0, vy, 0, vx, 0, vz],
                     [0, 0, vz, 0, vx, vy]])


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def C_mat(w):
    return skew(w) @ A_mat(w)


def estimate_naive(w_meas, tau):
    wd = np.gradient(w_meas, DT, axis=0)
    R = r0.stacked_regressor(w_meas, wd)
    y = tau.reshape(-1)
    theta, *_ = np.linalg.lstsq(R, y, rcond=None)
    return theta


def estimate_integral(w_meas, tau):
    rows, b = [], []
    for k in range(N - 1):
        dw = w_meas[k + 1] - w_meas[k]
        Cint = 0.5 * DT * (C_mat(w_meas[k]) + C_mat(w_meas[k + 1]))
        rows.append(A_mat(dw) + Cint)
        b.append(0.5 * DT * (tau[k] + tau[k + 1]))
    R = np.vstack(rows)
    y = np.concatenate(b)
    theta, *_ = np.linalg.lstsq(R, y, rcond=None)
    return theta


def stats(estim, omega, tau):
    hats = np.empty((N_MC, 6))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        hats[m] = estim(w_meas, tau)
    mean_hat = hats.mean(axis=0)
    nrm = np.linalg.norm(THETA)
    bias = np.linalg.norm(mean_hat - THETA) / nrm
    std = np.sqrt(np.mean(np.sum((hats - mean_hat) ** 2, axis=1))) / nrm
    rel = np.mean(np.sqrt(np.sum((hats - THETA) ** 2, axis=1))) / nrm
    return float(rel), float(bias), float(std)


def evaluate(torques):
    omega, omega_dot = r0.simulate(torques)
    tau = (r0.stacked_regressor(omega, omega_dot) @ THETA).reshape(N, 3)
    _, logdet = np.linalg.slogdet(r0.stacked_regressor(omega, omega_dot).T
                                  @ r0.stacked_regressor(omega, omega_dot))
    rn, bn, sn = stats(estimate_naive, omega, tau)
    ri, bi, si = stats(estimate_integral, omega, tau)
    return {"logdet": float(logdet),
            "naive": (rn, bn, sn), "integral": (ri, bi, si)}


def main():
    fam = r0.profile_family()
    rows = {n: evaluate(tq) for n, tq in fam.items()}
    names = list(rows.keys())
    ld = np.array([rows[n]["logdet"] for n in names])
    rel_n = np.array([rows[n]["naive"][0] for n in names])
    rel_i = np.array([rows[n]["integral"][0] for n in names])
    bias_n = np.array([rows[n]["naive"][1] for n in names])
    bias_i = np.array([rows[n]["integral"][1] for n in names])

    print(f"gyro sigma_w = {SIGMA_W:.0e}.  naive = diff+OLS,  integral = momentum-form (no diff)\n")
    print(f"{'profile':16s} {'logdetF':>9s} | {'naive rel':>9s} {'naive bias':>10s} "
          f"| {'integ rel':>9s} {'integ bias':>10s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet"]):
        r = rows[n]
        print(f"{n:16s} {r['logdet']:>9.2f} | {r['naive'][0]*100:>8.2f}% {r['naive'][1]*100:>9.2f}% "
              f"| {r['integral'][0]*100:>8.2f}% {r['integral'][1]*100:>9.2f}%")

    rho_n, _ = spearmanr(ld, rel_n)
    rho_i, _ = spearmanr(ld, rel_i)
    print(f"\n[rank] Spearman(logdet_F, rel_err)  naive    = {rho_n:+.3f}  (rung-0 ref: -0.93)")
    print(f"[rank] Spearman(logdet_F, rel_err)  integral = {rho_i:+.3f}")
    print(f"[bias] median bias  naive={np.median(bias_n)*100:.2f}%  integral={np.median(bias_i)*100:.3f}%  "
          f"-> {np.median(bias_n)/max(np.median(bias_i),1e-9):.0f}x reduction")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogy(ld, rel_n * 100, "o", color="crimson", label=f"naive diff+OLS (rho={rho_n:+.2f})")
    ax[0].semilogy(ld, rel_i * 100, "s", color="navy", label=f"integral form (rho={rho_i:+.2f})")
    for n, l, r in zip(names, ld, rel_i):
        ax[0].annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax[0].set_xlabel("oracle observability   log-det $F$")
    ax[0].set_ylabel("inertia rel-err [%]")
    ax[0].legend(fontsize=8)

    ax[1].semilogy(ld, bias_n * 100, "o", color="crimson", label="naive bias")
    ax[1].semilogy(ld, bias_i * 100, "s", color="navy", label="integral bias")
    ax[1].set_xlabel("oracle observability   log-det $F$")
    ax[1].set_ylabel("systematic error (bias) [%]")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung2_consistent_estimator.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
