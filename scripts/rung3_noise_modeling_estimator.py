import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402

rng = np.random.default_rng(0)

SIGMA_W = 1e-4
N_MC = 200
LAG = 3
DT = r0.DT
N = r0.N
THETA = r0.THETA_TRUE


def estimate_naive(w_meas, y):
    wd = np.gradient(w_meas, DT, axis=0)
    R = r0.stacked_regressor(w_meas, wd)
    theta, *_ = np.linalg.lstsq(R, y, rcond=None)
    return theta


def estimate_smoothed(w_meas, y):
    ws = savgol_filter(w_meas, window_length=21, polyorder=3, axis=0)
    wd = savgol_filter(w_meas, window_length=21, polyorder=3, deriv=1, delta=DT, axis=0)
    R = r0.stacked_regressor(ws, wd)
    theta, *_ = np.linalg.lstsq(R, y, rcond=None)
    return theta


def estimate_iv(w_meas, y):
    wd = np.gradient(w_meas, DT, axis=0)
    R = r0.stacked_regressor(w_meas, wd)
    d = 3 * LAG
    Rt, Z, yt = R[d:], R[:-d], y[d:]
    theta = np.linalg.solve(Z.T @ Rt, Z.T @ yt)
    return theta


def stats(estim, omega, y):
    hats = np.empty((N_MC, 6))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        hats[m] = estim(w_meas, y)
    mean_hat = hats.mean(axis=0)
    nrm = np.linalg.norm(THETA)
    bias = np.linalg.norm(mean_hat - THETA) / nrm
    std = np.sqrt(np.mean(np.sum((hats - mean_hat) ** 2, axis=1))) / nrm
    rel = np.mean(np.sqrt(np.sum((hats - THETA) ** 2, axis=1))) / nrm
    return float(rel), float(bias), float(std)


def evaluate(torques):
    omega, omega_dot = r0.simulate(torques)
    R0 = r0.stacked_regressor(omega, omega_dot)
    y = R0 @ THETA
    _, logdet = np.linalg.slogdet(R0.T @ R0)
    return {
        "logdet": float(logdet),
        "naive": stats(estimate_naive, omega, y),
        "smoothed": stats(estimate_smoothed, omega, y),
        "iv": stats(estimate_iv, omega, y),
    }


def main():
    fam = r0.profile_family()
    rows = {n: evaluate(tq) for n, tq in fam.items()}
    names = list(rows.keys())
    ld = np.array([rows[n]["logdet"] for n in names])

    def col(key, i):
        return np.array([rows[n][key][i] for n in names])

    print(f"gyro sigma_w = {SIGMA_W:.0e}.  IV lag = {LAG} steps\n")
    print(f"{'profile':16s} {'logdetF':>9s} | {'naive':>14s} | {'smoothed':>14s} | {'IV':>14s}")
    print(f"{'':16s} {'':>9s} | {'rel  / bias':>14s} | {'rel  / bias':>14s} | {'rel  / bias':>14s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet"]):
        r = rows[n]
        def fmt(k):
            return f"{r[k][0]*100:5.2f}/{r[k][1]*100:5.2f}%"
        print(f"{n:16s} {r['logdet']:>9.2f} | {fmt('naive'):>14s} | {fmt('smoothed'):>14s} | {fmt('iv'):>14s}")

    print()
    for key in ("naive", "smoothed", "iv"):
        rho, _ = spearmanr(ld, col(key, 0))
        print(f"[rank] Spearman(logdet_F, rel_err)  {key:9s} = {rho:+.3f}   "
              f"median bias={np.median(col(key,1))*100:.3f}%")
    print("       (rung-0 reference, clean LS: -0.93)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, c, m in (("naive", "crimson", "o"), ("smoothed", "darkorange", "^"), ("iv", "navy", "s")):
        rho, _ = spearmanr(ld, col(key, 0))
        ax[0].semilogy(ld, col(key, 0) * 100, m, color=c, label=f"{key} (rho={rho:+.2f})")
    ax[0].set_xlabel("oracle observability   log-det $F$")
    ax[0].set_ylabel("inertia rel-err [%]")
    ax[0].set_title("Does modeling the noise restore the link?")
    ax[0].legend(fontsize=8)

    for key, c, m in (("naive", "crimson", "o"), ("smoothed", "darkorange", "^"), ("iv", "navy", "s")):
        ax[1].semilogy(ld, col(key, 1) * 100, m, color=c, label=f"{key} bias")
    ax[1].set_xlabel("oracle observability   log-det $F$")
    ax[1].set_ylabel("systematic error (bias) [%]")
    ax[1].set_title("IV cancels the errors-in-variables bias")
    ax[1].legend(fontsize=8)

    fig.suptitle(f"Rung 3 — noise-modeling estimators (gyro sigma={SIGMA_W:.0e})")
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung3_noise_modeling_estimator.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
