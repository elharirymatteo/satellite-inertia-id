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

SIGMA_W = 1e-4        # gyro noise std [rad/s]
N_MC = 200
DT = r0.DT
N = r0.N
THETA = r0.THETA_TRUE


def central_diff(x):
    return np.gradient(x, DT, axis=0)


def evaluate(torques):
    omega, omega_dot = r0.simulate(torques)
    R_true = r0.stacked_regressor(omega, omega_dot)
    tau = (R_true @ THETA).reshape(N, 3)
    y = tau.reshape(-1)

    F_oracle = R_true.T @ R_true
    _, logdet_oracle = np.linalg.slogdet(F_oracle)

    sq_err = np.empty(N_MC)
    theta_hats = np.empty((N_MC, 6))
    logdet_data = np.empty(N_MC)
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        wd_meas = central_diff(w_meas)
        R = r0.stacked_regressor(w_meas, wd_meas)
        theta_hat, *_ = np.linalg.lstsq(R, y, rcond=None)
        theta_hats[m] = theta_hat
        sq_err[m] = np.sum((theta_hat - THETA) ** 2)
        _, ld = np.linalg.slogdet(R.T @ R)
        logdet_data[m] = ld

    mean_hat = theta_hats.mean(axis=0)
    bias = np.linalg.norm(mean_hat - THETA) / np.linalg.norm(THETA)
    std = np.sqrt(np.mean(np.sum((theta_hats - mean_hat) ** 2, axis=1)))
    std_rel = std / np.linalg.norm(THETA)
    rel_err = float(np.mean(np.sqrt(sq_err)) / np.linalg.norm(THETA))
    return {
        "logdet_oracle": float(logdet_oracle),
        "logdet_data": float(np.mean(logdet_data)),
        "rel_err": rel_err,
        "bias": float(bias),
        "std": float(std_rel),
    }


def main():
    fam = r0.profile_family()
    rows = {name: evaluate(tq) for name, tq in fam.items()}
    names = list(rows.keys())

    ld_o = np.array([rows[n]["logdet_oracle"] for n in names])
    ld_d = np.array([rows[n]["logdet_data"] for n in names])
    rel = np.array([rows[n]["rel_err"] for n in names])
    bias = np.array([rows[n]["bias"] for n in names])
    std = np.array([rows[n]["std"] for n in names])

    print(f"gyro sigma_w = {SIGMA_W:.0e} rad/s,  omega_dot = central-diff of noisy omega\n")
    print(f"{'profile':16s} {'logdetF_o':>10s} {'logdetF_d':>10s} "
          f"{'rel_err':>8s} {'bias':>8s} {'std':>8s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet_oracle"]):
        r = rows[n]
        print(f"{n:16s} {r['logdet_oracle']:>10.2f} {r['logdet_data']:>10.2f} "
              f"{r['rel_err']*100:>7.2f}% {r['bias']*100:>7.2f}% {r['std']*100:>7.2f}%")

    rho_o, _ = spearmanr(ld_o, rel)
    rho_d, _ = spearmanr(ld_d, rel)
    print(f"\n[rank]  Spearman(oracle logdet_F, rel_err)        = {rho_o:+.3f}")
    print(f"[rank]  Spearman(onboard data logdet_F, rel_err)   = {rho_d:+.3f}")
    print(f"[bias]  median bias/std ratio = {np.median(bias/np.maximum(std,1e-12)):.2f} "
          f"(>>1 => errors-in-variables bias dominates, CRLB no longer applies)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogy(ld_o, rel * 100, "o", color="navy", label="total rel-err")
    ax[0].semilogy(ld_o, bias * 100, "x", color="crimson", label="bias")
    ax[0].semilogy(ld_o, std * 100, "+", color="green", label="std")
    for n, l, r in zip(names, ld_o, rel):
        ax[0].annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax[0].set_xlabel("oracle observability   log-det $F$")
    ax[0].set_ylabel("inertia error [%]")
    ax[0].set_title(f"Does observability still rank accuracy?  (rho={rho_o:+.2f})")
    ax[0].legend(fontsize=8)

    ax[1].loglog(std * 100, bias * 100, "o", color="purple")
    lim = [min(std.min(), bias.min()) * 50, max(std.max(), bias.max()) * 200]
    ax[1].plot(lim, lim, "k--", lw=1, label="bias = std")
    for n, s, b in zip(names, std, bias):
        ax[1].annotate(n, (s * 100, b * 100), fontsize=6, alpha=0.6)
    ax[1].set_xlabel("random error (std) [%]")
    ax[1].set_ylabel("systematic error (bias) [%]")
    ax[1].set_title("Errors-in-variables: bias breaks the CRLB story")
    ax[1].legend(fontsize=8)

    fig.suptitle(f"Rung 1 — measurement noise (gyro sigma={SIGMA_W:.0e}, finite-diff omega_dot)")
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung1_measurement_noise.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
