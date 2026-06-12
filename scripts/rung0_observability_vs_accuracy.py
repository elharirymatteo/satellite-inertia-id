import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.observability import regression_rows_full  # noqa: E402

rng = np.random.default_rng(0)

DT = 0.1
T_MAX = 60.0
SIGMA = 2e-3          # torque-residual noise std [Nm]
E_TARGET = 1.0        # matched excitation energy  sum(||tau||^2) dt
N_MC = 400            # Monte-Carlo noise draws per profile
SUBSTEPS = 10

I_TRUE = np.array([
    [6.53, 0.45, -0.30],
    [0.45, 5.96, 0.22],
    [-0.30, 0.22, 4.53],
])
THETA_TRUE = np.array([I_TRUE[0, 0], I_TRUE[1, 1], I_TRUE[2, 2],
                       I_TRUE[0, 1], I_TRUE[0, 2], I_TRUE[1, 2]])

t_grid = np.arange(0.0, T_MAX + DT, DT)
N = len(t_grid)


def cross(a, b):
    return np.cross(a, b)


def simulate(torques):
    """True rigid-body rotation under a directly-applied body torque (no wheels,
    no disturbance). Returns omega, omega_dot on the sample grid."""
    Iinv = np.linalg.inv(I_TRUE)
    h = DT / SUBSTEPS
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    w = np.zeros(3)
    for k in range(N):
        tau = torques[k]
        omega[k] = w
        omega_dot[k] = Iinv @ (tau - cross(w, I_TRUE @ w))
        if k < N - 1:
            tau_next = torques[k + 1]
            for s in range(SUBSTEPS):
                frac = s / SUBSTEPS
                frac1 = (s + 1) / SUBSTEPS
                ta = tau + (tau_next - tau) * frac
                tb = tau + (tau_next - tau) * frac1
                tm = 0.5 * (ta + tb)
                k1 = Iinv @ (ta - cross(w, I_TRUE @ w))
                k2 = Iinv @ (tm - cross(w + 0.5 * h * k1, I_TRUE @ (w + 0.5 * h * k1)))
                k3 = Iinv @ (tm - cross(w + 0.5 * h * k2, I_TRUE @ (w + 0.5 * h * k2)))
                k4 = Iinv @ (tb - cross(w + h * k3, I_TRUE @ (w + h * k3)))
                w = w + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return omega, omega_dot


def match_energy(torques):
    e = np.sum(torques ** 2) * DT
    return torques * np.sqrt(E_TARGET / (e + 1e-30))


def profile_family():
    f = {}
    f["constant"] = np.tile([1.0, 0.3, -0.2], (N, 1))
    f["sine_x"] = np.column_stack([np.sin(2 * np.pi * 0.02 * t_grid),
                                   np.zeros(N), np.zeros(N)])
    f["sine_3axis_same"] = np.column_stack([np.sin(2 * np.pi * 0.02 * t_grid)] * 3)
    for fr in (0.005, 0.02, 0.08, 0.2):
        f[f"sine_iso_{fr}"] = np.column_stack([
            np.sin(2 * np.pi * fr * t_grid),
            np.sin(2 * np.pi * fr * t_grid + 2.1),
            np.sin(2 * np.pi * fr * t_grid + 4.2)])
    f["sine_multifreq"] = np.column_stack([
        np.sin(2 * np.pi * 0.01 * t_grid),
        np.sin(2 * np.pi * 0.035 * t_grid + 1.0),
        np.sin(2 * np.pi * 0.07 * t_grid + 2.0)])
    f["chirp"] = np.column_stack([
        np.sin(2 * np.pi * (0.005 + 0.05 * t_grid / T_MAX) * t_grid),
        np.sin(2 * np.pi * (0.01 + 0.06 * t_grid / T_MAX) * t_grid + 1.0),
        np.sin(2 * np.pi * (0.02 + 0.08 * t_grid / T_MAX) * t_grid + 2.0)])
    hold = int(4.0 / DT)
    for name, seed in (("prbs_a", 1), ("prbs_b", 2)):
        gen = np.random.default_rng(seed)
        steps = gen.choice([-1.0, 1.0], size=(N // hold + 1, 3))
        f[name] = np.repeat(steps, hold, axis=0)[:N]
    for name, seed in (("random_a", 3), ("random_b", 4), ("random_c", 5)):
        gen = np.random.default_rng(seed)
        steps = gen.standard_normal((N // hold + 1, 3))
        f[name] = np.repeat(steps, hold, axis=0)[:N]
    return {k: match_energy(v) for k, v in f.items()}


def stacked_regressor(omega, omega_dot):
    return np.vstack([np.asarray(regression_rows_full(omega[k], omega_dot[k]))
                      for k in range(N)])


def evaluate(torques):
    omega, omega_dot = simulate(torques)
    R = stacked_regressor(omega, omega_dot)
    F = R.T @ R
    y0 = R @ THETA_TRUE
    eig = np.linalg.eigvalsh(F)
    rank = int(np.sum(eig > eig[-1] * 1e-10))
    Finv = np.linalg.inv(F)
    pred_mse = SIGMA ** 2 * np.trace(Finv)
    sq_err = np.empty(N_MC)
    for m in range(N_MC):
        y = y0 + SIGMA * rng.standard_normal(y0.shape)
        theta_hat = np.linalg.solve(F, R.T @ y)
        sq_err[m] = np.sum((theta_hat - THETA_TRUE) ** 2)
    emp_mse = float(np.mean(sq_err))
    rel_err = float(np.mean(np.sqrt(sq_err)) / np.linalg.norm(THETA_TRUE))
    sign, logdet = np.linalg.slogdet(F)
    return {
        "logdet_F": float(logdet),
        "min_eig": float(eig[0]),
        "cond": float(eig[-1] / max(eig[0], 1e-30)),
        "rank": rank,
        "pred_mse": float(pred_mse),
        "emp_mse": emp_mse,
        "rel_err": rel_err,
    }


def main():
    fam = profile_family()
    rows = {name: evaluate(tq) for name, tq in fam.items()}

    names = list(rows.keys())
    logdet = np.array([rows[n]["logdet_F"] for n in names])
    pred = np.array([rows[n]["pred_mse"] for n in names])
    emp = np.array([rows[n]["emp_mse"] for n in names])
    rel = np.array([rows[n]["rel_err"] for n in names])

    print(f"{'profile':16s} {'rank':>4s} {'logdetF':>9s} {'cond':>10s} "
          f"{'predMSE':>10s} {'empMSE':>10s} {'rel_err':>8s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet_F"]):
        r = rows[n]
        print(f"{n:16s} {r['rank']:>4d} {r['logdet_F']:>9.2f} {r['cond']:>10.1e} "
              f"{r['pred_mse']:>10.2e} {r['emp_mse']:>10.2e} {r['rel_err']*100:>7.2f}%")

    ratio = emp / pred
    print(f"\n[theorem] empMSE / predMSE: mean={ratio.mean():.3f} "
          f"min={ratio.min():.3f} max={ratio.max():.3f}  (expect ~1.0)")
    rho_d, _ = spearmanr(logdet, rel)
    rho_a, _ = spearmanr(pred, emp)
    print(f"[D-opt]  Spearman(logdet_F, rel_err) = {rho_d:+.3f}  (expect strong -)")
    print(f"[A-opt]  Spearman(pred_mse, emp_mse) = {rho_a:+.3f}  (expect +1.0)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    lo = min(pred.min(), emp.min()) * 0.5
    hi = max(pred.max(), emp.max()) * 2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="identity (CRLB)")
    ax[0].loglog(pred, emp, "o", color="crimson")
    for n, p, e in zip(names, pred, emp):
        ax[0].annotate(n, (p, e), fontsize=6, alpha=0.6)
    ax[0].set_xlabel("predicted MSE  $\\sigma^2\\,\\mathrm{tr}(F^{-1})$")
    ax[0].set_ylabel("empirical MSE (Monte-Carlo)")
    ax[0].legend(fontsize=8)

    ax[1].semilogy(logdet, rel * 100, "o", color="navy")
    for n, l, r in zip(names, logdet, rel):
        ax[1].annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax[1].set_xlabel("observability   log-det $F$")
    ax[1].set_ylabel("inertia rel-err [%]")

    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung0_observability_vs_accuracy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
