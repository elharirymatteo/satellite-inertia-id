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
TAU_MAX = 0.5         # per-axis wheel torque limit [Nm]  (loose; momentum is the binding constraint)
H_MAX = 0.5           # per-axis wheel momentum limit [Nms]
DT = r0.DT
N = r0.N
I = r0.I_TRUE
IINV = np.linalg.inv(I)
THETA = r0.THETA_TRUE


def cross(a, b):
    return np.cross(a, b)


def simulate_rw(tau_cmd):
    """Rigid body driven by reaction wheels: commanding wheel torque tau applies
    -tau to the body. Wheel momentum h saturates at H_MAX (can't spin up forever)."""
    h = DT / r0.SUBSTEPS
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    h_rw = np.zeros((N, 3))
    sat = np.zeros(N)
    w = np.zeros(3)
    hr = np.zeros(3)

    def deriv(w, hr, tau):
        td = np.clip(tau, -TAU_MAX, TAU_MAX)
        hdot = np.where((np.abs(hr) >= H_MAX) & (np.sign(hr) == np.sign(td)), 0.0, td)
        wd = IINV @ (-cross(w, I @ w + hr) - hdot)
        return wd, hdot

    for k in range(N):
        tau = np.clip(tau_cmd[k], -TAU_MAX, TAU_MAX)
        wd, hdot = deriv(w, hr, tau)
        omega[k], omega_dot[k], h_rw[k] = w, wd, hr
        sat[k] = np.mean(np.abs(hr) >= H_MAX * 0.999)
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
                hr = hr + (h / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
                hr = np.clip(hr, -H_MAX, H_MAX)
    return omega, omega_dot, h_rw, float(sat.mean())


def iv_estimate(w_meas, y):
    wd = np.gradient(w_meas, DT, axis=0)
    R = r0.stacked_regressor(w_meas, wd)
    d = 3 * LAG
    return np.linalg.solve(R[:-d].T @ R[d:], R[:-d].T @ y[d:])


def evaluate(tau_cmd):
    omega, omega_dot, h_rw, sat = simulate_rw(tau_cmd)
    R0 = r0.stacked_regressor(omega, omega_dot)
    _, logdet = np.linalg.slogdet(R0.T @ R0)
    hdot = np.gradient(h_rw, DT, axis=0)
    hats = np.empty((N_MC, 6))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        y = (-np.cross(w_meas, h_rw) - hdot).reshape(-1)
        hats[m] = iv_estimate(w_meas, y)
    nrm = np.linalg.norm(THETA)
    rel = np.mean(np.sqrt(np.sum((hats - THETA) ** 2, axis=1))) / nrm
    return {"logdet": float(logdet), "rel": float(rel),
            "sat": sat, "wmax": float(np.abs(omega).max())}


def main():
    fam = r0.profile_family()
    rows = {n: evaluate(tq) for n, tq in fam.items()}
    names = list(rows.keys())
    ld = np.array([rows[n]["logdet"] for n in names])
    rel = np.array([rows[n]["rel"] for n in names])
    sat = np.array([rows[n]["sat"] for n in names])

    print(f"RW limits: tau_max={TAU_MAX} Nm, h_max={H_MAX} Nms.  estimator = IV (consistent)\n")
    print(f"{'profile':16s} {'logdetF':>9s} {'rel_err':>8s} {'wheel_sat':>10s} {'wmax':>9s}")
    for n in sorted(names, key=lambda k: rows[k]["logdet"]):
        r = rows[n]
        print(f"{n:16s} {r['logdet']:>9.2f} {r['rel']*100:>7.2f}% {r['sat']*100:>9.0f}% "
              f"{r['wmax']:>9.4f}")

    rho, _ = spearmanr(ld, rel)
    print(f"\n[rank] Spearman(logdet_F, rel_err) under constraints (IV) = {rho:+.3f}")
    print(f"[design] best observability:  {names[int(np.argmax(ld))]}  (logdet={ld.max():.1f})")
    print(f"[design] worst observability: {names[int(np.argmin(ld))]}  (logdet={ld.min():.1f})")

    # rung-0 (unconstrained) logdet ranking for the same profiles, to show the inversion
    fam0 = r0.profile_family()
    ld0 = {}
    for n, tq in fam0.items():
        om, omd = r0.simulate(tq)
        Rm = r0.stacked_regressor(om, omd)
        _, l = np.linalg.slogdet(Rm.T @ Rm)
        ld0[n] = float(l)
    print("\n[inversion] rung-0 (spin-up) top-3 vs rung-4 (momentum-limited) top-3:")
    top0 = sorted(names, key=lambda k: -ld0[k])[:3]
    top4 = sorted(names, key=lambda k: -rows[k]["logdet"])[:3]
    print(f"   unconstrained best: {top0}")
    print(f"   constrained   best: {top4}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = ax[0].scatter(ld, rel * 100, c=sat * 100, cmap="viridis", s=60)
    for n, l, r in zip(names, ld, rel):
        ax[0].annotate(n, (l, r * 100), fontsize=6, alpha=0.6)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("achieved observability   log-det $F$")
    ax[0].set_ylabel("inertia rel-err [%] (IV)")
    ax[0].set_title(f"Link holds under constraints (rho={rho:+.2f})")
    fig.colorbar(sc, ax=ax[0], label="wheel-sat %")

    o0 = np.array([ld0[n] for n in names])
    ax[1].scatter(o0, ld, c=sat * 100, cmap="viridis", s=60)
    lim = [min(o0.min(), ld.min()), max(o0.max(), ld.max())]
    ax[1].plot(lim, lim, "k--", lw=1, label="no change")
    for n, a, b in zip(names, o0, ld):
        ax[1].annotate(n, (a, b), fontsize=6, alpha=0.6)
    ax[1].set_xlabel("rung-0 observability (unconstrained spin-up)")
    ax[1].set_ylabel("rung-4 observability (momentum-limited)")
    ax[1].set_title("Constraints invert the ranking")
    ax[1].legend(fontsize=8)

    fig.suptitle(f"Rung 4 — reaction-wheel momentum budget (h_max={H_MAX} Nms)")
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung4_actuator_constraints.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
