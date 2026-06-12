import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402
import rung6_adaptive_excitation as r6  # noqa: E402

rng = np.random.default_rng(0)

SIGMA_W = 1e-4
N_MC = 40
LAG = 3
TAU_MAX = 0.5
H_MAX = 0.5
TAU_EXT = np.array([0.008, -0.005, 0.006])
WIN = 80                                       # tracking window [steps] (8 s)
STRIDE = 20
DT = r0.DT
N = r0.N
t = r0.t_grid
T_MAX = r0.T_MAX

I0 = r0.I_TRUE.copy()
I1 = r0.I_TRUE.copy()
I1[2, 2] = 6.5                                 # Izz ramps 4.53 -> 6.5 (~44%, e.g. fuel/deployable)


def I_at(tc):
    return I0 + (I1 - I0) * (tc / T_MAX)


def theta_at(tc):
    M = I_at(tc)
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


def simulate_tvi(tau_cmd):
    hsub = DT / r0.SUBSTEPS
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    h_rw = np.zeros((N, 3))
    w, hr = np.zeros(3), np.zeros(3)

    def deriv(w, hr, tau, tc):
        M = I_at(tc)
        td = np.clip(tau, -TAU_MAX, TAU_MAX)
        hdot = np.where((np.abs(hr) >= H_MAX) & (np.sign(hr) == np.sign(td)), 0.0, td)
        return np.linalg.solve(M, TAU_EXT - np.cross(w, M @ w + hr) - hdot), hdot

    for k in range(N):
        tau = np.clip(tau_cmd[k], -TAU_MAX, TAU_MAX)
        wd, hdot = deriv(w, hr, tau, t[k])
        omega[k], omega_dot[k], h_rw[k] = w, wd, hr
        if k < N - 1:
            tn = np.clip(tau_cmd[k + 1], -TAU_MAX, TAU_MAX)
            tc = t[k]
            for s in range(r0.SUBSTEPS):
                ta = tau + (tn - tau) * s / r0.SUBSTEPS
                tb = tau + (tn - tau) * (s + 1) / r0.SUBSTEPS
                tm = 0.5 * (ta + tb)
                k1w, k1h = deriv(w, hr, ta, tc)
                k2w, k2h = deriv(w + 0.5 * hsub * k1w, hr + 0.5 * hsub * k1h, tm, tc + hsub / 2)
                k3w, k3h = deriv(w + 0.5 * hsub * k2w, hr + 0.5 * hsub * k2h, tm, tc + hsub / 2)
                k4w, k4h = deriv(w + hsub * k3w, hr + hsub * k3h, tb, tc + hsub)
                w = w + (hsub / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
                hr = np.clip(hr + (hsub / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h), -H_MAX, H_MAX)
                tc += hsub
    return omega, omega_dot, h_rw


def windowed_iv(w_meas, y_rows, centers):
    """y_rows: (N,3) per-sample target. Returns theta_hat at each window center."""
    wd = np.gradient(w_meas, DT, axis=0)
    out = []
    for c in centers:
        lo, hi = c - WIN // 2, c + WIN // 2
        R = np.hstack([r6.fast_regressor(w_meas[lo:hi], wd[lo:hi]),
                       -np.tile(np.eye(3), (hi - lo, 1))])
        y = y_rows[lo:hi].reshape(-1)
        d = 3 * LAG
        Z = R.copy()
        Z[:, :6] = np.vstack([R[:d, :6], R[:-d, :6]])
        out.append(np.linalg.solve(Z.T @ R, Z.T @ y)[:6])
    return np.array(out)


def evaluate(tau_cmd):
    omega, omega_dot, h_rw = simulate_tvi(tau_cmd)
    hdot = np.gradient(h_rw, DT, axis=0)
    centers = np.arange(WIN // 2, N - WIN // 2, STRIDE)
    tc = t[centers]
    theta_true = np.array([theta_at(c) for c in tc])

    # per-window observability (oracle)
    wd0 = omega_dot
    ld = []
    for c in centers:
        lo, hi = c - WIN // 2, c + WIN // 2
        Raug = np.hstack([r6.fast_regressor(omega[lo:hi], wd0[lo:hi]),
                          -np.tile(np.eye(3), (hi - lo, 1))])
        _, l = np.linalg.slogdet(Raug.T @ Raug)
        ld.append(l)
    ld = np.array(ld)

    errs = np.zeros((N_MC, len(centers)))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        y = -np.cross(w_meas, h_rw) - hdot
        hat = windowed_iv(w_meas, y, centers)
        errs[m] = np.linalg.norm(hat - theta_true, axis=1) / np.linalg.norm(theta_true, axis=1)
    mean_err = errs.mean(0)
    return tc, theta_true, ld, mean_err


def main():
    print(f"Time-varying inertia: Izz {I0[2,2]:.2f} -> {I1[2,2]:.2f} over {T_MAX:.0f}s "
          f"(~{(I1[2,2]/I0[2,2]-1)*100:.0f}%).  windowed augmented-IV tracker (W={WIN}).\n")
    fam = r0.profile_family()
    probes = {"prbs_b": fam["prbs_b"], "chirp": fam["chirp"],
              "sine_x": fam["sine_x"], "constant": fam["constant"]}
    res = {n: evaluate(tq) for n, tq in probes.items()}

    print(f"{'profile':10s} {'track rel-err (mean over time)':>30s}")
    all_ld, all_err = [], []
    for n, (tc, th, ld, err) in res.items():
        print(f"{n:10s} {err.mean()*100:>28.2f}%")
        all_ld.append(ld); all_err.append(err)
    rho, _ = spearmanr(np.concatenate(all_ld), np.concatenate(all_err))
    print(f"\n[rank] Spearman(windowed observability, windowed track-err) = {rho:+.3f}")
    bb = res["prbs_b"][3].mean()
    print(f"[headroom] best broadband (prbs_b) tracks at {bb*100:.2f}% mean rel-err "
          f"across a {(I1[2,2]/I0[2,2]-1)*100:.0f}% inertia change")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    centers = np.arange(WIN // 2, N - WIN // 2, STRIDE)
    tc = res["prbs_b"][0]
    ax[0].plot(tc, [theta_at(c)[2] for c in tc], "k-", lw=2, label="true Izz(t)")
    for n, c in (("prbs_b", "navy"), ("chirp", "green"), ("sine_x", "crimson")):
        om, omd, hrw = simulate_tvi(probes[n])
        hd = np.gradient(hrw, DT, axis=0)
        w_meas = om + SIGMA_W * rng.standard_normal(om.shape)
        y = -np.cross(w_meas, hrw) - hd
        hat = windowed_iv(w_meas, y, centers)
        ax[0].plot(tc, hat[:, 2], "--", color=c, alpha=0.8, label=f"{n} est")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("Izz [kg m^2]")
    ax[0].legend(fontsize=8)

    for n, c in (("prbs_b", "navy"), ("chirp", "green"), ("sine_x", "crimson"), ("constant", "gray")):
        ax[1].semilogy(res[n][2], res[n][3] * 100, "o", color=c, alpha=0.6, label=n, ms=4)
    ax[1].set_xlabel("windowed observability  log-det F")
    ax[1].set_ylabel("windowed track rel-err [%]")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung7_time_varying_inertia.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
