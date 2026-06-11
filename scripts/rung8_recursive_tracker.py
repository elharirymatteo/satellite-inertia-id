import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402
import rung6_adaptive_excitation as r6  # noqa: E402
import rung7_time_varying_inertia as r7  # noqa: E402

rng = np.random.default_rng(0)

SIGMA_W = 1e-4
N_MC = 20
LAM = 0.99            # RLS forgetting factor (~100-step / 10s memory)
BURN = 120
DT = r0.DT
N = r0.N
t = r0.t_grid


def rls_track(w_meas, h_rw, hdot):
    ws = savgol_filter(w_meas, 21, 3, axis=0)
    wd = savgol_filter(w_meas, 21, 3, deriv=1, delta=DT, axis=0)
    R6 = r6.fast_regressor(ws, wd)                       # (3N, 6)
    y = (-np.cross(ws, h_rw) - hdot)                     # (N, 3)
    nI = -np.eye(3)
    theta = np.zeros(9)
    theta[:3] = 5.0
    P = np.eye(9) * 1e3
    hist = np.zeros((N, 6))
    for k in range(N):
        Rk = np.hstack([R6[3 * k:3 * k + 3], nI])        # (3, 9)
        yk = y[k]
        S = LAM * np.eye(3) + Rk @ P @ Rk.T
        K = P @ Rk.T @ np.linalg.inv(S)
        theta = theta + K @ (yk - Rk @ theta)
        P = (P - K @ Rk @ P) / LAM
        hist[k] = theta[:6]
    return hist


def evaluate(tau_cmd):
    omega, omega_dot, h_rw = r7.simulate_tvi(tau_cmd)
    hdot = np.gradient(h_rw, DT, axis=0)
    theta_true = np.array([r7.theta_at(tk) for tk in t])
    errs = np.zeros(N_MC)
    last = None
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        hist = rls_track(w_meas, h_rw, hdot)
        e = np.linalg.norm(hist - theta_true, axis=1) / np.linalg.norm(theta_true, axis=1)
        errs[m] = e[BURN:].mean()
        last = hist
    return float(errs.mean()), theta_true, last


def main():
    print(f"Recursive RLS tracker (lambda={LAM}, smoothed).  "
          f"Izz {r7.I0[2,2]:.2f}->{r7.I1[2,2]:.2f} (+{(r7.I1[2,2]/r7.I0[2,2]-1)*100:.0f}%)\n")
    fam = r0.profile_family()
    probes = ["chirp", "prbs_b", "sine_iso_0.02", "sine_x", "constant"]
    res = {n: evaluate(fam[n]) for n in probes}

    print(f"{'profile':14s} {'RLS track err':>14s} {'(rung7 windowed)':>18s}")
    rung7 = {"chirp": 14.07, "prbs_b": 63.34, "sine_x": 328.85, "constant": 368.79}
    for n in probes:
        r7v = f"{rung7[n]:.1f}%" if n in rung7 else "-"
        print(f"{n:14s} {res[n][0]*100:>13.2f}% {r7v:>18s}")

    print("\n[check] does chirp still beat prbs_b for tracking?  "
          f"chirp={res['chirp'][0]*100:.1f}%  prbs_b={res['prbs_b'][0]*100:.1f}%"
          f"  -> {'YES' if res['chirp'][0] < res['prbs_b'][0] else 'NO'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    tt = res["chirp"][1][:, 2]
    ax.plot(t, tt, "k-", lw=2.5, label="true Izz(t)")
    for n, c in (("chirp", "green"), ("prbs_b", "navy"), ("sine_x", "crimson")):
        ax.plot(t, res[n][2][:, 2], "--", color=c, alpha=0.85, label=f"{n} (RLS)")
    ax.axvspan(0, t[BURN], color="gray", alpha=0.12, label="burn-in")
    ax.set_ylim(0, 9)
    ax.set_xlabel("time [s]"); ax.set_ylabel("Izz [kg m^2]")
    ax.set_title(f"Rung 8 — recursive RLS tracking of time-varying inertia (lambda={LAM})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung8_recursive_tracker.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
