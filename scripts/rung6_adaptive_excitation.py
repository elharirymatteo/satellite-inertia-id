import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402

rng = np.random.default_rng(0)

SIGMA_W = 1e-4
N_MC = 200
LAG = 3
TAU_MAX = 0.5
H_MAX = 0.15                                   # tighter momentum budget (hard regime)
TAU_EXT = np.array([0.008, -0.005, 0.006])
N_TRIAL = 500
N_SINE = 4
DT = r0.DT
N = r0.N
t = r0.t_grid
I = r0.I_TRUE
IINV = np.linalg.inv(I)
THETA = r0.THETA_TRUE
I3 = np.tile(np.eye(3), (N, 1))


def fast_regressor(omega, omega_dot):
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    dx, dy, dz = omega_dot[:, 0], omega_dot[:, 1], omega_dot[:, 2]
    rx = np.stack([dx, -wy * wz, wy * wz, dy - wx * wz, dz + wx * wy, wy * wy - wz * wz], 1)
    ry = np.stack([wx * wz, dy, -wx * wz, dx + wy * wz, wz * wz - wx * wx, dz - wx * wy], 1)
    rz = np.stack([-wx * wy, wx * wy, dz, wx * wx - wy * wy, dx - wy * wz, dy + wx * wz], 1)
    n = omega.shape[0]
    R = np.empty((3 * n, 6))
    R[0::3], R[1::3], R[2::3] = rx, ry, rz
    return R


def simulate_rw(tau_cmd):
    hsub = DT / r0.SUBSTEPS
    omega = np.zeros((N, 3))
    omega_dot = np.zeros((N, 3))
    h_rw = np.zeros((N, 3))
    w, hr = np.zeros(3), np.zeros(3)

    def deriv(w, hr, tau):
        td = np.clip(tau, -TAU_MAX, TAU_MAX)
        hdot = np.where((np.abs(hr) >= H_MAX) & (np.sign(hr) == np.sign(td)), 0.0, td)
        return IINV @ (TAU_EXT - np.cross(w, I @ w + hr) - hdot), hdot

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
                k2w, k2h = deriv(w + 0.5 * hsub * k1w, hr + 0.5 * hsub * k1h, tm)
                k3w, k3h = deriv(w + 0.5 * hsub * k2w, hr + 0.5 * hsub * k2h, tm)
                k4w, k4h = deriv(w + hsub * k3w, hr + hsub * k3h, tb)
                w = w + (hsub / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
                hr = np.clip(hr + (hsub / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h), -H_MAX, H_MAX)
    return omega, omega_dot, h_rw


def aug_logdet(omega, omega_dot):
    Raug = np.hstack([fast_regressor(omega, omega_dot), -I3])
    _, ld = np.linalg.slogdet(Raug.T @ Raug)
    return ld


def aug_iv_relerr(tau_cmd):
    omega, omega_dot, h_rw = simulate_rw(tau_cmd)
    hdot = np.gradient(h_rw, DT, axis=0)
    d = 3 * LAG
    hats = np.empty((N_MC, 6))
    for m in range(N_MC):
        w_meas = omega + SIGMA_W * rng.standard_normal(omega.shape)
        y = (-np.cross(w_meas, h_rw) - hdot).reshape(-1)
        R = np.hstack([fast_regressor(w_meas, np.gradient(w_meas, DT, axis=0)), -I3])
        Z = R.copy()
        Z[:, :6] = np.vstack([R[:d, :6], R[:-d, :6]])
        hats[m] = np.linalg.solve(Z.T @ R, Z.T @ y)[:6]
    return float(np.mean(np.linalg.norm(hats - THETA, axis=1)) / np.linalg.norm(THETA))


def multisine(freqs, amps, phases):
    tau = np.zeros((N, 3))
    for a in range(3):
        for k in range(N_SINE):
            tau[:, a] += amps[a, k] * np.sin(2 * np.pi * freqs[k] * t + phases[a, k])
    return r0.match_energy(tau)


def optimize_excitation():
    best_ld, best_tau, best_p = -np.inf, None, None
    for _ in range(N_TRIAL):
        freqs = 10 ** rng.uniform(np.log10(0.005), np.log10(0.3), N_SINE)
        amps = rng.uniform(0.2, 1.0, (3, N_SINE))
        phases = rng.uniform(0, 2 * np.pi, (3, N_SINE))
        tau = multisine(freqs, amps, phases)
        om, omd, _ = simulate_rw(tau)
        ld = aug_logdet(om, omd)
        if ld > best_ld:
            best_ld, best_tau, best_p = ld, tau, (freqs, amps, phases)
    return best_tau, best_ld, best_p


def main():
    print(f"HARD regime: h_max={H_MAX} Nms, tau_ext={TAU_EXT}.  augmented-IV estimator.\n")

    fam = r0.profile_family()
    fixed = {}
    for n, tq in fam.items():
        om, omd, _ = simulate_rw(tq)
        fixed[n] = (aug_logdet(om, omd), aug_iv_relerr(tq))
    best_fixed = min(fixed, key=lambda k: fixed[k][1])

    print(f"{'fixed profile':16s} {'augFIM':>8s} {'rel_err':>9s}")
    for n in sorted(fixed, key=lambda k: fixed[k][1]):
        print(f"{n:16s} {fixed[n][0]:>8.1f} {fixed[n][1]*100:>8.2f}%")

    print(f"\noptimizing excitation ({N_TRIAL} trials, {N_SINE}-sine/axis, maximize augmented FIM)...")
    tau_opt, ld_opt, _ = optimize_excitation()
    rel_opt = aug_iv_relerr(tau_opt)

    bf_ld, bf_rel = fixed[best_fixed]
    print(f"\n{'method':22s} {'augFIM':>8s} {'rel_err':>9s}")
    print(f"{'best fixed ('+best_fixed+')':22s} {bf_ld:>8.1f} {bf_rel*100:>8.2f}%")
    print(f"{'FIM-optimized excitation':22s} {ld_opt:>8.1f} {rel_opt*100:>8.2f}%")
    gain = (bf_rel - rel_opt) / bf_rel * 100
    print(f"\n[verdict] optimized vs best-fixed: {gain:+.0f}% rel-err change, "
          f"augFIM {ld_opt - bf_ld:+.1f}")
    if abs(gain) < 15:
        print("          -> good fixed broadband excitation is near-optimal; "
              "adaptive layer adds little here.")
    elif gain > 0:
        print("          -> optimized excitation beats best fixed; adaptive layer has value.")

    flds = np.array([fixed[n][0] for n in fixed])
    frels = np.array([fixed[n][1] for n in fixed])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.semilogy(flds, frels * 100, "o", color="gray", label="fixed profiles")
    ax.semilogy([bf_ld], [bf_rel * 100], "o", color="navy", ms=12, label=f"best fixed ({best_fixed})")
    ax.semilogy([ld_opt], [rel_opt * 100], "*", color="crimson", ms=20, label="FIM-optimized")
    ax.set_xlabel("augmented observability   log-det $F$")
    ax.set_ylabel("augmented-IV rel-err [%]")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung6_adaptive_excitation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
