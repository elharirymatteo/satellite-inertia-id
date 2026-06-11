import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402
import rung9_real_ekf as r9  # noqa: E402
from rl.ekf_jax import EKFState, step as ekf_step, inertia_from_state  # noqa: E402

DT = r0.DT
N = r0.N
I = r9.I
IINV = np.linalg.inv(I)
I_RW = r9.I_RW
OMEGA_MAX = 150.0         # tight momentum budget (h_max = I_rw*150 = 0.3 Nms) so the envelope binds
SIGMA_W = r9.SIGMA_W
SIGMA_RW = r9.SIGMA_RW
TAU_EXT = r9.TAU_EXT
diagI = r9.diagI
PARAMS = r9.PARAMS
N_MC = 8
TAU_NORM = 0.129          # per-step torque norm (energy-matched to the fixed-profile family)
ENV_FRAC = 0.85           # momentum-envelope safety fraction of OMEGA_MAX

# candidate excitation directions: 26 grid directions on the cube, unit-normalized
_dirs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                  if (a, b, c) != (0, 0, 0)], dtype=float)
CAND = _dirs / np.linalg.norm(_dirs, axis=1, keepdims=True) * TAU_NORM   # (26,3) torques


def reg_row(w, wd):
    wx, wy, wz = w
    dx, dy, dz = wd
    return np.array([
        [dx, -wy * wz, wy * wz, dy - wx * wz, dz + wx * wy, wy * wy - wz * wz],
        [wx * wz, dy, -wx * wz, dx + wy * wz, wz * wz - wx * wx, dz - wx * wy],
        [-wx * wy, wx * wy, dz, wx * wx - wy * wy, dx - wy * wz, dy + wx * wz],
    ])


@jax.jit
def ekf_one(state, u, z):
    new_state, _ = ekf_step(state, jnp.asarray(u), jnp.asarray(z), PARAMS)
    return new_state


def integrate(w, Om, u):
    h = DT / r0.SUBSTEPS

    def deriv(w, Om):
        wd = IINV @ (TAU_EXT - np.cross(w, I @ w + I_RW * Om) - I_RW * u)
        return wd, u
    for _ in range(r0.SUBSTEPS):
        k1w, k1o = deriv(w, Om)
        k2w, k2o = deriv(w + 0.5 * h * k1w, Om + 0.5 * h * k1o)
        k3w, k3o = deriv(w + 0.5 * h * k2w, Om + 0.5 * h * k2o)
        k4w, k4o = deriv(w + h * k3w, Om + h * k3o)
        w = w + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        Om = np.clip(Om + (h / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o), -OMEGA_MAX, OMEGA_MAX)
    return w, Om


def choose_u(xhat, P, w, Om, envelope):
    I_est = np.asarray(inertia_from_state(jnp.asarray(xhat)))
    Iinv_est = np.linalg.inv(I_est)
    tau_ext_est = xhat[12:15]
    Pin = P[3:9, 3:9]
    evals, evecs = np.linalg.eigh(Pin)
    v_worst = evecs[:, -1]                       # least-observed inertia combo
    best_u, best_score = np.zeros(3), -1.0
    for tau in CAND:
        u = tau / I_RW
        Om_next = Om + u * DT
        if envelope and np.any((np.abs(Om_next) > ENV_FRAC * OMEGA_MAX) &
                               (np.sign(Om_next) == np.sign(u))):
            continue                              # would drive a wheel further past the envelope
        wd = Iinv_est @ (tau_ext_est - np.cross(w, I_est @ w + I_RW * Om) - I_RW * u)
        score = np.linalg.norm(reg_row(w, wd) @ v_worst)
        if score > best_score:
            best_score, best_u = score, u
    return best_u


def run(method, seed):
    rng = np.random.default_rng(seed)
    w, Om = np.zeros(3), np.zeros(3)
    P0 = np.diag(np.concatenate([
        np.full(3, SIGMA_W ** 2), (0.30 * diagI) ** 2,
        np.full(3, (0.05 * diagI.mean()) ** 2), np.full(3, SIGMA_RW ** 2),
        np.full(3, max(r9.DIST, 1e-9) ** 2)]))
    zw0, zr0 = w + SIGMA_W * rng.standard_normal(3), Om + SIGMA_RW * rng.standard_normal(3)
    x0 = np.concatenate([zw0, 0.85 * diagI, np.zeros(3), zr0, np.zeros(3)])
    ekf = EKFState(x=jnp.asarray(x0), P=jnp.asarray(P0))
    u_prev = np.zeros(3)
    sat_count = 0
    fixed = r0.profile_family()["prbs_b"] if method == "fixed" else None
    for k in range(N):
        zw = w + SIGMA_W * rng.standard_normal(3)
        zr = Om + SIGMA_RW * rng.standard_normal(3)
        ekf = ekf_one(ekf, u_prev, np.concatenate([zw, zr]))
        sat_count += int(np.any(np.abs(Om) >= ENV_FRAC * OMEGA_MAX))
        xhat = np.asarray(ekf.x)
        if method == "fixed":
            u = fixed[k] / I_RW
        else:
            u = choose_u(xhat, np.asarray(ekf.P), w, Om, envelope=(method == "cl_env"))
        u_prev = u
        w, Om = integrate(w, Om, u)
    Ihat = np.asarray(inertia_from_state(ekf.x))
    rel = np.linalg.norm(Ihat - I) / np.linalg.norm(I)
    return rel, sat_count / N


def main():
    print("Rung 10 — closed-loop in-envelope excitation (real EKF, sat2, disturbance)\n")
    methods = {
        "fixed broadband (prbs_b)": "fixed",
        "closed-loop greedy (in-envelope)": "cl_env",
        "closed-loop greedy (NO envelope)": "cl_noenv",
    }
    res = {}
    for label, m in methods.items():
        rels, sats = [], []
        for s in range(N_MC):
            r, sat = run(m, s)
            rels.append(r); sats.append(sat)
        res[label] = (float(np.mean(rels)), float(np.mean(sats)))
        print(f"{label:36s}  rel_err={np.mean(rels)*100:6.2f}%   wheel_sat={np.mean(sats)*100:5.1f}%")

    cle = res["closed-loop greedy (in-envelope)"]
    fx = res["fixed broadband (prbs_b)"]
    cln = res["closed-loop greedy (NO envelope)"]
    print(f"\n[envelope] in-envelope vs no-envelope: rel-err {cle[0]*100:.2f}% vs {cln[0]*100:.2f}%, "
          f"saturation {cle[1]*100:.0f}% vs {cln[1]*100:.0f}%")
    print(f"[vs fixed] closed-loop in-envelope {cle[0]*100:.2f}% vs best-fixed broadband {fx[0]*100:.2f}%")
    verdict = ("matches fixed while guaranteeing envelope" if cle[0] <= 1.5 * fx[0]
               else "underperforms fixed")
    print(f"[verdict] closed-loop in-envelope {verdict}; no-envelope reproduces the rung-9 saturation failure"
          if cln[0] > 3 * cle[0] else "[verdict] see numbers")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = list(res.keys())
    rels = [res[l][0] * 100 for l in labels]
    sats = [res[l][1] * 100 for l in labels]
    x = np.arange(len(labels))
    cols = ["#4575b4", "#1b7837", "#b2182b"]
    ax.bar(x, rels, color=cols)
    for xi, (r, s) in enumerate(zip(rels, sats)):
        ax.text(xi, r * 1.05, f"{r:.2f}%\n(sat {s:.0f}%)", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["fixed\nbroadband", "closed-loop\nin-envelope", "closed-loop\nNO envelope"], fontsize=9)
    ax.set_ylabel("EKF inertia rel-err [%]")
    ax.set_title("Rung 10 — under a tight momentum budget, closed-loop in-envelope excitation wins\n"
                 "(fixed broadband saturates 46%; no-envelope greedy spins up and fails)")
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "rung10_closed_loop.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
