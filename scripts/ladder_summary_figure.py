from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Final established Spearman(log-det F, inertia rel-err) per rung.
# Rungs 6 and 8 have no rho (comparison / tracking-error rungs) and are annotated separately.
rungs = [
    (0, -0.93, "clean LS", "anchor: theorem", "hold"),
    (1, -0.09, "+ gyro noise", "EIV bias breaks it", "broken"),
    (2, -0.20, "+ integral regression", "intrinsic, not algebra", "broken"),
    (3, -0.87, "consistent estimator (IV)", "link restored", "hold"),
    (4, -0.82, "+ RW momentum budget", "design problem appears", "hold"),
    (5, -0.81, "+ disturbance (augmented)", "augmentation absorbs it", "hold"),
    (7, -0.88, "time-varying inertia", "holds for tracking", "hold"),
    (9, -0.42, "real EKF (in-envelope)", "validated", "hold"),
]
ekf_saturating = (9, +0.34, "real EKF (wheels saturate)", "model-validity broken", "broken")

COL = {"hold": "#1b7837", "broken": "#b2182b"}

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.axhline(0.0, color="gray", lw=1, ls=":")
ax.axhspan(-1.0, -0.5, color="#1b7837", alpha=0.06)
ax.axhspan(-0.5, 0.5, color="#b2182b", alpha=0.06)
ax.text(0.05, -0.97, "link holds (observability predicts accuracy)", fontsize=8, color="#1b7837")
ax.text(0.05, 0.40, "link broken / inverted", fontsize=8, color="#b2182b")

xs = [r[0] for r in rungs]
ys = [r[1] for r in rungs]
ax.plot(xs, ys, "-", color="0.6", lw=1.2, zorder=1)
for x, y, inc, note, status in rungs:
    ax.scatter([x], [y], s=130, color=COL[status], zorder=3, edgecolor="k", linewidth=0.5)
    ax.annotate(f"{inc}\n{note}", (x, y),
                textcoords="offset points", xytext=(0, 14 if y < -0.4 else -28),
                ha="center", fontsize=7.2, color="0.15")

# rung 9 saturating point + the inversion arrow
xs9, ys9, inc9, note9, st9 = ekf_saturating
ax.scatter([xs9], [ys9], s=130, color=COL[st9], zorder=3, edgecolor="k", linewidth=0.5, marker="s")
ax.annotate(f"{inc9}\n{note9}", (xs9, ys9),
            textcoords="offset points", xytext=(0, 14), ha="center", fontsize=7.2, color="0.15")
ax.annotate("", xy=(9, 0.34), xytext=(9, -0.42),
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5))
ax.text(9.15, -0.05, "leaves the\nfilter's valid\nenvelope", fontsize=7, color="#b2182b", va="center")

ax.set_xlabel("realism ladder rung")
ax.set_ylabel(r"Spearman $\rho$(log-det $F$, inertia rel-err)")
ax.set_title("Does observability predict spacecraft inertia-ID accuracy?\n"
             "It is a theorem (rung 0), breaks under measurement noise (1–2), "
             "and is restored only by a consistent estimator (3+)")
ax.set_xticks(range(0, 10))
ax.set_ylim(-1.05, 0.6)
ax.set_xlim(-0.4, 10.4)
ax.grid(True, axis="x", alpha=0.2)
fig.tight_layout()
out = ROOT / "docs" / "figs" / "ladder_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
