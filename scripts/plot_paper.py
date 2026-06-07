"""Generate the paper's headline figures from the committed CSV results.

Reads:
  docs/t1_all_baselines_eval.txt              (constant disturbance, augmented EKF)
  docs/t1_all_baselines_eval_sinusoidal.txt   (sinusoidal disturbance, augmented EKF)
  docs/t1_inference_bench.txt                 (per-method microbenchmark)

Writes (under docs/figs/paper/):
  fig1_headline_bars.{pdf,png}    - rel_err per method per sat (main result)
  fig2_disturbance_effect.{pdf,png} - constant vs sinusoidal degradation
  fig3_inference_ms.{pdf,png}       - per-step inference cost
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _row_lookup(rows, sat, method, key="mean_rel_err"):
    sat_full = sat + ".yaml" if not sat.endswith(".yaml") else sat
    for r in rows:
        if r["sat_cfg"] == sat_full and r["method"] == method:
            v = r.get(key)
            return float(v) if v not in (None, "", "nan") else float("nan")
    return float("nan")


def _setup_axes(ax, title, ylabel="relative error (Frobenius)"):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


METHODS = ["multi step", "chirp", "sine", "prbs",
           "RL_DR", "PPO", "dual-MPC_RH", "dual-MPC_oneshot"]
LABELS = ["multi-step", "chirp", "sine", "PRBS",
          "RL DR", "PPO", "MPC RH", "MPC 1-shot"]
COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#8c564b",
          "#ff7f0e", "#e377c2", "#7f7f7f", "#bcbd22"]
SATS = ["config_sat1", "config_sat2", "config_sat3"]
SAT_LABELS = ["sat1 (CubeSat, I_min=0.16)", "sat2 (MicroSat, I_min=4.53)",
              "sat3 (SmallSat, I_min=10.6)"]


def plot_headline_bars():
    rows = _read_csv(ROOT / "docs" / "t1_all_baselines_eval.txt")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, sat, title in zip(axes, SATS, SAT_LABELS):
        values = [_row_lookup(rows, sat, m) * 100 for m in METHODS]
        x = np.arange(len(METHODS))
        bars = ax.bar(x, values, color=COLORS, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        _setup_axes(ax, title, ylabel="rel. err. (%)")
        for b, v in zip(bars, values):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0)
        # Cap y for readability
        finite = [v for v in values if v == v]
        if finite:
            ax.set_ylim(0, max(60, np.percentile(finite, 75) * 2))
    fig.suptitle("Inertia ID error under disturbance + augmented EKF "
                 "(16 seeds, dt=0.1 s, horizon=15 s)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_headline_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_headline_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig1_headline_bars.pdf'}")


def plot_disturbance_effect():
    rows_const = _read_csv(ROOT / "docs" / "t1_all_baselines_eval.txt")
    rows_sin = _read_csv(ROOT / "docs" / "t1_all_baselines_eval_sinusoidal.txt")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, sat, title in zip(axes, ["config_sat2", "config_sat3"],
                               ["sat2 (MicroSat)", "sat3 (SmallSat)"]):
        const = [_row_lookup(rows_const, sat, m) * 100 for m in METHODS]
        sin = [_row_lookup(rows_sin, sat, m) * 100 for m in METHODS]
        x = np.arange(len(METHODS))
        w = 0.4
        ax.bar(x - w / 2, const, w, label="trained: constant",
               color="#2ca02c", edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, sin, w, label="OOD: sinusoidal",
               color="#d62728", edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        _setup_axes(ax, title, ylabel="rel. err. (%)")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        finite = [v for v in const + sin if v == v]
        if finite:
            ax.set_ylim(0, np.percentile(finite, 90) * 1.5)
    fig.suptitle("Sim-to-sim generalization: constant-trained vs sinusoidal-tested",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_disturbance_effect.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_disturbance_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig2_disturbance_effect.pdf'}")


def plot_inference():
    rows = _read_csv(ROOT / "docs" / "t1_inference_bench.txt")
    names, vals = [], []
    short = {
        "RL DR (deterministic MLP)": "RL DR",
        "PPO (Gaussian mean)": "PPO",
        "Dual-MPC RH (per-step amortized)": "MPC RH\n(per step)",
        "Dual-MPC RH (per-plan call, every 10 steps)": "MPC RH\n(per plan)",
        "Scripted excitation": "scripted",
    }
    for r in rows:
        nm = r["method"]
        if nm in short:
            names.append(short[nm])
            vals.append(float(r["ms_per_step"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(names, vals,
                  color=["#ff7f0e", "#e377c2", "#7f7f7f", "#7f7f7f", "#2ca02c"],
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1e-3) * 1.1,
                f"{v*1000:.0f} μs" if v < 0.1 else f"{v:.2f} ms",
                ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("per-step latency (ms, log)")
    ax.set_title("Inference cost (Intel i5-14600K, one thread, JAX CPU, warm JIT)",
                 fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(1e-3, 50)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_inference_ms.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_inference_ms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig3_inference_ms.pdf'}")


if __name__ == "__main__":
    plot_headline_bars()
    plot_disturbance_effect()
    plot_inference()
    print(f"\nAll figures saved under {OUT}")
