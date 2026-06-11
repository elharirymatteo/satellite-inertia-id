import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rung0_observability_vs_accuracy as r0  # noqa: E402
import rung9_real_ekf as r9  # noqa: E402

# Diagnostic: is rung-9's inverted correlation caused by RW saturation
# (a filter-model mismatch the open-loop FIM ignores)? Re-run with the wheel
# speed limit effectively removed and compare.

fam = r0.profile_family()
names = list(fam.keys())


def peak_omega_wheel(tq, omax):
    r9.OMEGA_MAX = omax
    _, _, Om, _ = r9.simulate(tq)
    return float(np.abs(Om).max())


def run(omax):
    r9.OMEGA_MAX = omax
    rows = {n: r9.evaluate(fam[n]) for n in names}
    ld = np.array([rows[n]["logdet"] for n in names])
    rel = np.array([rows[n]["rel"] for n in names])
    return ld, rel, rows


def main():
    print("Diagnostic: RW saturation as the cause of rung-9's inverted link.\n")

    peaks = {n: peak_omega_wheel(fam[n], 1e12) for n in names}

    ld_s, rel_s, rows_s = run(900.0)        # saturating (rung-9 regime)
    ld_n, rel_n, rows_n = run(1e12)         # no saturation

    print(f"{'profile':16s} {'peak|Om|':>9s} {'sat?':>5s} {'rel(sat)':>9s} {'rel(no-sat)':>12s}")
    for n in sorted(names, key=lambda k: rows_s[k]["logdet"]):
        sat = "YES" if peaks[n] > 900 else "-"
        print(f"{n:16s} {peaks[n]:>9.0f} {sat:>5s} {rows_s[n]['rel']*100:>8.2f}% "
              f"{rows_n[n]['rel']*100:>11.2f}%")

    rho_s, _ = spearmanr(ld_s, rel_s)
    rho_n, _ = spearmanr(ld_n, rel_n)
    print(f"\n[saturating]    Spearman(FIM, EKF rel-err) = {rho_s:+.3f}")
    print(f"[no-saturation] Spearman(FIM, EKF rel-err) = {rho_n:+.3f}")
    # link among non-saturating profiles only, in the saturating run
    keep = [i for i, n in enumerate(names) if peaks[n] <= 900]
    if len(keep) >= 4:
        rho_sub, _ = spearmanr(ld_s[keep], rel_s[keep])
        print(f"[non-sat subset] Spearman(FIM, EKF rel-err) = {rho_sub:+.3f} "
              f"({len(keep)} profiles)")
    print("\n[verdict] saturation-mismatch confirmed as the cause "
          if rho_n < -0.4 else "\n[verdict] saturation is NOT the whole story ")


if __name__ == "__main__":
    main()
