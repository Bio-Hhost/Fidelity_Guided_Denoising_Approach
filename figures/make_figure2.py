import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import figure_style as fs

TABLE = os.path.join(ROOT, "results", "COMPARISON_simulated_unified.csv")
OUTPDF = os.path.join(ROOT, "figures", "figure2_simulated.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figure2_simulated.png")
OUTCSV = os.path.join(ROOT, "figures", "figure2_simulated_table.csv")

METHODS = [fs.NOISY, fs.N2V, fs.PN2V, fs.PPN2V, fs.DEEPCAD,
           fs.L0001_T1, fs.L0001_T3, fs.L01_T1, fs.L01_T3, fs.RL]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def boot_ci(tp, fp, fn, kind):
    n = int(tp + fp + fn)
    if n <= 0:
        return np.nan, np.nan
    p = np.array([tp, fp, fn], dtype=float) / n
    draws = RNG.multinomial(n, p, size=N_BOOT).astype(float)
    t, f_, m = draws[:, 0], draws[:, 1], draws[:, 2]
    if kind == "F1":
        val = 2 * t / np.clip(2 * t + f_ + m, 1e-9, None)
    else:
        val = t / np.clip(t + m, 1e-9, None)
    return float(np.percentile(val, 2.5)), float(np.percentile(val, 97.5))


def main():
    tab = pd.read_csv(TABLE)
    tab = tab[tab.method.isin(METHODS)].sort_values(["method", "scale"])
    print(f"table: {len(tab)} rows, {tab.method.nunique()} methods")

    ci = {}
    for kind in ("F1", "Recall"):
        for m in METHODS:
            s = tab[tab.method == m].sort_values("scale")
            lo, hi = [], []
            for _, r in s.iterrows():
                a, b = boot_ci(r["TP"], r["FP"], r["FN"], kind)
                lo.append(a); hi.append(b)
            ci[(kind, m)] = (np.array(lo), np.array(hi))

    # (column, y label, draw interval, conditional on detection)
    panels = [("PSNR", "PSNR (dB)", False, False), ("SSIM", "SSIM", False, False),
              ("AUC", "Detection PR-AUC", False, False), ("F1", "F1-Score", True, False),
              ("Loc_MedianAE", "Localization error (px)", False, True),
              ("Phot_MedianAE", "Photometry error (ADU)", False, True)]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
    })
    fig = plt.figure(figsize=(6.75, 4.35))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.36,
                          left=0.085, right=0.995, top=0.855, bottom=0.105)

    rows_out = []
    for i, (col, ylab, with_ci, conditional) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        for m in METHODS:
            s = tab[tab.method == m].sort_values("scale")
            x, y = s["scale"].values, s[col].values
            ret = (s["TP"] / (s["TP"] + s["FN"])).values     # fraction of ground truth matched
            c = fs.color_for(m)
            z = 3 if m == fs.RL else 2
            if conditional:
                # line without markers, then markers whose AREA is the retained fraction, so a
                # point computed over 52 spots cannot be read the same as one over 8374
                ax.plot(x, y, color=c, label=m, zorder=z)
                ax.scatter(x, y, s=1.0 + 21.0 * ret, color=c, zorder=z + 0.1,
                           linewidths=0, clip_on=False)
            else:
                ax.plot(x, y, marker="o", ms=2.2, color=c, label=m, zorder=z)
            if with_ci:
                lo, hi = ci[(col, m)]
                ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
            for xv, yv, rv in zip(x, y, ret):
                rows_out.append(dict(metric=col, method=m, scale=xv, value=yv,
                                     retained_fraction=rv if conditional else ""))
        ax.set_xlabel("Noise scale")
        ax.set_ylabel(ylab)
        ax.set_xticks([1, 4, 7, 10])
        ax.grid(color="#ececea", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(-0.30, 1.06, "ABCDEF"[i], transform=ax.transAxes, fontweight="bold",
                fontsize=8.5, va="bottom")
    handles = [plt.Line2D([], [], color=fs.color_for(m), marker="o", ms=2.6, lw=1.1, label=m)
               for m in METHODS]
    lm = fig.legend(handles, METHODS, loc="upper center", ncol=5, frameon=False,
                    bbox_to_anchor=(0.42, 1.005), columnspacing=1.1, handlelength=1.6)
    # marker AREA is an encoding, so it gets a key rather than a sentence on the panel.
    # Sizes are the mapping the panels use, s = 1.0 + 21.0 * retained fraction, and
    # the encoded quantity is TP/(TP+FN), which IS recall -- verified equal to the table's own
    # Recall column to 5e-7.
    size_keys = [plt.Line2D([], [], color="#57544e", marker="o", ls="none",
                            ms=np.sqrt(1.0 + 21.0 * r), label=f"{r:.0%}") for r in (0.25, 1.0)]
    ls = fig.legend(handles=size_keys, loc="upper right", bbox_to_anchor=(0.995, 1.005), ncol=2,
                    frameon=False, fontsize=6.0, handlelength=1.0, columnspacing=0.8,
                    title="Marker area on E and F:\nfraction of ground truth found",
                    title_fontsize=6.0)
    ls._legend_box.align = "right"
    fig.add_artist(lm)
    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=300, facecolor="white")
    print(f"  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame(rows_out).to_csv(OUTCSV, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
