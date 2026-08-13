import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "results", "r11_evidence")
SPOTS = os.path.join(EV, "shift_test_spots.csv")
SUMM = os.path.join(EV, "shift_test_biasscatter.csv")
PERD = os.path.join(EV, "shift_test_periods.csv")
OUTPDF = os.path.join(ROOT, "figures", "figureS_biasscatter.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_biasscatter.png")
OUTTAB = os.path.join(ROOT, "figures", "figureS_biasscatter_table.csv")

POOLING_PERIOD = 16       
NBINS = 8
LABEL = {"DeepCAD-RT": fs.DEEPCAD, "N2V": fs.N2V, "PN2V": fs.PN2V,
         "PPN2V": fs.PPN2V, "RL": fs.RL}
ORDER = ["DeepCAD-RT", "N2V", "PN2V", "PPN2V", "RL"]


def main():
    for p in (SPOTS, SUMM, PERD):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run figures/run_shift_test.py first")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    per = pd.read_csv(SPOTS)
    S = pd.read_csv(SUMM).set_index("method")
    P = pd.read_csv(PERD)

    fig = plt.figure(figsize=(7.09, 4.75))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.34,
                          left=0.088, right=0.985, top=0.845, bottom=0.098)

    def panel(ax, letter, title):
        ax.set_title(title, fontsize=7.1, pad=4)
        ax.text(-0.20, 1.13, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left")
        ax.grid(True, lw=0.35, color="#dcdad4", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A", "Bias against scatter")
    for m in ORDER:
        c = fs.PALETTE[LABEL[m]]
        ax.scatter(S.loc[m, "bias"], S.loc[m, "scatter"], s=46, color=c, zorder=5,
                   edgecolor="white", linewidth=0.7)
    for m, dxo, dyo, ha in (("DeepCAD-RT", 0.009, 0.042, "left"),
                            ("N2V", -0.009, 0.042, "right"),
                            ("PPN2V", 0.000, -0.050, "center"), ("RL", 0.007, 0.038, "left"),
                            ("PN2V", 0.008, -0.028, "left")):
        ax.text(S.loc[m, "bias"] + dxo, S.loc[m, "scatter"] + dyo, LABEL[m], fontsize=5.9,
                color=fs.PALETTE[LABEL[m]], ha=ha, va="center")
    ax.set_xlabel("bias (px)"); ax.set_ylabel("scatter (px)")
    ax.set_xlim(-0.016, 0.245); ax.set_ylim(0.28, 0.95)

    for j, m in enumerate(("DeepCAD-RT", "N2V")):
        ax = fig.add_subplot(gs[0, 1 + j])
        panel(ax, "BC"[j], f"{LABEL[m]} — where the fits land")
        d = per[(per.method == m) & per.keep]
        c = fs.PALETTE[LABEL[m]]
        ax.scatter(d.dx, d.dy, s=0.7, color=c, alpha=0.16, linewidths=0, zorder=3, rasterized=True)
        ax.axhline(0, color="#8a877f", lw=0.5, zorder=2)
        ax.axvline(0, color="#8a877f", lw=0.5, zorder=2)
        bx, by = S.loc[m, "mean_dx"], S.loc[m, "mean_dy"]
        ax.add_patch(Circle((bx, by), S.loc[m, "scatter"], fill=False, edgecolor="#121212",
                            lw=1.0, ls=(0, (4, 2)), zorder=6))
        ax.plot([bx], [by], marker="+", ms=9, mew=1.6, color="#121212", zorder=7)
        ax.set_xlabel("Δx (px)")
        ax.set_ylabel("Δy (px)" if j == 0 else "")
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "D", "Variation of bias across a period")
    periods = sorted(P.period.unique())
    xs = np.arange(len(periods))
    for m in ORDER:
        v = [P[(P.method == m) & (P.period == p)].peak_to_peak.iloc[0] for p in periods]
        ax.plot(xs, v, color=fs.PALETTE[LABEL[m]], lw=1.3, marker="o", ms=3.2, zorder=4)
    hl = periods.index(POOLING_PERIOD)
    ax.axvspan(hl - 0.32, hl + 0.32, color="#f0ece0", zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([str(p) for p in periods])
    ax.set_xlabel("period tested (px)")
    ax.set_ylabel("peak-to-peak of mean\nbias across bins (px)")
    ax.set_ylim(0, 0.175)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "E", f"Bias across the {POOLING_PERIOD}-pixel period")
    edges = np.linspace(0, 1, NBINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2 * POOLING_PERIOD
    profile = {}
    for m in ORDER:
        d = per[(per.method == m) & per.keep]
        ph = (d.gt_x.to_numpy() % POOLING_PERIOD) / POOLING_PERIOD
        dx = d.dx.to_numpy()
        v = [dx[(ph >= edges[i]) & (ph < edges[i + 1])].mean() for i in range(NBINS)]
        profile[m] = v
        ax.plot(centres, v, color=fs.PALETTE[LABEL[m]], lw=1.3, marker="o", ms=3.0, zorder=4)
    ax.axhline(0, color="#8a877f", lw=0.6, zorder=2)
    ax.set_xlabel("position within the period (px)")
    ax.set_ylabel("mean bias in x (px)")
    ax.set_ylim(-0.09, 0.09)

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "F", "Bias as a share of the error")
    yp = np.arange(len(ORDER))[::-1]
    for y, m in zip(yp, ORDER):
        ax.barh(y, S.loc[m, "bias_pct_of_rms"], height=0.6, color=fs.PALETTE[LABEL[m]], zorder=4)
        ax.text(S.loc[m, "bias_pct_of_rms"] + 1.2, y, f"{S.loc[m, 'bias_pct_of_rms']:.1f}%",
                va="center", fontsize=6.2, color="#3d3d3b")
    ax.set_yticks(yp); ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=6.4)
    ax.set_xlabel("|bias| as % of RMS error")
    ax.set_xlim(0, 60); ax.set_ylim(-0.6, 4.6)

    meth = [Line2D([], [], color=fs.PALETTE[LABEL[m]], lw=1.6, marker="o", ms=3.6,
                   label=LABEL[m]) for m in ORDER]
    marks = [
        Line2D([], [], color="#121212", marker="+", ms=8, mew=1.6, ls="none",
               label="mean of the fits (bias)"),
        Line2D([], [], color="#121212", lw=1.0, ls=(0, (4, 2)), label="r.m.s. spread (scatter)"),
        Patch(facecolor="#f0ece0", edgecolor="none",
              label=f"{POOLING_PERIOD} px, the period of interest"),
    ]
    l1 = fig.legend(handles=meth, loc="upper left", bbox_to_anchor=(0.062, 1.005), ncol=5,
                    frameon=False, fontsize=6.3, handlelength=1.9, columnspacing=1.1,
                    title="Method (colour)", title_fontsize=6.3)
    l1._legend_box.align = "left"
    l2 = fig.legend(handles=marks, loc="upper right", bbox_to_anchor=(0.988, 1.005), ncol=1,
                    frameon=False, fontsize=6.3, handlelength=1.9,
                    title="Marks", title_fontsize=6.3)
    l2._legend_box.align = "left"
    fig.add_artist(l1)

    fig.savefig(OUTPDF, dpi=400); fig.savefig(OUTPNG, dpi=400); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    tab = S.reset_index()
    tab["label"] = tab.method.map(LABEL)
    for i, p in enumerate(periods):
        tab[f"pp_period_{p}"] = [P[(P.method == m) & (P.period == p)].peak_to_peak.iloc[0]
                                 for m in tab.method]
    tab.to_csv(OUTTAB, index=False)
    pd.DataFrame(profile, index=[f"{c:.1f}" for c in centres]).to_csv(
        OUTTAB.replace(".csv", "_profile.csv"))

    print(S[["n", "retention_pct", "bias", "scatter", "rms", "bias_pct_of_rms",
             "bias_pct_of_rms_squared"]].round(4).to_string())


if __name__ == "__main__":
    main()
