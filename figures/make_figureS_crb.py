import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "results", "r11_evidence")
CELLS = os.path.join(EV, "cells")
SIM_SUMMARY = os.path.join(ROOT, "Data", "simulated_data_v2", "Gauss_Poisson_Est_summary.csv")
EXP_NOISE = os.path.join(ROOT, "trained_models", "static_models_new",
                         "static_T1_geo0.1_20260723-202607", "noise_parameters.npy")
OUTPDF = os.path.join(ROOT, "figures", "figureS_crb.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_crb.png")
OUTTAB = os.path.join(ROOT, "figures", "figureS_crb_table.csv")

SCALES = list(range(1, 11))
OUTLIER_PX = 3.0
AGREE_PX = 3.0
EXP_DS = ["19_Green", "20_Green", "29_Green", "30_Green"]
STACK_LABEL = {"19_Green": "Peptide (High Conc.) A", "20_Green": "Peptide (High Conc.) B",
               "29_Green": "Peptide (Low Conc.) A", "30_Green": "Peptide (Low Conc.) B"}

C_RAW = fs.PALETTE[fs.NOISY]      # the control, a neutral
C_RL = fs.PALETTE[fs.RL]          # amber, the method
C_CRB = "#121212"                 # theory, deliberately not a method colour
C_DIFF = "#0f6e78"                # the difference series (panel C); absent from PALETTE
C_PAIR = "#b3b0a7"                # the paired spot set (panel D); light against C_RAW's dark


def load_cell(scale, method, fitter, counts):
    key = f"sim_{scale}p0_{method}_{fitter}".replace("-", "")
    path = os.path.join(CELLS, key + ".csv")
    d = pd.read_csv(path)
    n0 = len(d)
    d = d[d["success"] == True]
    n1 = len(d)
    d = d[d["err_loc"].abs() < OUTLIER_PX]
    counts.append(dict(cell=key, n_attempted=n0, n_converged=n1, n_within_3px=len(d),
                       retention_pct=100.0 * len(d) / n0,
                       file=os.path.relpath(path, ROOT).replace("\\", "/")))
    return d.set_index("spot_id")


def calibration():
    s = pd.read_csv(SIM_SUMMARY)
    g, rv = s["Gain_ADU_per_photon"].unique(), s["ReadNoise_Var"].unique()
    if len(g) != 1 or len(rv) != 1:
        raise RuntimeError(f"{SIM_SUMMARY} carries more than one calibration: {g}, {rv}")
    npz = np.load(EXP_NOISE, allow_pickle=True).item()
    return (dict(gain_ADU_per_photon=float(g[0]), read_noise_var_ADU2=float(rv[0]),
                 source=os.path.relpath(SIM_SUMMARY, ROOT).replace("\\", "/")),
            dict(gain_ADU_per_photon=float(npz["gain_estimate"]),
                 read_noise_var_ADU2=float(npz["gaussian_variance"]),
                 source=os.path.relpath(EXP_NOISE, ROOT).replace("\\", "/")))


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    sim_cal, exp_cal = calibration()
    print(f"simulated calibration  gain {sim_cal['gain_ADU_per_photon']:.6f}, "
          f"read var {sim_cal['read_noise_var_ADU2']:.5f}")
    print(f"experimental           gain {exp_cal['gain_ADU_per_photon']:.6f}, "
          f"read var {exp_cal['read_noise_var_ADU2']:.5f}")

    crb = pd.read_csv(os.path.join(EV, "crb_sim.csv"))
    counts, rows = [], []
    for sc in SCALES:
        cells = {(m, f): load_cell(sc, m, f, counts)
                 for m in ("raw", "RL") for f in ("MLE", "LSE")}
        r = dict(scale=sc, crb_px=float(np.median(crb[crb.scale == sc]["crb_loc"])))
        for f in ("LSE", "MLE"):
            a, b = cells[("raw", f)], cells[("RL", f)]
            pair = sorted(set(a.index) & set(b.index))
            ea, eb = a.loc[pair, "err_loc"], b.loc[pair, "err_loc"]
            r[f"raw_{f}"] = float(np.median(ea))
            r[f"RL_{f}"] = float(np.median(eb))
            r[f"delta_{f}"] = float(np.median(ea) - np.median(eb))
            r[f"n_paired_{f}"] = len(pair)
            r[f"retention_paired_{f}_pct"] = 100.0 * len(pair) / 4000.0
        rows.append(r)
    T = pd.DataFrame(rows).set_index("scale")
    for f in ("LSE", "MLE"):
        T[f"ratio_raw_{f}_over_CRB"] = T[f"raw_{f}"] / T.crb_px
        T[f"ratio_RL_{f}_over_CRB"] = T[f"RL_{f}"] / T.crb_px

    ex = []
    for ds in EXP_DS:
        p = os.path.join(CELLS, f"exp_{ds}_RLmain_MLE.csv")
        d = pd.read_csv(p)
        den, raw = d["agree_mle_den"].dropna(), d["agree_mle_noisylse"].dropna()
        ex.append(dict(dataset=ds, label=STACK_LABEL[ds], n=len(d),
                       spots_per_frame=len(d) / d.frame.nunique(),
                       n_den_within=int((den < AGREE_PX).sum()),
                       n_raw_within=int((raw < AGREE_PX).sum()),
                       agree_den_px=float(den[den < AGREE_PX].median()),
                       agree_raw_px=float(raw[raw < AGREE_PX].median()),
                       file=os.path.relpath(p, ROOT).replace("\\", "/")))
    X = pd.DataFrame(ex).sort_values("label").reset_index(drop=True)

    fig = plt.figure(figsize=(7.09, 4.75))
    gs = fig.add_gridspec(2, 6, hspace=0.52, wspace=1.35,
                          left=0.062, right=0.988, top=0.845, bottom=0.185)
    sc = np.array(SCALES, float)
    DASH = (0, (2.6, 1.6))
    FIT = {"LSE": dict(ls="-", marker="o", ms=3.0),
           "MLE": dict(ls=DASH, marker="s", ms=3.0)}

    def panel(ax, letter, title, lx=-0.235):
        ax.set_title(title, fontsize=7.1, pad=3.5)
        ax.text(lx, 1.13, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left")
        ax.grid(True, lw=0.35, color="#dcdad4", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[0, 0:2])
    panel(ax, "A", "Localization error")
    ax.plot(sc, T.crb_px, color=C_CRB, lw=2.0, ls=(0, (3.2, 2.0)), zorder=5)
    for f in ("LSE", "MLE"):
        ax.plot(sc, T[f"raw_{f}"], color=C_RAW, lw=1.3, zorder=4, **FIT[f])
        ax.plot(sc, T[f"RL_{f}"], color=C_RL, lw=1.3, zorder=4, **FIT[f])
    ax.set_xlabel("noise scale"); ax.set_ylabel("median error (px)")
    ax.set_ylim(0, 0.68); ax.set_xticks(SCALES); ax.tick_params(labelsize=6.2)

    ax = fig.add_subplot(gs[0, 2:4])
    panel(ax, "B", "Error / Cramér–Rao bound")
    for f in ("LSE", "MLE"):
        ax.plot(sc, T[f"ratio_raw_{f}_over_CRB"], color=C_RAW, lw=1.3, zorder=4, **FIT[f])
        ax.plot(sc, T[f"ratio_RL_{f}_over_CRB"], color=C_RL, lw=1.3, zorder=4, **FIT[f])
    ax.set_xlabel("noise scale"); ax.set_ylabel("× the bound")
    ax.set_xticks(SCALES); ax.set_ylim(0, 5.2); ax.tick_params(labelsize=6.2)

    ax = fig.add_subplot(gs[0, 4:6])
    panel(ax, "C", "Effect of denoising")
    ax.axhline(0, color="#3d3d3b", lw=0.7, zorder=3)
    for f in ("LSE", "MLE"):
        ax.plot(sc, T[f"delta_{f}"], color=C_DIFF, lw=1.4, zorder=5, **FIT[f])
    ax.set_xlabel("noise scale"); ax.set_ylabel("raw − denoised (px)")
    ax.set_xticks(SCALES); ax.tick_params(labelsize=6.2)

    ax = fig.add_subplot(gs[1, 0:3])
    panel(ax, "D", "Spots retained", lx=-0.148)
    C = pd.DataFrame(counts)
    for m, c in (("raw", C_RAW), ("RL", C_RL)):
        for f in ("LSE", "MLE"):
            v = [C.loc[C.cell == f"sim_{s}p0_{m}_{f}", "retention_pct"].iloc[0] for s in SCALES]
            ax.plot(sc, v, color=c, lw=1.1, alpha=0.85, zorder=3, **FIT[f])
    for f in ("LSE", "MLE"):
        ax.plot(sc, T[f"retention_paired_{f}_pct"], color=C_PAIR, lw=2.2, zorder=2, **FIT[f])
    ax.set_xlabel("noise scale"); ax.set_ylabel("% of 4000 spots")
    ax.set_xticks(SCALES); ax.set_ylim(70, 101); ax.tick_params(labelsize=6.2)

    ax = fig.add_subplot(gs[1, 3:6])
    panel(ax, "E", "Experimental — estimator agreement", lx=-0.148)
    xp = np.arange(len(X))

    ax.plot(xp, X.agree_den_px, color=C_RL, lw=1.4, marker="D", ms=3.6, zorder=5)
    ax.plot(xp, X.agree_raw_px, color=C_RAW, lw=1.4, marker="D", ms=3.6, zorder=4)
    ax.set_xticks(xp)
    ax.set_xticklabels([l.replace("Peptide (", "").replace(") ", "\n") for l in X.label],
                       fontsize=6.0)
    ax.set_ylabel("median distance\nbetween fits (px)")
    ax.set_ylim(0.33, 0.43); ax.set_xlim(-0.4, len(X) - 0.6); ax.tick_params(labelsize=6.2)


    series = [
        Line2D([], [], color=C_CRB, lw=2.0, ls=(0, (3.2, 2.0)), label="Cramér–Rao bound"),
        Line2D([], [], color=C_RAW, lw=1.4, label="raw input"),
        Line2D([], [], color=C_RL, lw=1.4, label=f"denoised, {fs.RL}"),
        Line2D([], [], color=C_DIFF, lw=1.4, label="raw − denoised"),
        Line2D([], [], color=C_PAIR, lw=2.2, label="paired spot set"),
    ]
    fitter = [
        Line2D([], [], color="#57544e", lw=1.3, ls="-", marker="o", ms=3.2,
               label="least squares"),
        Line2D([], [], color="#57544e", lw=1.3, ls=DASH, marker="s", ms=3.0,
               label="maximum likelihood"),
    ]

    expair = [
        Line2D([], [], color=C_RL, lw=1.4, marker="D", ms=3.4,
               label="max. likelihood on raw  vs  least squares on denoised"),
        Line2D([], [], color=C_RAW, lw=1.4, marker="D", ms=3.4,
               label="max. likelihood on raw  vs  least squares on raw"),
    ]
    l1 = fig.legend(handles=series, loc="upper left", bbox_to_anchor=(0.055, 1.003), ncol=5,
                    frameon=False, fontsize=6.6, handlelength=2.4, columnspacing=1.3,
                    title="Series (colour)", title_fontsize=6.6)
    l1._legend_box.align = "left"
    l2 = fig.legend(handles=fitter, loc="upper right", bbox_to_anchor=(0.988, 1.003), ncol=1,
                    frameon=False, fontsize=6.6, handlelength=4.2,
                    title="Fitter (line, marker), panels A–D", title_fontsize=6.6)
    l2._legend_box.align = "left"
    l3 = fig.legend(handles=expair, loc="lower right", bbox_to_anchor=(0.988, 0.008), ncol=1,
                    frameon=False, fontsize=6.0, handlelength=2.4,
                    title="Panel E — distance between two estimators", title_fontsize=6.0)
    l3._legend_box.align = "left"
    fig.add_artist(l1); fig.add_artist(l2)

    fig.savefig(OUTPDF); fig.savefig(OUTPNG, dpi=400); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    T.to_csv(OUTTAB)
    X.drop(columns=["sha256"]).to_csv(OUTTAB.replace(".csv", "_experimental.csv"), index=False)

    print(T[["crb_px", "raw_LSE", "RL_LSE", "delta_LSE", "raw_MLE", "RL_MLE", "delta_MLE",
             "ratio_raw_MLE_over_CRB"]].round(4).to_string())


if __name__ == "__main__":
    main()
