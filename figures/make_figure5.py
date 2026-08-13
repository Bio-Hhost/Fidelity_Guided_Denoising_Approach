import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import figure_style as fs
from make_figure1 import zoom_spot_loc
from make_figureS_provenance import imagej_auto_range

EXPS = ["19_Green", "20_Green", "29_Green", "30_Green"]
STACK_LABEL = {"19_Green": "Peptide (High Conc.) A", "20_Green": "Peptide (High Conc.) B",
               "29_Green": "Peptide (Low Conc.) A", "30_Green": "Peptide (Low Conc.) B"}
PLOT_REGION_SIZE = 18
PIXELS_PER_MICRON = 3.9
SCALE_BAR_MICRONS = 1


KEY_FOR = {fs.NOISY: None, fs.N2V: "N2V", fs.PN2V: "PN2V", fs.PPN2V: "PPN2V",
           fs.DEEPCAD: "DeepCAD-RT",
           fs.L0001_T1: "static_seq1_lambda_geo=0.001",
           fs.L01_T1: "static_seq1_lambda_geo=0.1",
           fs.RL: "training_run_rlg17main"}
GROUPS = [("Input", [fs.NOISY]),
          ("Current state-of-the-art", [fs.N2V, fs.PN2V, fs.PPN2V, fs.DEEPCAD]),
          ("Fidelity-guided", [fs.L0001_T1, fs.L01_T1, fs.RL])]
PANELS = [m for _, g in GROUPS for m in g]

PUBLISHED = dict(exp="29_Green", spot_id=2973394, frame=970,
                 filename="Spot_2973394_Frame_970_Layout.pdf")

C_REF, C_FIT = "#e8112d", "#1f6fb4"
OUTPDF = os.path.join(ROOT, "figures", "figure5_qualitative.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figure5_qualitative.png")
OUTCSV = os.path.join(ROOT, "figures", "figure5_qualitative_table.csv")


def results_path(exp, key):
    k = key or KEY_FOR[fs.RL]         
    return os.path.join(ROOT, "results", "figure_rescan", "exp_results",
                        f"{exp}_denoised_{exp}_{k}_detailed_results.csv")


def stack_path(exp, key):
    d = os.path.join(ROOT, "results", "figure_rescan", "exp_data", exp)
    return os.path.join(d, f"{exp}.tif") if key is None else \
        os.path.join(d, f"denoised_{exp}_{key}.tif")


def delta_table(exp):
    """Per-spot |fit - reference| for all eight series, plus the frame and reference position."""
    cols, meta = {}, None
    for m in PANELS:
        key = KEY_FOR[m]
        d = pd.read_csv(results_path(exp, key))
        pre = "noisy_fit_" if key is None else "denoised_fit_"
        if meta is None:
            meta = d.set_index("ID")[["FRAME", "POSITION_X", "POSITION_Y"]]
        cols[m] = pd.Series(np.hypot(d[pre + "x"] - d["POSITION_X"],
                                     d[pre + "y"] - d["POSITION_Y"]).values, index=d.ID.values)
        cols[m + "|fitx"] = pd.Series(d[pre + "x"].values, index=d.ID.values)
        cols[m + "|fity"] = pd.Series(d[pre + "y"].values, index=d.ID.values)
    t = pd.DataFrame(cols).join(meta, how="inner")
    return t.dropna(subset=PANELS)


def select_spot():
    ranked, medians, best = [], {}, None
    for exp in EXPS:
        t = delta_table(exp)
        med = t[PANELS].median().values
        medians[exp] = {m: float(v) for m, v in zip(PANELS, med)}
        vals = t[PANELS].to_numpy()
        for i, sid in enumerate(t.index):
            tau = float(kendalltau(med, vals[i])[0])
            ranked.append((tau, int(sid), exp))
            cand = (tau, -int(sid))
            if best is None or cand > (best[0], -best[1]):
                best = (tau, int(sid), exp, t.loc[sid])
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return best, ranked, medians


def audit_published(medians):
    t = delta_table(PUBLISHED["exp"])
    if PUBLISHED["spot_id"] not in t.index:
        return dict(status="spot not present in the corrected-gain tables")
    row = t.loc[PUBLISHED["spot_id"]]
    med = medians[PUBLISHED["exp"]]
    tau = float(kendalltau([med[m] for m in PANELS], [row[m] for m in PANELS])[0])
    order_spot = [m for m in sorted(PANELS, key=lambda m: row[m])]
    order_med = [m for m in sorted(PANELS, key=lambda m: med[m])]
    return dict(
        **PUBLISHED, recovered_from="the published output filename; template "
                                    "Spot_{spot_id}_Frame_{frame_idx}_Layout.pdf, eval.ipynb cell 15",
        kendall_tau_against_its_acquisition_median=tau,
        delta_px={m: float(row[m]) for m in PANELS},
        acquisition_median_px=med,
        spot_ordering_best_first=order_spot, median_ordering_best_first=order_med,
        verdict="REPLACED. At corrected gain this spot inverts the aggregate for lambda = RL, "
                "so it no longer illustrates the aggregate.")


def main():
    (tau, sid, exp, row), ranked, medians = select_spot()
    frame = int(row["FRAME"])
    ref = (float(row["POSITION_X"]), float(row["POSITION_Y"]))
    n_tied = sum(1 for r in ranked if r[0] == tau)
    print(f"selected {exp} spot {sid} frame {frame}: Kendall tau {tau:+.4f} against its "
          f"acquisition median ordering ({n_tied} of {len(ranked)} candidates tie at this tau; "
          f"lowest spot ID wins)")
    for m in PANELS:
        print(f"   {m:16s} delta {row[m]:.4f} px   acquisition median {medians[exp][m]:.4f}")

    published = audit_published(medians)
    print(f"\npublished spot audit: tau {published['kendall_tau_against_its_acquisition_median']:+.4f}"
          f" -- {published['verdict'].split('.')[0]}")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.75, 2.02))
    gs = fig.add_gridspec(1, len(PANELS), wspace=0.055,
                          left=0.006, right=0.994, top=0.620, bottom=0.022)
    fig.suptitle("Qualitative Comparison of Denoising Methods", fontsize=8.4, weight="bold",
                 y=0.985)

    axes, per_panel, rows_out = [], [], []
    for i, m in enumerate(PANELS):
        key = KEY_FOR[m]
        img = tifffile.imread(stack_path(exp, key), key=frame).astype(float)
        patch, (gx1, gy1) = zoom_spot_loc(img, ref, PLOT_REGION_SIZE)
        vmin, vmax = imagej_auto_range(patch)
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        ax.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest",
                  rasterized=True)
        ax.plot(ref[0] - gx1 - 0.5, ref[1] - gy1 - 0.5, "+", color=C_REF, ms=8, mew=1.5, zorder=4)
        ax.plot(row[m + "|fitx"] - gx1 - 0.5, row[m + "|fity"] - gy1 - 0.5, "x", color=C_FIT,
                ms=6.5, mew=1.5, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor("#b8b8b4")
        ax.set_title(m, fontsize=6.8, pad=2.5, color=fs.color_for(m))
        ax.text(0.5, 0.03, f"$\\Delta$ = {row[m]:.2f} px", transform=ax.transAxes,
                fontsize=6.0, ha="center", va="bottom", color="white",
                bbox=dict(facecolor="#1a1a19", edgecolor="none", pad=1.2, alpha=0.72))
        if i == 0:
            bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
            h, w = patch.shape
            ax.plot([w - 1.2 - bar, w - 1.2], [1.3, 1.3], color="white", lw=1.6,
                    solid_capstyle="butt")
            ax.text(w - 1.2 - bar / 2, 2.0, f"{SCALE_BAR_MICRONS} µm", color="white",
                    fontsize=5.4, ha="center", va="top")
        per_panel.append(dict(method=m, delta_px=float(row[m]),
                              acquisition_median_px=medians[exp][m],
                              fit_xy=[float(row[m + "|fitx"]), float(row[m + "|fity"])],
                              vmin=float(vmin), vmax=float(vmax),
                              stack=os.path.relpath(stack_path(exp, key), ROOT).replace("\\", "/")))
        rows_out.append(dict(experiment=exp, spot_id=sid, frame=frame, method=m,
                             delta_px=float(row[m]),
                             acquisition_median_px=medians[exp][m]))

    fig.canvas.draw()
    idx, seps = 0, []
    for gi, (label, members) in enumerate(GROUPS):
        i0, i1 = idx, idx + len(members) - 1
        x0, x1 = axes[i0].get_position().x0, axes[i1].get_position().x1
        fig.text((x0 + x1) / 2, 0.800, label, fontsize=7.0, weight="bold", ha="center",
                 va="top", color="#333333")
        if gi:
            sx = (axes[i0 - 1].get_position().x1 + x0) / 2
            fig.add_artist(Line2D([sx, sx], [0.02, 0.80], transform=fig.transFigure,
                                  color="#808080", lw=0.9, linestyle=(0, (3, 2.5))))
            seps.append(float(sx))
        idx = i1 + 1

    fig.legend(handles=[
        Line2D([], [], color=C_REF, marker="+", ms=8, mew=1.5, ls="none",
               label="reference position"),
        Line2D([], [], color=C_FIT, marker="x", ms=7, mew=1.5, ls="none",
               label="fitted position"),
    ], loc="upper left", bbox_to_anchor=(0.006, 1.005), ncol=2, frameon=False, fontsize=6.6,
        handlelength=1.2, columnspacing=1.4)

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=400, facecolor="white")
    print(f"\n  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame(rows_out).to_csv(OUTCSV, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
