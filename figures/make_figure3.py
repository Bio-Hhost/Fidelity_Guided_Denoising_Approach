"""
Figure 3 -- spot detection on one simulated frame.

Run: python figures/make_figure3.py   (needs figures/stepP1_frame_scan.{json,csv})
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import figure_style as fs
from stepP1_frame_scan import GT_CSV, STACKS, TOLERANCE, classify, detect

SCAN_JSON = os.path.join(ROOT, "figures", "stepP1_frame_scan.json")
SCAN_CSV = os.path.join(ROOT, "figures", "stepP1_frame_scan.csv")
OUTPDF = os.path.join(ROOT, "figures", "figure3_detection.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figure3_detection.png")
OUTCSV = os.path.join(ROOT, "figures", "figure3_detection_table.csv")

GROUPS = [("Input", [fs.NOISY]),
          ("Current state-of-the-art", [fs.N2V, fs.PN2V, fs.PPN2V, fs.DEEPCAD]),
          ("Fidelity-guided", [fs.RL])]
SHOWN = [m for _, g in GROUPS for m in g]

C_TP, C_FP, C_FN = "#32d900", "#e8112d", "#2b7fff"     # lime / red / blue
CIRCLE_R, CIRCLE_LW = 4.0, 0.55
PIXELS_PER_MICRON, SCALE_BAR_MICRONS = 3.9, 10


def select_frame(agg_vec):
    """Re-score the scan over THE METHODS SHOWN, so the frame matches the ordering on display."""
    df = pd.read_csv(SCAN_CSV)
    df = df[df.method.isin(SHOWN)]
    ranked = []
    for f, g in df.groupby("frame"):
        s = g.set_index("method")["F1"]
        v = np.array([s[m] for m in SHOWN])
        tau, p = kendalltau(agg_vec, v)
        ranked.append(dict(frame=int(f), kendall_tau=float(tau), p_value=float(p),
                           n_gt=int(g.n_gt.iloc[0]),
                           order=list(np.array(SHOWN)[np.argsort(-v)])))
    ranked.sort(key=lambda r: (-r["kendall_tau"], r["frame"]))
    return ranked


def main():
    scan = json.load(open(SCAN_JSON, encoding="utf-8"))
    agg_all = pd.read_csv(os.path.join(ROOT, "results", "COMPARISON_simulated_unified.csv"))
    scale = float(scan["scale"])
    agg_all = agg_all[(agg_all.scale == scale) & (agg_all.method.isin(SHOWN))]
    agg = agg_all.set_index("method")["F1"]
    agg_vec = np.array([agg[m] for m in SHOWN])
    agg_order = list(np.array(SHOWN)[np.argsort(-agg_vec)])

    # Which Figure 2 panel does this figure correspond to? it corresponds to Figure 2's
    # F1 panel, and cannot correspond to the PR-AUC panel, which integrates over the whole
    # threshold sweep and ranks methods differently.
    agg_auc = agg_all.set_index("method")["AUC"]
    f1_order = list(np.array(SHOWN)[np.argsort(-agg_vec)])
    auc_order = list(np.array(SHOWN)[np.argsort(-np.array([agg_auc[m] for m in SHOWN]))])
    tau_f1_auc = float(kendalltau([agg[m] for m in SHOWN],
                                  [agg_auc[m] for m in SHOWN])[0])
    correspondence = dict(
        corresponds_to="Figure 2 F1 panel, at each method's F1-optimal threshold",
        figure2_F1_order=f1_order, figure2_AUC_order=auc_order,
        F1_and_AUC_orders_identical=bool(f1_order == auc_order),
        kendall_tau_F1_vs_AUC=tau_f1_auc,
        note="PR-AUC integrates over the six-point threshold sweep; F1 here is one operating "
             "point, so the two panels need not rank methods identically and at this scale they "
             "do not. No noise scale makes them agree except scale 1, where every method scores "
             "F1 0.95-0.99 and the unprocessed data ranks second, so nothing is illustrated.")
    print(f"  Figure 2 F1  order: " + " > ".join(f1_order))
    print(f"  Figure 2 AUC order: " + " > ".join(auc_order)
          + f"   (tau vs F1 {tau_f1_auc:+.3f})")

    ranked = select_frame(agg_vec)
    best = ranked[0]
    frame = best["frame"]
    ties = [r["frame"] for r in ranked if r["kendall_tau"] == best["kendall_tau"]]
    print(f"frame {frame}: Kendall tau {best['kendall_tau']:.4f} over the {len(SHOWN)} shown "
          f"methods (p={best['p_value']:.3g}), {best['n_gt']} GT spots; "
          f"tied at this tau: {ties}, lowest index wins")
    print("  aggregate order : " + " > ".join(agg_order))
    print("  frame order     : " + " > ".join(best["order"]))

    if best["order"] != f1_order:
        raise RuntimeError(f"frame ordering {best['order']} != Figure 2 F1 ordering {f1_order}; "
                           "this frame would contradict Figure 2")

    gt = pd.read_csv(GT_CSV)
    gt_xy = gt[gt.FRAME == frame][["POSITION_X", "POSITION_Y"]].values

    panels, rows_out = {}, []
    for m in SHOWN:
        thr = scan["thresholds"][m]["threshold"]
        img = tifffile.imread(os.path.join(ROOT, STACKS[m]), key=frame).astype(float)
        det = detect(img, thr)
        tp, fp, fn, det_tp, gt_hit = classify(det, gt_xy, TOLERANCE)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        panels[m] = dict(img=img, det=det, det_tp=det_tp, gt_hit=gt_hit, tp=tp, fp=fp, fn=fn,
                         f1=f1, threshold=thr)
        rows_out.append(dict(frame=frame, method=m, threshold=thr, TP=tp, FP=fp, FN=fn, F1=f1,
                             n_gt=len(gt_xy)))
        print(f"  {m:14s} thr {thr:5.1f}  TP {tp:3d}  FP {fp:4d}  FN {fn:3d}  F1 {f1:.4f}")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.75, 1.86))
    gs = fig.add_gridspec(1, len(SHOWN), wspace=0.055,
                          left=0.008, right=0.992, top=0.665, bottom=0.055)

    axes, per_panel = [], []
    for i, m in enumerate(SHOWN):
        p = panels[m]
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        vmin = float(np.percentile(p["img"], 25))
        vmax = float(np.percentile(p["img"], 99.9))
        ax.imshow(p["img"], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest",
                  rasterized=True)
        for xy in gt_xy[~p["gt_hit"]]:
            ax.add_patch(Circle(xy, radius=CIRCLE_R, ec=C_FN, fc="none", lw=CIRCLE_LW, alpha=0.9))
        if len(p["det"]):
            for xy in p["det"][~p["det_tp"]]:
                ax.add_patch(Circle(xy, radius=CIRCLE_R, ec=C_FP, fc="none", lw=CIRCLE_LW,
                                    alpha=0.9))
            for xy in p["det"][p["det_tp"]]:
                ax.add_patch(Circle(xy, radius=CIRCLE_R, ec=C_TP, fc="none", lw=CIRCLE_LW,
                                    alpha=0.9))
        h, w = p["img"].shape
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor("#b8b8b4")
        ax.set_title(f"{m}\nTP {p['tp']}   FP {p['fp']}   FN {p['fn']}", fontsize=6.5, pad=2.6,
                     linespacing=1.35, color=fs.color_for(m))
        if i == 0:
            bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
            x1, yb = w - 0.045 * w, h - 0.055 * h
            ax.plot([x1 - bar, x1], [yb, yb], color="white", lw=1.8, solid_capstyle="butt")
            ax.text(x1 - bar / 2, yb - 0.028 * h, f"{SCALE_BAR_MICRONS} µm", color="white",
                    fontsize=5.8, ha="center", va="bottom")
        per_panel.append(dict(method=m, aggregate_rank=agg_order.index(m) + 1,
                              frame_rank=best["order"].index(m) + 1,
                              **{k: p[k] for k in ("tp", "fp", "fn", "f1", "threshold")},
                              vmin=vmin, vmax=vmax,
                              background_grey=float((np.median(p["img"]) - vmin) / (vmax - vmin)),
                              clipped_pct=float(100.0 * ((p["img"] < vmin) |
                                                         (p["img"] > vmax)).mean()),
                              floor_value=float(p["img"].min()),
                              at_floor_pct=float(100.0 * (p["img"] == p["img"].min()).mean()),
                              vmin_equals_floor=bool(vmin == p["img"].min())))

    fig.canvas.draw()
    idx, seps = 0, []
    for gi, (label, members) in enumerate(GROUPS):
        i0, i1 = idx, idx + len(members) - 1
        x0, x1 = axes[i0].get_position().x0, axes[i1].get_position().x1
        fig.text((x0 + x1) / 2, 0.955, label, fontsize=7.4, weight="bold", ha="center",
                 va="top", color="#333333")
        if gi:
            sx = (axes[i0 - 1].get_position().x1 + x0) / 2
            fig.add_artist(Line2D([sx, sx], [0.035, 0.80], transform=fig.transFigure,
                                  color="#333333", lw=0.9, linestyle=(0, (3, 2.5))))
            seps.append(float(sx))
        idx = i1 + 1

    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec=c, mew=1.0, ms=4.5, label=l)
               for c, l in [(C_TP, "True positive"), (C_FP, "False positive"),
                            (C_FN, "False negative")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=6.5, handletextpad=0.35,
               columnspacing=1.6)

    fig.savefig(OUTPDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPNG, dpi=450, facecolor="white", bbox_inches="tight")
    print(f"\n  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame(rows_out).to_csv(OUTCSV, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
