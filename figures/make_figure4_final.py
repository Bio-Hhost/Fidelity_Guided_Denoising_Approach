import glob
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

ADA = os.path.join(ROOT, "results", "figure_rescan", "exp_results")
FIX = os.path.join(ROOT, "results", "figure_rescan", "exp_results_fixed")
IDA = os.path.join(ROOT, "results", "figure_rescan", "identity_control", "results_adaptive")
IDF = os.path.join(ROOT, "results", "figure_rescan", "identity_control", "results_fixed")
OUTPDF = os.path.join(ROOT, "figures", "figure4_experimental.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figure4_experimental.png")
OUTCSV = os.path.join(ROOT, "figures", "figure4_experimental_table.csv")

STACKS = ["19_Green", "20_Green", "29_Green", "30_Green"]
STACK_LABEL = {"19_Green": "Peptide (High Conc.) A", "20_Green": "Peptide (High Conc.) B",
               "29_Green": "Peptide (Low Conc.) A", "30_Green": "Peptide (Low Conc.) B"}
METHODS = fs.FIGURE4_METHODS


def load(directory):
    out, files = {}, []
    for p in sorted(glob.glob(os.path.join(directory, "*_detailed_results.csv"))):
        lab = fs.resolve_label(os.path.basename(p))
        d = pd.read_csv(p)
        d["key"] = d["experiment"].astype(str) + "#" + d["ID"].astype(str)
        out.setdefault(lab, []).append(d)
        files.append(p)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}, files


def main():
    fixed, f_fix = load(FIX)
    adapt, f_ada = load(ADA)
    id_ada, f_ia = load(IDA)
    id_fix, f_if = load(IDF)
    for name, d, src in (("fixed", fixed, FIX), ("adaptive", adapt, ADA),
                         ("identity/adaptive", id_ada, IDA), ("identity/fixed", id_fix, IDF)):
        if not d:
            raise SystemExit(f"no *_detailed_results.csv under {src}\n"
                             f"This figure needs the experimental re-scan tables, which are not "
                             f"part of the repository -- see README.md.")
    fixed[fs.NOISY] = id_fix[fs.NOISY]
    adapt[fs.NOISY] = id_ada[fs.NOISY]

    inner = {m: float(fixed[m]["bg_inner_radius_used"].mean()) for m in METHODS}
    outer = {m: float(fixed[m]["bg_outer_radius_used"].mean()) for m in METHODS}
    nbg = {m: float(fixed[m]["noisy_local_bg_std"].median()) for m in METHODS}
    spread = max(max(v.values()) - min(v.values()) for v in (inner, outer, nbg))
    if spread > 1e-9:
        raise SystemExit(
            f"ABORT: panel A requires a method-independent annulus and control, but the "
            f"cross-method spread is {spread:.6g} (inner {inner}, outer {outer}, noisy {nbg}). "
            f"Panel A must not be drawn from a control that moves with the method.")
    print(f"  assertion passed: cross-method spread {spread:.3g} "
          f"(annulus {inner[METHODS[0]]:.4f}/{outer[METHODS[0]]:.4f}, "
          f"noisy control {nbg[METHODS[0]]:.4f} ADU)")

    # common spot set
    common = None
    for m in METHODS:
        s = set(fixed[m]["key"]) & set(adapt[m]["key"])
        common = s if common is None else (common & s)
    n_before = max(len(fixed[m]) for m in METHODS)
    print(f"  spots: {n_before} max per method -> {len(common)} common across "
          f"{len(METHODS)} series in both passes")
    fixed = {m: d[d["key"].isin(common)].copy() for m, d in fixed.items() if m in METHODS}
    adapt = {m: d[d["key"].isin(common)].copy() for m, d in adapt.items() if m in METHODS}

    for m in METHODS:
        a = adapt[m]
        a["loc"] = np.hypot(pd.to_numeric(a["denoised_fit_x"], errors="coerce") - a["POSITION_X"],
                            pd.to_numeric(a["denoised_fit_y"], errors="coerce") - a["POSITION_Y"])
        a["phot"] = (pd.to_numeric(a["denoised_fit_amplitude"], errors="coerce")
                     - pd.to_numeric(a["noisy_fit_amplitude"], errors="coerce")).abs()

    # panel C: the identity control MEASURES zero, so Noisy is omitted as a consequence
    noisy_phot_max = float(adapt[fs.NOISY]["phot"].abs().max())
    noisy_phot_nonzero = int((adapt[fs.NOISY]["phot"] != 0).sum())
    print(f"  identity control photometry error: max {noisy_phot_max:g}, "
          f"non-zero {noisy_phot_nonzero} of {len(adapt[fs.NOISY])}")

    ref = adapt[fs.NOISY]
    ref = ref.assign(noisy_spot_radius=np.sqrt(
        pd.to_numeric(ref["noisy_fit_sx"], errors="coerce")
        * pd.to_numeric(ref["noisy_fit_sy"], errors="coerce")))
    base = ref.groupby("experiment").agg(
        median_noisy_radius=("noisy_spot_radius", "median"),
        median_noisy_amplitude=("noisy_fit_amplitude", "median"))

    panels = [("A", "Local background σ (ADU)", "bg", None),
              ("B", "Localization error (px)", "loc", "median_noisy_radius"),
              ("C", "Photometry error (ADU)", "phot", "median_noisy_amplitude")]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, axes = plt.subplots(3, 1, figsize=(6.75, 7.4), sharex=True)
    rows = []

    for ax, (tag, ylab, metric, basecol) in zip(axes, panels):
        series = [m for m in METHODS if not (metric == "phot" and m == fs.NOISY)]
        width = 0.86 / len(series)
        annot = []        
        for si, m in enumerate(series):
            data, positions = [], []
            for xi, st in enumerate(STACKS):
                src = fixed[m] if metric == "bg" else adapt[m]
                col = ("noisy_local_bg_std" if (metric == "bg" and m == fs.NOISY)
                       else "denoised_local_bg_std" if metric == "bg" else metric)
                v = src.loc[src["experiment"] == st, col].dropna().values
                data.append(v)
                positions.append(xi + (si - (len(series) - 1) / 2) * width)
                q1, q2, q3 = np.percentile(v, [25, 50, 75])
                rec = dict(panel=tag, metric=metric, method=m, stack=st, n=len(v),
                           median=q2, q1=q1, q3=q3, iqr=q3 - q1)
                if basecol:
                    rec["relative_pct"] = 100 * q2 / float(base.loc[st, basecol])
                if metric == "bg":
                    rec["censored_fraction"] = float((v == 0).mean())
                rows.append(rec)
            bp = ax.boxplot(data, positions=positions, widths=width * 0.8, showfliers=False,
                            patch_artist=True, medianprops=dict(color="#1a1a19", lw=1.0),
                            whiskerprops=dict(color="#5c5c5a", lw=0.6),
                            capprops=dict(color="#5c5c5a", lw=0.6))
            for patch in bp["boxes"]:
                patch.set(facecolor=fs.color_for(m), edgecolor="#1a1a19", linewidth=0.5)
            if basecol:
                # Each percentage sits directly above ITS OWN box, at that box's upper
                # whisker, so the label is read against the distribution it describes.
                for bi, (pos, st) in enumerate(zip(positions, STACKS)):
                    cap_top = float(bp["caps"][2 * bi + 1].get_ydata()[0])
                    pct = 100 * np.median(data[bi]) / float(base.loc[st, basecol])
                    annot.append((pos, cap_top, f"{pct:.1f}%"))

        ax.set_ylabel(ylab)
        ax.set_title(f"{tag}", loc="left", fontweight="bold", fontsize=8.5, pad=3)
        ax.grid(axis="y", color="#e8e8e6", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if annot:
            lo, hi = ax.get_ylim()
            pad = 0.012 * (hi - lo)
            for x, ytop, txt in annot:
                ax.text(x, ytop + pad, txt, rotation=90, fontsize=7,
                        ha="center", va="bottom", color="#1a1a19")
            ax.set_ylim(lo, hi + 0.24 * (hi - lo))

    axes[-1].set_xticks(range(len(STACKS)))
    axes[-1].set_xticklabels([STACK_LABEL[s] for s in STACKS])
    axes[-1].set_xlim(-0.75, len(STACKS) - 0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fs.color_for(m), edgecolor="#1a1a19",
                             linewidth=0.5) for m in METHODS]
    fig.legend(handles, METHODS, loc="upper center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.1)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=300, facecolor="white")
    print(f"  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False, float_format="%.6f")

    return tab


if __name__ == "__main__":
    main()
