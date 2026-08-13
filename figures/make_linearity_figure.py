import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import figure_style as fs

RESCAN = os.path.join(ROOT, "results", "figure_rescan", "exp_results")
OUTPDF = os.path.join(ROOT, "figures", "figureS_linearity.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_linearity.png")
OUTCSV = os.path.join(ROOT, "figures", "figureS_linearity_table.csv")

STACKS = ["19_Green", "20_Green", "29_Green", "30_Green"]
SPOTS_TO_PROCESS = 10000
ROI_SIZE = 5

METHODS = [fs.L0001_T1, fs.L01_T1, fs.RL, fs.N2V, fs.PN2V, fs.PPN2V, fs.DEEPCAD]

def stack_path(method, st):
    if method in (fs.L0001_T1, fs.L01_T1):
        lam = "0.001" if method == fs.L0001_T1 else "0.1"
        return os.path.join(ROOT, "results", "ablation_static_v2", "exp_data", st,
                            f"denoised_{st}_static_seq1_lambda_geo={lam}.tif")
    if method == fs.RL:
        return os.path.join(ROOT, "results", "rl_gain17689", "exp_data", st,
                            f"denoised_{st}_training_run_rlg17main.tif")
    return os.path.join(ROOT, "Data", "experimental_data", st, f"{st}_denoised_{method}.tif")


def normalize_image(img):
    img = img.astype(float)
    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99.9)
    if vmax - vmin < 1e-9: return img
    return (img - vmin) / (vmax - vmin)


def get_spot_intensities(original_stack, denoised_stack, spots_df):
    orig_sums = []
    den_sums = []

    norm_orig = normalize_image(original_stack)
    norm_den = normalize_image(denoised_stack)

    half_size = ROI_SIZE // 2
    max_t = norm_orig.shape[0]
    max_h = norm_orig.shape[1]
    max_w = norm_orig.shape[2]

    for _, spot in spots_df.iterrows():
        try:
            t = int(float(spot['FRAME']))
            x = int(round(float(spot['POSITION_X'])))
            y = int(round(float(spot['POSITION_Y'])))

            if t >= max_t: continue
            if x < half_size or x >= max_w - half_size: continue
            if y < half_size or y >= max_h - half_size: continue

            y1, y2 = y - half_size, y + half_size + 1
            x1, x2 = x - half_size, x + half_size + 1

            patch_orig = norm_orig[t, y1:y2, x1:x2]
            patch_den = norm_den[t, y1:y2, x1:x2]

            orig_sums.append(np.sum(patch_orig))
            den_sums.append(np.sum(patch_den))
        except ValueError:
            continue

    return np.array(orig_sums), np.array(den_sums)


def evaluator_loader():
    spec = importlib.util.spec_from_file_location(
        "ee", os.path.join(ROOT, "evaluation", "evaluate_experimental.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_and_clean_trackmate_csv


def main():
    load_tm = evaluator_loader()
    from pathlib import Path

    per_method, used_files, counts = {}, [], {}
    for st in STACKS:
        raw = tifffile.imread(os.path.join(ROOT, "Data", "experimental_data", st, f"{st}.tif"))
        spots = load_tm(Path(os.path.join(ROOT, "Data", "experimental_data", st, f"{st}.csv")))
        n_all = len(spots)
        if len(spots) > SPOTS_TO_PROCESS:
            spots = spots.sample(SPOTS_TO_PROCESS, random_state=42)
        counts[st] = dict(spots_in_csv=int(n_all), sampled=int(len(spots)))
        for m in METHODS:
            p = stack_path(m, st)
            den = tifffile.imread(p)
            x, y = get_spot_intensities(raw, den, spots)
            per_method.setdefault(m, {"x": [], "y": []})
            per_method[m]["x"].append(x)
            per_method[m]["y"].append(y)
            used_files.append(p)
        print(f"  {st}: {len(spots)} spots sampled from {n_all}")

    rows = []
    for m in METHODS:
        x = np.concatenate(per_method[m]["x"])
        y = np.concatenate(per_method[m]["y"])
        sl, ic, r, _, _ = stats.linregress(x, y)
        per_method[m]["xa"], per_method[m]["ya"] = x, y
        per_method[m]["fit"] = (sl, ic, r ** 2)
        rows.append(dict(route="normalized_pixel_sum", method=m, n=int(len(x)),
                         slope=sl, intercept=ic, r_squared=r ** 2, unit="normalized"))
        print(f"  route1 {m:18s} n={len(x):6d} slope={sl:.4f} R2={r**2:.4f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False, float_format="%.6f")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    LIM = max(max(np.percentile(per_method[m]["xa"], 99.9),
                  np.percentile(per_method[m]["ya"], 99.9)) for m in METHODS)
    LIM = float(np.ceil(LIM))
    clipped = {m: int(((per_method[m]["xa"] > LIM) | (per_method[m]["ya"] > LIM)).sum())
               for m in METHODS}
    n_tot = len(per_method[METHODS[0]]["xa"])
    print(f"  common axis limit {LIM:.0f}; points outside the view per method "
          f"(of {n_tot}): {clipped}")

    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.7), constrained_layout=True)
    for ax, m in zip(axes.ravel(), METHODS):
        x, y = per_method[m]["xa"], per_method[m]["ya"]
        sl, ic, r2 = per_method[m]["fit"]
        ax.scatter(x, y, s=1.2, alpha=0.06, color=fs.color_for(m), edgecolors="none",
                   rasterized=True)
        lim = LIM
        ax.plot([0, lim], [0, lim], color="#9a9a97", lw=0.7, ls=(0, (3, 2)), zorder=3)

        xs = np.linspace(float(x.min()), min(float(x.max()), lim), 50)
        ax.plot(xs, sl * xs + ic, color="#1a1a19", lw=0.9, zorder=4)
        ax.set_title(m, fontsize=8, pad=3)
        ax.text(0.04, 0.95, f"slope {sl:.3f}\n$R^2$ {r2:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5, color="#1a1a19")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].plot([], [], color="#9a9a97", lw=0.7, ls=(0, (3, 2)), label="identity")
    axes.ravel()[-1].plot([], [], color="#1a1a19", lw=0.9, label="linear fit")
    axes.ravel()[-1].legend(loc="center", frameon=False, fontsize=6.5)
    for ax in axes[1, :]:
        ax.set_xlabel("Original, normalized 5×5 sum")
    for ax in axes[:, 0]:
        ax.set_ylabel("Denoised, normalized 5×5 sum")
    fig.savefig(OUTPDF, format="pdf", dpi=600)
    fig.savefig(OUTPNG, dpi=300, facecolor="white")
    print(f"\nwrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    return tab


if __name__ == "__main__":
    main()
