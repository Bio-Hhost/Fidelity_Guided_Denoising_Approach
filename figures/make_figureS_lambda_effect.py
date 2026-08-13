import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
from make_figureS_provenance import imagej_auto_range


STATIC_STUDY = os.path.join(ROOT, "results", "ablation_static_v2", "sim_data")
RL_STUDY = os.path.join(ROOT, "results", "rl_gain17689", "sim_data")
RAW = os.path.join(ROOT, "Data", "simulated_data_v2", "Gauss_Poisson_Est")
GT = os.path.join(ROOT, "Data", "simulated_data", "GT",
                  "synthetic_ground_truth_airy_corr_randsiz_scaled_0.1.tif")
OUTPDF = os.path.join(ROOT, "figures", "figureS_lambda_effect.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_lambda_effect.png")

FRAME, SCALE = 111, 7.0
PIXELS_PER_MICRON, SCALE_BAR_MICRONS = 3.9, 10
DIFF_PCT = 99.5

VARIANTS = [("seq1_mask5_geo0.001", fs.L0001_T1, 1, "STATIC"),
            ("seq1_mask5_geo0.1", fs.L01_T1, 1, "STATIC"),
            ("training_run_rlg17main", fs.RL, 5, "RL")]
PAIRS = [(1, 0), (2, 0), (2, 1)]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    frame, scale = FRAME, SCALE
    print(f"frame {frame} at scale {scale:g}, inherited from Figure 3 (not re-selected here)")

    raw_path = os.path.join(RAW, f"sim_Gauss_Poisson_Est_scale_{scale:.2f}.tif")
    raw = tifffile.imread(raw_path, key=frame).astype(float)
    gt = tifffile.imread(GT, key=frame).astype(float)
    imgs = []
    for tag, label, _, study in VARIANTS:
        p = os.path.join(STATIC_STUDY if study == "STATIC" else RL_STUDY,
                         f"denoised_sim_{scale:.2f}_{tag}.tif")
        imgs.append(tifffile.imread(p, key=frame).astype(float))

    fig = plt.figure(figsize=(7.09, 3.72))
    gs = fig.add_gridspec(2, 5, hspace=0.14, wspace=0.045,
                          left=0.010, right=0.990, top=0.945, bottom=0.185)
    panels = []

    def imshow(ax, a, vmin, vmax, cmap="gray"):
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)

    def label(ax, letter, title, sub, colour="white"):
        ax.set_title(title, fontsize=7.0, pad=2.5)
        ax.text(0.018, 0.982, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                va="top", ha="left", color="white" if colour == "white" else "#1a1a19")
        ax.text(0.5, 0.018, sub, transform=ax.transAxes, fontsize=5.6, ha="center", va="bottom",
                color=colour,
                bbox=dict(facecolor="#1a1a19", edgecolor="none", pad=1.1, alpha=0.62)
                if colour == "white" else None)


    row1 = [("Noisy input", raw, ""), ("Ground truth", gt, "")] +            [(v[1], im, "T = %d" % v[2]) for v, im in zip(VARIANTS, imgs)]

    vmin, vmax = imagej_auto_range(np.concatenate([a.ravel() for n, a, _ in row1
                                                   if n != "Ground truth"]))
    print(f"row 1 shared ImageJ stack range [{vmin:.1f}, {vmax:.1f}] ADU")
    for j, (name, a, tsub) in enumerate(row1):
        ax = fig.add_subplot(gs[0, j])
        gtp = (name == "Ground truth")
        own = imagej_auto_range(a)
        imshow(ax, a, *(own if gtp else (vmin, vmax)))

        base = re.sub(r"\s*\(T\s*=\s*\d+\)", "", name).strip()
        title = base if tsub == "" else f"{base}  ({tsub})"
        mad = float(np.median(np.abs(a - np.median(a))) * 1.4826)
        rng = float(np.percentile(a, 99.9) - np.percentile(a, 0.1))
        clamped = float((a <= a.min()).mean() * 100)

        if gtp:
            sub = "ImageJ auto [%.0f, %.0f]" % own
        else:
            sub = "spread %s ADU" % ("%.1f" % rng).rstrip("0").rstrip(".")
        label(ax, "ABCDE"[j], title, sub)
        panels.append(dict(panel="ABCDE"[j], content=name, T=tsub or None, kind="image",
                           display=("ImageJ stretchHistogram, 0.35% saturation, own range"
                                    if gtp else
                                    "ImageJ stretchHistogram, 0.35% saturation, ONE range shared "
                                    "across the four real images (stack mode)"),
                           vmin=float(own[0] if gtp else vmin),
                           vmax=float(own[1] if gtp else vmax),
                           own_auto_range_if_shown_alone=[float(own[0]), float(own[1])],
                           mad_sigma_ADU=mad, spread_p0p1_to_p99p9_ADU=rng,
                           pct_at_minimum=clamped,
                           mad_note=("MAD sigma is exactly 0 on every denoised panel because the "
                                     "output is clamped at the background floor; the statistic "
                                     "degenerates and is not shown on the figure")))
        if j == 0:
            bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
            h, w = a.shape
            x1, yb = w - 0.055 * w, h - 0.065 * h
            ax.plot([x1 - bar, x1], [yb, yb], color="white", lw=1.8, solid_capstyle="butt")
            ax.text(x1 - bar / 2, yb - 0.03 * h, f"{SCALE_BAR_MICRONS} µm", color="white",
                    fontsize=5.9, ha="center", va="bottom")

    pooled = np.abs(np.concatenate([(imgs[b] - imgs[a_]).ravel() for b, a_ in PAIRS]))
    lim = float(np.percentile(pooled, DIFF_PCT))
    n_pooled = int(pooled.size)
    per_panel_p995 = [float(np.percentile(np.abs(imgs[b] - imgs[a_]), DIFF_PCT))
                      for b, a_ in PAIRS]
    print(f"common difference limit +/-{lim:.0f} ADU = P{DIFF_PCT} of the {n_pooled:,} pooled "
          f"|differences| (the three panels alone would give "
          f"{', '.join('%.0f' % v for v in per_panel_p995)})")
    for k, (b, a_) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[1, 2 + k])
        d = imgs[b] - imgs[a_]
        imshow(ax, d, -lim, lim, cmap="RdBu_r")
        name = "%s − %s" % (re.sub(r"\s*\(T\s*=\s*\d+\)", "", VARIANTS[b][1]).strip(),
                             re.sub(r"\s*\(T\s*=\s*\d+\)", "", VARIANTS[a_][1]).strip())
        sub = "max |diff| %.0f ADU" % float(np.abs(d).max())
        label(ax, "FGH"[k], name, sub, colour="#4a2208")
        panels.append(dict(panel="FGH"[k], content=name, kind="difference",
                           display="diverging, symmetric about zero",
                           vmin=-lim, vmax=lim,
                           limit_rule="one COMMON limit, shared by all three difference panels; "
                                      "see difference_convention for how it is derived",
                           symmetric_about_zero=True,
                           max_abs_difference=float(np.abs(d).max()),
                           median_abs_difference=float(np.median(np.abs(d)))))


    import matplotlib as mpl
    
    cax = fig.add_axes([0.470, 0.072, 0.30, 0.016])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("RdBu_r"),
                                   norm=mpl.colors.Normalize(-1, 1), orientation="horizontal")
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels([f"−{lim:.0f}", "0", f"+{lim:.0f}"])
    cb.ax.tick_params(labelsize=5.6, length=1.6, pad=1.2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("difference panels F–H, one common limit", fontsize=5.8, labelpad=2)

    fig.savefig(OUTPDF); fig.savefig(OUTPNG, dpi=400); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    for p in panels:
        print(f"  {p['panel']}  {p['content'][:38]:<40s} [{p['vmin']:.1f}, {p['vmax']:.1f}]")


if __name__ == "__main__":
    main()
