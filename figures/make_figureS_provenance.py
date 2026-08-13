import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "Data", "experimental_data", "Cy3_Best", "Cy3_Best.tif")
GT = os.path.join(ROOT, "Data", "simulated_data", "GT",
                  "synthetic_ground_truth_airy_corr_randsiz_scaled_0.1.tif")
SIM = os.path.join(ROOT, "Data", "simulated_data_v2", "Gauss_Poisson_Est")
SRC_CSV = os.path.join(ROOT, "Data", "experimental_data", "Cy3_Best", "Cy3_Best.csv")
LOG = os.path.join(ROOT, "REGENERATION_LOG_simulation_v2.md")
OUTPDF = os.path.join(ROOT, "figures", "figureS_provenance.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_provenance.png")

CROP = 56
SHOW_SCALES = [1.0, 10.0]
PIXELS_PER_MICRON, SCALE_BAR_MICRONS = 3.9, 10
C_REF, C_GT, C_S1, C_S10 = "#6b6b68", "#000000", "#1f6fb4", "#d1600f"
C_NOISEBOX, C_ZOOMBOX = "#00c853", "#e5007d"


def noise_regions_from_log():
    m = re.search(r"--noise_regions\s+([\d\s]+?)\s*\\", open(LOG, encoding="utf-8").read())
    if m is None:
        raise RuntimeError(f"could not read --noise_regions from {LOG} -- do not substitute")
    v = [int(x) for x in m.group(1).split()]
    if len(v) % 4:
        raise RuntimeError(f"--noise_regions in {LOG} is not a multiple of 4: {v}")
    return [tuple(v[i:i + 4]) for i in range(0, len(v), 4)]


def imagej_auto_range(a, saturated=0.35, nbins=256):
    a = np.asarray(a, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return lo, lo + 1.0
    binsize = (hi - lo) / nbins
    hist, _ = np.histogram(a, bins=nbins, range=(lo, hi))
    threshold = a.size * saturated / 200.0
    c = np.cumsum(hist)
    below = np.nonzero(c > threshold)[0]
    hmin = int(below[0]) if below.size else 0
    c_rev = np.cumsum(hist[::-1])
    above = np.nonzero(c_rev > threshold)[0]
    hmax = int(nbins - 1 - above[0]) if above.size else nbins - 1
    if hmax <= hmin:
        return lo, hi
    return lo + hmin * binsize, lo + hmax * binsize


def check_gt_source(spots):
    src = pd.read_csv(SRC_CSV, low_memory=False)

    meta_rows = int(pd.to_numeric(src["FRAME"], errors="coerce").isna().sum())
    joined = spots.set_index("SOURCE_CSV_INDEX").join(
        src.set_index(pd.RangeIndex(len(src))), rsuffix="_src", how="inner")
    if len(joined) != len(spots):
        raise RuntimeError(f"{len(spots) - len(joined)} GT spots do not join to {SRC_CSV}")

    diffs = {}
    for gt_col, src_col, tol in [("POSITION_X", "POSITION_X", 5e-5),
                                 ("POSITION_Y", "POSITION_Y", 5e-5),
                                 ("SOURCE_INTENSITY_MEASURE", "TOTAL_INTENSITY_CH1", 0.0)]:
        gt_key = gt_col if gt_col in joined else gt_col + "_src"
        src_key = src_col + "_src" if src_col + "_src" in joined else src_col
        if gt_key == src_key:
            raise RuntimeError(f"comparison of {gt_col} would compare a column with itself")
        a = pd.to_numeric(joined[gt_key], errors="coerce")
        b = pd.to_numeric(joined[src_key], errors="coerce")
        d = float((a - b).abs().max())
        diffs[f"{gt_col}_vs_{src_col}"] = dict(max_abs_difference=d, tolerance=tol,
                                               n=int((a - b).notna().sum()))
        if d > tol:
            raise RuntimeError(f"{gt_col} disagrees with {SRC_CSV}:{src_col} by {d} (tol {tol})")
    return dict(source_csv=os.path.relpath(SRC_CSV, ROOT).replace("\\", "/"),
                looks_like="TrackMate export -- columns LABEL, ID, TRACK_ID, QUALITY, "
                           "POSITION_X/Y/Z/T, FRAME, RADIUS, MEAN/MEDIAN/MIN/MAX/"
                           "TOTAL_INTENSITY_CH1",
                rows_in_file=int(len(src)), trackmate_metadata_rows=meta_rows,
                detections=int(len(src) - meta_rows), gt_spots=int(len(spots)),
                uses_every_detection=bool(len(spots) == len(src) - meta_rows),
                source_index_range=[int(spots.SOURCE_CSV_INDEX.min()),
                                    int(spots.SOURCE_CSV_INDEX.max())],
                frames_covered=int(spots.FRAME.nunique()),
                max_abs_difference=diffs)


def check_registration(spots, n_frames=20, seed=0):
    rng = np.random.default_rng(seed)
    hit, rnd = [], []
    for f in range(5, 5 + n_frames):
        s = spots[spots.FRAME == f]
        if len(s) == 0:
            continue
        a = tifffile.imread(REF, key=int(f)).astype(float)
        med = np.median(a)
        y = np.clip(s.POSITION_Y.round().astype(int), 0, a.shape[0] - 1)
        x = np.clip(s.POSITION_X.round().astype(int), 0, a.shape[1] - 1)
        hit.append(a[y, x] - med)
        rnd.append(a[rng.integers(0, a.shape[0], len(s)), rng.integers(0, a.shape[1], len(s))] - med)
    hit, rnd = np.concatenate(hit), np.concatenate(rnd)
    return dict(n_spots_tested=int(hit.size), mean_at_gt_spots_ADU=float(hit.mean()),
                mean_at_random_ADU=float(rnd.mean()),
                ratio=float(hit.mean() / max(rnd.mean(), 1e-9)),
                frame_indexing="0-indexed (tested against the 1-indexed alternative)",
                verdict="registered" if hit.mean() > 5 * rnd.mean() else "NOT REGISTERED")


def check_alignment(spots, frames=range(100, 140), sigma=17.79, thresh=6.0):
    d_all = []
    for f in frames:
        s = spots[spots.FRAME == f]
        if len(s) == 0:
            continue
        a = tifffile.imread(REF, key=int(f)).astype(float)
        peak = (a == maximum_filter(a, size=5)) & (a > np.median(a) + thresh * sigma)
        yy, xx = np.nonzero(peak)
        if len(yy) == 0:
            continue
        for _, row in s.iterrows():
            d_all.append(float(np.hypot(xx - row.POSITION_X, yy - row.POSITION_Y).min()))
    d = np.array(d_all)
    return dict(n_spots=int(d.size), median_px=float(np.median(d)),
                within_2px_pct=float(100 * (d < 2).mean()),
                within_3px_pct=float(100 * (d < 3).mean()),
                frames=[int(min(frames)), int(max(frames))],
                reference_peak_criterion=f"local max in a 5x5 window, >{thresh:g} sigma above the "
                                         f"frame median with sigma={sigma}")


def main():
    regions = noise_regions_from_log()
    spots = pd.read_csv(GT.replace(".tif", "_spot_info.csv"))

    src = check_gt_source(spots)
    print(f"GT source: {src['source_csv']} -- {src['detections']} TrackMate detections, "
          f"GT uses every one: {src['uses_every_detection']}")
    reg = check_registration(spots)
    print(f"registration: {reg['verdict']} -- {reg['mean_at_gt_spots_ADU']:.1f} ADU at GT spots "
          f"vs {reg['mean_at_random_ADU']:.1f} random ({reg['ratio']:.1f}x)")
    align = check_alignment(spots)
    print(f"alignment: median {align['median_px']:.2f} px to the nearest reference maximum, "
          f"{align['within_2px_pct']:.1f}% within 2 px (n={align['n_spots']})")
    if reg["verdict"] != "registered":
        raise RuntimeError("GT is not registered to the reference; the zoom row would be "
                           "comparing different fields -- stopping rather than drawing it")

    counts = spots.groupby("FRAME").size()
    frame = int(counts.idxmax())
    inframe = spots[spots.FRAME == frame]
    best = inframe.loc[inframe.GT_AMPLITUDE_DRAWN.idxmax()]
    cy, cx = int(round(best.POSITION_Y)), int(round(best.POSITION_X))
    print(f"frame {frame} ({counts.max()} GT spots); crop centred on the brightest spot "
          f"at ({cx}, {cy}), amplitude {best.GT_AMPLITUDE_DRAWN:.0f} ADU")

    imgs = {"Reference (Cy3)": tifffile.imread(REF, key=frame).astype(float),
            "Ground truth": tifffile.imread(GT, key=frame).astype(float)}
    for s in SHOW_SCALES:
        imgs[f"Simulated, scale {s:.0f}"] = tifffile.imread(
            os.path.join(SIM, f"sim_Gauss_Poisson_Est_scale_{s:.2f}.tif"), key=frame).astype(float)

    windows = {name: imagej_auto_range(a) for name, a in imgs.items()}
    for name, (lo, hi) in windows.items():
        print(f"  ImageJ auto range, {name:<20s} [{lo:7.1f}, {hi:7.1f}] ADU")

    y0, y1 = max(0, cy - CROP // 2), max(0, cy - CROP // 2) + CROP
    x0, x1 = max(0, cx - CROP // 2), max(0, cx - CROP // 2) + CROP

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.4,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig = plt.figure(figsize=(6.75, 5.55))
    gs = fig.add_gridspec(2, 4, hspace=0.075, wspace=0.05,
                          left=0.052, right=0.988, top=0.945, bottom=0.435)
    gsb = fig.add_gridspec(1, 2, wspace=0.27,
                           left=0.088, right=0.988, top=0.345, bottom=0.085)

    per_panel, letters, axmap = [], "ABCDEFGH", {}
    for j, (name, im) in enumerate(imgs.items()):
        vmin, vmax = windows[name]
        for row, (a, tag) in enumerate([(im, "full"), (im[y0:y1, x0:x1], "zoom")]):
            ax = fig.add_subplot(gs[row, j])
            axmap[(row, j)] = ax
            ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest",
                      rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_linewidth(0.5)
            clip = float(100.0 * ((a < vmin) | (a > vmax)).mean())
            ax.text(0.965, 0.03, f"{vmin:.0f}-{vmax:.0f}", transform=ax.transAxes, fontsize=5.4,
                    ha="right", va="bottom", color="white",
                    bbox=dict(facecolor="#1a1a19", edgecolor="none", pad=1.0, alpha=0.68))
            if row == 0:
                ax.set_title(name, fontsize=7.2, pad=3)
                if j == 0:
                    bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
                    h_, w_ = a.shape
                    xb, yb = w_ - 0.05 * w_, h_ - 0.06 * h_
                    ax.plot([xb - bar, xb], [yb, yb], color="white", lw=1.8,
                            solid_capstyle="butt", zorder=6)
                    ax.text(xb - bar / 2, yb - 0.028 * h_, f"{SCALE_BAR_MICRONS} µm",
                            color="white", fontsize=5.9, ha="center", va="bottom", zorder=6)
                if j == 0:
                    for (rx0, ry0, rx1, ry1) in regions:
                        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                                               ec=C_NOISEBOX, lw=0.7))
                ax.add_patch(Rectangle((x0, y0), CROP, CROP, fill=False, ec=C_ZOOMBOX, lw=0.9))
            else:
                for sp_ in ax.spines.values():
                    sp_.set_edgecolor(C_ZOOMBOX); sp_.set_linewidth(0.9)
            ax.text(0.032, 0.965, letters[row * 4 + j], transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="top", color="white")
            if j == 0:
                ax.set_ylabel("Full 256 x 256 field" if row == 0 else f"Zoom, {CROP} x {CROP} px",
                              fontsize=6.8, labelpad=3)
            per_panel.append(dict(panel=letters[row * 4 + j], image=name, view=tag,
                                  vmin=float(vmin), vmax=float(vmax), clipped_pct=clip))

    for j in range(len(imgs)):
        axA, axB = axmap[(0, j)], axmap[(1, j)]
        for xy_a, xy_b in [((x0, y0 + CROP), (0.0, 1.0)), ((x0 + CROP, y0 + CROP), (1.0, 1.0))]:
            fig.add_artist(ConnectionPatch(
                xyA=xy_a, coordsA=axA.transData, xyB=xy_b, coordsB=axB.transAxes,
                color=C_ZOOMBOX, lw=0.6, ls=(0, (2.5, 2)), zorder=5, clip_on=False))

    ax = fig.add_subplot(gsb[0, 0])
    ref_bg = np.concatenate([imgs["Reference (Cy3)"][ry0:ry1, rx0:rx1].ravel()
                             for (rx0, ry0, rx1, ry1) in regions])
    s1_bg = np.concatenate([imgs["Simulated, scale 1"][ry0:ry1, rx0:rx1].ravel()
                            for (rx0, ry0, rx1, ry1) in regions])
    bins = np.arange(min(ref_bg.min(), s1_bg.min()), max(ref_bg.max(), s1_bg.max()) + 2, 2)
    for d, c, lab in [(ref_bg, C_REF, "Reference (Cy3)"), (s1_bg, C_S1, "Simulated, scale 1")]:
        sd = float(np.median(np.abs(d - np.median(d))) * 1.4826)
        ax.hist(d, bins=bins, density=True, histtype="step", color=c, lw=1.1,
                label=f"{lab}  median {np.median(d):.0f}, $\\sigma$ {sd:.1f}")
    ax.set_xlabel("Pixel value in the noise-estimation regions (ADU)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right", handlelength=1.3, borderpad=0.2)
    ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.text(-0.20, 1.04, "I", transform=ax.transAxes, fontweight="bold", fontsize=8.5,
            va="bottom")

    ax = fig.add_subplot(gsb[0, 1])
    xs = np.arange(x0, x1)
    for name, c, lw in [("Reference (Cy3)", C_REF, 0.9), ("Ground truth", C_GT, 1.3),
                        ("Simulated, scale 1", C_S1, 0.9), ("Simulated, scale 10", C_S10, 0.9)]:
        ax.plot(xs, imgs[name][cy, x0:x1], color=c, lw=lw, label=name)
    ax.axvline(cx, color="#c9c9c6", lw=0.6, zorder=0)
    ax.set_xlabel(f"x (px), row y = {cy}")
    ax.set_ylabel("Intensity (ADU)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3, borderpad=0.2)
    ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.text(-0.20, 1.04, "J", transform=ax.transAxes, fontweight="bold", fontsize=8.5,
            va="bottom")

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=300, facecolor="white")
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    def stat(a):
        med = float(np.median(a))
        return dict(median_ADU=med, mad_sigma_ADU=float(np.median(np.abs(a - med)) * 1.4826),
                    max_ADU=float(a.max()))


if __name__ == "__main__":
    main()
