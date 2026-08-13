import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  
import tifffile  
from matplotlib.lines import Line2D 
from matplotlib.patches import Ellipse, Rectangle 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
from make_figureS_provenance import imagej_auto_range

EXP = "20_Green"
SPOT_ID = 473252
FRAME = 806
DENOISED_KEY = "training_run_rlg17main"          # lambda = RL at gain 17.689
ROW_LABEL = "Denoised"

EXPDIR = os.path.join(ROOT, "results", "figure_rescan", "exp_data", EXP)
RAW = os.path.join(EXPDIR, f"{EXP}.tif")
DEN = os.path.join(EXPDIR, f"denoised_{EXP}_{DENOISED_KEY}.tif")
FITS = os.path.join(ROOT, "results", "figure_rescan", "exp_results",
                    f"{EXP}_denoised_{EXP}_{DENOISED_KEY}_detailed_results.csv")
TRACKMATE = os.path.join(ROOT, "Data", "experimental_data", EXP, f"{EXP}.csv")

OUTPDF = os.path.join(ROOT, "figures", "figure1_multilevel.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figure1_multilevel.png")

ZOOM_REGION_SIZE = 64
SPOT_REGION_SIZE = 9
PIXELS_PER_MICRON = 3.9        # = 1 / 0.25641 um; agrees with create_ground_truth.py:28
SCALE_BAR_MICRONS = 5

C_NOISY_FIT = "#e8112d"        # red  '+' -- fit to the unprocessed data  } Figure 5's
C_DEN_FIT = "#1f6fb4"          # blue 'x' -- fit to the denoised data     } convention
C_CHAIN = "#00c853"            # green -- magnification boxes and their leaders


def zoom_spot_loc(frame, spot_xy, region_size):
    x_int, y_int = int(round(spot_xy[0])), int(round(spot_xy[1]))
    half = region_size // 2
    y1, y2 = y_int - half, y_int - half + region_size
    x1, x2 = x_int - half, x_int - half + region_size
    h, w = frame.shape
    pt, pb = -min(0, y1), max(0, y2 - h)
    pl, pr = -min(0, x1), max(0, x2 - w)
    region = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if any(p > 0 for p in (pt, pb, pl, pr)):
        region = np.pad(region, ((pt, pb), (pl, pr)), "edge")
    return region, (x1, y1)


def check_subject():
    with open(TRACKMATE, errors="ignore") as f:
        for i, line in enumerate(f):
            if "LABEL" in line:
                hdr = i
                break
        else:
            raise RuntimeError(f"no TrackMate header row in {TRACKMATE}")
    tm = pd.read_csv(TRACKMATE, header=hdr, low_memory=False)
    tm.columns = tm.columns.str.strip().str.upper()
    for c in ("ID", "FRAME", "POSITION_X", "POSITION_Y", "MAX_INTENSITY_CH1"):
        tm[c] = pd.to_numeric(tm[c], errors="coerce")
    row = tm[tm.ID == SPOT_ID]
    if len(row) != 1:
        raise RuntimeError(f"spot ID {SPOT_ID} matched {len(row)} rows in {TRACKMATE}, expected 1")
    r = row.iloc[0]
    if int(r.FRAME) != FRAME:
        raise RuntimeError(f"spot {SPOT_ID} is on frame {int(r.FRAME)}, filename says {FRAME}")
    a = tifffile.imread(RAW, key=FRAME).astype(float)
    y, x = int(round(r.POSITION_Y)), int(round(r.POSITION_X))
    peak = float(a[max(0, y - 3):y + 4, max(0, x - 3):x + 4].max())
    med = float(np.median(a))
    sig = float(np.median(np.abs(a - med)) * 1.4826)
    if abs(peak - float(r.MAX_INTENSITY_CH1)) > 1e-6:
        raise RuntimeError(f"stack 7x7 peak {peak} != TrackMate MAX_INTENSITY_CH1 "
                           f"{r.MAX_INTENSITY_CH1}; table and pixels disagree")
    return dict(recovered_from="published filename "
                               f"{EXP}_spot_ID{SPOT_ID}_frame_{FRAME}_qualitative_zoom_compact.pdf",
                trackmate_matches=1, frame_agrees_with_filename=True,
                position_xy=[float(r.POSITION_X), float(r.POSITION_Y)],
                trackmate_max_intensity=float(r.MAX_INTENSITY_CH1),
                stack_7x7_peak=peak, peak_matches_table=True,
                frame_shape=list(a.shape), frame_median=med, frame_mad_sigma=sig,
                peak_sigma_above_median=(peak - med) / sig,
                original_selection="sample(n=10, random_state=42) per acquisition -- seeded "
                                   "random draw, not eye-selected")


def main():
    subject = check_subject()
    print(f"subject verified: {EXP} spot {SPOT_ID} frame {FRAME} at "
          f"({subject['position_xy'][0]:.4f}, {subject['position_xy'][1]:.4f}); "
          f"peak {subject['peak_sigma_above_median']:.1f} sigma")

    fits = pd.read_csv(FITS)
    fr = fits[fits.ID == SPOT_ID]
    if len(fr) != 1:
        raise RuntimeError(f"spot {SPOT_ID} matched {len(fr)} rows in {FITS}, expected 1")
    fr = fr.iloc[0]
    if int(fr.FRAME) != FRAME:
        raise RuntimeError(f"fit row frame {int(fr.FRAME)} != {FRAME}")

    def params(prefix):
        return dict(x0=float(fr[prefix + "x"]), y0=float(fr[prefix + "y"]),
                    sigma_x=float(fr[prefix + "sx"]), sigma_y=float(fr[prefix + "sy"]),
                    theta_deg=float(fr[prefix + "theta"]),
                    amplitude=float(fr[prefix + "amplitude"]))

    p_noisy, p_den = params("noisy_fit_"), params("denoised_fit_")
    d_loc = float(np.hypot(p_den["x0"] - p_noisy["x0"], p_den["y0"] - p_noisy["y0"]))
    print(f"  fits: noisy ({p_noisy['x0']:.4f}, {p_noisy['y0']:.4f})  "
          f"denoised ({p_den['x0']:.4f}, {p_den['y0']:.4f})  delta {d_loc:.4f} px")

    frames = {"Unprocessed": tifffile.imread(RAW, key=FRAME).astype(float),
              ROW_LABEL: tifffile.imread(DEN, key=FRAME).astype(float)}

    vmin, vmax = imagej_auto_range(frames["Unprocessed"])
    print(f"  shared display window [{vmin:.1f}, {vmax:.1f}] ADU, ImageJ auto on the raw frame")

    def floor_stats(a):
        v, c = np.unique(a, return_counts=True)
        modal = float(v[int(c.argmax())])
        above = a[a > a.min()]
        return dict(distinct_values=int(len(v)), modal_value=modal,
                    modal_fraction_pct=float(100.0 * c.max() / a.size),
                    min=float(a.min()), max=float(a.max()),
                    above_floor_pct=float(100.0 * (a > a.min()).mean()),
                    mad_sigma_above_floor=(float(np.median(np.abs(above - np.median(above)))
                                                 * 1.4826) if above.size else float("nan")))

    def corner_stats(a, k=25):
        reg = np.concatenate([a[:k, :k].ravel(), a[-k:, -k:].ravel(),
                              a[:k, -k:].ravel(), a[-k:, :k].ravel()])
        q1, q3 = np.percentile(reg, [25, 75])
        return dict(n_px=int(reg.size), median=float(np.median(reg)), iqr=float(q3 - q1),
                    std=float(reg.std()), at_floor_pct=float(100.0 * (reg <= a.min()).mean()))

    background = {k: dict(whole_frame=floor_stats(v), signal_free_corners=corner_stats(v))
                  for k, v in frames.items()}
    b_raw, b_den = background["Unprocessed"], background[ROW_LABEL]
    print(f"  denoised frame: modal value {b_den['whole_frame']['modal_value']:.0f} occupies "
          f"{b_den['whole_frame']['modal_fraction_pct']:.1f}% of pixels -- MAD sigma degenerates")
    print(f"  corners IQR: unprocessed {b_raw['signal_free_corners']['iqr']:.1f} -> denoised "
          f"{b_den['signal_free_corners']['iqr']:.1f} ADU, with "
          f"{b_den['signal_free_corners']['at_floor_pct']:.1f}% of corner pixels at the floor")

    spot_xy = (p_noisy["x0"], p_noisy["y0"])

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.75, 4.72))
    gs = fig.add_gridspec(2, 3, hspace=0.045, wspace=0.045,
                          left=0.075, right=0.995, top=0.915, bottom=0.012)

    col_titles = ["Full frame", f"Zoom region ({ZOOM_REGION_SIZE} x {ZOOM_REGION_SIZE})",
                  f"Spot region ({SPOT_REGION_SIZE} x {SPOT_REGION_SIZE})"]
    axmap, per_panel = {}, []

    for r, (rowname, frame) in enumerate(frames.items()):
        zoom_img, zoom_tl = zoom_spot_loc(frame, spot_xy, ZOOM_REGION_SIZE)
        spot_img, spot_tl = zoom_spot_loc(frame, spot_xy, SPOT_REGION_SIZE)
        want = ((ZOOM_REGION_SIZE, ZOOM_REGION_SIZE), (SPOT_REGION_SIZE, SPOT_REGION_SIZE))
        if (zoom_img.shape, spot_img.shape) != want:
            raise RuntimeError("crop size does not match the column titles: "
                               f"zoom {zoom_img.shape} spot {spot_img.shape}, expected {want}")
        for c, (img, tl) in enumerate([(frame, (0, 0)), (zoom_img, zoom_tl), (spot_img, spot_tl)]):
            ax = fig.add_subplot(gs[r, c])
            axmap[(r, c)] = ax
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest",
                      rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7.4, pad=3)
            if c == 0:
                ax.set_ylabel(rowname, fontsize=7.8, labelpad=6)
                bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
                h, w = img.shape
                x1 = w - 0.07 * w
                yb = h - 0.075 * h
                ax.plot([x1 - bar, x1], [yb, yb], color="white", lw=2.0, solid_capstyle="butt")
                ax.text(x1 - bar / 2, yb - 0.035 * h, f"{SCALE_BAR_MICRONS} µm", color="white",
                        fontsize=6.2, ha="center", va="bottom")
            if c == 0:
                ax.add_patch(Rectangle(zoom_tl, ZOOM_REGION_SIZE, ZOOM_REGION_SIZE, fill=False,
                                       ec=C_CHAIN, lw=0.9))
            if c == 1:
                ax.add_patch(Rectangle((spot_tl[0] - zoom_tl[0], spot_tl[1] - zoom_tl[1]),
                                       SPOT_REGION_SIZE, SPOT_REGION_SIZE, fill=False,
                                       ec=C_CHAIN, lw=0.9))
            if c == 2:    
                pairs = ([(p_noisy, C_NOISY_FIT, "+", 12.0, 1.5, "-")] if r == 0
                         else [(p_noisy, C_NOISY_FIT, "+", 12.0, 1.5, "-"),
                               (p_den, C_DEN_FIT, "x", 6.0, 1.9, (0, (2.2, 1.4)))])
                for p, col, mk, ms, mew, ls in pairs:
                    cx, cy = p["x0"] - spot_tl[0] - 0.5, p["y0"] - spot_tl[1] - 0.5
                    ax.add_patch(Ellipse((cx, cy), width=p["sigma_x"] * 2,
                                         height=p["sigma_y"] * 2, angle=p["theta_deg"],
                                         ec=col, fc="none", lw=1.3, ls=ls))
                    ax.plot(cx, cy, mk, color=col, ms=ms, mew=mew)
                if r == 1:
                    ax.text(0.5, 0.025, f"$\\Delta$ = {d_loc:.3f} px", transform=ax.transAxes,
                            fontsize=6.6, ha="center", va="bottom", color="white",
                            bbox=dict(facecolor="#1a1a19", edgecolor="none", pad=1.6, alpha=0.72))
            per_panel.append(dict(row=rowname, column=col_titles[c], shape=list(img.shape),
                                  clipped_pct=float(100.0 * ((img < vmin) | (img > vmax)).mean())))

    fig.canvas.draw()
    for r in range(2):
        for c, (tl, size) in enumerate([(zoom_tl, ZOOM_REGION_SIZE), (spot_tl, SPOT_REGION_SIZE)]):
            src_ax, dst_ax = axmap[(r, c)], axmap[(r, c + 1)]
            off = (0, 0) if c == 0 else zoom_tl
            x_r = tl[0] - off[0] + size
            for y_corner, dst_y in [(tl[1] - off[1], 1.0), (tl[1] - off[1] + size, 0.0)]:
                p = fig.transFigure.inverted().transform(
                    src_ax.transData.transform((x_r, y_corner)))
                bb = dst_ax.get_position()
                fig.add_artist(Line2D((p[0], bb.x0), (p[1], bb.y1 if dst_y else bb.y0),
                                      transform=fig.transFigure, color=C_CHAIN, lw=1.3,
                                      linestyle=(0, (3.2, 2.0)), zorder=5,
                                      solid_capstyle="butt"))

    handles = [Line2D([], [], color=C_NOISY_FIT, ls="-", lw=1.3, marker="+", ms=8.0, mew=1.5,
                      label="Fit to unprocessed data"),
               Line2D([], [], color=C_DEN_FIT, ls=(0, (2.2, 1.4)), lw=1.3, marker="x", ms=5.5,
                      mew=1.9, label="Fit to denoised data")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handlelength=1.8, columnspacing=1.6)

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=400, facecolor="white")
    print(f"  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")


if __name__ == "__main__":
    main()