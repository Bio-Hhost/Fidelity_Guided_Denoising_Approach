import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "simulated_data_v2", "Gauss_Poisson_Est")
OUTPDF = os.path.join(ROOT, "figures", "figureS_frames.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_frames.png")

SCALES = [float(s) for s in range(1, 11)]
PIXELS_PER_MICRON, SCALE_BAR_MICRONS = 3.9, 10
sys.path.insert(0, os.path.join(ROOT, "figures"))
from make_figureS_provenance import imagej_auto_range
GT = os.path.join(ROOT, "Data", "simulated_data", "GT",
                  "synthetic_ground_truth_airy_corr_randsiz_scaled_0.1.tif")


def read_recorded_noise_params():
    log = os.path.join(ROOT, "REGENERATION_LOG_simulation_v2.md")
    txt = open(log, encoding="utf-8").read()
    want = {"gain_ADU_per_photon": r"gain_g \(ADU/photon\)",
            "read_noise_sigma_ADU": r"read-noise σ \(ADU\)",
            "read_noise_var_ADU2": r"read-noise var \(ADU²\)"}
    out = {}
    for key, pat in want.items():
        m = re.search(pat + r"\s*\|\s*([\d.]+)", txt)
        if m is None:
            raise RuntimeError(f"could not read '{key}' from {log} -- do not substitute a value")
        out[key] = float(m.group(1))
    out["source"] = os.path.relpath(log, ROOT).replace("\\", "/")
    return out


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.75, 3.05))
    gs = fig.add_gridspec(2, 5, hspace=0.16, wspace=0.045,
                          left=0.012, right=0.988, top=0.925, bottom=0.012)

    gt = tifffile.imread(GT, key=0).astype(float)
    core = (gt - np.median(gt)) > np.percentile(gt - np.median(gt), 99.9)

    frames = [tifffile.imread(os.path.join(RAW, f"sim_Gauss_Poisson_Est_scale_{s:.2f}.tif"),
                              key=0).astype(float) for s in SCALES]
    vmin, vmax = imagej_auto_range(np.concatenate([f.ravel() for f in frames]))
    print(f"  ImageJ stack auto range [{vmin:.1f}, {vmax:.1f}] ADU, over all ten frames pooled")

    stats = []
    for i, sc in enumerate(SCALES):
        path = os.path.join(RAW, f"sim_Gauss_Poisson_Est_scale_{sc:.2f}.tif")
        a = frames[i]
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)) * 1.4826)
        clipped = float(100.0 * ((a < vmin) | (a > vmax)).mean())
        core_ampl = float(a[core].mean() - med)

        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        ax.set_title(f"scale {sc:.0f}", fontsize=7.2, pad=2.5)
        ax.text(0.035, 0.035, f"$\\sigma$ {mad:.0f} ADU", transform=ax.transAxes, fontsize=5.9,
                va="bottom", ha="left", color="white",
                bbox=dict(facecolor="#1a1a19", edgecolor="none", pad=1.1, alpha=0.72))
        if i == 0:
            bar = SCALE_BAR_MICRONS * PIXELS_PER_MICRON
            h, w = a.shape
            x1, yb = w - 0.055 * w, h - 0.065 * h
            ax.plot([x1 - bar, x1], [yb, yb], color="white", lw=1.8, solid_capstyle="butt")
            ax.text(x1 - bar / 2, yb - 0.03 * h, f"{SCALE_BAR_MICRONS} µm", color="white",
                    fontsize=5.9, ha="center", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

        stats.append(dict(scale=sc, median_ADU=med, mad_sigma_ADU=mad,
                          clipped_pct=clipped, emitter_core_ADU_above_bg=core_ampl,
                          emitter_core_over_sigma=core_ampl / mad,
                          shape=list(a.shape)))
        print(f"  scale {sc:4.1f}  median {med:7.2f}  MAD sigma {mad:7.2f}  "
              f"clipped {clipped:5.2f}%  core {core_ampl:6.1f} ADU = {core_ampl/mad:5.2f} sigma")

    sc = np.array([d["scale"] for d in stats])
    var = np.array([d["mad_sigma_ADU"] for d in stats]) ** 2
    slope, icept = (float(v) for v in np.polyfit(sc, var, 1))
    r2 = float(np.corrcoef(sc, var)[0, 1] ** 2)
    rec = read_recorded_noise_params()
    fit = dict(model="MAD variance = slope * scale + intercept", slope_ADU2=slope,
               intercept_ADU2=icept, r_squared=r2,
               recorded_in="REGENERATION_LOG_simulation_v2.md", recorded=rec,
               agreement_pct=(100.0 * abs(slope - rec["read_noise_var_ADU2"])
                              / rec["read_noise_var_ADU2"]),
               interpretation="slope reproduces the recorded read-noise variance and the "
                              "intercept is ~0, as expected when `scale` multiplies the read "
                              "term only and the MAD is dominated by background pixels")
    print(f"\n  variance vs scale: {slope:.2f}*scale {icept:+.2f}, r2 {r2:.5f}")
    print(f"  recorded read-noise variance {rec['read_noise_var_ADU2']} ADU^2 "
          f"-> agreement {fit['agreement_pct']:.2f}%")

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=300, facecolor="white")
    print(f"\nwrote {os.path.relpath(OUTPDF, ROOT)} and .png")


if __name__ == "__main__":
    main()
