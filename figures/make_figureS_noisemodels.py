import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import tifffile

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PN2V_NM = os.path.join(ROOT, "trained_models", "PN2V_model", "noiseModel.npy")
PN2V_SPEC = os.path.join(ROOT, "trained_models", "PN2V_model", "pn2v_run_config_and_specs.json")
PPN2V_NM = os.path.join(ROOT, "trained_models", "PPN2V_model",
                        "GMMNoiseModel_cy3_3_2_bootstrap.npz")
PPN2V_SPEC = os.path.join(ROOT, "trained_models", "PPN2V_model",
                          "ppn2v_run_config_and_specs.json")
REF = os.path.join(ROOT, "Data", "experimental_data", "Cy3_Best", "Cy3_Best.tif")
OUTPDF = os.path.join(ROOT, "figures", "figureS_noisemodels.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_noisemodels.png")
OUTTAB = os.path.join(ROOT, "figures", "figureS_noisemodels_table.csv")

C_PN2V, C_PPN2V = fs.PALETTE[fs.PN2V], fs.PALETTE[fs.PPN2V]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    nm = np.load(PN2V_NM, allow_pickle=True)
    hist, lo_plane, hi_plane = nm[0], nm[1], nm[2]
    sig_lo, sig_hi = float(lo_plane.min()), float(hi_plane.max())
    nbins = hist.shape[0]
    pspec = json.load(open(PN2V_SPEC))
    rec_lo, rec_hi = pspec["noise_model"]["range"]
    rec_bins = pspec["noise_model"]["bins"]
    if rec_bins != nbins:
        raise RuntimeError(f"specs say {rec_bins} bins, file has {nbins} -- resolve before drawing")
    if not (np.isclose(sig_lo, rec_lo, atol=0.51) and np.isclose(sig_hi, rec_hi, atol=6.0)):
        raise RuntimeError(f"file edges [{sig_lo}, {sig_hi}] disagree with the recorded range "
                           f"[{rec_lo}, {rec_hi}] -- do not draw either until resolved")
    floor = hist.min()

    row_mass = hist.sum(axis=1)
    populated = row_mass > 0.5
    empty = int((~populated).sum())
    srt = -np.sort(-hist, axis=1)
    n99 = (np.cumsum(srt, axis=1) < 0.99).sum(axis=1) + 1
    n99 = np.where(populated, n99, 0)

    z = np.load(PPN2V_NM, allow_pickle=True)
    g_lo, g_hi = float(z["min_signal"][0]), float(z["max_signal"][0])
    weight = z["trained_weight"]
    gspec = json.load(open(PPN2V_SPEC))["noise_model"]
    n_gauss, n_coeff = int(gspec["n_gaussian"]), int(gspec["n_coeff"])
    if weight.shape != (n_gauss * 3, n_coeff):
        raise RuntimeError(f"trained_weight is {weight.shape}, not ({n_gauss * 3}, {n_coeff}) "
                           f"-- the layout is not what the specs record")

    # what the data actually looks like
    stack = tifffile.imread(REF)
    vals = stack.reshape(-1).astype(np.float32)
    n_px = vals.size
    vmax_data = float(vals.max())
    edges = np.geomspace(float(vals.min()), vmax_data * 1.02, 400)
    counts, _ = np.histogram(vals, bins=edges)
    cover_pn2v = float(((vals >= rec_lo) & (vals <= rec_hi)).mean() * 100)
    cover_ppn2v = float(((vals >= g_lo) & (vals <= g_hi)).mean() * 100)
    above_ppn2v = float((vals > g_hi).mean() * 100)
    n_above_pn2v = int((vals > rec_hi).sum())
    n_above_ppn2v = int((vals > g_hi).sum())

    def pct(x):
        """Never round a coverage up to a flat 100%: PN2V's is 99.999918%, not 100%."""
        return f"{x:.4f}%" if x > 99.9 else f"{x:.1f}%"
    print(f"reference stack {stack.shape}, {n_px:,} pixels")
    print(f"  data range      [{vals.min():.0f}, {vmax_data:.0f}] ADU")
    print(f"  PN2V  histogram binning range [{rec_lo}, {rec_hi}] ADU -- bounds BOTH axes, so "
          f"{n_above_pn2v} observed pixels fall outside it and cannot be looked up")
    print(f"  PPN2V mixture signal range    [{g_lo:.2f}, {g_hi:.2f}] ADU -- bounds the SIGNAL "
          f"only; the mixture is unbounded over observations")
    print("  NOT COMPUTED: the fraction of SIGNAL each model spans. The stage-1 N2V pseudo-clean "
          "stack is not on disk.")
    centres_all = np.linspace(rec_lo, rec_hi, nbins)
    pop_lo, pop_hi = float(centres_all[populated].min()), float(centres_all[populated].max())
    n99p = n99[populated]
    hi_third = populated & (centres_all > pop_lo + 2 * (pop_hi - pop_lo) / 3)
    print(f"  PN2V occupancy, from row normalisation: {empty} of {nbins} signal bins are EMPTY")
    print(f"    empty at ADU: {np.round(centres_all[~populated], 1).tolist()}")
    print(f"    populated signal range {pop_lo:.1f} to {pop_hi:.1f} ADU")
    print(f"    among populated bins, observation bins carrying 99 pct of the row: "
          f"min {n99p.min()} median {int(np.median(n99p))} max {n99p.max()}")
    print(f"    upper third of the populated range: median {int(np.median(n99[hi_third]))}, "
          f"min {int(n99[hi_third].min())}   <-- thinly-populated high-signal region")


    fig = plt.figure(figsize=(7.28, 2.95))
    outer = fig.add_gridspec(1, 2, wspace=0.62, left=0.070, right=0.982, top=0.855, bottom=0.185)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 4.6], hspace=0.06)

    def panel(ax, letter, title, lx=-0.165):
        ax.set_title(title, fontsize=7.1, pad=4)
        ax.text(lx, 1.16, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axm = fig.add_subplot(left[0, 0])
    ax = fig.add_subplot(left[1, 0], sharex=axm)
    centres = np.linspace(rec_lo, rec_hi, nbins)
    half = (rec_hi - rec_lo) / (2 * nbins)

    for x in centres[~populated]:
        axm.axvspan(x - half, x + half, color="#c9491f", alpha=0.32, lw=0, zorder=2)
    axm.fill_between(centres, 0, n99, where=populated, step="mid",
                     color=C_PN2V, alpha=0.30, lw=0, zorder=3)
    axm.plot(centres[populated], n99[populated], color=C_PN2V, lw=0.9, zorder=4)
    axm.set_ylim(0, 92); axm.set_yticks([0, 40, 80])
    axm.tick_params(labelsize=5.0, length=1.8, pad=1.2)
    axm.set_ylabel("bins at\n99% mass", fontsize=5.4, labelpad=2)
    plt.setp(axm.get_xticklabels(), visible=False)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    axm.set_title("PN2V noise model", fontsize=7.1, pad=4)
    axm.text(-0.165, 1.42, "A", transform=axm.transAxes, fontsize=9, fontweight="bold",
             va="top", ha="left")

    shown = np.where(hist > floor * 1.0001, hist, np.nan)
    im = ax.imshow(shown.T, origin="lower", aspect="auto", cmap="magma",
                   norm=mcolors.LogNorm(vmin=np.nanmin(shown), vmax=np.nanmax(shown)),
                   extent=[rec_lo, rec_hi, rec_lo, rec_hi], interpolation="nearest",
                   rasterized=True)
    cax = ax.inset_axes([1.028, 0.0, 0.032, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=4.8, length=1.6, pad=1.0)

    cb.set_label("P(observed value | signal)", fontsize=5.5, rotation=270, labelpad=7,
                 color="#3d3d3b")
    ax.set_xlabel("signal (ADU)"); ax.set_ylabel("observed value (ADU)")
    ax.tick_params(labelsize=6.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(outer[0, 1])
    panel(ax, "B", "Signal range of each model", lx=-0.150)
    ax.grid(True, axis="x", lw=0.35, color="#dcdad4", zorder=0); ax.set_axisbelow(True)

    bars = [(2, rec_lo, rec_hi, C_PN2V, "PN2V histogram\ndeclared binning"),
            (1, pop_lo, pop_hi, C_PN2V, "PN2V histogram\nbins with samples"),
            (0, g_lo, g_hi, C_PPN2V, "PPN2V mixture\nsignal range")]
    for i, (y, x0, x1, c, _) in enumerate(bars):
        al = 0.32 if i == 0 else 1.0
        ax.plot([x0, x1], [y, y], color=c, lw=7.0, solid_capstyle="butt", zorder=5, alpha=al)
        for x in (x0, x1):
            ax.plot([x, x], [y - 0.26, y + 0.26], color=c, lw=1.1, zorder=6, alpha=al)
    ax.set_xscale("log")
    ax.set_xlim(rec_lo * 0.94, rec_hi * 1.08)
    ax.set_ylim(-0.65, 2.65)
    ax.set_yticks([b[0] for b in bars])
    ax.set_yticklabels([b[4] for b in bars], fontsize=6.0)
    for tick, (_, _, _, c, _) in zip(ax.get_yticklabels(), bars):
        tick.set_color(c)
    ax.set_xlabel("signal (ADU)")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([150, 200, 300, 500, 800])

    fig.savefig(OUTPDF); fig.savefig(OUTPNG, dpi=400); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    import pandas as pd
    pd.DataFrame([
        dict(model="PN2V", kind="histogram", bins=nbins, range_min=rec_lo, range_max=rec_hi,
             range_bounds="both axes (signal and observation)",
             observed_pixels_outside=n_above_pn2v, empty_signal_bins=empty,
             populated_signal_min=pop_lo, populated_signal_max=pop_hi,
             bins_at_99pct_min=int(n99p.min()), bins_at_99pct_median=int(np.median(n99p)),
             density_drawn=True),
        dict(model="PPN2V", kind="gaussian_mixture", bins=np.nan, range_min=g_lo, range_max=g_hi,
             range_bounds="signal only; unbounded over observations",
             observed_pixels_outside=np.nan, empty_signal_bins=np.nan,
             populated_signal_min=np.nan, populated_signal_max=np.nan,
             bins_at_99pct_min=np.nan, bins_at_99pct_median=np.nan, density_drawn=False),
    ]).to_csv(OUTTAB, index=False)


if __name__ == "__main__":
    main()
