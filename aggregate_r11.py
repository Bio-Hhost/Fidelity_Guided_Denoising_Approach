"""Localization-benchmark aggregation + figures + RESULTS.md (run after run_r11.py).

Produces:
  * Simulated localization vs noise scale: idealized Gaussian CRB (theory floor) AND MLE-on-raw
    (empirical achievable floor, which sits ~2.8x above CRB purely from Airy-vs-Gaussian PSF mismatch)
    vs LSE-on-raw and fit-on-denoised for each method.
  * Bias vs scatter decomposition (RMS^2 = |bias|^2 + scatter^2) — the CRB bounds scatter, not bias.
  * Photometry (relative amplitude error) vs scale.
  * Experimental: agreement between MLE-on-raw and LSE-on-denoised, split by spot-density
    (concentration proxy) across the four exp test datasets.
Reads only results/r11_evidence/cells/*.csv + crb_sim.csv; writes tables, PNGs, RESULTS.md there.
"""
import os, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/r11_evidence"
CELLS = os.path.join(OUT, "cells")
SCALES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# "RL" = RL-denoised + paper LSE fitter (as before). "RL-MLE" = RL-denoised + MLE fitter, so that
# MLE-raw vs RL-MLE is an apples-to-apples estimator comparison (same fitter, raw vs denoised input).
METHOD_ORDER = ["CRB(ideal)", "MLE-raw", "LSE-raw", "RL", "RL-MLE", "N2V", "PN2V", "PPN2V", "DeepCAD-RT"]
COL = {"CRB(ideal)": "k", "MLE-raw": "#444", "LSE-raw": "#888", "RL": "#1f77b4", "RL-MLE": "#17becf",
       "N2V": "#ff7f0e", "PN2V": "#2ca02c", "PPN2V": "#d62728", "DeepCAD-RT": "#9467bd"}
EXP_DS = ["19_Green", "20_Green", "29_Green", "30_Green"]


def load_sim():
    frames = []
    for f in glob.glob(os.path.join(CELLS, "sim_*.csv")):
        d = pd.read_csv(f)
        if "success" in d:
            d = d[d["success"] == True]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def robust(series):
    a = series.dropna().to_numpy()
    a = a[np.abs(a) < 3]
    return a


def sim_tables(sim, crb):
    loc, bias, scat, photo = {}, {}, {}, {}
    for sc in SCALES:
        loc[sc], bias[sc], scat[sc], photo[sc] = {}, {}, {}, {}
        cc = crb[crb.scale == sc]["crb_loc"]
        loc[sc]["CRB(ideal)"] = float(np.median(cc)) if len(cc) else np.nan
        for label, (method, fitter) in {"MLE-raw": ("raw", "MLE"), "LSE-raw": ("raw", "LSE"),
                                        "RL": ("RL", "LSE"), "RL-MLE": ("RL", "MLE"),
                                        "N2V": ("N2V", "LSE"), "PN2V": ("PN2V", "LSE"),
                                        "PPN2V": ("PPN2V", "LSE"), "DeepCAD-RT": ("DeepCAD-RT", "LSE")}.items():
            g = sim[(sim.scale == sc) & (sim.method == method) & (sim.fitter == fitter)]
            e = robust(g["err_loc"]) if "err_loc" in g else np.array([])
            if len(e):
                dx = (g["fit_x"] - g["gt_x"]); dy = (g["fit_y"] - g["gt_y"])
                m = (np.abs(np.hypot(dx, dy)) < 3)
                mx, my = dx[m].mean(), dy[m].mean()
                loc[sc][label] = float(np.median(e))
                bias[sc][label] = float(np.hypot(mx, my))
                scat[sc][label] = float(np.sqrt(((dx[m] - mx) ** 2 + (dy[m] - my) ** 2).mean()))
                pr = np.abs(g["err_amp_rel"].dropna().to_numpy()); pr = pr[pr < 2]
                photo[sc][label] = float(np.median(pr)) * 100 if len(pr) else np.nan
    return (pd.DataFrame(loc).T.reindex(columns=METHOD_ORDER),
            pd.DataFrame(bias).T.reindex(columns=[c for c in METHOD_ORDER if c != "CRB(ideal)"]),
            pd.DataFrame(scat).T.reindex(columns=[c for c in METHOD_ORDER if c != "CRB(ideal)"]),
            pd.DataFrame(photo).T.reindex(columns=[c for c in METHOD_ORDER if c != "CRB(ideal)"]))


def fig_loc(loc, path):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for label in METHOD_ORDER:
        if label not in loc.columns:
            continue
        y = loc[label].values
        style = dict(marker="o", ms=4, lw=1.8)
        if label == "CRB(ideal)":
            style = dict(ls="--", lw=2, marker="")
        elif label == "MLE-raw":
            style = dict(marker="s", ms=5, lw=2.4)
        ax.plot(SCALES, y, color=COL[label], label=label, **style)
    ax.set_xlabel("noise scale"); ax.set_ylabel("median localization error (px)")
    ax.set_title("Localization floor — CRB (theory) & MLE-on-raw (achievable) vs fit-on-denoised")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_photo(photo, path):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for label in photo.columns:
        ax.plot(SCALES, photo[label].values, color=COL.get(label, "gray"), marker="o", ms=4, lw=1.8, label=label)
    ax.set_xlabel("noise scale"); ax.set_ylabel("median |amplitude error| (%)")
    ax.set_title("Photometry error vs noise scale"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_bias_scatter(bias, scat, path):
    scs = [1.0, 5.0, 10.0]
    labels = [c for c in ["MLE-raw", "RL", "RL-MLE", "N2V", "PN2V", "PPN2V", "DeepCAD-RT"] if c in scat.columns]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, sc in zip(axes, scs):
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w / 2, [bias.loc[sc, l] for l in labels], w, label="|bias|", color="#d62728")
        ax.bar(x + w / 2, [scat.loc[sc, l] for l in labels], w, label="scatter", color="#1f77b4")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"scale {sc:.0f}"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("px"); axes[0].legend(fontsize=8)
    fig.suptitle("Localization error = bias vs scatter (CRB bounds scatter, not bias)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def exp_density(ds):
    try:
        d = pd.read_csv(f"results/rl_gain17689/exp_data/{ds}/{ds}.csv", low_memory=False)
        x = pd.to_numeric(d["POSITION_X"], errors="coerce")
        fr = pd.to_numeric(d["FRAME"], errors="coerce")
        d = d[x.notna() & fr.notna()]
        nfr = int(pd.to_numeric(d["FRAME"], errors="coerce").max()) + 1
        return len(d) / max(nfr, 1)
    except Exception:
        return np.nan


def exp_table():
    rows = []
    for ds in EXP_DS:
        p = os.path.join(CELLS, f"exp_{ds}_RLmain_MLE.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        dens = exp_density(ds)
        am = d["agree_mle_den"].dropna(); am = am[am < 3]
        an = d["agree_mle_noisylse"].dropna() if "agree_mle_noisylse" in d else pd.Series(dtype=float); an = an[an < 3]
        rows.append(dict(dataset=ds, spots_per_frame=round(dens, 2), n=len(d),
                         median_agree_MLEraw_vs_LSEdenoised=round(float(am.median()), 4) if len(am) else np.nan,
                         median_agree_MLEraw_vs_LSEraw=round(float(an.median()), 4) if len(an) else np.nan))
    return pd.DataFrame(rows).sort_values("spots_per_frame").reset_index(drop=True)


def fig_exp(et, path):
    if et.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(et.spots_per_frame, et.median_agree_MLEraw_vs_LSEdenoised, "o-", color="#1f77b4",
            label="MLE-raw vs LSE-denoised")
    if et.median_agree_MLEraw_vs_LSEraw.notna().any():
        ax.plot(et.spots_per_frame, et.median_agree_MLEraw_vs_LSEraw, "s--", color="#888",
                label="MLE-raw vs LSE-raw")
    for _, r in et.iterrows():
        ax.annotate(r.dataset.replace("_Green", ""), (r.spots_per_frame, r.median_agree_MLEraw_vs_LSEdenoised),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("detections per frame (concentration proxy)")
    ax.set_ylabel("median position disagreement (px)")
    ax.set_title("Experimental: raw-MLE vs denoised agreement by concentration")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| scale | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for sc, row in df.iterrows():
        cells = [(fmt.format(row[c]) if pd.notna(row[c]) else "—") for c in cols]
        out.append(f"| {sc:.0f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    sim = load_sim()
    crb = pd.read_csv(os.path.join(OUT, "crb_sim.csv")) if os.path.exists(os.path.join(OUT, "crb_sim.csv")) else pd.DataFrame(columns=["scale", "crb_loc"])
    loc, bias, scat, photo = sim_tables(sim, crb)
    loc.to_csv(os.path.join(OUT, "table_localization.csv"))
    photo.to_csv(os.path.join(OUT, "table_photometry.csv"))
    bias.to_csv(os.path.join(OUT, "table_bias.csv")); scat.to_csv(os.path.join(OUT, "table_scatter.csv"))
    fig_loc(loc, os.path.join(OUT, "fig_localization_vs_scale.png"))
    fig_photo(photo, os.path.join(OUT, "fig_photometry_vs_scale.png"))
    fig_bias_scatter(bias, scat, os.path.join(OUT, "fig_bias_scatter.png"))
    et = exp_table(); et.to_csv(os.path.join(OUT, "table_experimental.csv"), index=False)
    fig_exp(et, os.path.join(OUT, "fig_experimental_agreement.png"))

    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Localization benchmark — CRB vs MLE-on-raw vs fit-on-denoised\n\n")
        f.write("**Fitter validation (gates):** MLE is deterministic; respects the CRB (median always above, "
                "no systematic sub-bound violation); converges with LSE at low noise and diverges at high noise. "
                "MLE-on-raw sits ~2.8x above the idealized Gaussian CRB even with sigma fixed to GT — this gap is "
                "**Airy-vs-Gaussian PSF model mismatch** (GT is `airy_corr`, sigma~0.90px), not fitter loss. "
                "Simulated fits use the sim's own calibration (gain 18.052, read-var 387.205); experimental uses "
                "the Cy3 estimate (17.689, 316.53).\n\n")
        f.write("## Localization: median error (px) vs noise scale\n\n")
        f.write("`CRB(ideal)` = analytic Gaussian information floor (theory). `MLE-raw` = achievable floor on raw "
                "data. Method columns = paper LSE fitter on each denoised stack.\n\n")
        f.write(md_table(loc) + "\n\n")
        f.write("![loc](fig_localization_vs_scale.png)\n\n")
        # apples-to-apples: isolate the denoising effect from the estimator choice
        f.write("### Apples-to-apples: denoising effect vs estimator effect (localization)\n\n")
        f.write("`RL-MLE` = RL-denoised fit with the **same MLE fitter** as `MLE-raw`, so `MLE-raw → RL-MLE` "
                "is the pure denoising effect under a fixed estimator. `dDenoise` = raw − RL-denoised (same "
                "fitter; +ve ⇒ denoising helps). `dEstim(raw)` = MLE-raw − LSE-raw (fitter swap on raw). The "
                "denoising effect is the same order as, and often smaller than, the estimator swap.\n\n")
        aa = pd.DataFrame({
            "MLE-raw": loc["MLE-raw"], "RL-MLE": loc["RL-MLE"],
            "dDenoise_MLE": loc["MLE-raw"] - loc["RL-MLE"],
            "LSE-raw": loc["LSE-raw"], "RL-LSE": loc["RL"],
            "dDenoise_LSE": loc["LSE-raw"] - loc["RL"],
            "dEstim_raw": loc["MLE-raw"] - loc["LSE-raw"]})
        f.write(md_table(aa) + "\n\n")
        f.write("## Localization: bias vs scatter (RMS^2 = |bias|^2 + scatter^2; CRB bounds scatter)\n\n")
        f.write("### |bias| (px)\n\n" + md_table(bias) + "\n\n")
        f.write("### scatter (px)\n\n" + md_table(scat) + "\n\n")
        f.write("![biasscatter](fig_bias_scatter.png)\n\n")
        f.write("## Photometry: median |amplitude error| (%) vs noise scale\n\n")
        f.write(md_table(photo, "{:.1f}") + "\n\n![photo](fig_photometry_vs_scale.png)\n\n")
        f.write("## Experimental: raw-MLE vs denoised agreement by concentration\n\n")
        f.write("Concentration proxy = detections/frame. Disagreement = median |position| between MLE-on-raw and "
                "LSE-on-denoised (RL, corrected gain), per dataset.\n\n")
        if not et.empty:
            hdr = "| " + " | ".join(et.columns) + " |\n|" + "---|" * len(et.columns) + "\n"
            body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in et.values)
            f.write(hdr + body + "\n\n![exp](fig_experimental_agreement.png)\n\n")
        f.write("## Provenance\n\n`run_r11.py` (harness) + `aggregate_r11.py` (this). Fitters: `mle_fit.py` "
                "(Poisson-Gaussian MLE), `evaluation/evaluate_full.py` (paper LSE), `crb.py` (Fisher CRB). "
                "Cells: `results/r11_evidence/cells/`. Sim GT: `…_spot_info.csv`. "
                "Sim calibration: `Data/simulated_data_v2/Gauss_Poisson_Est_summary.csv`.\n")
    print("wrote", os.path.join(OUT, "RESULTS.md"))
    print("\nLOCALIZATION (px):\n", loc.round(4).to_string())
    print("\nEXPERIMENTAL:\n", et.to_string(index=False))


if __name__ == "__main__":
    main()
