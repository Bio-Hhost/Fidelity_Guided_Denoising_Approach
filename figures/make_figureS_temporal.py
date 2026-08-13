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

EXPS = ["19_Green", "20_Green", "29_Green", "30_Green"]
STACK_LABEL = {"19_Green": "Peptide (High Conc.) A", "20_Green": "Peptide (High Conc.) B",
               "29_Green": "Peptide (Low Conc.) A", "30_Green": "Peptide (Low Conc.) B"}

STACK_SHORT = {"19_Green": "High Conc. A", "20_Green": "High Conc. B",
               "29_Green": "Low Conc. A", "30_Green": "Low Conc. B"}
OUTPDF = os.path.join(ROOT, "figures", "figureS_temporal.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_temporal.png")
OUTCSV = os.path.join(ROOT, "figures", "figureS_temporal_table.csv")

FR = "figure_rescan/exp_results"
RLG = "rl_gain17689/exp_results"
PAIRS = [
    ("$\\lambda$ = 0.001", "#13518e", fs.L0001_T1, fs.L0001_T3,
     "static_seq1_lambda_geo=0.001", "static_seq3_lambda_geo=0.001", FR, "T=1", "T=3"),
    ("$\\lambda$ = 0.1", "#006643", fs.L01_T1, fs.L01_T3,
     "static_seq1_lambda_geo=0.1", "static_seq3_lambda_geo=0.1", FR, "T=1", "T=3"),
    ("frozen $\\lambda$", "#a3651f", fs.FROZEN_T1, fs.FROZEN_T5,
     "training_run_rlg17frozfixedT1", "training_run_rlg17frozfixed", RLG, "T=1", "T=5"),
]


def exp_metrics(key, which, root):
    out = {}
    for e in EXPS:
        p = os.path.join(ROOT, "results", *root.split("/"),
                         f"{e}_denoised_{e}_{key}_detailed_results.csv")
        d = pd.read_csv(p)
        if which == "bg":
            v = d["denoised_local_bg_std"]
        elif which == "loc":
            v = np.hypot(d["denoised_fit_x"] - d["POSITION_X"],
                         d["denoised_fit_y"] - d["POSITION_Y"])
        else:
            v = (d["denoised_fit_amplitude"] - d["noisy_fit_amplitude"]).abs()
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        q1, q3 = (np.percentile(v, [25, 75]) if v.size else (np.nan, np.nan))
        out[e] = dict(values=v, median=float(np.median(v)) if v.size else float("nan"),
                      q1=float(q1), q3=float(q3), n=int(v.size),
                      censored_fraction=float((v == 0).mean()) if v.size else float("nan"),
                      inner_r=float(d.bg_inner_radius_used.median()),
                      outer_r=float(d.bg_outer_radius_used.median()),
                      source=os.path.relpath(p, ROOT).replace("\\", "/"))
    return out


def main():
    sim = pd.read_csv(os.path.join(ROOT, "results", "COMPARISON_simulated_unified.csv"))
    rows, exp_cache = [], {}

    print("simulated, mean over the ten noise scales:")
    sim_summary = {}
    for label, _, m_s, m_l, _, _, _, ts, tl in PAIRS:
        a = sim[sim.method == m_s].sort_values("scale")
        b = sim[sim.method == m_l].sort_values("scale")
        d = {}
        for col in ("AUC", "F1", "Loc_MedianAE"):
            d[col] = dict(short=float(a[col].mean()), long=float(b[col].mean()),
                          delta=float(b[col].mean() - a[col].mean()),
                          long_ahead_scales=int((b[col].values > a[col].values).sum())
                          if col != "Loc_MedianAE"
                          else int((b[col].values < a[col].values).sum()),
                          max_abs_per_scale=float(np.max(np.abs(b[col].values - a[col].values))))
        sim_summary[label] = d
        print(f"  {label:16s} {ts}->{tl}   AUC {d['AUC']['short']:.4f}->{d['AUC']['long']:.4f} "
              f"(long ahead {d['AUC']['long_ahead_scales']}/10)   "
              f"F1 {d['F1']['short']:.4f}->{d['F1']['long']:.4f}   "
              f"Loc {d['Loc_MedianAE']['short']:.4f}->{d['Loc_MedianAE']['long']:.4f}")
        for col in ("AUC", "F1", "Loc_MedianAE"):
            for sc, vs, vl in zip(a.scale.values, a[col].values, b[col].values):
                rows.append(dict(domain="simulated", pair=label, metric=col, scale=sc,
                                 short_T=ts, long_T=tl, short=vs, long=vl, delta=vl - vs))

    print("\nexperimental, median per acquisition:")
    for label, _, _, _, k_s, k_l, root, ts, tl in PAIRS:
        for which in ("bg", "loc", "phot"):
            s_, l_ = exp_metrics(k_s, which, root), exp_metrics(k_l, which, root)
            exp_cache[(label, which)] = (s_, l_)
            ds = [l_[e]["median"] - s_[e]["median"] for e in EXPS]
            if which == "bg":
                cz = [max(s_[e]["censored_fraction"], l_[e]["censored_fraction"]) for e in EXPS]
                if max(cz) > 0.01:
                    print(f"  {label:16s} bg   CENSORED: max zero-background fraction per "
                          f"acquisition " + ", ".join(f"{100*c:.1f}%" for c in cz))
            print(f"  {label:16s} {which:4s} {ts}->{tl}  median delta per acquisition: "
                  + ", ".join(f"{d:+.3f}" for d in ds))
            for e in EXPS:
                rows.append(dict(domain="experimental", pair=label, metric=which, scale=np.nan,
                                 experiment=e, short_T=ts, long_T=tl,
                                 short=s_[e]["median"], long=l_[e]["median"],
                                 delta=l_[e]["median"] - s_[e]["median"]))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.4, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(6.75, 4.62))
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.34,
                          left=0.083, right=0.988, top=0.865, bottom=0.185)
    gsb = gs

    for i, (col, ylab) in enumerate([("AUC", "Detection PR-AUC"), ("F1", "F1-Score"),
                                     ("Loc_MedianAE", "Localization error (px)")]):
        ax = fig.add_subplot(gs[0, i])
        for label, colour, m_s, m_l, _, _, _, ts, tl in PAIRS:
            a = sim[sim.method == m_s].sort_values("scale")
            b = sim[sim.method == m_l].sort_values("scale")
            ax.plot(a.scale, a[col], color=colour, lw=1.0, ls="-", marker="o", ms=1.8)
            ax.plot(b.scale, b[col], color=colour, lw=1.0, ls=(0, (2.6, 1.6)), marker="s", ms=1.8)
        ax.set_xlabel("Noise scale"); ax.set_ylabel(ylab)
        ax.set_xticks([1, 4, 7, 10]); ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(-0.28, 1.09, "ABC"[i], transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    x = np.arange(len(EXPS))
    for i, (which, ylab) in enumerate([("bg", "Background $\\sigma$ (ADU)"),
                                       ("loc", "Localization error (px)"),
                                       ("phot", "Photometry error (ADU)")]):
        ax = fig.add_subplot(gsb[1, i])
        for label, colour, _, _, _, _, _, ts, tl in PAIRS:
            s_, l_ = exp_cache[(label, which)]
            ys = [s_[e]["median"] for e in EXPS]
            yl = [l_[e]["median"] for e in EXPS]
            ax.plot(x, ys, color=colour, lw=1.0, ls="-", marker="o", ms=2.6)
            ax.plot(x, yl, color=colour, lw=1.0, ls=(0, (2.6, 1.6)), marker="s", ms=2.6,
                    mfc="white", mew=0.9)
        ax.set_ylabel(ylab)
        ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
        ax.set_xlim(-0.25, len(EXPS) - 0.75)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([STACK_SHORT[e] for e in EXPS], fontsize=6.2, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.text(-0.28, 1.09, "DEF"[i], transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.4, label=l) for l, c, *_ in PAIRS]
    handles += [Line2D([], [], color="#5a5a57", ls="-", marker="o", ms=3.0, lw=1.0,
                       label="Single frame (T = 1)"),
                Line2D([], [], color="#5a5a57", ls=(0, (2.6, 1.6)), marker="s", ms=3.0, lw=1.0,
                       mfc="white", label="Multi-frame (T = 3; T = 5 for frozen $\\lambda$)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.004), fontsize=6.5, handlelength=2.0, columnspacing=1.25)

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=400, facecolor="white")
    print(f"\n  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame(rows).to_csv(OUTCSV, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
