import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import figure_style as fs

EXPS = ["19_Green", "20_Green", "29_Green", "30_Green"]
STACK_SHORT = {"19_Green": "High Conc. A", "20_Green": "High Conc. B",
               "29_Green": "Low Conc. A", "30_Green": "Low Conc. B"}
STACK_LABEL = {"19_Green": "Peptide (High Conc.) A", "20_Green": "Peptide (High Conc.) B",
               "29_Green": "Peptide (Low Conc.) A", "30_Green": "Peptide (Low Conc.) B"}

RUN = os.path.join(ROOT, "trained_models", "rl_models_gain17689", "training_run_20260724-023428")
EXP_ROOT = os.path.join(ROOT, "results", "rl_gain17689", "exp_results")
KEY_RL, KEY_FROZEN = "training_run_rlg17main", "training_run_rlg17frozfixed"

C_RL, C_FROZEN = fs.color_for(fs.RL), "#a3651f"

SCAN_DIR = os.path.join(ROOT, "results", "rl_gain17689", "sim_scan")

OUTPDF = os.path.join(ROOT, "figures", "figureS_cotraining.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_cotraining.png")
OUTCSV = os.path.join(ROOT, "figures", "figureS_cotraining_table.csv")


def _series(d, which):
    if which == "bg":
        v = d["denoised_local_bg_std"]
    elif which == "loc":
        v = np.hypot(d["denoised_fit_x"] - d["POSITION_X"],
                     d["denoised_fit_y"] - d["POSITION_Y"])
    else:
        v = (d["denoised_fit_amplitude"] - d["noisy_fit_amplitude"]).abs()
    return pd.Series(np.asarray(v, float), index=d.ID.values)


def exp_medians(which):
    out = {}
    for e in EXPS:
        pa = os.path.join(EXP_ROOT, f"{e}_denoised_{e}_{KEY_RL}_detailed_results.csv")
        pb = os.path.join(EXP_ROOT, f"{e}_denoised_{e}_{KEY_FROZEN}_detailed_results.csv")
        a, b = _series(pd.read_csv(pa), which), _series(pd.read_csv(pb), which)
        j = pd.concat([a.rename("rl"), b.rename("fz")], axis=1).dropna()
        va, vb = j["rl"].to_numpy(), j["fz"].to_numpy()
        out[e] = dict(n=len(va), rl=float(np.median(va)), frozen=float(np.median(vb)),
                      diff=float(np.median(va) - np.median(vb)),
                      censored_fraction_rl=float((va == 0).mean()),
                      censored_fraction_frozen=float((vb == 0).mean()),
                      sources=[os.path.relpath(pa, ROOT).replace("\\", "/"),
                               os.path.relpath(pb, ROOT).replace("\\", "/")])
    return out


def lambda_trace():
    d = pd.read_csv(os.path.join(RUN, "lambda_trace.csv"))
    cfg = json.load(open(os.path.join(RUN, "config.json"), encoding="utf-8"))
    lo, hi = cfg["lambda_geo_bounds"]
    tr = d[d.phase == "train"]
    per_step = tr.groupby(["epoch", "step"])["lambda_actor"].std(ddof=0)
    last10 = per_step[per_step.index.get_level_values(0) >= tr.epoch.max() - 9]
    by_ep = tr.groupby("epoch")["lambda_actor"].agg(["mean", "std"])
    converged = int(by_ep[by_ep["mean"] >= hi - 1e-4].index.min())
    return d, cfg, dict(
        bounds=[lo, hi], warmup_epochs=int(cfg["rl_warmup_epochs"]),
        train_epochs=[int(tr.epoch.min()), int(tr.epoch.max())],
        within_batch_std_actor=dict(median=float(np.nanmedian(per_step)),
                                    max_last10_epochs=float(np.nanmax(last10)),
                                    note="lambda_actor is the policy output; lambda_used adds "
                                         "exploration noise and has std ~1e-2 by construction"),
        at_bound_fraction=float(tr.at_bound.mean()),
        converged_at_train_epoch_0indexed=converged,
        converged_at_train_epoch_1indexed=converged + 1,
        epoch_indexing=("lambda_trace.csv and training_history.csv are 0-indexed; specs.json's "
                        "best_reward_epoch is 1-INDEXED. Panel G plots the 1-indexed axis so both "
                        "agree: lambda reaches the bound at epoch 2, and the selected checkpoint "
                        "is epoch 11, matching specs.json."),
        mean_actor_final=float(by_ep["mean"].iloc[-1]),
        warmup_lambda_used=dict(mean=float(d[d.phase == "warmup"].lambda_used.mean()),
                                min=float(d[d.phase == "warmup"].lambda_used.min()),
                                max=float(d[d.phase == "warmup"].lambda_used.max())),
        source=os.path.relpath(os.path.join(RUN, "lambda_trace.csv"), ROOT).replace("\\", "/"))


def main():
    sim = pd.read_csv(os.path.join(ROOT, "results", "COMPARISON_simulated_unified.csv"))
    rl = sim[sim.method == fs.RL].sort_values("scale")
    fz = sim[sim.method == fs.FROZEN_T5].sort_values("scale")

    specs = json.load(open(os.path.join(RUN, "specs.json"), encoding="utf-8"))
    best_ep, epochs_run = int(specs["best_reward_epoch"]), int(specs["epochs_run"])
    d_trace, cfg, tr = lambda_trace()
    print(f"run: {epochs_run} training epochs; checkpoint SELECTED at epoch {best_ep} "
          f"(specs.json best_reward_epoch, reward {specs['best_reward']:.6f}) -- training did not "
          f"stop there")
    print(f"lambda: converges to the upper bound {tr['bounds'][1]} at train epoch "
          f"{tr['converged_at_train_epoch_1indexed']} (1-indexed); within-batch std of the policy output "
          f"{tr['within_batch_std_actor']['max_last10_epochs']:.2e} over the last 10 epochs; "
          f"at_bound {tr['at_bound_fraction']:.3f}")

    rows, summary = [], {}

    print("\nsimulated -- values and their differences, no inferential band:")
    for col, lab in (("AUC", "PR-AUC"), ("F1", "F1"), ("Loc_MedianAE", "localization")):
        diff = rl[col].values - fz[col].values
        summary[f"simulated|{col}"] = dict(
            n=int(len(diff)), max_abs_diff=float(np.max(np.abs(diff))),
            mean_diff=float(np.mean(diff)),
            max_abs_diff_pct=float(np.max(np.abs(diff) / ((rl[col].values + fz[col].values) / 2))
                                   * 100))
        print(f"   {lab:13s} max |diff| {np.max(np.abs(diff)):.4f}  "
              f"({summary[f'simulated|{col}']['max_abs_diff_pct']:.2f}% of the value)")
        for sc, x, y in zip(rl.scale.values, rl[col].values, fz[col].values):
            rows.append(dict(domain="simulated", metric=col, x=sc, rl=x, frozen=y, diff=x - y))

    print("\nexperimental -- median over spots per acquisition:")
    exp = {}
    for which in ("bg", "loc", "phot"):
        r = exp_medians(which)
        exp[which] = r
        d = np.array([r[e]["diff"] for e in EXPS])
        mid = np.array([(r[e]["rl"] + r[e]["frozen"]) / 2 for e in EXPS])
        summary[f"experimental|{which}"] = dict(
            n=len(EXPS), max_abs_diff=float(np.max(np.abs(d))), mean_diff=float(np.mean(d)),
            max_abs_diff_pct=float(np.max(np.abs(d) / mid) * 100),
            per_acquisition={STACK_LABEL[e]: r[e] for e in EXPS})
        print(f"   {which:5s} max |diff| {np.max(np.abs(d)):.4f}  "
              f"({summary[f'experimental|{which}']['max_abs_diff_pct']:.2f}% of the value)")
        for e in EXPS:
            rows.append(dict(domain="experimental", metric=which, x=e, rl=r[e]["rl"],
                             frozen=r[e]["frozen"], diff=r[e]["diff"]))
    cens = {e: dict(rl=exp["bg"][e]["censored_fraction_rl"],
                    frozen=exp["bg"][e]["censored_fraction_frozen"]) for e in EXPS}
    print("   background censoring (zero-sigma spots): "
          + ", ".join(f"{STACK_SHORT[e]} RL {100*cens[e]['rl']:.1f}% / frozen "
                      f"{100*cens[e]['frozen']:.1f}%" for e in EXPS))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.4, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig = plt.figure(figsize=(6.75, 6.15))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.36,
                          left=0.085, right=0.988, top=0.905, bottom=0.395)
    gsb = fig.add_gridspec(1, 1, left=0.085, right=0.988, top=0.285, bottom=0.075)

    for i, (col, ylab) in enumerate([("AUC", "Detection PR-AUC"), ("F1", "F1-Score"),
                                     ("Loc_MedianAE", "Localization error (px)")]):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(fz.scale, fz[col], color=C_FROZEN, lw=1.0, ls=(0, (2.6, 1.6)), marker="s", ms=2.4,
                mfc="white", mew=0.9)
        ax.plot(rl.scale, rl[col], color=C_RL, lw=1.0, ls="-", marker="o", ms=2.4)
        ax.set_xlabel("Noise scale"); ax.set_ylabel(ylab); ax.set_xticks([1, 4, 7, 10])
        ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        m = summary[f"simulated|{col}"]
        ax.set_title(f"max |difference| {m['max_abs_diff']:.4f}", fontsize=6.2, pad=2.5,
                     color="#5a5a57")
        ax.text(-0.30, 1.12, "ABC"[i], transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    x = np.arange(len(EXPS))
    for i, (which, ylab) in enumerate([("bg", "Background $\\sigma$ (ADU)"),
                                       ("loc", "Localization error (px)"),
                                       ("phot", "Photometry error (ADU)")]):
        ax = fig.add_subplot(gs[1, i])
        r = exp[which]
        ya = np.array([r[e]["rl"] for e in EXPS])
        yb = np.array([r[e]["frozen"] for e in EXPS])
        ax.plot(x, yb, color=C_FROZEN, lw=1.0, ls=(0, (2.6, 1.6)), marker="s", ms=3.0,
                mfc="white", mew=0.9)
        ax.plot(x, ya, color=C_RL, lw=1.0, ls="-", marker="o", ms=3.0)
        ax.set_xticks(x)
        ax.set_xticklabels([STACK_SHORT[e] for e in EXPS], fontsize=6.2, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab); ax.set_xlim(-0.25, len(EXPS) - 0.75)
        ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        m = summary[f"experimental|{which}"]
        ax.set_title(f"max |difference| {m['max_abs_diff']:.4g}", fontsize=6.2, pad=2.5,
                     color="#5a5a57")
        ax.text(-0.30, 1.12, "DEF"[i], transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    ax = fig.add_subplot(gsb[0, 0])
    lo, hi = tr["bounds"]
    n_warm = tr["warmup_epochs"]
    warm = d_trace[d_trace.phase == "warmup"].copy()
    trn = d_trace[d_trace.phase == "train"].copy()
    for part, off in ((warm, 1.0 - n_warm), (trn, 1.0)):
        per_ep = part.groupby("epoch").size().iloc[0]
        part["t"] = off + part.epoch + (np.arange(len(part)) % per_ep) / per_ep
    conv_ep = tr["converged_at_train_epoch_1indexed"]          # 0-indexed trace -> 1-indexed axis
    ax.axvspan(1 - n_warm, 1, color="#f0f0ee", lw=0)
    ax.plot(warm.t, warm.lambda_used, lw=0.3, color="#9a9a97")
    ax.plot(trn.t, trn.lambda_used, lw=0.3, color="#d9b98a")
    g = trn.groupby("epoch")["lambda_actor"].mean()
    ax.plot(g.index.values + 1, g.values, color=C_RL, lw=1.6, zorder=4)
    ax.axhline(hi, color="#3d3d3b", lw=0.7, ls=":", zorder=3)
    # the checkpoint that was actually selected and used -- specs.json best_reward_epoch, 1-indexed
    ax.axvline(best_ep, color="#3d3d3b", lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.text(best_ep + 0.6, hi - 0.055, f"ep {best_ep} of {epochs_run}", fontsize=5.8,
            ha="left", va="top", color="#3d3d3b")
    ax.set_ylim(lo - 0.035, hi + 0.095)
    ax.set_xlim(1 - n_warm, trn.epoch.max() + 1)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("$\\lambda$")
    ax.grid(color="#ececea", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.075, 1.06, "G", transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    handles = [Line2D([], [], color=C_RL, lw=1.2, marker="o", ms=3.2,
                      label="$\\lambda$ = RL (co-trained)"),
               Line2D([], [], color=C_FROZEN, lw=1.2, ls=(0, (2.6, 1.6)), marker="s", ms=3.2,
                      mfc="white", mew=0.9, label="$\\lambda$ = 0.5 (fixed), T = 5"),
]
    lm = fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                    bbox_to_anchor=(0.070, 1.002), fontsize=6.8, handlelength=2.2,
                    title="Panels A–F", title_fontsize=6.8)
    lm._legend_box.align = "left"
    gk = [Line2D([], [], color=C_RL, lw=1.6, label="$\\lambda$ selected by the agent"),
          Line2D([], [], color="#d9b98a", lw=1.0, label="$\\lambda$ applied, per sample"),
          Line2D([], [], color="#9a9a97", lw=1.0, label="$\\lambda$ applied during warm-up"),
          Line2D([], [], color="#3d3d3b", lw=0.7, ls=":", label="upper bound of the range"),
          Line2D([], [], color="#3d3d3b", lw=0.8, ls=(0, (3, 2)), label="epoch used in the paper"),
          Patch(facecolor="#f0f0ee", edgecolor="none", label="warm-up epochs")]
    lg = fig.legend(handles=gk, loc="upper right", ncol=2, frameon=False,
                    bbox_to_anchor=(0.988, 1.002), fontsize=6.2, handlelength=2.0,
                    columnspacing=1.2, title="Panel G", title_fontsize=6.2)
    lg._legend_box.align = "left"
    fig.add_artist(lm)

    fig.savefig(OUTPDF, format="pdf")
    fig.savefig(OUTPNG, dpi=400, facecolor="white")
    print(f"\n  wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame(rows).to_csv(OUTCSV, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
