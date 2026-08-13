import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLDIR = os.path.join(ROOT, "trained_models", "rl_models_gain17689")
LOGS = os.path.join(ROOT, "trained_models", "static_models_new", "batch_logs")
TM = os.path.join(ROOT, "trained_models")
CM = os.path.join(ROOT, "comp_methods")
OUTPDF = os.path.join(ROOT, "figures", "figureS_curves.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_curves.png")
OUTTAB = os.path.join(ROOT, "figures", "figureS_curves_table.csv")

RL_RUNS = [("training_run_20260724-023428", fs.RL, 5),
           ("training_run_20260724-080854", fs.FROZEN_T5, 5),
           ("training_run_20260724-101233", fs.FROZEN_T1, 1)]
STATIC_RUNS = [("static_T1_geo0.001", fs.L0001_T1), ("static_T3_geo0.001", fs.L0001_T3),
               ("static_T1_geo0.1", fs.L01_T1), ("static_T3_geo0.1", fs.L01_T3)]

DISPLAY = {fs.FROZEN_T5: "λ = 0.5 (fixed), T = 5",
           fs.FROZEN_T1: "λ = 0.5 (fixed), T = 1"}


def disp(label):
    return DISPLAY.get(label, label)

C_TRAIN, C_VAL, C_SEL = "#8a877f", "#13518e", "#b03a11"


def parse_static_log(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"- loss: ([0-9.eE+-]+) - val_loss: ([0-9.eE+-]+)", txt)
    if not m:
        raise RuntimeError(f"no epoch summaries in {path} -- do not substitute a curve")
    tr = np.array([float(a) for a, _ in m])
    va = np.array([float(b) for _, b in m])
    return tr, va


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    models = []

    # RL family: validation REWARD is the selection criterion
    for run, label, seq in RL_RUNS:
        d = os.path.join(RLDIR, run)
        hist = pd.read_csv(os.path.join(d, "training_history.csv"))
        specs = json.load(open(os.path.join(d, "specs.json")))
        best = int(specs["best_reward_epoch"])                 # 1-indexed
        argmax1 = int(np.argmax(hist["reward"].to_numpy())) + 1
        if argmax1 != best:
            raise RuntimeError(f"{run}: specs.json best_reward_epoch={best} but argmax(reward)"
                               f"={argmax1} -- resolve before drawing, do not assume")
        held = hist["val_reward"].to_numpy() if "val_reward" in hist else None
        models.append(dict(label=disp(label), family="RL", kind="reward",
                           epochs=np.arange(1, len(hist) + 1),
                           train=hist["reward"].to_numpy(), val=held,
                           loss=hist["unet_loss"].to_numpy(),
                           selected=best, n_epochs=len(hist), rule="reward",
                           extra=f"T = {seq}",
                           held_out=None if held is None else int(np.argmax(held)) + 1))

    for stem, label in STATIC_RUNS:
        p = os.path.join(LOGS, stem + ".log")
        tr, va = parse_static_log(p)
        best = int(np.argmin(va)) + 1
        models.append(dict(label=label, family="static", kind="loss",
                           epochs=np.arange(1, len(va) + 1), val=va, train=tr,
                           selected=best, n_epochs=len(va),
                           rule="val loss", extra="", held_out=None))

    for tag, label, sub in (("pn2v", fs.PN2V, "PN2V_model"), ("ppn2v", fs.PPN2V, "PPN2V_model")):
        p = os.path.join(TM, sub, f"history{tag}.npy")

        h = np.load(p, allow_pickle=True)
        tr, va = np.asarray(h[1], float), np.asarray(h[2], float)
        spec = json.load(open(os.path.join(TM, sub, f"{tag}_run_config_and_specs.json")))["final_losses"]
        for got, want, what in ((tr[-1], spec[f"{tag}_train_last"], "train_last"),
                                (va[-1], spec[f"{tag}_val_last"], "val_last"),
                                (va.min(), spec[f"{tag}_val_best"], "val_best")):
            if not np.isclose(got, want, rtol=1e-6):
                raise RuntimeError(f"{sub}: {what} is {got} in history{tag}.npy but {want} in the "
                                   f"specs -- the row layout is not (epoch, train, val)")
        best = int(np.argmin(va)) + 1
        models.append(dict(label=label, family="comp", kind="loss",
                           epochs=np.arange(1, len(va) + 1), val=va, train=tr,
                           selected=best, n_epochs=len(va),
                           rule="val loss", extra="", held_out=None))


    p = os.path.join(CM, "n2v_training_curve.csv")
    d = pd.read_csv(p)
    best = int(d.val_loss.idxmin()) + 1
    models.append(dict(label=fs.N2V, family="comp", kind="loss", epochs=d.epoch.to_numpy(),
                       val=d.val_loss.to_numpy(), train=d.train_loss.to_numpy(),
                       selected=best, n_epochs=len(d), rule="val loss", extra="",
                       held_out=None))


    p = os.path.join(CM, "deepcad_training_curve.csv")
    d = pd.read_csv(p)
    ck = [f for f in os.listdir(os.path.join(TM, "DeepCAD_model")) if f.endswith(".pth")]
    if len(ck) != 1:
        raise RuntimeError(f"expected one DeepCAD checkpoint, found {ck}")
    sel = int(re.search(r"E_(\d+)", ck[0]).group(1))

    argmin_total = int(d.loc[d.total_loss.idxmin(), "epoch"])
    argmin_l1 = int(d.loc[d.l1_loss.idxmin(), "epoch"])
    later = d[d.epoch > sel]
    beat_total = int((later.total_loss < d.loc[d.epoch == sel, "total_loss"].iloc[0]).sum())
    beat_l1 = int((later.l1_loss < d.loc[d.epoch == sel, "l1_loss"].iloc[0]).sum())
    if not (argmin_total == sel and argmin_l1 == sel and beat_total == 0 and beat_l1 == 0):
        raise RuntimeError(
            f"the selected epoch {sel} should be the minimum of both the total loss and its L1 "
            f"component with no later epoch improving on either, but argmin(total)={argmin_total}, "
            f"argmin(L1)={argmin_l1}, later epochs beating it: {beat_total} / {beat_l1}.")
    models.append(dict(label=fs.DEEPCAD, family="comp", kind="train_only", epochs=d.epoch.to_numpy(),
                       val=None, train=d.total_loss.to_numpy(), selected=sel, n_epochs=len(d),
                       rule="training loss", extra="", held_out=None))

    fig = plt.figure(figsize=(7.09, 6.5))
    gs = fig.add_gridspec(3, 4, hspace=0.60, wspace=0.38,
                          left=0.072, right=0.987, top=0.925, bottom=0.062)
    letters = "ABCDEFGHIJKL"

    for i, m in enumerate(models):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        extra = m["extra"] if (m["extra"] and "T=" not in m["label"].replace(" ", "")) else ""

        ax.set_title(m["label"] + (f"  ({extra})" if extra else ""), fontsize=6.8, pad=12.5)
        ax.text(-0.30, 1.20, letters[i], transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", va="top", ha="left")
        ax.grid(True, lw=0.35, color="#dcdad4", zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ep = m["epochs"]

        if m["kind"] == "reward":
            ax.plot(ep, m["train"], color=C_TRAIN, lw=1.2, zorder=3)
            if m["val"] is not None:
                ax.plot(ep, m["val"], color=C_VAL, lw=1.2, zorder=4)
            ax.set_ylabel("reward", fontsize=6.2)
            ins = ax.inset_axes([0.46, 0.13, 0.50, 0.35])
            ins.plot(ep, m["loss"], color="#595650", lw=0.9)
            ins.tick_params(labelsize=4.6, length=1.5, pad=1.0)
            ins.set_title("U-Net training loss", fontsize=4.9, pad=1.2, color="#595650")
            for s in ("top", "right"):
                ins.spines[s].set_visible(False)
        else:
            ax.plot(ep, m["train"], color=C_TRAIN, lw=1.1, zorder=3)
            if m["val"] is not None:
                ax.plot(ep, m["val"], color=C_VAL, lw=1.2, zorder=4)
            ax.set_ylabel("loss", fontsize=6.2)
        ax.axvline(m["selected"], color=C_SEL, lw=0.9, ls=(0, (3, 2)), zorder=5)
        ax.set_xlabel("epoch", fontsize=6.2)
        ax.tick_params(labelsize=5.8)
        ax.text(0.5, 1.028, "ep %d of %d · %s" % (m["selected"], m["n_epochs"], m["rule"]),
                transform=ax.transAxes, fontsize=5.6, ha="center", va="bottom",
                color="#57544e")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_TRAIN, lw=1.2, label="training"),
               Line2D([], [], color=C_VAL, lw=1.2, label="validation"),
               Line2D([], [], color=C_SEL, lw=1.0, ls=(0, (3, 2)),
                      label="epoch carried into the paper")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.002), ncol=3,
               frameon=False, fontsize=7, handlelength=2.6, columnspacing=2.2)


    fig.savefig(OUTPDF); fig.savefig(OUTPNG, dpi=400); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")

    pd.DataFrame([dict(model=m["label"], family=m["family"], epochs_run=m["n_epochs"],
                       selected_epoch=m["selected"], selection_rule=m["rule"],
                       has_validation_curve=m["val"] is not None) for m in models]
                 ).to_csv(OUTTAB, index=False)
    for m in models:
        print(f"  {m['label']:<18s} {m['n_epochs']:3d} epochs, selected {m['selected']:3d}  "
              f"({m['rule']})")


if __name__ == "__main__":
    main()
