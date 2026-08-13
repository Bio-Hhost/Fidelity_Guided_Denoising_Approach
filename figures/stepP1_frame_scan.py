import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from skimage.feature import blob_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import figure_style as fs

SCALE = float(os.environ.get("FIG3_SCALE", "7.0"))
MIN_SIGMA, MAX_SIGMA, NUM_SIGMA, OVERLAP = 1.5, 3.0, 10, 0.5
TOLERANCE = 2.0
GT_CSV = os.path.join(ROOT, "Data", "simulated_data", "GT",
                      "synthetic_ground_truth_airy_corr_randsiz_scaled_0.1_spot_info.csv")
OUT_JSON = os.path.join(ROOT, "figures", "stepP1_frame_scan.json")
OUT_CSV = os.path.join(ROOT, "figures", "stepP1_frame_scan.csv")

S = f"{SCALE:.2f}"
STACKS = {
    fs.NOISY: f"Data/simulated_data_v2/Gauss_Poisson_Est/sim_Gauss_Poisson_Est_scale_{S}.tif",
    fs.N2V: f"Data/simulated_data_v2/Denoised/sim_Gauss_Poisson_Est_scale_{S}_denoised_N2V.tif",
    fs.PN2V: f"Data/simulated_data_v2/Denoised/sim_Gauss_Poisson_Est_scale_{S}_denoised_PN2V.tif",
    fs.PPN2V: f"Data/simulated_data_v2/Denoised/sim_Gauss_Poisson_Est_scale_{S}_denoised_PPN2V.tif",
    fs.DEEPCAD: f"Data/simulated_data_v2/Denoised/sim_Gauss_Poisson_Est_scale_{S}_denoised_DeepCAD-RT.tif",
    fs.L0001_T1: f"results/ablation_static_v2/sim_data/denoised_sim_{S}_seq1_mask5_geo0.001.tif",
    fs.L0001_T3: f"results/ablation_static_v2/sim_data/denoised_sim_{S}_seq3_mask5_geo0.001.tif",
    fs.L01_T1: f"results/ablation_static_v2/sim_data/denoised_sim_{S}_seq1_mask5_geo0.1.tif",
    fs.L01_T3: f"results/ablation_static_v2/sim_data/denoised_sim_{S}_seq3_mask5_geo0.1.tif",
    fs.RL: f"results/rl_gain17689/sim_data/denoised_sim_{S}_training_run_rlg17main.tif",
}
SCAN_STUDY = {fs.NOISY: "rl_gain17689", fs.N2V: "crossmethod_comparison",
              fs.PN2V: "crossmethod_comparison", fs.PPN2V: "crossmethod_comparison",
              fs.DEEPCAD: "crossmethod_comparison", fs.L0001_T1: "ablation_static_v2",
              fs.L0001_T3: "ablation_static_v2", fs.L01_T1: "ablation_static_v2",
              fs.L01_T3: "ablation_static_v2", fs.RL: "rl_gain17689"}


def optimal_thresholds():
    out = {}
    for m, study in SCAN_STUDY.items():
        p = os.path.join(ROOT, "results", study, "sim_scan", f"Group_Scale_{S}",
                         "detection_metrics_all_variants.csv")
        d = pd.read_csv(p)
        best = None
        for dt, g in d.groupby("data_type"):
            try:
                if fs.resolve_label(str(dt)) != m:
                    continue
            except fs.UnknownMethodError:
                continue
            r = g.loc[g.F1.idxmax()]
            best = dict(threshold=float(r.threshold), pooled_F1=float(r.F1),
                        source=os.path.relpath(p, ROOT).replace("\\", "/"))
        if best is None:
            raise RuntimeError(f"no scan row for {m} at scale {S} in {p}")
        out[m] = best
    return out


def classify(det_xy, gt_xy, tol):
    if len(det_xy) == 0:
        return 0, 0, len(gt_xy), np.zeros(0, bool), np.zeros(len(gt_xy), bool)
    if len(gt_xy) == 0:
        return 0, len(det_xy), 0, np.zeros(len(det_xy), bool), np.zeros(0, bool)
    d = np.hypot(gt_xy[:, 0][:, None] - det_xy[None, :, 0],
                 gt_xy[:, 1][:, None] - det_xy[None, :, 1])
    cost = np.where(d <= tol, d, 1e6)
    gi, di = linear_sum_assignment(cost)
    ok = d[gi, di] <= tol
    gi, di = gi[ok], di[ok]
    det_tp = np.zeros(len(det_xy), bool); det_tp[di] = True
    gt_hit = np.zeros(len(gt_xy), bool); gt_hit[gi] = True
    return int(ok.sum()), int((~det_tp).sum()), int((~gt_hit).sum()), det_tp, gt_hit


def detect(frame, threshold):
    f = frame.astype(float)
    b = blob_log(f, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, num_sigma=NUM_SIGMA,
                 threshold=threshold, overlap=OVERLAP, log_scale=True)
    return np.zeros((0, 2)) if b.size == 0 else b[:, [1, 0]]      # -> (x, y)


def main():
    for m, rel in STACKS.items():
        if not os.path.exists(os.path.join(ROOT, rel)):
            raise RuntimeError(f"missing stack for {m}: {rel}")
    thr = optimal_thresholds()
    print("optimal thresholds at scale 9 (F1-max on the six-point grid):")
    for m, v in thr.items():
        print(f"  {m:18s} {v['threshold']:5.1f}   pooled F1 {v['pooled_F1']:.4f}")

    gt = pd.read_csv(GT_CSV)
    counts = gt.groupby("FRAME").size()
    cut = float(np.percentile(counts.values, 95))
    candidates = sorted(int(f) for f in counts[counts >= cut].index)
    print(f"\ncandidate frames: {len(candidates)} with >= {cut:.0f} GT spots "
          f"(95th percentile of {len(counts)} frames)")

    agg = pd.read_csv(os.path.join(ROOT, "results", "COMPARISON_simulated_unified.csv"))
    agg = agg[(agg.scale == SCALE) & (agg.method.isin(STACKS))].set_index("method")["F1"]
    methods = list(STACKS)
    agg_vec = np.array([agg[m] for m in methods])
    print("aggregate F1 ranking at scale 9: "
          + " > ".join(np.array(methods)[np.argsort(-agg_vec)]))

    rows = []
    for mi, m in enumerate(methods):
        path = os.path.join(ROOT, STACKS[m])
        t = thr[m]["threshold"]
        with tifffile.TiffFile(path) as tf:
            for f in candidates:
                frame = tf.pages[f].asarray()
                g = gt[gt.FRAME == f]
                gt_xy = g[["POSITION_X", "POSITION_Y"]].values
                det = detect(frame, t)
                tp, fp, fn, _, _ = classify(det, gt_xy, TOLERANCE)
                f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
                rows.append(dict(frame=f, method=m, threshold=t, TP=tp, FP=fp, FN=fn, F1=f1,
                                 n_gt=len(gt_xy)))
        print(f"  [{mi + 1}/{len(methods)}] {m:18s} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    scored = []
    for f in candidates:
        sub = df[df.frame == f].set_index("method")["F1"]
        vec = np.array([sub[m] for m in methods])
        tau, p = kendalltau(agg_vec, vec)
        scored.append(dict(frame=int(f), kendall_tau=float(tau), p_value=float(p),
                           n_gt=int(df[df.frame == f].n_gt.iloc[0]),
                           order=list(np.array(methods)[np.argsort(-vec)]),
                           f1={m: float(sub[m]) for m in methods}))
    # first, exact ordering agreement with the aggregate.
    # second, among frames that tie on it: the frame whose per-method F1 VALUES sit closest to
    # the aggregate, so the panel is a typical frame.
    for r in scored:
        v = np.array([r["f1"][m] for m in methods])
        r["mean_abs_dev_from_aggregate"] = float(np.mean(np.abs(v - agg_vec)))
    scored.sort(key=lambda r: (-r["kendall_tau"], r["mean_abs_dev_from_aggregate"], r["frame"]))
    best = scored[0]
    print(f"\nSELECTED frame {best['frame']}  Kendall tau {best['kendall_tau']:.4f} "
          f"(p={best['p_value']:.2g}), {best['n_gt']} GT spots")
    print("  frame ordering : " + " > ".join(best["order"]))
    print("  runners-up     : " + ", ".join(f"frame {r['frame']} tau {r['kendall_tau']:.3f}"
                                            for r in scored[1:5]))

    json.dump(dict(
        scale=SCALE, rule="candidates = frames with GT spot count >= the 95th percentile; score = "
                          "Kendall tau-b of the frame's per-method F1 ranking against the "
                          "aggregate F1 ranking at the same scale; highest tau wins, ties to the "
                          "lowest frame index",
        rule_is_method_neutral=True,
        contrast_with_published="paper_figure_plotting_reference.copy.py:2575 required every "
                                "denoised method to beat Noisy AND the top two to be exactly "
                                "{lambda = RL, lambda = 0.1 (T=1)}",
        detection=dict(min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, num_sigma=NUM_SIGMA,
                       overlap=OVERLAP, log_scale=True, matching_tolerance_px=TOLERANCE,
                       source="evaluation/evaluate_detection_threshold_scan.py:73-78"),
        thresholds=thr, candidate_count=len(candidates), gt_spot_cut=cut,
        aggregate_order=list(np.array(methods)[np.argsort(-agg_vec)]),
        aggregate_f1={m: float(agg[m]) for m in methods},
        selected=best, ranked_candidates=scored,
        stacks={m: STACKS[m] for m in methods}),
        open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)} and {os.path.relpath(OUT_CSV, ROOT)}")


if __name__ == "__main__":
    main()
