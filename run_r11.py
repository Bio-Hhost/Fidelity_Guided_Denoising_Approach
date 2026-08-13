"""Localization benchmark

Fits localization + photometry for the CRB vs MLE-on-raw vs fit-on-denoised comparison,
across the full simulated v2 noise-scale grid and the experimental test datasets.

Run:  python run_r11.py            (resumes; safe to Ctrl-C and rerun)
      python run_r11.py --sim-spots 4000 --exp-spots 6000
"""
import os, sys, time, argparse, numpy as np, pandas as pd, tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
from evaluate_full import fit_rotated_gaussian_2d, zoom_spot_loc    
from mle_fit import fit_mle_gaussian_2d, GAIN as CY3_GAIN, READ_VAR_BASE as CY3_RVB
import crb as CRB

OUT = "results/r11_evidence"
CELLS = os.path.join(OUT, "cells")
FRS = 7                                  
SCALES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SIM_GAIN, SIM_RVB, SIM_BG = 18.052034, 387.20514, 198.0    
GT_CSV = "Data/simulated_data/GT/synthetic_ground_truth_airy_corr_randsiz_scaled_0.1_spot_info.csv"
BASELINES = ["N2V", "PN2V", "PPN2V", "DeepCAD-RT"]
EXP_DS = ["19_Green", "20_Green", "29_Green", "30_Green"]


def sim_path(scale, method):
    S = f"{scale:.2f}"
    if method == "raw":
        return f"Data/simulated_data_v2/Gauss_Poisson_Est/sim_Gauss_Poisson_Est_scale_{S}.tif"
    if method == "RL":
        return f"results/rl_gain17689/sim_data/denoised_sim_{S}_training_run_rlg17main.tif"
    return f"results/crossmethod_comparison/sim_data/denoised_sim_{S}_{method}.tif"


def atomic_write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sim_spots(n):
    p = os.path.join(OUT, "spots_sim.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    g = pd.read_csv(GT_CSV).dropna(subset=["FRAME", "POSITION_X", "POSITION_Y",
                                           "GT_AMPLITUDE_DRAWN", "APPLIED_TOTAL_BLUR_SIGMA"])
    rng = np.random.default_rng(1101)
    idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
    idx.sort()
    s = g.iloc[idx][["GT_SPOT_ID", "FRAME", "POSITION_X", "POSITION_Y",
                     "GT_AMPLITUDE_DRAWN", "APPLIED_TOTAL_BLUR_SIGMA", "GT_BACKGROUND_LEVEL"]].reset_index(drop=True)
    s.columns = ["spot_id", "frame", "gt_x", "gt_y", "gt_amp", "gt_sigma", "gt_bg"]
    atomic_write_csv(s, p)
    return s


def exp_spots(ds, n):
    p = os.path.join(OUT, f"spots_exp_{ds}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    dcsv = f"results/rl_gain17689/exp_results/{ds}_denoised_{ds}_training_run_rlg17main_detailed_results.csv"
    d = pd.read_csv(dcsv)
    keep = ["POSITION_X", "POSITION_Y", "FRAME", "noisy_fit_x", "noisy_fit_y", "noisy_fit_amplitude",
            "denoised_fit_x", "denoised_fit_y", "denoised_fit_amplitude"]
    d = d[keep].dropna(subset=["POSITION_X", "POSITION_Y", "FRAME"]).reset_index(drop=True)
    if len(d) > n:
        rng = np.random.default_rng(1102)
        d = d.iloc[np.sort(rng.choice(len(d), size=n, replace=False))].reset_index(drop=True)
    atomic_write_csv(d, p)
    return d


# (computed once, sim's own calibration)
def build_crb(spots):
    p = os.path.join(OUT, "crb_sim.csv")
    if os.path.exists(p):
        return
    rows = []
    for scale in SCALES:
        for _, s in spots.iterrows():
            c = CRB.crb_from_gt_row(float(s["gt_amp"]), SIM_BG, float(s["gt_sigma"]),
                                    scale=scale, gain=SIM_GAIN, read_var_base=SIM_RVB)
            rows.append(dict(scale=scale, spot_id=s["spot_id"], gt_sigma=s["gt_sigma"], gt_amp=s["gt_amp"],
                             crb_x=c["crb_x"], crb_y=c["crb_y"],
                             crb_loc=float(np.hypot(c["crb_x"], c["crb_y"])),
                             crb_N=c["crb_N"], crb_amp=c["crb_amp"]))
    atomic_write_csv(pd.DataFrame(rows), p)


def run_sim_cell(scale, method, fitter, spots):
    path = sim_path(scale, method)
    stack = tifffile.imread(path).astype(np.float32)
    valid = (fitter == "LSE") or (method == "raw")   
    rows = []
    for _, s in spots.iterrows():
        fr = int(s["frame"])
        if fr >= len(stack):
            continue
        patch, (x1, y1) = zoom_spot_loc(stack[fr], (s["gt_x"], s["gt_y"]), FRS)
        if fitter == "MLE":
            ok, p = fit_mle_gaussian_2d(patch, x1, y1, scale=scale, gain=SIM_GAIN, read_var_base=SIM_RVB)
        else:
            ok, p = fit_rotated_gaussian_2d(patch, x1, y1)
        if not ok:
            rows.append(dict(spot_id=s["spot_id"], frame=fr, success=False))
            continue
        rows.append(dict(spot_id=s["spot_id"], frame=fr, success=True,
                         gt_x=s["gt_x"], gt_y=s["gt_y"], gt_amp=s["gt_amp"], gt_sigma=s["gt_sigma"],
                         fit_x=p["fit_x"], fit_y=p["fit_y"], fit_amplitude=p["fit_amplitude"],
                         fit_sx=p["fit_sx"], fit_sy=p["fit_sy"], fit_theta=p["fit_theta"], fit_offset=p["fit_offset"],
                         err_loc=float(np.hypot(p["fit_x"] - s["gt_x"], p["fit_y"] - s["gt_y"])),
                         err_amp=float(p["fit_amplitude"] - s["gt_amp"]),
                         err_amp_rel=float((p["fit_amplitude"] - s["gt_amp"]) / s["gt_amp"]),
                         noise_model_valid=valid))
    df = pd.DataFrame(rows)
    df["dataset"] = "sim"; df["scale"] = scale; df["method"] = method; df["fitter"] = fitter
    return df


def run_exp_cell(ds, spots):
    stack = tifffile.imread(f"Data/experimental_data/{ds}/{ds}.tif").astype(np.float32)
    rows = []
    for _, s in spots.iterrows():
        fr = int(s["FRAME"])
        if fr >= len(stack):
            continue
        patch, (x1, y1) = zoom_spot_loc(stack[fr], (s["POSITION_X"], s["POSITION_Y"]), FRS)
        ok, p = fit_mle_gaussian_2d(patch, x1, y1, scale=1.0, gain=CY3_GAIN, read_var_base=CY3_RVB)
        r = dict(dataset=ds, frame=fr, det_x=s["POSITION_X"], det_y=s["POSITION_Y"],
                 noisy_lse_x=s["noisy_fit_x"], noisy_lse_y=s["noisy_fit_y"], noisy_lse_amp=s["noisy_fit_amplitude"],
                 den_lse_x=s["denoised_fit_x"], den_lse_y=s["denoised_fit_y"], den_lse_amp=s["denoised_fit_amplitude"],
                 mle_raw_ok=ok)
        if ok:
            r.update(mle_raw_x=p["fit_x"], mle_raw_y=p["fit_y"], mle_raw_amp=p["fit_amplitude"])
            if pd.notna(s["denoised_fit_x"]):
                r["agree_mle_den"] = float(np.hypot(p["fit_x"] - s["denoised_fit_x"], p["fit_y"] - s["denoised_fit_y"]))
            if pd.notna(s["noisy_fit_x"]):
                r["agree_mle_noisylse"] = float(np.hypot(p["fit_x"] - s["noisy_fit_x"], p["fit_y"] - s["noisy_fit_y"]))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-spots", type=int, default=4000)
    ap.add_argument("--exp-spots", type=int, default=6000)
    args = ap.parse_args()
    os.makedirs(CELLS, exist_ok=True)

    spots = sim_spots(args.sim_spots)
    build_crb(spots)
    print(f"[setup] sim spots={len(spots)}  CRB table ready", flush=True)

    prio_scales = [1.0, 10.0, 5.0, 2.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0]
    cells = []
    for sc in prio_scales:
        cells.append(("sim", sc, "raw", "MLE"))
        cells.append(("sim", sc, "raw", "LSE"))
        cells.append(("sim", sc, "RL", "LSE"))
        cells.append(("sim", sc, "RL", "MLE"))     
        for b in BASELINES:
            cells.append(("sim", sc, b, "LSE"))
    for ds in EXP_DS:
        cells.append(("exp", ds, "RLmain", "MLE"))

    manifest_path = os.path.join(OUT, "manifest.csv")
    man = pd.read_csv(manifest_path).set_index("cell_id").to_dict("index") if os.path.exists(manifest_path) else {}

    total = len(cells); t0 = time.time()
    for i, (dset, key, method, fitter) in enumerate(cells):
        cell_id = f"{dset}_{key}_{method}_{fitter}".replace(".", "p").replace("-", "")
        cpath = os.path.join(CELLS, cell_id + ".csv")
        if os.path.exists(cpath) and man.get(cell_id, {}).get("status") == "done":
            print(f"[{i+1}/{total}] skip {cell_id} (done)", flush=True)
            continue
        ts = time.time()
        try:
            if dset == "sim":
                df = run_sim_cell(key, method, fitter, spots)
            else:
                df = run_exp_cell(key, exp_spots(key, args.exp_spots))
            atomic_write_csv(df, cpath)
            nok = int(df["success"].sum()) if "success" in df else int(df.get("mle_raw_ok", pd.Series(dtype=bool)).sum())
            man[cell_id] = dict(dataset=dset, key=key, method=method, fitter=fitter,
                                status="done", n=len(df), n_ok=nok, seconds=round(time.time() - ts, 1))
            print(f"[{i+1}/{total}] {cell_id}: n={len(df)} ok={nok} ({man[cell_id]['seconds']}s)", flush=True)
        except Exception as e:
            man[cell_id] = dict(dataset=dset, key=key, method=method, fitter=fitter,
                                status=f"ERROR:{type(e).__name__}", n=0, n_ok=0, seconds=round(time.time() - ts, 1))
            print(f"[{i+1}/{total}] {cell_id}: ERROR {e}", flush=True)
        pd.DataFrame([{"cell_id": k, **v} for k, v in man.items()]).to_csv(manifest_path, index=False)

    print(f"\n[done] {total} cells in {round(time.time()-t0,1)}s -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
