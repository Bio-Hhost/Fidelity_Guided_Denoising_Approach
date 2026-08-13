"""
verify_mle_snr.py - step-6 sanity check for the MLE localization arm.

Stratifies raw-LSQ vs raw-MLE localization error by GT amplitude (SNR proxy) for a
single noisy scale, and overlays the CRLB reference (mle_localization.crlb_xy).
Reuses evaluate_full's detect/fit split so detection is shared between the two arms.

Example:
    python verify_mle_snr.py \
        --noisy "Data/simulated_data/Noisy/sim_Gauss_Poisson_Est_scale_1.00.tif" \
        --gt_spots_csv "Data/simulated_data/GT/synthetic_ground_truth_airy_corr_randsiz_scaled_0.1_spot_info.csv" \
        --noise_params_csv "Data/simulated_data/noise_model_params.csv" \
        --scale 1.0 --threshold 24.8 --sample_frames 200 \
        --out_png "Data/simulated_data/_gate_full_mle/mle_vs_lsq_snr_scale1.png"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
import evaluate_full as ef
from mle_localization import make_mle_fitter, crlb_xy

LOG_PARAMS = {'min_sigma': 1.5, 'max_sigma': 3.0, 'num_sigma': 10, 'blob_overlap': 0.5}
FIT_PARAMS = {'match_tolerance': 2.0, 'fit_region_size': 7}


def per_spot_error(df):
    d = df.dropna(subset=['fit_x', 'fit_y', 'GT_X', 'GT_Y']).copy()
    d['err'] = np.hypot(d['fit_x'] - d['GT_X'], d['fit_y'] - d['GT_Y'])
    return d


def main():
    ap = argparse.ArgumentParser(description="SNR-stratified LSQ vs MLE localization sanity plot.")
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--gt_spots_csv", required=True)
    ap.add_argument("--noise_params_csv", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=24.8)
    ap.add_argument("--sample_frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--out_png", required=True)
    a = ap.parse_args()

    row = pd.read_csv(a.noise_params_csv).iloc[0]
    gain_g, read_var, bg = float(row['gain_g']), float(row['read_noise_variance']), float(row['background_level'])
    K = gain_g   # ADU/photon (alpha), matching mle_localization's Var = alpha*signal + read_var
    read_var_scaled = read_var * a.scale
    print(f"[params] K={K:.4f}, read_var(scale={a.scale})={read_var_scaled:.2f}, bg={bg:.2f}")

    stack_full = tifffile.imread(a.noisy)
    gt_full = pd.read_csv(a.gt_spots_csv)
    n = stack_full.shape[0]
    rng = np.random.RandomState(a.seed)
    idx = np.sort(rng.choice(n, size=min(a.sample_frames, n), replace=False))
    stack = stack_full[idx]
    gt = gt_full[gt_full['FRAME'].isin(idx)].copy()

    matches, _ = ef.detect_and_match(stack, gt, idx, a.threshold, "Noisy", **LOG_PARAMS, **FIT_PARAMS)
    lsq = per_spot_error(ef.fit_matches(stack, gt, matches, idx, ef.fit_rotated_gaussian_2d, FIT_PARAMS['fit_region_size']))
    mle = per_spot_error(ef.fit_matches(stack, gt, matches, idx, make_mle_fitter(K, read_var_scaled, bg), FIT_PARAMS['fit_region_size']))
    print(f"[fits] LSQ n={len(lsq)}  MLE n={len(mle)}")

    edges = np.quantile(lsq['GT_Amplitude'], np.linspace(0, 1, a.nbins + 1))
    edges = np.unique(edges)
    centers, lsq_med, mle_med, crlb_ref = [], [], [], []
    typ_sx = float(np.nanmedian(lsq.get('fit_sx', pd.Series([1.3])))) if 'fit_sx' in lsq else 1.3
    typ_sx = typ_sx if np.isfinite(typ_sx) and typ_sx > 0.2 else 1.3
    fr = FIT_PARAMS['fit_region_size']

    for lo, hi in zip(edges[:-1], edges[1:]):
        m_l = (lsq['GT_Amplitude'] >= lo) & (lsq['GT_Amplitude'] < hi)
        m_m = (mle['GT_Amplitude'] >= lo) & (mle['GT_Amplitude'] < hi)
        if m_l.sum() < 5:
            continue
        amp = float(np.median(lsq.loc[m_l, 'GT_Amplitude']))
        centers.append(amp)
        lsq_med.append(float(np.median(lsq.loc[m_l, 'err'])))
        mle_med.append(float(np.median(mle.loc[m_m, 'err'])) if m_m.sum() else np.nan)
        theta = (amp, fr / 2.0, fr / 2.0, typ_sx, typ_sx, 0.0, bg)
        cx, cy = crlb_xy(theta, (fr, fr), 0, 0, K, read_var_scaled, bg)
        crlb_ref.append(float(np.hypot(cx, cy) / np.sqrt(2)))  # per-axis -> radial-ish

    tbl = pd.DataFrame({'amp': centers, 'LSQ_medAE': lsq_med, 'MLE_medAE': mle_med, 'CRLB': crlb_ref})
    tbl['MLE<=LSQ'] = tbl['MLE_medAE'] <= tbl['LSQ_medAE'] + 1e-9
    print("\n=== median localization error by GT amplitude bin ===")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plt.figure(figsize=(8, 6))
    plt.plot(centers, lsq_med, 'o-', label='LSQ', color='#0072B2')
    plt.plot(centers, mle_med, 's-', label='MLE', color='#D55E00')
    plt.plot(centers, crlb_ref, 'k--', alpha=0.7, label='CRLB (matched model)')
    plt.xlabel('GT amplitude (SNR proxy)')
    plt.ylabel('Median localization error (px)')
    plt.title(f'LSQ vs MLE vs CRLB — noisy, scale {a.scale}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out_png, dpi=200)
    print(f"\nSaved: {a.out_png}")


if __name__ == "__main__":
    main()
