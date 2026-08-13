"""
estimate_noise_params.py -- estimate the Poisson-Gaussian noise model that
add_noise_to_gt.py used to build the simulated sweep, and persist it to a CSV.

evaluate_full.py's MLE localization arms read this CSV (--noise_params_csv) so the
per-pixel MLE weights match the simulation's own noise model, and can cross-check
it against the source video. The estimators are imported directly from
add_noise_to_gt.py, so the numbers are identical to what a simulation run recorded.

Noise model (add_noise_to_gt.py, 'gp_estimated'):
    signal_adu = max(0, gt - background)
    sim = background + Poisson(signal_adu * gain_g) / gain_g + N(0, (sigma_read*sqrt(scale))**2)
=>  Var_i(scale) = signal_adu / gain_g + read_noise_variance * scale
so the matched MLE uses K = 1/gain_g and read_var(scale) = read_noise_variance * scale.

NOTE: `read_noise_variance` here is the *intercept* of the variance-vs-mean regression (387.2 ADU^2,
sigma 19.68) -- the value the simulation was built with. It is NOT the read-noise variance used in
the training loss, which is a MAD estimate on background regions (316.53 ADU^2, sigma 17.79) stored
in each model's `noise_parameters.npy`. Both are estimated from the same Cy3 recording and both are
printed when the estimators run; do not interchange them. See
`results/gain_correction_disclosure.md`.

Example (paper Cy3_Best params):
    python estimate_noise_params.py \
        --original_video "Data/experimental_data/Cy3_Best/Cy3_Best.tif" \
        --noise_regions 0 190 50 250 200 190 250 250 0 10 40 50 220 10 250 50 \
        --out_csv "Data/simulated_data/noise_model_params.csv"
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.append(str(Path(__file__).resolve().parent.parent / "simulated_data"))
from add_noise_to_gt import analyze_noise_regions, analyze_intensity_variance_relationship


def main():
    ap = argparse.ArgumentParser(description="Estimate and save the simulation's Poisson-Gaussian noise model.")
    ap.add_argument("--original_video", required=True,
                    help="Original experimental video add_noise_to_gt.py estimated the noise model from.")
    ap.add_argument("--noise_regions", type=int, nargs='+', required=True,
                    help="Background noise-region coords 'x1 y1 x2 y2 ...' (same as the add_noise run).")
    ap.add_argument("--patch_size", type=int, default=32, help="PTC patch size (match the add_noise run).")
    ap.add_argument("--robust_regression", action='store_true', help="Theil-Sen gain estimation (match add_noise run).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path for the noise-model parameters.")
    a = ap.parse_args()

    if len(a.noise_regions) % 4 != 0:
        raise SystemExit(f"--noise_regions must be a multiple of 4 (x1 y1 x2 y2 ...); got {len(a.noise_regions)} values.")
    regions = [tuple(a.noise_regions[i:i + 4]) for i in range(0, len(a.noise_regions), 4)]

    print(f"Loading source video: {a.original_video}")
    stack = tifffile.imread(a.original_video).astype(np.float32)
    print(f"  shape={stack.shape}, dtype={stack.dtype}")

    background_level, _, _ = analyze_noise_regions(stack, regions, plot=False)
    gain_g, read_noise_variance = analyze_intensity_variance_relationship(
        stack, background_level, patch_size=a.patch_size,
        use_robust_regression=a.robust_regression, plot=False)

    row = {
        'gain_g': float(gain_g),
        'read_noise_variance': float(read_noise_variance),
        'sigma_read': float(np.sqrt(max(read_noise_variance, 0.0))),
        'background_level': float(background_level),
        'mle_K': float(1.0 / gain_g),  # convenience: K used by the MLE weights
        'source_video': os.path.abspath(a.original_video),
        'noise_regions': ' '.join(map(str, a.noise_regions)),
        'patch_size': a.patch_size,
        'robust_regression': bool(a.robust_regression),
    }

    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"\nWrote noise model params to: {out}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
