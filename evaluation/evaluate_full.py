"""
Runs a full evaluation at optimal thresholds.

This script implements the main quantitative analysis (Sec 2.4.1).
It requires the output from the threshold scan ('evaluate_detection_threshold_scan.py').

Workflow:
1. Scans the '--threshold_scan_dir' for all 'detection_metrics_all_variants.csv'
   files created by 'evaluate_detection_threshold_scan.py'.
2. Combines them into a single master DataFrame, adding 'scale' metadata.
3. Saves this combined CSV (e.g., 'combined_threshold_scan_results.csv')
   to the '--output_dir'.
2. For each method and noise scale, it identifies the *optimal threshold*
   (e.g., the one that maximized F1-Score).
3. It re-loads the GT video and the simulated/denoised video (sampled).
4. It re-runs the LoG detector *only at that optimal threshold*.
5. It then performs a full analysis:
   - Pixel-level metrics (PSNR, SSIM)
   - Detection metrics (F1, Precision, Recall)
   - 2D Gaussian fitting on all True Positives.
   - Localization error (RMSE, MedianAE).
   - Photometry error (R-squared, Gain, MedianAE).
6. Saves a final summary CSV and plots for all metrics vs. noise.
"""

import numpy as np
import pandas as pd
import tifffile
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import traceback
import glob
import re
from pathlib import Path
import argparse

try:
    from skimage.feature import blob_log
except ImportError:
    print("ERROR: scikit-image not found or feature module missing.")
    print("Please install it: pip install scikit-image")
    exit()

print(f"Comprehensive Evaluation Script Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# --- Plotting Style Configuration (should be consistent with other scripts) ---
ALL_COLORS = {
    'Noisy': '#D55E00', 'N2V': '#56B4E9', 'DeepCAD-RT': '#F0E442',
    'λ = RL': '#E69F00', 'λ = 0.001 (T=1)': '#0072B2', 'λ = 0.1 (T=1)': '#009E73',
    'λ = 0 (T=3)': '#CC79A7', 'λ = 0.1 (T=3)': '#2F4F4F'
}

def get_method_style(data_type_key):
    key_lower = data_type_key.lower()
    if 'n2v' in key_lower: return 'N2V', ALL_COLORS.get('N2V', '#888888')
    if 'deepcad-rt' in key_lower: return 'DeepCAD-RT', ALL_COLORS.get('DeepCAD-RT', '#888888')
    if 'training_run' in key_lower: return 'λ = RL', ALL_COLORS.get('λ = RL', '#888888')
    if key_lower == 'noisy': return 'Noisy', ALL_COLORS.get('Noisy', '#888888')
    seq_match = re.search(r'seq(\d+)', key_lower); geo_match = re.search(r'geo(\d+\.?\d*)', key_lower)
    t_val = seq_match.group(1) if seq_match else '1'; t_val_str = f"(T={t_val})"
    if geo_match: name = f"λ = {geo_match.group(1)} {t_val_str}".strip()
    elif seq_match: name = f"λ = 0 {t_val_str}".strip()
    else: name = data_type_key
    return name, ALL_COLORS.get(name, '#999999')


def zoom_spot_loc(video_frame, spot_position, region_size):
    x_int, y_int = int(round(spot_position[0])), int(round(spot_position[1]))
    half_size = region_size // 2
    y1, y2 = y_int - half_size, y_int + half_size + 1
    x1, x2 = x_int - half_size, x_int + half_size + 1
    y1_c, y2_c = max(0, y1), min(video_frame.shape[0], y2)
    x1_c, x2_c = max(0, x1), min(video_frame.shape[1], x2)
    return video_frame[y1_c:y2_c, x1_c:x2_c], (x1, y1)

def rotated_2d_gaussian(coords, amp, x0, y0, sx, sy, theta_deg, offset):
    (x, y) = coords; theta = np.deg2rad(theta_deg); X, Y = x - x0, y - y0
    x_prime = X * np.cos(theta) + Y * np.sin(theta)
    y_prime = -X * np.sin(theta) + Y * np.cos(theta)
    sx2, sy2 = sx**2, sy**2; exponent = 0.0
    if sx2 > 1e-6 and sy2 > 1e-6:
        exponent = (x_prime**2) / (2 * sx2) + (y_prime**2) / (2 * sy2)
    return offset + amp * np.exp(-exponent)

def fit_rotated_gaussian_2d(region, gx1, gy1):
    h, w = region.shape
    if h == 0 or w == 0: return False, None
    Y, X = np.mgrid[gy1:gy1+h, gx1:gx1+w]
    x_flat, y_flat, z_flat = X.ravel(), Y.ravel(), region.ravel().astype(float)
    
    min_r, max_r = np.min(region), np.max(region)
    amp_g = float(max_r - min_r) if max_r > min_r else 1.0
    off_g = float(min_r); x0_g = gx1 + w / 2.0; y0_g = gy1 + h / 2.0
    sig_g = max(1.0, min(w, h) / 4.0)
    init_g = (amp_g, x0_g, y0_g, sig_g, sig_g, 0.0, off_g)
    
    bounds = ([0, gx1-2, gy1-2, 0.1, 0.1, -180, -np.inf],
              [np.inf, gx1+w+2, gy1+h+2, w*2, h*2, 180, np.inf])
    
    def gaussian_func_flat(coords, *p):
        return rotated_2d_gaussian(coords, *p).ravel()
    
    try:
        popt, _ = curve_fit(gaussian_func_flat, (X, Y), z_flat, p0=init_g, bounds=bounds, maxfev=5000, method='trf')
        return True, {'fit_x': popt[1], 'fit_y': popt[2], 'fit_amplitude': popt[0], 'fit_sx': popt[3], 'fit_sy': popt[4], 'fit_theta': popt[5], 'fit_offset': popt[6]}
    except (RuntimeError, ValueError, TypeError):
        return False, None

def detect_spots_log(frame, min_sigma, max_sigma, num_sigma, threshold, overlap):
    try:
        frame_float = frame.astype(float)
        if np.ptp(frame_float) == 0: return []
        blobs = blob_log(frame_float, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap, log_scale=True)
        return [(int(round(x)), int(round(y)), s * np.sqrt(2)) for y, x, s in blobs]
    except Exception: return []

def match_detections_to_gt(detections_df, gt_df, tolerance):
    all_matches = []; detected_indices_matched = set()
    
    if 'frame' not in detections_df.columns: detections_df = pd.DataFrame(columns=['frame', 'x_int', 'y_int'])
    
    for frame_idx in gt_df['FRAME'].unique():
        gt_in_frame = gt_df[gt_df['FRAME'] == frame_idx]
        dets_in_frame = detections_df[detections_df['frame'] == frame_idx]
        if gt_in_frame.empty and dets_in_frame.empty: continue
        
        gt_coords = gt_in_frame[['POSITION_Y', 'POSITION_X']].values
        det_coords = dets_in_frame[['y_int', 'x_int']].values if not dets_in_frame.empty else np.empty((0,2))
        
        if not gt_in_frame.empty and not dets_in_frame.empty:
            distances = cdist(gt_coords, det_coords)
            gt_indices, det_indices = np.where(distances <= tolerance)
            match_candidates = sorted([(distances[gi, di], gt_in_frame.index[gi], dets_in_frame.index[di]) for gi, di in zip(gt_indices, det_indices)])
            assigned_gt, assigned_det = set(), set()
            for dist, gt_idx, det_idx in match_candidates:
                if gt_idx not in assigned_gt and det_idx not in assigned_det:
                    gt_s, det_s = gt_df.loc[gt_idx], detections_df.loc[det_idx]
                    all_matches.append({
                        'GT_SPOT_ID': gt_s['GT_SPOT_ID'], 'frame': frame_idx,
                        'GT_X': gt_s['POSITION_X'], 'GT_Y': gt_s['POSITION_Y'],
                        'GT_Amplitude': gt_s.get('GT_AMPLITUDE_DRAWN', np.nan),
                        'det_x_int': det_s['x_int'], 'det_y_int': det_s['y_int'],
                        'distance': dist, 'match_status': 'TP', 'detection_index': det_idx
                    })
                    assigned_gt.add(gt_idx); assigned_det.add(det_idx); detected_indices_matched.add(det_idx)
        
        matched_gt_ids = {m['GT_SPOT_ID'] for m in all_matches if m['frame'] == frame_idx}
        for gt_idx in gt_in_frame.index:
            if gt_df.loc[gt_idx, 'GT_SPOT_ID'] not in matched_gt_ids:
                gt_s = gt_df.loc[gt_idx]
                all_matches.append({
                    'GT_SPOT_ID': gt_s['GT_SPOT_ID'], 'frame': frame_idx,
                    'GT_X': gt_s['POSITION_X'], 'GT_Y': gt_s['POSITION_Y'],
                    'GT_Amplitude': gt_s.get('GT_AMPLITUDE_DRAWN', np.nan),
                    'match_status': 'FN', 'detection_index': -1
                })
                
    fp_df = pd.DataFrame()
    if not detections_df.empty:
        fp_indices = set(detections_df.index) - detected_indices_matched
        if fp_indices:
            fp_df = detections_df.loc[list(fp_indices)].copy()
            fp_df['match_status'] = 'FP'
            
    matches_df = pd.DataFrame(all_matches)
    req_cols = ['GT_SPOT_ID', 'frame', 'GT_X', 'GT_Y', 'GT_Amplitude', 'det_x_int', 'det_y_int', 'distance', 'match_status', 'detection_index']
    for col in req_cols:
        if col not in matches_df.columns:
            matches_df[col] = np.nan
            
    return matches_df, fp_df

def combine_results_to_master_df(base_dir, output_file):
    print("\n--- Starting CSV Combination ---")
    search_pattern = os.path.join(base_dir, '**', 'detection_metrics_all_variants.csv')
    all_csv_paths = glob.glob(search_pattern, recursive=True)
    
    if not all_csv_paths:
        print(f"Error: No 'detection_metrics_all_variants.csv' files found in '{base_dir}'.")
        return None

    print(f"Found {len(all_csv_paths)} CSV files to combine.")
    
    all_dfs = []
    for path in all_csv_paths:
        try:
            df = pd.read_csv(path)
            # Extract scale from the directory name, e.g., "Group_Scale_1.0"
            scale_str = os.path.basename(os.path.dirname(path)).replace('Group_Scale_', '')
            df['scale'] = float(scale_str)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not process file '{path}'. Reason: {e}")
            
    if not all_dfs:
        print("Error: Failed to read any CSV files successfully.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)    
    style_info = combined_df['data_type'].apply(get_method_style)
    combined_df['Method'] = style_info.apply(lambda x: x[0])
    combined_df['Color'] = style_info.apply(lambda x: x[1])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined all results into:\n{output_file}")
    return combined_df

def get_optimal_threshold(summary_df, scale_value, data_type_key, default_thresh, metric_col='F1'):
    try:
        subset = summary_df[(np.isclose(summary_df['scale'], scale_value)) & (summary_df['data_type'] == data_type_key)]
        if subset.empty:
            print(f"        WARNING: No threshold data for '{data_type_key}' at scale {scale_value:.2f}. Using default: {default_thresh}")
            return default_thresh
        
        best_row = subset.loc[subset[metric_col].idxmax()]
        threshold = best_row['threshold']
        metric_val = best_row[metric_col]
        print(f"        Found optimal threshold for '{data_type_key}' (Scale {scale_value:.2f}): {threshold:.4f} (Best {metric_col}={metric_val:.4f})")
        return threshold
    except Exception as e:
        print(f"        ERROR looking up threshold for '{data_type_key}' (Scale {scale_value:.2f}): {e}. Using default: {default_thresh}")
        return default_thresh

def evaluate_video(video_stack, gt_spots_df, frame_indices_map, threshold, name, **kwargs):
    print(f"    Evaluating: {name}...")
    
    detections = []
    for i, frame in enumerate(video_stack):
        original_frame_idx = frame_indices_map[i]
        for x, y, r in detect_spots_log(frame, kwargs['min_sigma'], kwargs['max_sigma'], kwargs['num_sigma'], threshold, kwargs['blob_overlap']):
            detections.append({'frame': original_frame_idx, 'x_int': x, 'y_int': y})
    detections_df = pd.DataFrame(detections)

    matches_df, fp_df = match_detections_to_gt(detections_df, gt_spots_df, kwargs['match_tolerance'])
    
    detailed_results = []
    border = kwargs['fit_region_size'] // 2 + 1
    gt_spots_for_fitting = gt_spots_df[(gt_spots_df['POSITION_X'] >= border) & (gt_spots_df['POSITION_X'] < video_stack.shape[2] - border) &
                                       (gt_spots_df['POSITION_Y'] >= border) & (gt_spots_df['POSITION_Y'] < video_stack.shape[1] - border)]

    for _, gt_spot in gt_spots_for_fitting.iterrows():
        gt_id, frame_idx = gt_spot['GT_SPOT_ID'], int(gt_spot['FRAME'])
        result_entry = {'GT_SPOT_ID': gt_id, 'frame': frame_idx, 'GT_X': gt_spot['POSITION_X'], 'GT_Y': gt_spot['POSITION_Y'], 'GT_Amplitude': gt_spot['GT_AMPLITUDE_DRAWN']}
        
        try:
            sampled_frame_idx = np.where(frame_indices_map == frame_idx)[0][0]
        except IndexError:
            continue 
        
        match = matches_df[(matches_df['GT_SPOT_ID'] == gt_id) & (matches_df['match_status'] == 'TP')]
        
        if not match.empty:
            det_x, det_y = match.iloc[0]['det_x_int'], match.iloc[0]['det_y_int']
            patch, (gx1, gy1) = zoom_spot_loc(video_stack[sampled_frame_idx], (det_x, det_y), kwargs['fit_region_size'])
            success, params = fit_rotated_gaussian_2d(patch, gx1, gy1)
            if success:
                result_entry.update(params)
        detailed_results.append(result_entry)
        
    detailed_results_df = pd.DataFrame(detailed_results)
    return matches_df, fp_df, detailed_results_df

def calculate_summary_metrics(matches_df, fp_df, detailed_results_df, gt_stack, video_stack):
    metrics = {}
    data_range = gt_stack.max() - gt_stack.min() if np.ptp(gt_stack) > 0 else 1.0

    try:
        metrics['PSNR'] = psnr(gt_stack, video_stack.astype(gt_stack.dtype), data_range=data_range)
    except Exception:
        metrics['PSNR'] = np.nan
        
    try:
        min_dim = min(gt_stack.shape[1], gt_stack.shape[2])
        win_size = min(7, min_dim)
        if win_size % 2 == 0: win_size -= 1
        
        if win_size >= 3:
             metrics['SSIM'] = ssim(gt_stack, video_stack.astype(gt_stack.dtype), data_range=data_range, channel_axis=0, win_size=win_size)
        else:
             metrics['SSIM'] = np.nan
    except Exception:
        metrics['SSIM'] = np.nan

    tp = len(matches_df[matches_df['match_status'] == 'TP'])
    fn = len(matches_df[matches_df['match_status'] == 'FN'])
    fp = len(fp_df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    metrics.update({'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1})
    
    valid_fits = detailed_results_df.dropna(subset=['fit_x', 'GT_X'])
    if not valid_fits.empty:
        loc_errors = np.sqrt((valid_fits['fit_x'] - valid_fits['GT_X'])**2 + (valid_fits['fit_y'] - valid_fits['GT_Y'])**2)
        metrics.update({'Loc_RMSE': np.sqrt(np.mean(loc_errors**2)), 'Loc_MedianAE': np.median(loc_errors)})
        
        phot_fits = valid_fits.dropna(subset=['fit_amplitude', 'GT_Amplitude'])
        if len(phot_fits) > 1:
            gt_amps, est_amps = phot_fits['GT_Amplitude'], phot_fits['fit_amplitude']
            phot_errors = np.abs(est_amps - gt_amps)
            metrics.update({
                'Phot_R_squared': np.corrcoef(gt_amps, est_amps)[0, 1]**2 if np.std(gt_amps) > 1e-6 and np.std(est_amps) > 1e-6 else np.nan,
                'Phot_Gain': np.polyfit(gt_amps, est_amps, 1)[0],
                'Phot_MedianAE': np.median(phot_errors)
            })
    all_metric_keys = ['PSNR', 'SSIM', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1',
                       'Loc_RMSE', 'Loc_MedianAE', 'Phot_R_squared', 'Phot_Gain', 'Phot_MedianAE']
    for key in all_metric_keys:
        if key not in metrics:
            metrics[key] = np.nan
            
    return metrics

def create_summary_plots(all_results_df, output_dir):
    print("\n--- Creating Final Summary Plots ---")
    metrics_to_plot = {
        'F1': 'F1 Score',
        'Loc_MedianAE': 'Localization Median Error (pixels)',
        'Phot_MedianAE': 'Photometry Median Error (amplitude)',
        'Phot_R_squared': 'Photometry R-squared',
        'PSNR': 'PSNR vs. Ground Truth'
    }
    for metric, y_label in metrics_to_plot.items():
        if metric not in all_results_df.columns or all_results_df[metric].isnull().all():
            print(f"  Skipping plot for '{metric}': No valid data available.")
            continue

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 9))
        palette = {row['Method']: row['Color'] for _, row in all_results_df.drop_duplicates('Method').iterrows()}
        
        is_error_metric = 'Error' in y_label or 'RMSE' in metric
        
        sns.lineplot(data=all_results_df, x='scale', y=metric, hue='Method', palette=palette, marker='o', ax=ax)
        ax.set_title(f'{y_label} vs. Noise Scale', fontsize=18, weight='bold')
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_xlabel('Simulated Noise Scale (Higher is more noise)', fontsize=14)
        
        if not is_error_metric:
            ax.set_ylim(-0.05, 1.05)
            
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig_path = output_dir / f"Summary_{metric}_vs_Scale.png"
        plt.savefig(fig_path, dpi=300); plt.close(fig)
        print(f"  Saved summary plot: {fig_path}")

def main(args):
    overall_start_time = time.time()
    
    gt_video_path = Path(args.gt_video)
    gt_spots_csv_path = Path(args.gt_spots_csv)
    input_data_dir = Path(args.input_dir)
    base_output_dir = Path(args.output_dir)
    optimal_threshold_summary_path = Path(args.threshold_summary_csv)
    
    METHODS_TO_EVALUATE = args.methods if args.methods else []
    NUM_FRAMES_TO_SAMPLE = args.sample_frames
    RANDOM_SEED = args.seed
    OPTIMIZATION_METRIC = args.opt_metric
    DEFAULT_THRESHOLD = args.default_thresh
    
    LOG_PARAMS = {
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma,
        'num_sigma': args.num_sigma,
        'blob_overlap': args.overlap
    }
    FIT_PARAMS = {
        'match_tolerance': args.tolerance,
        'fit_region_size': args.fit_region_size
    }

    base_output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv_path = base_output_dir / "combined_threshold_scan_results.csv"
    
    print(f"--- Step 1: Combining threshold scan results ---")
    print(f"Reading from: {threshold_scan_dir}")
    
    optimal_thresholds_df = combine_results_to_master_df(
        threshold_scan_dir, 
        combined_csv_path
    )
    
    if optimal_thresholds_df is None:
        print(f"FATAL ERROR: Failed to combine threshold scan CSVs from '{threshold_scan_dir}'. Cannot proceed.")
        exit()
        
    print(f"Successfully combined threshold results into: {combined_csv_path}")

   print(f"\n--- Step 2: Running Comprehensive Evaluation ---") 
   print(f"\nSearching for TIF files in: {input_data_dir}")
    all_tif_files = [p for p in input_data_dir.glob("*.tif") if "softmasked" not in p.name]
    scale_pattern = re.compile(r"scale_(\d+\.\d+)|sim_(\d+\.\d+)")
    
    file_groups = defaultdict(list)
    for f_path in all_tif_files:
        match = scale_pattern.search(f_path.name)
        if match:
            scale_str = match.group(1) or match.group(2)
            file_groups[scale_str].append(f_path)

    print(f"Found {len(file_groups)} noise scale groups to process.")
    
    all_evaluation_results = []
    
    for scale_str, file_paths in file_groups.items():
        scale_val = float(scale_str)
        print(f"\n--- Processing Group: Scale {scale_val:.2f} ---")
        
        noisy_path = next((p for p in file_paths if f"sim_Gauss_Poisson_Est_scale_{scale_str}" in p.name), None)
        if not noisy_path:
            print(f"  WARNING: Noisy base file not found for scale {scale_str}. Skipping group.")
            continue
            
        denoised_paths = [p for p in file_paths if p != noisy_path]
        
        try:
            gt_stack_full = tifffile.imread(gt_video_path).astype(np.float32)
            gt_spots_df_full = pd.read_csv(gt_spots_csv_path)
            noisy_stack_full = tifffile.imread(noisy_path)
            num_frames_total = gt_stack_full.shape[0]
            
            np.random.seed(RANDOM_SEED)
            if num_frames_total > NUM_FRAMES_TO_SAMPLE:
                frame_indices = np.random.choice(num_frames_total, size=NUM_FRAMES_TO_SAMPLE, replace=False)
            else:
                frame_indices = np.arange(num_frames_total)
            frame_indices.sort()
            
            gt_stack = gt_stack_full[frame_indices]
            gt_spots_df = gt_spots_df_full[gt_spots_df_full['FRAME'].isin(frame_indices)].copy()
            noisy_stack = noisy_stack_full[frame_indices]
            
            print(f"  Loaded and sampled {len(frame_indices)} frames for this group.")
        except Exception as e:
            print(f"  FATAL ERROR loading base data for scale {scale_str}: {e}"); continue
            
        thresh_noisy = get_optimal_threshold(optimal_thresholds_df, scale_val, 'noisy', DEFAULT_THRESHOLD, OPTIMIZATION_METRIC)
        matches_n, fp_n, details_n = evaluate_video(noisy_stack, gt_spots_df, frame_indices, thresh_noisy, "Noisy", **LOG_PARAMS, **FIT_PARAMS)
        metrics_n = calculate_summary_metrics(matches_n, fp_n, details_n, gt_stack, noisy_stack)
        metrics_n.update({'scale': scale_val, 'Method': 'Noisy', 'Color': ALL_COLORS['Noisy'], 'data_type': 'noisy'})
        all_evaluation_results.append(metrics_n)

        for d_path in denoised_paths:
            data_type_key = d_path.name
            method_name, color = get_method_style(data_type_key)
            
            if METHODS_TO_EVALUATE and method_name not in METHODS_TO_EVALUATE:
                continue

            try:
                denoised_stack = tifffile.imread(d_path)[frame_indices]
                thresh_d = get_optimal_threshold(optimal_thresholds_df, scale_val, data_type_key, DEFAULT_THRESHOLD, OPTIMIZATION_METRIC)
                matches_d, fp_d, details_d = evaluate_video(denoised_stack, gt_spots_df, frame_indices, thresh_d, method_name, **LOG_PARAMS, **FIT_PARAMS)
                metrics_d = calculate_summary_metrics(matches_d, fp_d, details_d, gt_stack, denoised_stack)
                metrics_d.update({'scale': scale_val, 'Method': method_name, 'Color': color, 'data_type': data_type_key})
                all_evaluation_results.append(metrics_d)
            except Exception as e:
                print(f"  ERROR processing variant {d_path.name}: {e}")

    if not all_evaluation_results:
        print("\nNo evaluation results were generated. Exiting.")
    else:
        final_summary_df = pd.DataFrame(all_evaluation_results)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        summary_csv_path = base_output_dir / "evaluation_summary_all_methods.csv"
        final_summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"\nSaved final summary of all evaluations to:\n{summary_csv_path}")
        
        create_summary_plots(final_summary_df, base_output_dir)

    print(f"\n--- Overall Evaluation Script Finished in {time.time() - overall_start_time:.2f} seconds ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation at optimal thresholds.")
    
    # --- I/O Arguments ---
    parser.add_argument("--gt_video", type=str, required=True,
                        help="Path to the *pristine* ground truth TIF video.")
    parser.add_argument("--gt_spots_csv", type=str, required=True,
                        help="Path to the ground truth CSV file (e.g., '..._spot_info.csv').")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing all noisy and denoised .tif files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the base directory where comprehensive results (CSVs, plots) will be saved.")
   parser.add_argument("--threshold_scan_dir", type=str, required=True,
                           help="Path to the base directory containing the 'Group_Scale_...' folders from the 'evaluate_detection_threshold_scan.py' scan.")

    # --- Filtering ---
    parser.add_argument("--methods", type=str, nargs='+', default=None,
                        help="Optional: List of method display names to evaluate (e.g., 'N2V' 'DeepCAD-RT'). If empty, evaluates all.")

    # --- Evaluation Parameters ---
    parser.add_argument("--sample_frames", type=int, default=500,
                        help="Number of frames to randomly sample from each video.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible frame sampling.")
    parser.add_argument("--fit_region_size", type=int, default=7,
                        help="Pixel size of the square region for 2D Gaussian fitting (must be odd).")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Matching tolerance in pixels to count a detection as a True Positive.")
    parser.add_argument("--opt_metric", type=str, default='F1', choices=['F1', 'Precision', 'Recall'],
                        help="Metric used to select the optimal threshold from the summary CSV.")
    parser.add_argument("--default_thresh", type=float, default=10.0,
                        help="Fallback threshold if a method is not found in the summary CSV.")

    # --- LoG Detector Parameters (used for the optimal threshold run) ---
    parser.add_argument("--min_sigma", type=float, default=1.5, help="LoG detector: minimum sigma.")
    parser.add_argument("--max_sigma", type=float, default=3.0, help="LoG detector: maximum sigma.")
    parser.add_argument("--num_sigma", type=int, default=10, help="LoG detector: number of sigma steps.")
    parser.add_argument("--overlap", type=float, default=0.5, help="LoG detector: blob overlap threshold.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fit_region_size % 2 == 0:
        print(f"Warning: --fit_region_size ({args.fit_region_size}) must be odd. Using {args.fit_region_size + 1} instead.")
        args.fit_region_size += 1
        
    main(args)
