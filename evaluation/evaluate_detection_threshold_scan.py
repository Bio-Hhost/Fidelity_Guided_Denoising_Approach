"""
Runs a multi-variant detection threshold scan on simulated data.

This script implements the quantitative evaluation described in
Section 2.4.1 of the paper. It takes a ground truth (GT) spot list
and a directory of corresponding videos (one noisy, many denoised)
for each noise level ('scale').

For each video, it:
1. Samples a subset of frames for efficiency.
2. Runs a Laplacian of Gaussian (LoG) blob detector
   across a range of detection thresholds.
3. Matches detections to the GT spots (TP, FP, FN).
4. Calculates Precision, Recall, and F1-Score.

At the end, it saves a CSV of all metrics and generates
Precision-Recall (PR) curves and Metrics-vs-Threshold plots for
each noise level group.
"""

import numpy as np
import pandas as pd
import tifffile
import os
import time
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict
import traceback
import glob
import re
import argparse # <<< Added

try:
    from skimage.feature import blob_log
except ImportError:
    print("ERROR: scikit-image not found or feature module missing.")
    print("Please install it: pip install scikit-image")
    exit()
try:
    from sklearn.metrics import precision_recall_curve, auc
except ImportError:
    print("WARNING: scikit-learn not found. Precision-Recall curve plotting will be disabled.")
    print("Please install it: pip install scikit-learn")
    precision_recall_curve = None
    auc = None


print(f"Detection Threshold Scan Script Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# This dictionary maps the exact display name to a unique color for plotting.
# NOTE: If you use different denoising methods, you MUST update this dictionary and the `get_method_style` function below.
ALL_COLORS = {
    'Noisy': '#D55E00',             # Orange
    'N2V': '#56B4E9',               # Sky Blue
    'DeepCAD-RT': '#F0E442',         # Yellow
    'λ = RL': '#E69F00',             # Amber
    'λ = 0.001 (T=1)': '#0072B2',    # Blue
    'λ = 0.1 (T=1)': '#009E73',      # Green
    'λ = 0 (T=3)': '#CC79A7',        # Reddish Purple
    'λ = 0.1 (T=3)': '#2F4F4F'       # Dark Slate Gray
}

def get_method_style(data_type_key):
    key_lower = data_type_key.lower()

    if 'n2v' in key_lower: return 'N2V', ALL_COLORS.get('N2V', '#888888')
    if 'deepcad-rt' in key_lower: return 'DeepCAD-RT', ALL_COLORS.get('DeepCAD-RT', '#888888')
    if 'training_run' in key_lower: return 'λ = RL', ALL_COLORS.get('λ = RL', '#888888')
    if key_lower == 'noisy': return 'Noisy', ALL_COLORS.get('Noisy', '#888888')

    seq_match = re.search(r'seq(\d+)', key_lower)
    geo_match = re.search(r'geo(\d+\.?\d*)', key_lower) 
    
    t_val = seq_match.group(1) if seq_match else '1'
    t_val_str = f"(T={t_val})"
    
    if geo_match:
        lambda_val_str = geo_match.group(1)
        name = f"λ = {lambda_val_str} {t_val_str}".strip()
    elif seq_match: 
        name = f"λ = 0 {t_val_str}".strip()
    else: 
        name = data_type_key

    color = ALL_COLORS.get(name, '#999999') 
            
    return name, color

def detect_spots_log(frame, min_sigma, max_sigma, num_sigma, threshold, overlap=0.5):
    try:
        frame_float = frame.astype(float)
        if np.ptp(frame_float) == 0: return []
        blobs = blob_log(frame_float, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold, overlap=overlap, log_scale=True)
        # Return (x, y, radius)
        return [(int(round(x)), int(round(y)), s * np.sqrt(2)) for y, x, s in blobs]
    except Exception as e:
        print(f"Warning: blob_log failed: {e}")
        return []

def run_log_detection_on_stack(stack, min_sigma, max_sigma, num_sigma, threshold, blob_overlap, frame_indices_map=None):
    all_detections = []
    for f_idx, frame in enumerate(stack):
        detections_in_frame = detect_spots_log(
            frame, min_sigma, max_sigma, num_sigma, threshold, blob_overlap
        )
        original_frame_idx = frame_indices_map[f_idx] if frame_indices_map is not None else f_idx
        for x, y, r in detections_in_frame:
            all_detections.append({'frame': original_frame_idx, 'x_int': x, 'y_int': y, 'radius_est': r})
    
    detections_df = pd.DataFrame(all_detections)
    required_cols = ['frame', 'x_int', 'y_int', 'radius_est']
    if detections_df.empty:
        detections_df = pd.DataFrame(columns=required_cols)
    else:
        for col in required_cols:
            if col not in detections_df.columns:
                detections_df[col] = np.nan
    return detections_df

def match_detections_to_gt(detections_df, gt_df, tolerance):
    all_matches = []
    detected_indices_matched = set()
    
    if 'frame' not in detections_df.columns:
        detections_df['frame'] = []
    if 'y_int' not in detections_df.columns:
        detections_df['y_int'] = []
    if 'x_int' not in detections_df.columns:
        detections_df['x_int'] = []

    for frame_idx in gt_df['FRAME'].unique():
        gt_in_frame = gt_df[gt_df['FRAME'] == frame_idx]
        dets_in_frame = detections_df[detections_df['frame'] == frame_idx]
        
        if gt_in_frame.empty and dets_in_frame.empty: continue
        
        gt_coords = gt_in_frame[['POSITION_Y', 'POSITION_X']].values
        det_coords = dets_in_frame[['y_int', 'x_int']].values
        
        frame_matches = []
        
        if not gt_in_frame.empty and not dets_in_frame.empty:
            distances = cdist(gt_coords, det_coords)
            gt_indices, det_indices = np.where(distances <= tolerance)
            
            match_candidates = sorted([(distances[gi, di], gt_in_frame.index[gi], dets_in_frame.index[di])
                                      for gi, di in zip(gt_indices, det_indices)])
            
            assigned_gt, assigned_det = set(), set()
            for dist, gt_orig_idx, det_orig_idx in match_candidates:
                if gt_orig_idx not in assigned_gt and det_orig_idx not in assigned_det:
                    gt_s, det_s = gt_df.loc[gt_orig_idx], detections_df.loc[det_orig_idx]
                    frame_matches.append({
                        'GT_SPOT_ID': gt_s['GT_SPOT_ID'], 'frame': frame_idx,
                        'GT_X': gt_s['POSITION_X'], 'GT_Y': gt_s['POSITION_Y'],
                        'GT_Amplitude': gt_s.get('GT_AMPLITUDE_DRAWN', np.nan),
                        'det_x_int': det_s['x_int'], 'det_y_int': det_s['y_int'],
                        'distance': dist, 'match_status': 'TP',
                        'detection_index': det_orig_idx
                    })
                    assigned_gt.add(gt_orig_idx)
                    assigned_det.add(det_orig_idx)
                    detected_indices_matched.add(det_orig_idx)
        
        matched_gt_ids_in_frame = {m['GT_SPOT_ID'] for m in frame_matches}
        for gt_orig_idx in gt_in_frame.index:
            gt_s = gt_df.loc[gt_orig_idx]
            if gt_s['GT_SPOT_ID'] not in matched_gt_ids_in_frame:
                frame_matches.append({
                    'GT_SPOT_ID': gt_s['GT_SPOT_ID'], 'frame': frame_idx,
                    'GT_X': gt_s['POSITION_X'], 'GT_Y': gt_s['POSITION_Y'],
                    'GT_Amplitude': gt_s.get('GT_AMPLITUDE_DRAWN', np.nan),
                    'det_x_int': np.nan, 'det_y_int': np.nan,
                    'distance': np.inf, 'match_status': 'FN',
                    'detection_index': -1
                })
        
        all_matches.extend(frame_matches)

    fp_df = pd.DataFrame()
    if not detections_df.empty:
        fp_indices = set(detections_df.index) - detected_indices_matched
        if fp_indices:
            fp_df_temp = detections_df.loc[list(fp_indices)].copy()
            fp_df_temp['match_status'] = 'FP'
            cols_k = ['frame', 'x_int', 'y_int', 'match_status']
            fp_df = fp_df_temp[[c for c in cols_k if c in fp_df_temp.columns]].rename(
                columns={'x_int': 'det_x_int', 'y_int': 'det_y_int'}, errors='ignore')

    matches_df = pd.DataFrame(all_matches)
    req_cols = ['GT_SPOT_ID', 'frame', 'GT_X', 'GT_Y', 'GT_Amplitude', 'det_x_int', 'det_y_int', 'distance', 'match_status', 'detection_index']
    for col in req_cols:
        if col not in matches_df.columns:
            matches_df[col] = np.nan if col != 'match_status' else 'Unknown'
            
    return matches_df, fp_df

def calculate_detection_metrics(matches_df, fp_df):
    """Calculates TP, FP, FN, Precision, Recall, and F1."""
    tp = len(matches_df[matches_df['match_status'] == 'TP'])
    fn = len(matches_df[matches_df['match_status'] == 'FN'])
    fp = len(fp_df)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}

def plot_metrics_vs_threshold(results_df, filename, title_suffix):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"Detection Metrics vs. LoG Threshold\n{title_suffix}", fontsize=16)
    
    data_types = results_df['data_type'].unique()
    if len(data_types) == 0:
        print("      Warning: No data found to plot metrics vs threshold.")
        return

    for data_type in sorted(data_types):
        df_subset = results_df[results_df['data_type'] == data_type].sort_values('threshold')
        if df_subset.empty: continue
        
        display_name, color = get_method_style(data_type)
        
        axes[0].plot(df_subset['threshold'], df_subset['Precision'], 'o-', label=display_name, color=color, alpha=0.8)
        axes[1].plot(df_subset['threshold'], df_subset['Recall'], 'o-', label=display_name, color=color, alpha=0.8)
        axes[2].plot(df_subset['threshold'], df_subset['F1'], 'o-', label=display_name, color=color, alpha=0.8)

    axes[0].set_ylabel("Precision"); axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.4)
    axes[1].set_ylabel("Recall (Sensitivity)"); axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.4)
    axes[2].set_ylabel("F1 Score"); axes[2].set_xlabel("LoG Threshold"); axes[2].set_ylim(0, 1.05); axes[2].grid(True, alpha=0.4)
    
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 0.95), title="Method")
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    
    try:
        plt.savefig(filename, dpi=300); print(f"      Saved metrics vs threshold plot: {filename}")
    except Exception as e:
        print(f"      Error saving metrics vs threshold plot {filename}: {e}")
    plt.close(fig)

def plot_precision_recall_curve(results_df, filename, title_suffix):
    if auc is None:
        print("      Skipping PR curve plot: scikit-learn's auc function not found.")
        return
    
    plt.figure(figsize=(10, 10))
    
    data_types = results_df['data_type'].unique()
    if len(data_types) == 0:
        print("      Warning: No data found to plot PR curve.")
        return

    for data_type in sorted(data_types):
        df_subset = results_df[results_df['data_type'] == data_type]
        if df_subset.empty: continue
        
        pr_auc_value = df_subset['AUC'].iloc[0] if 'AUC' in df_subset.columns and not df_subset['AUC'].empty else np.nan
        display_name, color = get_method_style(data_type)
        
        df_sorted = df_subset.sort_values('Recall')
        recall_values = np.clip(df_sorted['Recall'].values, 0, 1)
        precision_values = np.clip(df_sorted['Precision'].values, 0, 1)
        
        label_text = f'{display_name} (AUC = {pr_auc_value:.3f})' if not np.isnan(pr_auc_value) else f'{display_name} (AUC: N/A)'
        plt.plot(recall_values, precision_values, 'o-', label=label_text, color=color, alpha=0.8)

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve\n{title_suffix}", fontsize=16)
    plt.legend(title="Method")
    plt.grid(True, alpha=0.4)
    plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300); print(f"      Saved Precision-Recall curve plot: {filename}")
    except Exception as e:
        print(f"      Error saving PR curve plot {filename}: {e}")
    plt.close()

def calculate_auc_for_df(df_subset):
    if df_subset.empty or auc is None or len(df_subset) < 2:
        return np.nan
    
    df_sorted = df_subset.sort_values('threshold', ascending=False)
    recall_values = np.clip(df_sorted['Recall'].values, 0, 1)
    precision_values = np.clip(df_sorted['Precision'].values, 0, 1)

    recall_for_auc = np.insert(recall_values, 0, 0.0)
    precision_for_auc = np.insert(precision_values, 0, 1.0)

    sort_indices = np.lexsort((-precision_for_auc, recall_for_auc))
    recall_sorted = recall_for_auc[sort_indices]
    precision_sorted = precision_for_auc[sort_indices]
    
    return auc(recall_sorted, precision_sorted)

def main(args):
    overall_start_time = time.time()
    
    LOG_PARAMS = {
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma,
        'num_sigma': args.num_sigma,
        'blob_overlap': args.overlap
    }
    THRESHOLDS_TO_SCAN = np.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    print(f"Using GT CSV: {args.gt_spots_csv}")
    try:
        if not os.path.exists(args.gt_spots_csv):
            raise FileNotFoundError(f"GT CSV not found: {args.gt_spots_csv}")
        gt_spots_df_global = pd.read_csv(args.gt_spots_csv)
        gt_spots_df_global['FRAME'] = gt_spots_df_global['FRAME'].astype(int)
        if not all(c in gt_spots_df_global.columns for c in ['FRAME', 'POSITION_X', 'POSITION_Y', 'GT_SPOT_ID']):
            raise ValueError("GT Spots CSV missing required columns.")
        print(f"  Loaded GT Spots CSV: {len(gt_spots_df_global)} spots globally.")
    except Exception as e:
        print(f"FATAL ERROR loading Ground Truth Spots CSV: {e}"); traceback.print_exc(); exit()

    print(f"\nSearching for TIF files in: {args.input_dir}")
    print(f"Saving detection scan results to subdirectories within: {args.output_dir}")
    print(f"Will sample {args.sample_frames} frames from each video group for analysis.")
    os.makedirs(args.output_dir, exist_ok=True)

    all_tif_files = glob.glob(os.path.join(args.input_dir, "*.tif"))
    all_tif_files = [f for f in all_tif_files if "softmasked" not in os.path.basename(f)]

    if not all_tif_files:
        print(f"\nERROR: No suitable .tif files found in {args.input_dir}."); exit()

    noisy_files_map = {}
    denoised_files_dict = defaultdict(list)
    scale_pattern = re.compile(r"scale_(\d+\.\d+)|sim_(\d+\.\d+)")

    for f_path in all_tif_files:
        basename = os.path.basename(f_path)
        match = scale_pattern.search(basename)

        if not match:
            print(f"  - Skipping file with un-parsable scale: {basename}")
            continue

        scale_str = match.group(1) or match.group(2)

        if basename.startswith(f"sim_") and basename.endswith(f"scale_{scale_str}.tif"):
             noisy_files_map[scale_str] = f_path
        else:
             denoised_files_dict[scale_str].append({'path': f_path, 'key': basename})

    print(f"\nFound {len(noisy_files_map)} noisy base files to process as groups.")
    for scale_key, variants in denoised_files_dict.items():
        print(f"  - Group for scale '{scale_key}' has {len(variants)} denoised variants.")

    processed_groups_count, successful_groups_count = 0, 0
    failed_items_list = []

    for scale_str, noisy_video_path in noisy_files_map.items():
        processed_groups_count += 1
        print(f"\n--- Processing Group {processed_groups_count}/{len(noisy_files_map)}: Scale {scale_str} ---")
        group_start_time = time.time()
        
        all_group_results = []
        
        try:
            print(f"  Analyzing source noisy file: {os.path.basename(noisy_video_path)}")
            if not os.path.exists(noisy_video_path):
                raise FileNotFoundError("Noisy video not found")
            
            noisy_stack_full = tifffile.imread(noisy_video_path)
            num_frames_total = noisy_stack_full.shape[0]

            np.random.seed(args.seed)
            if num_frames_total > args.sample_frames:
                frame_indices = np.random.choice(num_frames_total, size=args.sample_frames, replace=False)
                frame_indices.sort()
            else:
                frame_indices = np.arange(num_frames_total)
            
            noisy_stack_sampled = noisy_stack_full[frame_indices, :, :]
            gt_spots_df_sampled = gt_spots_df_global[gt_spots_df_global['FRAME'].isin(frame_indices)].copy()
            
            print(f"    Loaded noisy video {noisy_stack_full.shape}, randomly sampled to {noisy_stack_sampled.shape} on {len(frame_indices)} frames.")
            
            for i, thresh in enumerate(THRESHOLDS_TO_SCAN):
                print(f"      Noisy Threshold {i+1}/{len(THRESHOLDS_TO_SCAN)}: {thresh:.4f}", end='\r')
                detections_df = run_log_detection_on_stack(
                    noisy_stack_sampled, **LOG_PARAMS, threshold=thresh, frame_indices_map=frame_indices)
                matches_df, fp_df = match_detections_to_gt(detections_df, gt_spots_df_sampled, args.tolerance)
                metrics = calculate_detection_metrics(matches_df, fp_df)
                metrics['data_type'] = 'noisy'
                metrics['threshold'] = thresh
                all_group_results.append(metrics)
            print("\n    Noisy scan complete.                               ")

        except Exception as e:
            print(f"\n    FATAL ERROR processing noisy file {os.path.basename(noisy_video_path)}: {e}")
            traceback.print_exc()
            failed_items_list.append(f"{os.path.basename(noisy_video_path)} (Noisy file processing failed)")
            continue 

        if scale_str in denoised_files_dict:
            variants = denoised_files_dict[scale_str]
            print(f"  Analyzing {len(variants)} denoised variants...")
            
            for variant_info in variants:
                variant_path, variant_key = variant_info['path'], variant_info['key']
                print(f"\n    Variant: {os.path.basename(variant_path)}")
                try:
                    if not os.path.exists(variant_path):
                        raise FileNotFoundError("Denoised variant not found")
                    
                    denoised_stack_full = tifffile.imread(variant_path)
                    denoised_stack_sampled = denoised_stack_full[frame_indices, :, :]
                    print(f"      Loaded variant {denoised_stack_full.shape}, sampled to {denoised_stack_sampled.shape}.")

                    for i, thresh in enumerate(THRESHOLDS_TO_SCAN):
                        print(f"        Threshold {i+1}/{len(THRESHOLDS_TO_SCAN)}: {thresh:.4f}", end='\r')
                        detections_df = run_log_detection_on_stack(
                            denoised_stack_sampled, **LOG_PARAMS, threshold=thresh, frame_indices_map=frame_indices)
                        matches_df, fp_df = match_detections_to_gt(detections_df, gt_spots_df_sampled, args.tolerance)
                        metrics = calculate_detection_metrics(matches_df, fp_df)
                        metrics['data_type'] = variant_key 
                        metrics['threshold'] = thresh
                        all_group_results.append(metrics)
                    print("\n        Variant scan complete.                        ")

                except Exception as e:
                    print(f"\n      ERROR processing variant {variant_key}: {e}")
                    failed_items_list.append(f"{scale_str} / {variant_key}")
        else:
            print("  No denoised variants found for this group.")

        if not all_group_results:
            print("  No results generated for this group. Skipping finalization.")
            continue

        print("\n  Finalizing and saving results for the group...")
        group_results_df = pd.DataFrame(all_group_results)

        auc_series = group_results_df.groupby('data_type').apply(calculate_auc_for_df, include_groups=False)
        group_results_df['AUC'] = group_results_df['data_type'].map(auc_series)
        
        output_dir = os.path.join(args.output_dir, f"Group_Scale_{scale_str}")
        os.makedirs(output_dir, exist_ok=True)
        
        output_csv_path = os.path.join(output_dir, "detection_metrics_all_variants.csv")
        cols_order = ['data_type', 'threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'AUC']
        for col in cols_order:
            if col not in group_results_df.columns:
                group_results_df[col] = np.nan
        group_results_df[cols_order].to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"    Saved combined group results to: {output_csv_path}")

        plot_title_suffix = f"Scale = {scale_str}"
        mvt_plot_path = os.path.join(output_dir, "metrics_vs_threshold_comparison.png")
        plot_metrics_vs_threshold(group_results_df, mvt_plot_path, plot_title_suffix)
        pr_plot_path = os.path.join(output_dir, "precision_recall_curve_comparison.png")
        plot_precision_recall_curve(group_results_df, pr_plot_path, plot_title_suffix)
        
        successful_groups_count += 1
        print(f"  Group processing finished in {time.time() - group_start_time:.2f} seconds.")

    overall_end_time = time.time()
    print(f"\n--- Overall Script Finished in {time.time() - overall_start_time:.2f} seconds ---")
    print(f"Attempted to process {processed_groups_count} groups.")
    print(f"Successfully completed processing for {successful_groups_count} groups.")
    if failed_items_list:
        print(f"Failed to process {len(failed_items_list)} items:")
        for item_str in failed_items_list: print(f"  - {item_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run detection performance evaluation on simulated data.")
    
    # --- I/O Arguments ---
    parser.add_argument("--gt_spots_csv", type=str, required=True,
                        help="Path to the ground truth CSV file (e.g., '..._spot_info.csv') generated by create_ground_truth.py.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing all noisy and denoised .tif files to be evaluated.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the base directory where results (subfolders, CSVs, plots) will be saved.")

    # --- Evaluation Parameters ---
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Matching tolerance in pixels to count a detection as a True Positive.")
    parser.add_argument("--sample_frames", type=int, default=100,
                        help="Number of frames to randomly sample from each video for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible frame sampling.")

    # --- LoG Detector Parameters ---
    parser.add_argument("--min_sigma", type=float, default=1.5,
                        help="LoG detector: minimum sigma.")
    parser.add_argument("--max_sigma", type=float, default=3.0,
                        help="LoG detector: maximum sigma.")
    parser.add_argument("--num_sigma", type=int, default=10,
                        help="LoG detector: number of sigma steps.")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="LoG detector: blob overlap threshold.")

    # --- Threshold Scan Parameters ---
    parser.add_argument("--thresh_min", type=float, default=1.0,
                        help="Minimum LoG threshold to scan.")
    parser.add_argument("--thresh_max", type=float, default=100.0,
                        help="Maximum LoG threshold to scan.")
    parser.add_argument("--thresh_steps", type=int, default=20,
                        help="Number of threshold steps to scan.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
