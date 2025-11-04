"""
Runs the full evaluation pipeline for EXPERIMENTAL data.

This script implements the quantitative analysis described in Sec 2.4.2
of the paper. It is a single, integrated script that performs both
analysis and plotting.

Workflow:
1.  Iterates through all experiment folders in the '--base_dir'.
2.  For each experiment, it loads the original noisy .tif video and the
    corresponding TrackMate _spots.csv file.
3.  It finds all 'denoised.tif' variants in the same folder.
4.  For each denoised variant, it iterates through the TrackMate spots.
5.  For each spot, it performs a 2D Gaussian fit on both the noisy
    and denoised video at the reference coordinate.
6.  It calculates local background statistics (mean, std dev) using
    an adaptive annulus, as described in the paper.
7.  It saves a detailed CSV file of these per-spot metrics for the variant.
8.  (Optional) It saves visualization images for a subset of spots.
9.  After all experiments and variants are processed, it combines all
    results into a single master DataFrame.
10. It calculates summary metrics (median local noise, median local
    brightness, median localization error, median photometry error).
11. It generates and saves the final summary bar plots comparing all
    methods across all experiments.
"""

import numpy as np
import pandas as pd
import tifffile
import os
from pathlib import Path
import time
import traceback
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import glob
import re
import argparse
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: 'tqdm' not found. Progress bars will be disabled.")
    print("Install with: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

print(f"Experimental Data Evaluation & Plotting Script Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Mapping keywords in filenames to a clean display name
METHOD_MAP = {
    'noisy': 'Noisy', 'geo0.001': 'λ = 0.001 (T=1)', 'geo0.1': 'λ = 0.1 (T=1)',
    'training_run': 'λ = RL', 'n2v': 'N2V',
    'deepcad-rt': 'DeepCAD-RT', 
    # Add other mappings if needed, e.g., 'geo0.1_seq3': 'λ = 0.1 (T=3)'
    # 'geo0.001_seq1': 'λ = 0.001 (T=1)', 'geo0.1_seq1': 'λ = 0.1 (T=1)',
    # 'geo0.001_seq3': 'λ = 0.001 (T=3)', 'geo0.1_seq3': 'λ = 0.1 (T=3)'
}

# Defining the colors for each method
ALL_COLORS = {
    'Noisy': '#D55E00',             # Orange
    'N2V': '#56B4E9',               # Sky Blue
    'DeepCAD-RT': '#F0E442',         # Yellow
    'λ = RL': '#E69F00',             # Amber
    'λ = 0.001 (T=1)': '#0072B2',    # Blue
    'λ = 0.1 (T=1)': '#009E73',      # Green
    # 'λ = 0 (T=3)': '#CC79A7',        # Reddish Purple
    # 'λ = 0.1 (T=3)': '#2F4F4F',      # Dark Slate Gray
    # 'λ = 0.001 (T=3)': '#0072B2',    # Blue (re-use)
}

def get_variant_display_name_from_filename(path):
    name = Path(path).name.lower()
    for keyword, display_name in sorted(METHOD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in name:
            return display_name
    return 'Other'

def rotated_2d_gaussian(coords, amplitude, x0, y0, sigma_x, sigma_y, theta_deg, offset):
    (x, y) = coords; theta = np.deg2rad(theta_deg)
    X, Y = x - x0, y - y0
    x_prime = X * np.cos(theta) + Y * np.sin(theta)
    y_prime = -X * np.sin(theta) + Y * np.cos(theta)
    sx2, sy2 = sigma_x**2, sigma_y**2
    exponent = 0.0
    if sx2 > 1e-9 and sy2 > 1e-9:
        exponent = (x_prime**2) / (2 * sx2) + (y_prime**2) / (2 * sy2)
    return offset + amplitude * np.exp(-exponent)

def fit_rotated_gaussian_2d(region, global_x1_region_origin, global_y1_region_origin, data_max):
    height, width = region.shape
    if height == 0 or width == 0: return False, None
    
    Y_mesh, X_mesh = np.mgrid[global_y1_region_origin : global_y1_region_origin + height,
                              global_x1_region_origin : global_x1_region_origin + width]
    z_data_flat = region.ravel().astype(float)
    
    min_region, max_region = np.min(region), np.max(region)
    amplitude_guess = float(max_region - min_region) if max_region > min_region else 1.0
    offset_guess = float(min_region)
    x0_guess = global_x1_region_origin + width / 2.0
    y0_guess = global_y1_region_origin + height / 2.0
    sigma_guess = max(1.0, min(width, height) / 4.0)
    initial_guess = (amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0.0, offset_guess)
    
    lower_bounds = [0, global_x1_region_origin - 2, global_y1_region_origin - 2, 0.1, 0.1, -180, 0]
    upper_bounds = [data_max*1.5, global_x1_region_origin + width + 2, global_y1_region_origin + height + 2, width/2, height/2, 180, data_max]

    def gaussian_func_flat(coords_mesh, amp, x0, y0, sx, sy, theta, offset):
        return rotated_2d_gaussian(coords_mesh, amp, x0, y0, sx, sy, theta, offset).ravel()
    
    try:
        popt, _ = curve_fit(gaussian_func_flat, (X_mesh, Y_mesh), z_data_flat,
                            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=5000)
        fit_params = {'fit_x': popt[1], 'fit_y': popt[2], 'fit_amplitude': popt[0],
                      'fit_sx': popt[3], 'fit_sy': popt[4], 'fit_theta': popt[5], 'fit_offset': popt[6]}
        if fit_params['fit_sx'] > width/1.5 or fit_params['fit_sy'] > height/1.5:
            return False, None
        return True, fit_params
    except (RuntimeError, ValueError):
        return False, None

def zoom_spot_loc(video_frame, spot_position, region_size):
    x_int, y_int = int(round(spot_position[0])), int(round(spot_position[1]))
    half_size = region_size // 2
    y1, y2 = y_int - half_size, y_int + half_size + 1
    x1, x2 = x_int - half_size, x_int + half_size + 1
    y1_clipped, y2_clipped = max(0, y1), min(video_frame.shape[0], y2)
    x1_clipped, x2_clipped = max(0, x1), min(video_frame.shape[1], x2)
    region = video_frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
    return region, (x1, y1)

def analyze_local_background(frame, spot_center_xy, inner_radius, outer_radius):
    h, w = frame.shape
    x_c, y_c = int(round(spot_center_xy[0])), int(round(spot_center_xy[1]))
    
    y, x = np.ogrid[0:h, 0:w]
    dist_sq = (x - x_c)**2 + (y - y_c)**2
    inner_rad_sq, outer_rad_sq = inner_radius**2, outer_radius**2
    
    annulus_mask = (dist_sq > inner_rad_sq) & (dist_sq <= outer_rad_sq)
    local_bg_pixels = frame[annulus_mask]
    
    if local_bg_pixels.size < 10:
        return np.nan, np.nan
        
    background_median = np.median(local_bg_pixels)
    mad = np.median(np.abs(local_bg_pixels - background_median))
    background_std_dev = mad * 1.4826 if mad > 1e-9 else 0.0
    
    return background_median, background_std_dev

def load_and_clean_trackmate_csv(path):
    try:
        with open(path, 'r', errors='ignore') as f:
            for i, line in enumerate(f):
                if 'LABEL' in line:
                    header_row = i
                    break
            else:
                raise ValueError(f"Could not find header row in {path.name}.")
        df = pd.read_csv(path, header=header_row, low_memory=False)
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
        return pd.DataFrame()
        
    df.columns = df.columns.str.strip().str.upper()
    required_cols = ['POSITION_X', 'POSITION_Y', 'FRAME']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: CSV file {path.name} is missing required columns. Skipping.")
        return pd.DataFrame()
        
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols)
    df['FRAME'] = df['FRAME'].astype(int)
    return df


def save_background_visualization(output_path, full_frame, spot_xy, roi_size, inner_rad, outer_rad):
    try:
        x_c, y_c = spot_xy
        h, w = full_frame.shape
        half_size = roi_size // 2
        y1, y2 = max(0, int(y_c) - half_size), min(h, int(y_c) + half_size)
        x1, x2 = max(0, int(x_c) - half_size), min(w, int(x_c) + half_size)
        roi_patch = full_frame[y1:y2, x1:x2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        fig.suptitle("Background Annulus Visualization", fontsize=16)

        ax1.imshow(full_frame, cmap='gray')
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title("Full Frame with ROI")
        ax1.axis('off')

        ax2.imshow(roi_patch, cmap='gray')
        spot_in_patch_x = x_c - x1 - 0.5
        spot_in_patch_y = y_c - y1 - 0.5
        center_of_annulus = (spot_in_patch_x, spot_in_patch_y)

        outer_circle = Circle(center_of_annulus, outer_rad, linewidth=1.5, edgecolor='#00ff00', facecolor='none', label='Outer Radius')
        inner_circle = Circle(center_of_annulus, inner_rad, linewidth=1.5, edgecolor='#ff8c00', facecolor='none', label='Inner Radius')
        ax2.add_patch(outer_circle)
        ax2.add_patch(inner_circle)
        ax2.set_title("Zoomed ROI with Background Annulus")
        ax2.legend(handles=[outer_circle, inner_circle], loc='upper right')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save background visualization: {e}")

def save_gaussian_fit_visualization(output_path, region, fit_params):
    try:
        if not fit_params: return
        h, w = region.shape
        y_mesh, x_mesh = np.mgrid[0:h, 0:w]

        vis_params = fit_params.copy()
        vis_params['fit_x'] = (w - 1) / 2.0 # Center of patch
        vis_params['fit_y'] = (h - 1) / 2.0 # Center of patch
        
        key_map = {'fit_amplitude': 'amplitude', 'fit_x': 'x0', 'fit_y': 'y0',
                   'fit_sx': 'sigma_x', 'fit_sy': 'sigma_y', 'fit_theta': 'theta_deg',
                   'fit_offset': 'offset'}
        unpacked_params = {key_map[k]: v for k, v in vis_params.items() if k in key_map}
        
        model_data = rotated_2d_gaussian((x_mesh, y_mesh), **unpacked_params)
        residual = region - model_data

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        fig.suptitle("Gaussian Fit Diagnostics", fontsize=16)

        axes[0].imshow(region, cmap='gray')
        axes[0].set_title(f"Raw Data (θ={fit_params['fit_theta']:.1f}°)")
        center = ((w - 1) / 2.0, (h - 1) / 2.0)

        ellipse = plt.matplotlib.patches.Ellipse(xy=center, width=fit_params['fit_sx']*2, height=fit_params['fit_sy']*2,
                                                 angle=-fit_params['fit_theta'], edgecolor='red', facecolor='none', lw=1.5)
        axes[0].add_patch(ellipse)
        axes[1].imshow(model_data, cmap='gray')
        axes[1].set_title(f"Fitted Model (Amp={fit_params['fit_amplitude']:.1f})")

        resid_vmax = np.abs(residual).max()
        im3 = axes[2].imshow(residual, cmap='coolwarm', vmin=-resid_vmax, vmax=resid_vmax)
        axes[2].set_title("Residual (Data - Model)")
        fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        for ax in axes: ax.axis('off')

        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save fit visualization: {e}")


def process_spot(spot_tuple, original_stack, denoised_stack, exp_folder_name, data_max, variant_name, vis_dirs, args):
    spot_idx, spot = spot_tuple
    frame_idx = spot['FRAME']
    if frame_idx >= len(original_stack) or frame_idx >= len(denoised_stack):
        return None
        
    ref_x, ref_y = spot['POSITION_X'], spot['POSITION_Y']
    result_entry = {'experiment': exp_folder_name, **spot.to_dict()}
    
    frame_noisy = original_stack[frame_idx]
    frame_denoised = denoised_stack[frame_idx]

    patch_denoised_gauss, (gx1_d, gy1_d) = zoom_spot_loc(frame_denoised, (ref_x, ref_y), args.fit_region_size)
    success_d, params_d = fit_rotated_gaussian_2d(patch_denoised_gauss, gx1_d, gy1_d, data_max)
    if success_d:
        result_entry.update({f"denoised_{key}": val for key, val in params_d.items()})

    patch_noisy_gauss, (gx1_n, gy1_n) = zoom_spot_loc(frame_noisy, (ref_x, ref_y), args.fit_region_size)
    success_n, params_n = fit_rotated_gaussian_2d(patch_noisy_gauss, gx1_n, gy1_n, data_max)
    if success_n:
        result_entry.update({f"noisy_{key}": val for key, val in params_n.items()})

    if not success_d and not success_n:
        return None

    base_params_for_radius = params_d if success_d else params_n
    if args.adaptive_radii:
        effective_radius = np.clip(np.sqrt(base_params_for_radius['fit_sx'] * base_params_for_radius['fit_sy']), 0.5, 5.0)
        inner_rad = (args.adapt_inner_mult * effective_radius) + args.adapt_inner_base
        outer_rad = (args.adapt_outer_mult * effective_radius) + args.adapt_outer_base
    else:
        inner_rad, outer_rad = args.fixed_inner_radius, args.fixed_outer_radius
    result_entry['bg_inner_radius_used'] = inner_rad
    result_entry['bg_outer_radius_used'] = outer_rad

    
    # 1. Local Background Std Dev (Noise)
    bg_mean_local_n, bg_std_local_n = analyze_local_background(frame_noisy, (ref_x, ref_y), inner_rad, outer_rad)
    result_entry.update({'noisy_local_bg_mean': bg_mean_local_n, 'noisy_local_bg_std': bg_std_local_n})
    bg_mean_local_d, bg_std_local_d = analyze_local_background(frame_denoised, (ref_x, ref_y), inner_rad, outer_rad)
    result_entry.update({'denoised_local_bg_mean': bg_mean_local_d, 'denoised_local_bg_std': bg_std_local_d})

    # (Localization Error and Photometry Error are calculated in the main plotting part)

    if args.save_visuals and spot_idx < args.spots_to_visualize:
        vis_filename_base = f"{exp_folder_name}_{variant_name}_frame{frame_idx}_spot{spot_idx}"
        if success_d:
            vis_filename_d = f"{vis_filename_base}_denoised.png"
            save_background_visualization(vis_dirs['denoised_bg'] / vis_filename_d, frame_denoised, (ref_x, ref_y), args.bg_vis_size, inner_rad, outer_rad)
            save_gaussian_fit_visualization(vis_dirs['denoised_fit'] / vis_filename_d, patch_denoised_gauss, params_d)
        if success_n:
            vis_filename_n = f"{vis_filename_base}_noisy.png"
            save_background_visualization(vis_dirs['noisy_bg'] / vis_filename_n, frame_noisy, (ref_x, ref_y), args.bg_vis_size, inner_rad, outer_rad)
            save_gaussian_fit_visualization(vis_dirs['noisy_fit'] / vis_filename_n, patch_noisy_gauss, params_n)

    return result_entry

def main(args):
    base_data_dir = Path(args.base_dir)
    output_dir = base_data_dir / args.output_dir_name
    output_dir.mkdir(exist_ok=True)
    
    vis_base_dir = output_dir / args.vis_subdir_name
    vis_fit_dir = vis_base_dir / args.fit_vis_subdir_name

    print(f"Base data directory: {base_data_dir}")
    print(f"Output will be saved to: {output_dir}")

    experiment_folders = [d for d in base_data_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != args.output_dir_name]
    
    all_detailed_dfs = [] 

    for exp_folder in experiment_folders:
        print(f"\n{'='*20} Processing Experiment: {exp_folder.name} {'='*20}")
        try:
            original_video_path = next(exp_folder.glob(f"{exp_folder.name}.tif"))
            trackmate_csv_path = next(exp_folder.glob(f"{exp_folder.name}.csv"))
            original_stack = tifffile.imread(original_video_path)
            trackmate_df = load_and_clean_trackmate_csv(trackmate_csv_path)
            if trackmate_df.empty:
                print(f"  No valid spots found in {trackmate_csv_path.name}. Skipping experiment.")
                continue
                
            data_max = np.iinfo(original_stack.dtype).max if np.issubdtype(original_stack.dtype, np.integer) else np.percentile(original_stack, 99.9)

            if args.spots_to_process is not None and len(trackmate_df) > args.spots_to_process:
                trackmate_df_sampled = trackmate_df.sample(n=args.spots_to_process, random_state=42)
            else:
                trackmate_df_sampled = trackmate_df
            trackmate_df_sampled = trackmate_df_sampled.reset_index(drop=True)
            print(f"  Loaded noisy video: {original_video_path.name} ({original_stack.shape})")
            print(f"  Loaded {len(trackmate_df)} spots, processing {len(trackmate_df_sampled)}.")

        except StopIteration:
            print(f"  WARNING: Could not find matching .tif and .csv for '{exp_folder.name}'. Skipping.")
            continue
        except Exception as e:
            print(f"  ERROR loading data for '{exp_folder.name}': {e}")
            traceback.print_exc()
            continue

        for denoised_video_path in sorted(list(exp_folder.glob("*denoised*.tif"))):
            variant_name_stem = denoised_video_path.stem
            variant_display_name = get_variant_display_name_from_filename(variant_name_stem)
            
            if args.methods and variant_display_name not in args.methods:
                continue

            print(f"\n  --- Evaluating Variant: {variant_name_stem} (as '{variant_display_name}') ---")
            try:
                denoised_stack = tifffile.imread(denoised_video_path)
                if denoised_stack.shape[0] != original_stack.shape[0]:
                    print(f"    WARNING: Frame count mismatch! Noisy={original_stack.shape[0]}, Denoised={denoised_stack.shape[0]}. Skipping.")
                    continue

                vis_dirs = {}
                if args.save_visuals:
                    vis_dirs['denoised_bg'] = vis_base_dir / f"{variant_name_stem}_denoised_bg_examples"
                    vis_dirs['noisy_bg'] = vis_base_dir / f"{variant_name_stem}_noisy_bg_examples"
                    vis_dirs['denoised_fit'] = vis_fit_dir / f"{variant_name_stem}_denoised"
                    vis_dirs['noisy_fit'] = vis_fit_dir / f"{variant_name_stem}_noisy"
                    for d in vis_dirs.values():
                        d.mkdir(exist_ok=True, parents=True)

                detailed_results = []
                for spot_tuple in tqdm(trackmate_df_sampled.iterrows(), total=len(trackmate_df_sampled), desc="Processing Spots", leave=False):
                    result = process_spot(spot_tuple, original_stack, denoised_stack, exp_folder.name, data_max, variant_name_stem, vis_dirs, args)
                    if result:
                        detailed_results.append(result)

                if not detailed_results:
                    print("    No spots were successfully processed for this variant.")
                    continue
                    
                detailed_results_df = pd.DataFrame(detailed_results)
                detailed_results_df['variant_display_name'] = variant_display_name
                
                detailed_csv_path = output_dir / f"{exp_folder.name}_{variant_name_stem}_detailed_results.csv"
                detailed_results_df.to_csv(detailed_csv_path, index=False, float_format='%.6f')
                print(f"    - Saved detailed results to: {detailed_csv_path.name}")
                
                all_detailed_dfs.append(detailed_results_df) 
                
            except Exception as e:
                print(f"    ERROR processing variant {variant_name_stem}: {e}")
                traceback.print_exc()
                continue
                
    print(f"\n{'='*20} All experiments processed. Starting summary plotting. {'='*20}")
    
    if not all_detailed_dfs:
        print("No detailed results were generated. Exiting plot generation.")
        return

    master_df = pd.concat(all_detailed_dfs, ignore_index=True)

    if args.methods:
        master_df = master_df[master_df['variant_display_name'].isin(args.methods)].copy()
        
    if args.exclude_training_data:
        master_df = master_df[master_df['experiment'] != 'Cy3_Best'].copy()
        print(f"Excluding 'Cy3_Best' training data from plots.")

    if master_df.empty:
        print("No data left after filtering. Exiting.")
        return

    print("Calculating localization and photometry errors...")
    # 1. Localization Error
    loc_cols = ['denoised_fit_x', 'POSITION_X', 'denoised_fit_y', 'POSITION_Y']
    for col in loc_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    valid_loc_rows = master_df.dropna(subset=loc_cols)
    master_df['localization_error'] = np.sqrt(
        (valid_loc_rows['denoised_fit_x'] - valid_loc_rows['POSITION_X'])**2 +
        (valid_loc_rows['denoised_fit_y'] - valid_loc_rows['POSITION_Y'])**2
    )
    
    # 2. Photometry Error
    phot_cols = ['denoised_fit_amplitude', 'noisy_fit_amplitude']
    for col in phot_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    valid_phot_rows = master_df.dropna(subset=phot_cols)
    master_df['photometry_error'] = (
        valid_phot_rows['denoised_fit_amplitude'] - valid_phot_rows['noisy_fit_amplitude']
    ).abs()

    metric_cols = [
        'noisy_local_bg_mean', 'noisy_local_bg_std', 'denoised_local_bg_mean',
        'denoised_local_bg_std', 'localization_error', 'photometry_error'
    ]
    summary_df = master_df.groupby(['experiment', 'variant_display_name'])[metric_cols].median().reset_index()
    summary_df.columns = ['experiment', 'variant_display_name'] + [f"{col}_median" for col in metric_cols]

    plot_data_list = []
    noisy_summary = summary_df.groupby('experiment')[['noisy_local_bg_std_median', 'noisy_local_bg_mean_median']].first().reset_index()
    for _, row in noisy_summary.iterrows():
        plot_data_list.append({'Experiment': row['experiment'], 'Variant': 'Noisy', 'Value': row['noisy_local_bg_std_median'], 'Metric': 'Local Background Std Dev (Noise)'})
        plot_data_list.append({'Experiment': row['experiment'], 'Variant': 'Noisy', 'Value': row['noisy_local_bg_mean_median'], 'Metric': 'Local Background Mean (Brightness)'})

    for _, row in summary_df.iterrows():
        plot_data_list.append({'Experiment': row['experiment'], 'Variant': row['variant_display_name'], 'Value': row.get('denoised_local_bg_std_median'), 'Metric': 'Local Background Std Dev (Noise)'})
        plot_data_list.append({'Experiment': row['experiment'], 'Variant': row['variant_display_name'], 'Value': row.get('denoised_local_bg_mean_median'), 'Metric': 'Local Background Mean (Brightness)'})

    plot_df = pd.DataFrame(plot_data_list).dropna(subset=['Value'])

    print("\nGenerating and saving plots...")
    
    denoised_hue_order = [method for method in args.methods if method in ALL_COLORS]
    all_hue_order = ['Noisy'] + denoised_hue_order
    fixed_color_map = {k: v for k, v in ALL_COLORS.items() if k in all_hue_order}

    # Plots 1-2: Background Std and Mean
    for metric, title, ylabel in [
        ('Local Background Std Dev (Noise)', 'Local Background Noise Reduction', 'Median Local Background Std Dev'),
        ('Local Background Mean (Brightness)', 'Local Background Mean Preservation', 'Median Local Background Mean'),
    ]:
        plt.figure(figsize=(16, 8))
        data_subset = plot_df[plot_df['Metric'] == metric]
        sns.barplot(data=data_subset, x='Experiment', y='Value', hue='Variant', hue_order=all_hue_order, palette=fixed_color_map)
        plt.title(title, fontsize=16, weight='bold')
        plt.ylabel(ylabel)
        plt.xlabel('Experiment')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()
        print(f"  Saved plot: {title.lower().replace(' ', '_')}.png")

    # Plot 3: Localization Error
    if 'localization_error_median' in summary_df.columns:
        plt.figure(figsize=(16, 8))
        sns.barplot(data=summary_df, x='experiment', y='localization_error_median', hue='variant_display_name', hue_order=denoised_hue_order, palette=fixed_color_map)
        plt.title('Median Localization Error', fontsize=16, weight='bold')
        plt.ylabel('Median Localization Error (pixels)\nLower is Better')
        plt.xlabel('Experiment')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / "median_localization_error.png", dpi=300)
        plt.close()
        print("  Saved plot: median_localization_error.png")

    # Plot 4: Photometry Error (Newly Added)
    if 'photometry_error_median' in summary_df.columns:
        plt.figure(figsize=(16, 8))
        sns.barplot(data=summary_df, x='experiment', y='photometry_error_median', hue='variant_display_name', hue_order=denoised_hue_order, palette=fixed_color_map)
        plt.title('Median Photometry Error (vs. Noisy Fit)', fontsize=16, weight='bold')
        plt.ylabel('Median Absolute Amplitude Error (ADU)\nLower is Better')
        plt.xlabel('Experiment')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / "median_photometry_error.png", dpi=300)
        plt.close()
        print("  Saved plot: median_photometry_error.png")

    print(f"\n--- Analysis complete. Plots saved to {output_dir} ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline on experimental data.")
    
    # --- I/O Paths ---
    parser.add_argument("--base_dir", type=str, required=True,
                        help="The base directory containing all experimental data folders (e.g., '.../exp_data3').")
    parser.add_argument("--output_dir_name", type=str, default="Experimental_Evaluation_Results",
                        help="Name of the subfolder to create within --base_dir to save all results and plots.")
    
    # --- Method Filtering ---
    parser.add_argument("--methods", type=str, nargs='+',
                        default=['λ = 0.001 (T=1)', 'λ = 0.1 (T=1)', 'λ = RL', 'N2V', 'DeepCAD-RT'],
                        help=f"List of method display names to process and plot. Default: 'λ = 0.001 (T=1)', 'λ = 0.1 (T=1)', etc. Available: {list(METHOD_MAP.values())}")
    parser.add_argument("--exclude_training_data", action="store_true",
                        help="If set, excludes the 'Cy3_Best' experiment from the final plots.")
                        
    # --- Analysis Parameters ---
    parser.add_argument("--spots_to_process", type=int, default=None,
                        help="Number of spots to sample from each experiment. Default: None (process all spots).")
    parser.add_argument("--fit_region_size", type=int, default=7,
                        help="Pixel size of the square region for 2D Gaussian fitting (must be odd).")
    
    # --- Background Annulus Parameters ---
    parser.add_argument("--adaptive_radii", action="store_true",
                        help="Use adaptive radii for background annulus (as per paper).")
    parser.add_argument("--fixed_inner_radius", type=float, default=2.0,
                        help="Inner radius if not using adaptive.")
    parser.add_argument("--fixed_outer_radius", type=float, default=5.0,
                        help="Outer radius if not using adaptive.")
    parser.add_argument("--adapt_inner_mult", type=float, default=2.0,
                        help="Adaptive inner radius multiplier (sigma * mult + base).")
    parser.add_argument("--adapt_inner_base", type=float, default=1.0,
                        help="Adaptive inner radius base (sigma * mult + base).")
    parser.add_argument("--adapt_outer_mult", type=float, default=4.0,
                        help="Adaptive outer radius multiplier (sigma * mult + base).")
    parser.add_argument("--adapt_outer_base", type=float, default=2.0,
                        help="Adaptive outer radius base (sigma * mult + base).")
    
     --- Visualization Parameters ---
    parser.add_argument("--save_visuals", action="store_true",
                        help="Save visualization images for a subset of spots.")
    parser.add_argument("--spots_to_visualize", type=int, default=5,
                        help="Number of spots to save visualizations for (per variant).")
    parser.add_argument("--vis_subdir_name", type=str, default="visualizations",
                        help="Name of the subfolder for background visualizations.")
    parser.add_argument("--fit_vis_subdir_name", type=str, default="gaussian_fit_visualizations",
                        help="Name of the subfolder for fit visualizations.")
    parser.add_argument("--bg_vis_size", type=int, default=40,
                        help="Pixel size of the background annulus visualization ROI.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.fit_region_size % 2 == 0:
        print(f"Warning: --fit_region_size ({args.fit_region_size}) must be odd. Using {args.fit_region_size + 1} instead.")
        args.fit_region_size += 1
        
    main(args)
