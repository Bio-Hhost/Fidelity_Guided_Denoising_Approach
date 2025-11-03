import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import tifffile
import os
import time
import argparse 
from sklearn.linear_model import LinearRegression, TheilSenRegressor

print(f"Noise Addition Script Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def validate_region(region, frame_shape):
    """Checks if a noise region is valid."""
    height, width = frame_shape
    try:
        x1, y1, x2, y2 = map(int, region)
    except (ValueError, TypeError):
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height: return False
    if (x2 - x1) < 5 or (y2 - y1) < 5: return False
    return True

def analyze_noise_regions(frames, noise_regions, plot=False, plot_title_suffix="", save_path=None):
    if not isinstance(frames, list):
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            frames = [frame for frame in frames]
        else: raise TypeError("Input 'frames' must be a list/stack")
    if not frames: raise ValueError("Input frames list is empty")
    height, width = frames[0].shape; valid_regions = []
    for region in noise_regions:
        r = tuple(map(int, region))
        if validate_region(r, (height, width)): valid_regions.append(r)
    if not valid_regions: raise ValueError("No valid noise regions found")
    pixel_collector = []
    for region_idx, (x1, y1, x2, y2) in enumerate(valid_regions):
        for frame in frames:
            region = frame[y1:y2, x1:x2]
            if region.size > 0: pixel_collector.append(region.flatten())
    if not pixel_collector: raise ValueError("No valid pixels found")
    noise_pixels = np.concatenate(pixel_collector)
    background_level = np.median(noise_pixels)
    mad = np.median(np.abs(noise_pixels - background_level))
    if mad < 1e-9: gaussian_std_dev = 0.0; print("Warning: MAD is near-zero.")
    else: gaussian_std_dev = mad * 1.4826
    gaussian_variance = gaussian_std_dev ** 2
    print(f"  Background Level (Median): {background_level:.2f}")
    print(f"  Est. Noise Variance (from MAD): {gaussian_variance:.2f}")
    print(f"  Est. Noise Std Dev (from MAD): {gaussian_std_dev:.2f}")
    if plot:
        fig = plt.figure(figsize=(10, 5)); p_min, p_max = np.percentile(noise_pixels, [0.1, 99.9])
        if np.isclose(p_min, p_max): p_min -= 1; p_max += 1
        bins = np.linspace(p_min, p_max, 100)
        plt.hist(noise_pixels, bins=bins, density=True, alpha=0.7, label=f'Pixel Histogram\n(N={len(noise_pixels)})')
        xmin, xmax = plt.xlim(); x = np.linspace(xmin, xmax, 200)
        if gaussian_std_dev > 1e-6:
             try:
                 p = norm.pdf(x, background_level, gaussian_std_dev)
                 plt.plot(x, p, 'k', linewidth=2, label=f'Fitted Normal\n(mu={background_level:.2f}, sigma={gaussian_std_dev:.2f})')
             except Exception as e: print(f"Could not plot Gaussian fit: {e}")
        else: plt.axvline(background_level, color='k', linestyle='--', label=f'Median={background_level:.2f} (StdDev~0)')
        plt.title(f'Background Noise Distribution{plot_title_suffix}'); plt.xlabel('Pixel Intensity'); plt.ylabel('Density')
        plt.legend(); plt.grid(True, alpha=0.3)
        if save_path:
             plt.savefig(save_path); print(f"  Saved plot to {save_path}")
             plt.close(fig)
        else: plt.show()
    return background_level, gaussian_variance, gaussian_std_dev

def analyze_intensity_variance_relationship(frames, background_level, patch_size=32, use_robust_regression=False, plot=True, save_path=None):
    if not isinstance(frames, list):
        if isinstance(frames, np.ndarray) and frames.ndim == 3: frames = [frame for frame in frames]
        else: raise TypeError("Input 'frames' must be a list/stack")
    means = []; variances = []; height, width = frames[0].shape; processed_patches = 0
    print(f"Analyzing variance vs mean using patch size {patch_size}x{patch_size}...")
    frames_to_process = frames
    all_pixels_corrected = np.concatenate([(frame - background_level).flatten() for frame in frames_to_process])
    if all_pixels_corrected.size == 0: raise ValueError("No pixels found after background correction.")
    p01 = np.percentile(all_pixels_corrected, 0.1); p999 = np.percentile(all_pixels_corrected, 99.9)
    for frame in frames_to_process:
        frame_corrected = frame.astype(np.float32) - background_level
        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                patch = frame_corrected[y:y+patch_size, x:x+patch_size]
                if patch.size == 0: continue
                if np.max(patch) > p999 or np.min(patch) < p01: continue
                patch_mean = np.mean(patch); patch_variance = np.var(patch, ddof=1)
                if not np.isnan(patch_mean) and not np.isnan(patch_variance) and \
                   not np.isinf(patch_mean) and not np.isinf(patch_variance):
                    means.append(patch_mean); variances.append(patch_variance); processed_patches += 1
    if not means: raise ValueError("No valid patches found for analysis")
    print(f"Collected mean/variance from {processed_patches} patches.")
    means = np.array(means); variances = np.array(variances)
    valid_mask = (means > 1e-6) & (variances > 1e-9)
    if np.sum(valid_mask) < 10: raise ValueError(f"Insufficient valid data points ({np.sum(valid_mask)}) for regression")
    means_filtered = means[valid_mask]; variances_filtered = variances[valid_mask]
    print(f"Using {len(means_filtered)} data points for linear regression.")
    X = means_filtered.reshape(-1, 1); y = variances_filtered
    try:
        if use_robust_regression:
            print("Using robust Theil-Sen regressor.")
            model = TheilSenRegressor(random_state=42, n_jobs=-1)
        else:
            print("Using standard Linear Regression.")
            model = LinearRegression(n_jobs=-1)
        model.fit(X, y)
    except Exception as e: raise ValueError(f"Linear regression failed. Error: {e}")
    gain_estimate = model.coef_[0]
    read_noise_variance_estimate = model.intercept_
    if read_noise_variance_estimate < 0:
        print(f"Warning: Intercept (Read Noise Var) negative ({read_noise_variance_estimate:.2f}). Clamping to 0.")
        read_noise_variance_estimate = 0.0
    read_noise_std_dev_estimate = np.sqrt(read_noise_variance_estimate)
    print(f"Estimated Gain (Slope, alpha): {gain_estimate:.3f} ADU/photon")
    print(f"Estimated Read Noise Variance (Intercept, sigma^2): {read_noise_variance_estimate:.2f} ADU^2")
    print(f"Estimated Read Noise Std Dev (Intercept, sigma): {read_noise_std_dev_estimate:.2f} ADU")
    if plot:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(means_filtered, variances_filtered, alpha=0.2, s=5, label='Patch Data (Filtered)')
        means_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        plt.plot(means_range, model.predict(means_range), 'r-', label=f'Linear Fit (Gain={gain_estimate:.3f}, ReadVar={read_noise_variance_estimate:.2f})')
        plt.xlabel('Mean Intensity (Background Corrected ADU)'); plt.ylabel('Variance (ADU^2)')
        plt.title('Variance vs. Mean Intensity (Photon Transfer)'); plt.legend(); plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path); print(f"  Saved plot to {save_path}"); plt.close(fig)
        else: plt.show()
    return gain_estimate, read_noise_variance_estimate

def save_histogram_comparison(original_pixels, sim_pixels, bins, title, filename):
    try:
        fig = plt.figure(figsize=(12, 6))
        plt.hist(original_pixels, bins=bins, density=True, alpha=0.7, label='Original Video')
        plt.hist(sim_pixels, bins=bins, density=True, alpha=0.7, label=os.path.basename(filename).replace('.tif',''))
        plt.title(title); plt.xlabel('Pixel Intensity'); plt.ylabel('Density')
        plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
        plt.savefig(filename.replace('.tif', '_hist.png'))
        print(f"  Saved histogram comparison: {filename.replace('.tif', '_hist.png')}")
        plt.close(fig)
    except Exception as e: print(f"    Error saving histogram comparison: {e}")

def save_visual_comparison(frame_orig, frame_sim, frame_idx, title_suffix, filename):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        vmin = np.percentile([frame_orig, frame_sim], 1); vmax = np.percentile([frame_orig, frame_sim], 99)
        im0 = axes[0].imshow(frame_orig, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Original Video (Frame {frame_idx})'); axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(frame_sim, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Simulated ({title_suffix}, Frame {frame_idx})'); axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(filename.replace('.tif', f'_frame{frame_idx}_comp.png'))
        print(f"  Saved visual comparison: {filename.replace('.tif', f'_frame{frame_idx}_comp.png')}")
        plt.close(fig)
    except IndexError: print(f"    Error: comparison_frame_idx {frame_idx} out of bounds.")
    except Exception as e: print(f"    Error saving visual comparison: {e}")

def save_psd_comparison(frame_orig, frame_sim, frame_idx, title_suffix, filename):
    try:
        fft_orig = np.fft.fftshift(np.fft.fft2(frame_orig)); psd_orig = np.abs(fft_orig)**2
        fft_sim = np.fft.fftshift(np.fft.fft2(frame_sim)); psd_sim = np.abs(fft_sim)**2
        fig, axes = plt.subplots(1, 2, figsize=(14, 7)); epsilon = 1e-9
        psd_orig_log = np.log10(psd_orig + epsilon); psd_sim_log = np.log10(psd_sim + epsilon)
        valid_psd_pixels = np.concatenate((psd_orig_log[np.isfinite(psd_orig_log)], psd_sim_log[np.isfinite(psd_sim_log)]))
        if valid_psd_pixels.size > 0:
             psd_min = np.percentile(valid_psd_pixels, 1)
             psd_max = np.percentile(valid_psd_pixels, 99.9)
        else: psd_min, psd_max = 0, 1
        im0 = axes[0].imshow(psd_orig_log, cmap='magma', aspect='equal', vmin=psd_min, vmax=psd_max)
        axes[0].set_title(f'Log(PSD) - Original (Frame {frame_idx})'); axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(psd_sim_log, cmap='magma', aspect='equal', vmin=psd_min, vmax=psd_max)
        axes[1].set_title(f'Log(PSD) - Simulated ({title_suffix})'); axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(filename.replace('.tif', f'_frame{frame_idx}_psd.png'))
        print(f"  Saved PSD comparison: {filename.replace('.tif', f'_frame{frame_idx}_psd.png')}")
        plt.close(fig)
    except IndexError: print(f"    Error: comparison_frame_idx {frame_idx} out of bounds.")
    except Exception as e: print(f"    Error saving PSD comparison: {e}")

def distort_gt(gt_stack, background_level, sigma=1.0):
    print(f"    Applying GT distortion: Gaussian blur sigma={sigma:.2f}")
    gt_signal = gt_stack - background_level
    blurred_signal = gaussian_filter(gt_signal, sigma=(0, sigma, sigma))
    distorted_gt = blurred_signal + background_level
    return distorted_gt.astype(np.float32)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created base output directory: {args.output_dir}")

    print("--- Loading Input Data ---")
    try:
        gt_video_stack_pristine = tifffile.imread(args.gt_video_in).astype(np.float32)
        print(f"Loaded pristine ground truth video: {gt_video_stack_pristine.shape}, dtype={gt_video_stack_pristine.dtype}")
        
        noise_stack_residual = None
        if args.noise_model == 'norm_residual':
            noise_stack_residual = tifffile.imread(args.residual_noise_in).astype(np.float32)
            print(f"Loaded residual noise video: {noise_stack_residual.shape}, dtype={noise_stack_residual.dtype}")
        
        ret_orig, frames_original_list = cv2.imreadmulti(args.original_video_in, flags=cv2.IMREAD_UNCHANGED)
        if not ret_orig or not frames_original_list:
            raise ValueError(f"Could not load original video from {args.original_video_in}.")
        original_stack = np.array(frames_original_list)
        original_dtype = original_stack.dtype
        print(f"Loaded original video: {original_stack.shape}, dtype={original_dtype}")
        
        if noise_stack_residual is not None and not (gt_video_stack_pristine.shape == noise_stack_residual.shape == original_stack.shape):
             raise ValueError("ERROR: Input video dimensions do not match!")
        elif not (gt_video_stack_pristine.shape == original_stack.shape):
             raise ValueError("ERROR: GT video and Original video dimensions do not match!")
             
        num_frames, height, width = gt_video_stack_pristine.shape
        original_pixels_flat = original_stack.flatten()
    except Exception as e:
        print(f"Error loading input files: {e}"); exit()

    print("\n--- Estimating Noise Parameters from Original Video ---")
    gain_g = None; sigma_read = None; background_level = None; target_bg_std_orig = None
    
    if not args.noise_regions or len(args.noise_regions) % 4 != 0:
        print(f"ERROR: --noise_regions must be provided in multiples of 4 (x1 y1 x2 y2). Got: {args.noise_regions}")
        exit()
    noise_regions_parsed = [tuple(args.noise_regions[i:i+4]) for i in range(0, len(args.noise_regions), 4)]
    print(f"Using noise regions: {noise_regions_parsed}")
    
    try:
        print("Analyzing background level and variance...")
        background_level, _, target_bg_std_orig = analyze_noise_regions(
            original_stack, noise_regions_parsed, plot=False
        )
        if background_level is None or target_bg_std_orig is None:
            raise ValueError("Background analysis failed.")
            
        print("\nAnalyzing variance vs mean for gain/read noise...")
        gain_ptc_plot_path = os.path.join(args.output_dir, "variance_vs_mean_fit.png")
        gain_g, read_noise_variance = analyze_intensity_variance_relationship(
            original_stack, background_level,
            patch_size=args.patch_size,
            use_robust_regression=args.robust_regression,
            plot=True,
            save_path=gain_ptc_plot_path
        )
        
        if gain_g is not None and gain_g > 1e-7 and read_noise_variance is not None:
            sigma_read = np.sqrt(read_noise_variance)
            print(f"Using Estimated: Gain={gain_g:.3f}, ReadNoiseStdDev={sigma_read:.2f}")
        else:
            print(f"Warning: Gain/ReadNoise estimation failed or invalid gain ({gain_g}). Using fallback.")
            sigma_read = target_bg_std_orig
            gain_g = 1e9
            print(f"Fallback: Using TotalStdDev={sigma_read:.2f} for Gaussian noise, disabling Poisson.")
            
    except Exception as e:
        print(f"Fatal Error during noise parameter estimation: {e}"); exit()

    noise_normalized_residual = None
    if args.noise_model == 'norm_residual':
        print("\n--- Pre-calculating Normalized Residual Noise ---")
        if noise_stack_residual is None:
            print(f"Error: --noise_model is 'norm_residual' but --residual_noise_in was not provided or failed to load.")
            exit()
        try:
            print("Analyzing residual noise stats...")
            bg_lvl_resid, _, bg_std_residual = analyze_noise_regions(noise_stack_residual, noise_regions_parsed)
            if bg_std_residual is not None and bg_std_residual > 1e-6:
                print(f"Residual Median={bg_lvl_resid:.2f}, Residual StdDev={bg_std_residual:.2f}")
                noise_normalized_residual = (noise_stack_residual - bg_lvl_resid) / bg_std_residual
                print("Successfully calculated normalized residual noise.")
            else:
                print("Warning: Could not reliably estimate residual noise std dev. 'norm_residual' scenario will fail.")
        except Exception as e:
            print(f"Error analyzing residual noise, 'norm_residual' scenario will be skipped: {e}")
            noise_normalized_residual = None

    summary_data = []
    summary_file_path = os.path.join(args.output_dir, f"{args.scenario_name}_summary.csv")

    scenarios = [
        {
            'name': args.scenario_name,
            'noise_model': args.noise_model,
            'scales': args.scales,
            'distort_gt': args.distort_gt,
            'gt_distort_sigma': args.gt_distort_sigma
        }
    ]
    print("\n--- Generating Simulation Scenario ---")

    for scenario in scenarios:
        scenario_name = scenario['name']
        noise_model = scenario['noise_model']
        scales = scenario['scales']
        distort_gt_flag = scenario['distort_gt']
        print(f"\n>>> Processing Scenario: {scenario_name} <<<")

        scenario_dir = os.path.join(args.output_dir, scenario_name)
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)

        if distort_gt_flag:
            dist_sigma = scenario.get('gt_distort_sigma', 1.0)
            gt_video_to_use = distort_gt(gt_video_stack_pristine, background_level, sigma=dist_sigma)
            gt_desc = f"DistortedGT(sigma={dist_sigma:.2f})"
        else:
            gt_video_to_use = gt_video_stack_pristine
            gt_desc = "PristineGT"
        print(f"  Using Ground Truth: {gt_desc}")
        
        if noise_model == 'norm_residual' and noise_normalized_residual is None:
            print(f"  Skipping scenario {scenario_name}: Normalized residual noise not available.")
            continue

        for scale in scales:
            print(f"\n  -- Processing Scale (Variance Factor): {scale:.2f} --")
            output_basename = f"sim_{scenario_name}_scale_{scale:.2f}"
            output_sim_video_path = os.path.join(scenario_dir, f"{output_basename}.tif")
            
            noise_std_scaling = math.sqrt(scale)

            try:
                if noise_model == 'gaussian':
                    print(f"      Model: Artificial Gaussian")
                    base_gaussian_total_noise = np.random.normal(loc=0.0, scale=target_bg_std_orig, size=gt_video_to_use.shape).astype(np.float32)
                    final_noise = base_gaussian_total_noise * noise_std_scaling
                    noise_desc = f"Artificial Gaussian (BaseStd={target_bg_std_orig:.2f} * {noise_std_scaling:.2f})"
                    simulated_float_stack = gt_video_to_use + final_noise

                elif noise_model == 'gp_estimated':
                    print(f"      Model: Poisson-Gaussian (Estimated)")
                    if gain_g < 1e-7 : 
                         print("      Using fallback: Applying scaled Gaussian noise only.")
                         current_gaussian_noise = np.random.normal(loc=0.0, scale=(sigma_read * noise_std_scaling), size=gt_video_to_use.shape).astype(np.float32)
                         final_noise = current_gaussian_noise
                         noise_desc = f"Fallback Gaussian (BaseStd={sigma_read:.2f} * {noise_std_scaling:.2f})"
                         simulated_float_stack = gt_video_to_use + final_noise
                    else:
                         # 1. Scale the READOUT noise std dev
                         scaled_sigma_read = sigma_read * noise_std_scaling
                         current_gaussian_noise = np.random.normal(loc=0.0, scale=scaled_sigma_read, size=gt_video_to_use.shape).astype(np.float32)
                         
                         # 2. Apply Poisson shot noise
                         gt_non_negative = np.maximum(0, gt_video_to_use - background_level) # Signal above background
                         
                         # Convert signal (ADU) to photons
                         lambda_photons = gt_non_negative * gain_g
                         
                         # Poisson process
                         poisson_realization_photons = np.random.poisson(lam=lambda_photons).astype(np.float32)
                         
                         # Convert back to ADU (signal part)
                         poisson_realization_adu = poisson_realization_photons / gain_g
                         
                         # 3. Combine: GT Background + Poisson Signal + Readout Noise
                         simulated_float_stack = background_level + poisson_realization_adu + current_gaussian_noise
                         
                         noise_desc = f"G+P Est (g={gain_g:.3f}, ReadStd={sigma_read:.2f} * {noise_std_scaling:.2f})"

                elif noise_model == 'norm_residual':
                    print(f"      Model: Normalized Residual")
                    if noise_normalized_residual is None: continue
                    final_noise = noise_normalized_residual * target_bg_std_orig * noise_std_scaling
                    noise_desc = f"Norm. Residual (TargetStd={target_bg_std_orig:.2f} * {noise_std_scaling:.2f})"
                    simulated_float_stack = gt_video_to_use + final_noise
                
                else:
                     print(f"    Skipping unknown noise model: {noise_model}")
                     continue

                min_val, max_val = 0, np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else (0, 1)
                simulated_clipped_stack = np.clip(simulated_float_stack, min_val, max_val)
                simulated_uint16_stack = simulated_clipped_stack.astype(original_dtype)

                tifffile.imwrite(output_sim_video_path, simulated_uint16_stack, imagej=True, metadata={'axes': 'TYX'})
                print(f"    Saved simulation video: {output_sim_video_path}")

                print("    Running validations...")
                print("      Analyzing background stats...")
                sim_bg_lvl, _, sim_bg_std = analyze_noise_regions(simulated_uint16_stack, noise_regions_parsed, plot=False)
                summary_data.append({
                    'Scenario': scenario_name, 'Scale': scale, 'GT_Type': gt_desc,
                    'Noise_Model': noise_desc, 'Output_Video': output_sim_video_path,
                    'Sim_BG_Median': sim_bg_lvl, 'Sim_BG_StdDev': sim_bg_std,
                    'Orig_BG_Median': background_level, 'Orig_BG_StdDev': target_bg_std_orig
                })

                print("      Generating histogram...")
                sim_pixels_flat = simulated_uint16_stack.flatten()
                hist_min = min(np.percentile(original_pixels_flat, 0.1), np.percentile(sim_pixels_flat, 0.1))
                hist_max = max(np.percentile(original_pixels_flat, 99.9), np.percentile(sim_pixels_flat, 99.9))
                if np.isclose(hist_min, hist_max): hist_min-=1; hist_max+=1
                hist_bins = np.linspace(hist_min, hist_max, 100)
                save_histogram_comparison(original_pixels_flat, sim_pixels_flat, hist_bins,
                                          f'Intensity Distribution ({scenario_name} Scale={scale:.2f})',
                                          output_sim_video_path)

                print("      Generating visual comparison...")
                save_visual_comparison(original_stack[args.plot_frame_idx],
                                       simulated_uint16_stack[args.plot_frame_idx],
                                       args.plot_frame_idx, f"{scenario_name} Scale={scale:.2f}",
                                       output_sim_video_path)

                print("      Generating PSD comparison...")
                save_psd_comparison(original_stack[args.plot_frame_idx],
                                    simulated_uint16_stack[args.plot_frame_idx],
                                    args.plot_frame_idx, f"{scenario_name} Scale={scale:.2f}",
                                    output_sim_video_path)

            except Exception as e:
                print(f"    ERROR processing scale {scale} for scenario {scenario_name}: {e}")

    print("\n--- Saving Summary Statistics ---")
    try:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file_path, index=False)
        print(f"Summary data saved to: {summary_file_path}")
    except Exception as e:
        print(f"Error saving summary data: {e}")

    print(f"\n--- Noise Addition Script Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply noise models to a ground truth video.")

    # --- I/O Arguments ---
    parser.add_argument("--gt_video_in", type=str, required=True,
                        help="Path to the input pristine ground truth TIF video (from create_ground_truth.py).")
    parser.add_argument("--original_video_in", type=str, required=True,
                        help="Path to the *original experimental* TIF video (for noise analysis).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the base directory to save noisy videos and validation plots.")
    parser.add_argument("--residual_noise_in", type=str, default=None,
                        help="Path to the residual noise video (required for 'norm_residual' model).")

    # --- Analysis Arguments ---
    parser.add_argument("--noise_regions", type=int, nargs='+', required=True,
                        help="List of noise region coordinates (x1 y1 x2 y2 ...) for noise analysis.")
    parser.add_argument("--patch_size", type=int, default=32,
                        help="Patch size for gain estimation (photon transfer).")
    parser.add_argument("--robust_regression", action='store_true',
                        help="Use robust Theil-Sen regression for gain estimation instead of OLS.")
    parser.add_argument("--plot_frame_idx", type=int, default=0,
                        help="Index of the frame to use for visual/PSD comparison plots.")

    # --- Scenario Arguments ---
    parser.add_argument("--scenario_name", type=str, default="Gauss_Poisson_Est",
                        help="Name for the simulation scenario (used for output folder).")
    parser.add_argument("--noise_model", type=str, default="gp_estimated",
                        choices=['gp_estimated', 'gaussian', 'norm_residual'],
                        help="The noise model to apply.")
    parser.add_argument("--scales", type=float, nargs='+',
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                        help="List of variance scaling factors for the readout noise.")
    parser.add_argument("--distort_gt", action='store_true',
                        help="Apply a slight Gaussian blur to the GT video before adding noise.")
    parser.add_argument("--gt_distort_sigma", type=float, default=1.0,
                        help="Sigma for the GT distortion blur (if --distort_gt is used).")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run main function
    args = parse_args()
    main(args)
