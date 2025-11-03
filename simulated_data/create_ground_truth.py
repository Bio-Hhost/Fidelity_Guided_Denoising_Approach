import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import j1 
from scipy.signal import convolve2d
try:
    from skimage.draw import line_aa 
    print("Successfully imported skimage.draw.line_aa")
except ImportError:
    print("ERROR: scikit-image not found or skimage.draw module missing.")
    print("Please ensure scikit-image is installed correctly ('pip install scikit-image')")
    exit()
import tifffile
import os
import time
import traceback
import argparse 

print(f"Script started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# --- PSF Parameters (Airy focus) ---
LAMBDA_EM = 0.571  # Emission wavelength in micrometers (µm)
NA = 1.49          # Numerical Aperture of the objective
N_MEDIUM = 1.33    # Refractive index of the sample medium (water/buffer)
PIXEL_SIZE_XY_UM = 1.0 / 3.9 # approx 0.2564 µm/pixel


def validate_region(region, frame_shape):
    height, width = frame_shape
    try:
        x1, y1, x2, y2 = map(int, region)
    except (ValueError, TypeError):
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height: return False
    if (x2 - x1) < 5 or (y2 - y1) < 5: return False 
    return True

def analyze_noise_regions(frames, noise_regions, plot=False, plot_title_suffix="", save_path=None):
    is_sequence = isinstance(frames, (list, tuple))
    is_stack = isinstance(frames, np.ndarray) and frames.ndim == 3
    if not (is_sequence or is_stack):
        raise TypeError("Input 'frames' must be sequence or 3D stack.")

    frames_sequence = [frame for frame in frames] if is_stack else frames
    if not frames_sequence: raise ValueError("Input frames sequence is empty.")
    if frames_sequence[0] is None or not isinstance(frames_sequence[0], np.ndarray):
        raise ValueError("First frame invalid.")

    try:
        height, width = frames_sequence[0].shape
    except Exception as e:
        raise ValueError(f"Could not determine shape from first frame. Error: {e}")

    valid_regions = [r for region in noise_regions if validate_region(r := tuple(map(int, region)), (height, width))]
    if not valid_regions: raise ValueError("No valid noise regions found")

    pixel_collector = [frame[y1:y2, x1:x2].flatten()
                       for x1, y1, x2, y2 in valid_regions
                       for frame in frames_sequence
                       if frame[y1:y2, x1:x2].size > 0]

    if not pixel_collector: raise ValueError("No valid pixels found")

    noise_pixels = np.concatenate(pixel_collector)
    background_level = np.median(noise_pixels)
    mad = np.median(np.abs(noise_pixels - background_level))
    gaussian_std_dev = 0.0 if mad < 1e-9 else mad * 1.4826
    gaussian_variance = gaussian_std_dev ** 2

    print(f"  Background Level (Median): {background_level:.2f}")
    print(f"  Est. Noise Variance (from MAD): {gaussian_variance:.2f}")
    print(f"  Est. Noise Std Dev (from MAD): {gaussian_std_dev:.2f}")

    return background_level, gaussian_variance, gaussian_std_dev

def draw_psf_spot(canvas, x0, y0, amplitude_total,
                  # PSF params
                  lambda_em, na, n_medium, pixel_size_xy_um, kernel_size,
                  # Brightness-Size Coupling & Randomness params
                  apply_brightness_blur=False,
                  blur_sigma_base_mean=0.0,
                  base_sigma_std_min=0.0,    # Std dev control params
                  base_sigma_std_scaler=0.0,
                  base_sigma_std_max=0.5,
                  blur_sigma_scaler=0.0,    # Deterministic control params
                  amp_reference=1.0,
                  max_add_sigma=1.0,
                  # Motion blur params passed via dict
                  motion_blur_params_dict={}):
    if amplitude_total <= 0: return 0, 0, 0, 0

    height, width = canvas.shape
    k = (2 * np.pi) / lambda_em

    roi_hw_half = kernel_size // 2
    x0_int, y0_int = int(round(x0)), int(round(y0))
    x_min = max(0, x0_int - roi_hw_half); x_max = min(width, x0_int + roi_hw_half + 1)
    y_min = max(0, y0_int - roi_hw_half); y_max = min(height, y0_int + roi_hw_half + 1)
    roi_shape = (y_max - y_min, x_max - x_min)
    if roi_shape[0] <= 0 or roi_shape[1] <= 0: return 0, 0, 0, 0

    try:
        x_coords_roi = np.arange(x_min, x_max); y_coords_roi = np.arange(y_min, y_max)
        xx_roi, yy_roi = np.meshgrid(x_coords_roi, y_coords_roi)
        dx_physical = (xx_roi - x0) * pixel_size_xy_um; dy_physical = (yy_roi - y0) * pixel_size_xy_um
        r_physical = np.sqrt(dx_physical**2 + dy_physical**2)
        arg_factor = (k * na) / n_medium
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = np.divide(r_physical * arg_factor, 1.0, where=r_physical > 0, out=np.zeros_like(r_physical))
            j1_arg = j1(arg)
            airy_term = np.divide(2.0 * j1_arg, arg, where=arg!=0, out=np.ones_like(arg))
        psf_patch_unnormalized = airy_term**2
        psf_patch_unnormalized[np.isnan(psf_patch_unnormalized)] = 0 

    except Exception as e: print(f"ERROR calculating Airy disk at ({x0:.1f},{y0:.1f}): {e}"); return 0,0,0,0

    total_applied_sigma = 0.0
    current_spot_base_sigma = blur_sigma_base_mean
    additional_sigma = 0.0

    if apply_brightness_blur and amp_reference > 0:
        log_ratio = 0.0
        try:
            if amplitude_total > 1e-9: log_ratio = np.log10(amplitude_total / amp_reference)
            else: log_ratio = -np.inf
        except Exception as e_log: print(f"Warning: Error calculating log_ratio: {e_log}")

        current_spot_std_dev = base_sigma_std_min + base_sigma_std_scaler * max(0, log_ratio)
        current_spot_std_dev = min(current_spot_std_dev, base_sigma_std_max)
        current_spot_std_dev = max(0, current_spot_std_dev)

        if current_spot_std_dev > 1e-6:
             current_spot_base_sigma = np.random.normal(loc=blur_sigma_base_mean, scale=current_spot_std_dev)
        else:
             current_spot_base_sigma = blur_sigma_base_mean
        current_spot_base_sigma = max(0.1, current_spot_base_sigma)

        try:
            additional_sigma = blur_sigma_scaler * max(0, log_ratio)
            additional_sigma = min(additional_sigma, max_add_sigma)
        except Exception as e_add_sig: print(f"Warning: Error calculating additional_sigma: {e_add_sig}")

    total_applied_sigma = current_spot_base_sigma + additional_sigma

    if total_applied_sigma > 0.05:
        try:
            k_size = int(np.ceil(total_applied_sigma * 6)) | 1; k_size = max(3, k_size)
            k_hw = k_size // 2; ky, kx = np.indices((k_size, k_size)) - k_hw
            with np.errstate(over='ignore'):
                 exp_term = -(kx**2 + ky**2) / (2 * total_applied_sigma**2)
                 g_kernel = np.exp(exp_term)
            kernel_sum = np.sum(g_kernel)
            if kernel_sum > 1e-9:
                g_kernel /= kernel_sum
                psf_patch_unnormalized = convolve2d(psf_patch_unnormalized, g_kernel,
                                                    mode='same', boundary='fill', fillvalue=0)
        except Exception as e_gauss_blur:
            print(f"Warning: Error during Gaussian blur convolution at ({x0:.1f},{y0:.1f}): {e_gauss_blur}")

    motion_dx_pix, motion_dy_pix, motion_length_pix = 0.0, 0.0, 0.0
    final_psf_patch = psf_patch_unnormalized

    apply_blur = motion_blur_params_dict.get('apply_motion_blur', False)
    if apply_blur:
        try:
            exp_time_s=motion_blur_params_dict['exp_time_s']; diff_coeff_um2_s=motion_blur_params_dict['diff_coeff_um2_s']; fixed_blur_pixels=motion_blur_params_dict['fixed_blur_pixels'];
            if diff_coeff_um2_s > 0 and exp_time_s > 0: motion_length_pix = np.sqrt(4*diff_coeff_um2_s*exp_time_s)/pixel_size_xy_um
            elif fixed_blur_pixels > 0: motion_length_pix = fixed_blur_pixels
            if motion_length_pix > 0.1:
                 theta = np.random.uniform(0, 2 * np.pi); motion_dx_pix = motion_length_pix*np.cos(theta); motion_dy_pix = motion_length_pix*np.sin(theta)
                 line_kernel_size = int(np.ceil(motion_length_pix))+2;
                 if line_kernel_size % 2 == 0: line_kernel_size += 1
                 line_kernel_half = line_kernel_size // 2; line_kernel = np.zeros((line_kernel_size, line_kernel_size), dtype=np.float32)
                 x_start_lk = line_kernel_half - motion_dx_pix/2.0; y_start_lk = line_kernel_half - motion_dy_pix/2.0
                 x_end_lk = line_kernel_half + motion_dx_pix/2.0; y_end_lk = line_kernel_half + motion_dy_pix/2.0
                 rr, cc, val = line_aa(int(round(y_start_lk)), int(round(x_start_lk)), int(round(y_end_lk)), int(round(x_end_lk)))
                 valid_idx = (rr >= 0) & (rr < line_kernel_size) & (cc >= 0) & (cc < line_kernel_size); rr, cc, val = rr[valid_idx], cc[valid_idx], val[valid_idx]
                 if len(rr) > 0:
                      line_kernel[rr, cc] = val; line_kernel_sum = np.sum(line_kernel)
                      if line_kernel_sum > 1e-6:
                          line_kernel /= line_kernel_sum
                          final_psf_patch = convolve2d(final_psf_patch, line_kernel, mode='same', boundary='fill', fillvalue=0)
        except NameError: print("ERROR: skimage.draw.line_aa not available. Motion blur skipped.")
        except KeyError as e_blur: print(f"Warning: Motion blur param missing: {e_blur}.")
        except Exception as e_blur_generic: print(f"Warning: Error during motion blur: {e_blur_generic}.")

    patch_sum = np.sum(final_psf_patch)
    if patch_sum > 1e-9: psf_patch_normalized = final_psf_patch / patch_sum
    else: return 0, 0, 0, total_applied_sigma
    scaled_patch = psf_patch_normalized * amplitude_total

    kernel_center_y, kernel_center_x = kernel_size // 2, kernel_size // 2
    rel_y_min = y_min - (y0_int - kernel_center_y)
    rel_x_min = x_min - (x0_int - kernel_center_x)
    patch_slice_y_start = max(0, rel_y_min)
    patch_slice_x_start = max(0, rel_x_min)
    patch_slice_y_end = patch_slice_y_start + roi_shape[0]
    patch_slice_x_end = patch_slice_x_start + roi_shape[1]

    canvas_slice_y_start = y_min
    canvas_slice_x_start = x_min
    canvas_slice_y_end = y_max
    canvas_slice_x_end = x_max

    try:
        patch_slice_y_end = min(patch_slice_y_end, scaled_patch.shape[0])
        patch_slice_x_end = min(patch_slice_x_end, scaled_patch.shape[1])
        patch_to_add = scaled_patch[patch_slice_y_start:patch_slice_y_end, patch_slice_x_start:patch_slice_x_end]
        target_roi = canvas[canvas_slice_y_start:canvas_slice_y_end, canvas_slice_x_start:canvas_slice_x_end]

        if target_roi.shape == patch_to_add.shape:
            canvas[canvas_slice_y_start:canvas_slice_y_end, canvas_slice_x_start:canvas_slice_x_end] += patch_to_add
        else:
            min_h = min(target_roi.shape[0], patch_to_add.shape[0])
            min_w = min(target_roi.shape[1], patch_to_add.shape[1])
            print(f"Warning: Shape mismatch adding PSF at ({x0:.1f},{y0:.1f}). Target: {target_roi.shape}, Patch: {patch_to_add.shape}. Using ({min_h},{min_w}).")
            canvas[canvas_slice_y_start:canvas_slice_y_start+min_h, canvas_slice_x_start:canvas_slice_x_start+min_w] += patch_to_add[:min_h, :min_w]

    except IndexError as e_idx: print(f"ERROR: IndexError adding PSF at ({x0:.1f},{y0:.1f}). ROI:({y_min}:{y_max}, {x_min}:{x_max}), PatchSlice:({patch_slice_y_start}:{patch_slice_y_end}, {patch_slice_x_start}:{patch_slice_x_end}), ScaledPatchShape: {scaled_patch.shape}. Error: {e_idx}"); return 0, 0, 0, total_applied_sigma
    except ValueError as e_val: print(f"ERROR: ValueError adding PSF at ({x0:.1f},{y0:.1f}). TargetShape: {target_roi.shape}, PatchShape: {patch_to_add.shape}. Error: {e_val}"); return 0, 0, 0, total_applied_sigma
    except Exception as e_gen: print(f"ERROR: Generic error adding PSF at ({x0:.1f},{y0:.1f}): {e_gen}"); return 0, 0, 0, total_applied_sigma

    return motion_dx_pix, motion_dy_pix, motion_length_pix, total_applied_sigma


def main(args):
    print(f"Loading original video from: {args.video_in}")
    try:
        with tifffile.TiffFile(args.video_in) as tif:
            frames_original_stack = tif.asarray()
            if frames_original_stack.ndim == 2:
                frames_original_stack = frames_original_stack[np.newaxis, :, :]
            if frames_original_stack.ndim != 3:
                raise ValueError(f"Expected 2D/3D TIF, got {frames_original_stack.ndim}D.")
        num_frames, height, width = frames_original_stack.shape
        original_dtype = frames_original_stack.dtype
        print(f"Loaded {num_frames} frames. Dims: {height}x{width}, dtype={original_dtype}")
    except Exception as e:
        print(f"Error loading original video: {e}"); exit()

    print("\nEstimating background noise parameters...")
    if not args.noise_regions or len(args.noise_regions) % 4 != 0:
        print(f"ERROR: --noise_regions must be provided in multiples of 4 (x1 y1 x2 y2). Got: {args.noise_regions}")
        exit()
    noise_regions_parsed = [tuple(args.noise_regions[i:i+4]) for i in range(0, len(args.noise_regions), 4)]
    print(f"Using noise regions: {noise_regions_parsed}")

    background_level, _, background_std_dev = analyze_noise_regions(frames_original_stack, noise_regions_parsed, plot=False)
    if background_level is None or background_std_dev is None:
        print("Error: Failed to estimate background parameters."); exit()

    print(f"\nLoading spot data from: {args.spots_in}")
    try:
        spots_df = pd.read_csv(args.spots_in)
        print(f"Loaded {len(spots_df)} spot detections.")
        col_map = {'X': 'POSITION_X', 'Y': 'POSITION_Y', 'Slice': 'FRAME'}
        for old, new in col_map.items():
            if old in spots_df.columns and new not in spots_df.columns:
                spots_df.rename(columns={old: new}, inplace=True)
        
        required = ['POSITION_X', 'POSITION_Y', 'FRAME', args.intensity_col]
        if not all(c in spots_df.columns for c in required):
            raise ValueError(f"CSV missing columns: {[c for c in required if c not in spots_df.columns]}")
        
        print("Cleaning columns...")
        cols_convert = required + (['RADIUS'] if 'RADIUS' in spots_df.columns else [])
        for col in cols_convert:
            if col in spots_df.columns:
                spots_df[col] = pd.to_numeric(spots_df[col], errors='coerce')
            if col == 'FRAME':
                spots_df[col] = spots_df[col].astype(pd.Int64Dtype()) 
        
        original_rows = len(spots_df)
        spots_df.dropna(subset=[c for c in required if c in spots_df.columns], inplace=True)
        removed = original_rows - len(spots_df)
        if removed > 0: print(f"Removed {removed} rows due to NaNs in required columns.")
        
        if 'FRAME' in spots_df.columns and spots_df['FRAME'].notna().all():
             spots_df['FRAME'] = spots_df['FRAME'].astype(int)
        else:
             print("Warning: NaNs remain in FRAME column after initial dropna, cannot convert to int.")
             spots_df.dropna(subset=['FRAME'], inplace=True)
             if spots_df['FRAME'].notna().all():
                 spots_df['FRAME'] = spots_df['FRAME'].astype(int)
             else:
                 raise ValueError("Could not clean FRAME column to integer type.")

        print(f"Processing {len(spots_df)} spots."); print("Calculating amplitude reference...")
        amps = spots_df[args.intensity_col] - background_level
        pos_amps = amps[amps > 0]
        if len(pos_amps) > 0:
            amplitude_reference = np.median(pos_amps)
            print(f"  Using Median Positive Amplitude Ref: {amplitude_reference:.2f}")
        else:
            amplitude_reference = 1.0
            print("Warning: No positive amplitudes found. Using fallback ref=1.0")
        
        globals()['amplitude_reference'] = amplitude_reference
        
    except Exception as e:
        print(f"Error loading/cleaning spots CSV: {e}"); traceback.print_exc(); exit()

    print(f"\nInitializing ground truth video with background level: {background_level:.2f}")
    gt_video = np.full((num_frames, height, width), background_level, dtype=np.float32)
    ground_truth_spot_data = []
    gt_spot_id_counter = 0

    psf_params = {
        'lambda_em': LAMBDA_EM, 'na': NA, 'n_medium': N_MEDIUM,
        'pixel_size_xy_um': PIXEL_SIZE_XY_UM, 'kernel_size': args.kernel_size
    }
    brightness_blur_params = {
        'apply_brightness_blur': True, 
        'blur_sigma_base_mean': args.blur_sigma_base,
        'base_sigma_std_min': args.blur_std_min,
        'base_sigma_std_scaler': args.blur_std_scale,
        'base_sigma_std_max': args.blur_std_max,
        'blur_sigma_scaler': args.blur_bright_scale,
        'amp_reference': amplitude_reference, 
        'max_add_sigma': args.blur_max_add
    }
    motion_blur_params_dict = {
        'apply_motion_blur': True, 
        'exp_time_s': args.exposure,
        'diff_coeff_um2_s': args.diff_coeff,
        'fixed_blur_pixels': 0.0 
    }
    print("\nUsing PSF Parameters:"); [print(f"  {k}: {v}") for k, v in psf_params.items()]
    print("\nUsing Brightness-Size Coupling & Correlated Randomness Parameters:"); [print(f"  {k}: {v}") for k, v in brightness_blur_params.items()]
    print("\nUsing Motion Blur Parameters:"); [print(f"  {k}: {v}") for k, v in motion_blur_params_dict.items()]

    print(f"\nDrawing {len(spots_df)} PSF spots...")
    print(f"  Intensity Adjustment Enabled: Mode='scale'")
    print(f"    Global Scaling Factor: {args.intensity_scale}")

    spots_drawn = 0; skipped_nan = 0; skipped_low_amp = 0; skipped_other_error = 0
    draw_start_time = time.time()

    for index, spot in spots_df.iterrows():
        x = spot['POSITION_X']; y = spot['POSITION_Y']
        frame_idx = spot['FRAME']; intensity_measure = spot[args.intensity_col]

        if pd.isna(x) or pd.isna(y) or pd.isna(frame_idx) or pd.isna(intensity_measure):
            skipped_nan += 1; continue

        try:
            try: frame_idx = int(frame_idx)
            except (ValueError, TypeError):
                print(f"Warning: Skipping spot index {index} due to invalid FRAME value: {frame_idx}")
                skipped_nan += 1
                continue

            if not (0 <= frame_idx < num_frames):
                skipped_other_error += 1
                continue

            amplitude_original = intensity_measure - background_level
            adjusted_amplitude = amplitude_original

            if amplitude_original > 0:
                adjusted_amplitude = amplitude_original * args.intensity_scale
                adjusted_amplitude = max(0, adjusted_amplitude)

            if adjusted_amplitude > 1e-6 :
                motion_dx, motion_dy, motion_len, applied_sigma = draw_psf_spot(
                    canvas=gt_video[frame_idx], x0=x, y0=y,
                    amplitude_total=adjusted_amplitude, 
                    **psf_params,
                    **brightness_blur_params,
                    motion_blur_params_dict=motion_blur_params_dict
                )
                spots_drawn += 1

                spot_info = {
                    'GT_SPOT_ID': gt_spot_id_counter, 'FRAME': frame_idx,
                    'POSITION_X': x, 'POSITION_Y': y,
                    'APPLIED_TOTAL_BLUR_SIGMA': applied_sigma,
                    'GT_AMPLITUDE_DRAWN': adjusted_amplitude,
                    'SOURCE_AMPLITUDE_CALC': amplitude_original,
                    'GT_BACKGROUND_LEVEL': background_level,
                    'SNR_ESTIMATE': adjusted_amplitude / background_std_dev if background_std_dev > 1e-6 else np.inf,
                    'MOTION_BLUR_DX_PIX': motion_dx, 'MOTION_BLUR_DY_PIX': motion_dy, 'MOTION_BLUR_LEN_PIX': motion_len,
                    'SOURCE_CSV_INDEX': index, 'SOURCE_INTENSITY_MEASURE': intensity_measure, 'SOURCE_INTENSITY_COLUMN': args.intensity_col
                }
                ground_truth_spot_data.append(spot_info)
                gt_spot_id_counter += 1
            elif amplitude_original > 0:
                 skipped_low_amp += 1

            total_processed = index + 1
            if total_processed > 0 and total_processed % 1000 == 0:
                elapsed = time.time() - draw_start_time
                print(f"  Processed {total_processed}/{len(spots_df)} spots input [{elapsed:.1f}s]...")

        except Exception as e:
            skipped_other_error += 1
            print(f"\n--- ERROR processing spot index {index} (Frame {frame_idx}, Pos ({x:.1f},{y:.1f})) ---")
            traceback.print_exc()
            print("--- Continuing ---")

    draw_time = time.time() - draw_start_time
    total_accounted = spots_drawn + skipped_nan + skipped_low_amp + skipped_other_error
    print("\n--- Drawing Summary ---")
    print(f"Total spots from CSV: {len(spots_df)}")
    print(f"Total accounted for (drawn + skipped): {total_accounted}")
    print(f"  Successfully drawn (final amplitude > 0): {spots_drawn}")
    print(f"  Skipped (NaN in essential cols or invalid frame): {skipped_nan}")
    print(f"  Skipped (Original or Adjusted Amplitude <= 0): {skipped_low_amp}")
    print(f"  Skipped (Other processing error): {skipped_other_error}")
    print(f"Total time for drawing loop: {draw_time:.2f}s")
    print(f"Ground truth spots recorded: {len(ground_truth_spot_data)}")

    print("\n--- DEBUG: Checking GT Video Data Range ---")
    try:
        finite_gt_video = gt_video[np.isfinite(gt_video)]
        if finite_gt_video.size > 0:
            min_val_gt=np.min(finite_gt_video); max_val_gt=np.max(finite_gt_video)
            mean_val_gt=np.mean(finite_gt_video); median_val_gt=np.median(finite_gt_video)
            print(f"GT Video Finite Min/Max: {min_val_gt:.2f} / {max_val_gt:.2f}")
            print(f"GT Video Finite Mean/Median: {mean_val_gt:.2f} / {median_val_gt:.2f}")
        else: print("GT Video contains no finite values!")
        nan_count = np.sum(np.isnan(gt_video)); inf_count = np.sum(np.isinf(gt_video))
        if nan_count > 0: print(f"WARNING: GT Video contains {nan_count} NaNs!")
        if inf_count > 0: print(f"WARNING: GT Video contains {inf_count} Infs!")
    except Exception as e: print(f"Error checking GT video range: {e}")
    print("--- END DEBUG ---")

    print(f"\nSaving synthetic ground truth video to: {args.video_out}")
    print(f"Video dtype: {gt_video.dtype}")
    try:
        os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
        tifffile.imwrite(args.video_out, gt_video, imagej=True, metadata={'axes': 'TYX'})
        print("Synthetic ground truth video saved successfully.")
    except Exception as e:
        print(f"Error saving ground truth video: {e}"); traceback.print_exc()

    print(f"\nSaving ground truth spot info to: {args.spots_out}")
    if ground_truth_spot_data:
        try:
            gt_spots_df = pd.DataFrame(ground_truth_spot_data)
            cols_order = [
                'GT_SPOT_ID', 'FRAME', 'POSITION_X', 'POSITION_Y',
                'APPLIED_TOTAL_BLUR_SIGMA',
                'GT_AMPLITUDE_DRAWN',
                'SOURCE_AMPLITUDE_CALC',
                'GT_BACKGROUND_LEVEL', 'SNR_ESTIMATE',
                'MOTION_BLUR_DX_PIX', 'MOTION_BLUR_DY_PIX', 'MOTION_BLUR_LEN_PIX',
                'SOURCE_CSV_INDEX', 'SOURCE_INTENSITY_MEASURE', 'SOURCE_INTENSITY_COLUMN'
            ]
            cols_order = [col for col in cols_order if col in gt_spots_df.columns]
            gt_spots_df = gt_spots_df[cols_order]

            os.makedirs(os.path.dirname(args.spots_out), exist_ok=True)
            gt_spots_df.to_csv(args.spots_out, index=False, float_format='%.4f')
            print(f"Ground truth spot info saved successfully ({len(gt_spots_df)} spots).")
        except Exception as e:
            print(f"Error saving ground truth spot info CSV: {e}"); traceback.print_exc()
    else:
        print("No valid ground truth spots were generated to save.")

    print(f"\n--- Script Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pristine ground truth videos from real spot data.")
    
    # --- I/O Arguments ---
    parser.add_argument("--video_in", type=str, required=True,
                        help="Path to the original experimental TIF video (e.g., 'Cy3_Best/Cy3 best SN 30ms per frame.tif')")
    parser.add_argument("--spots_in", type=str, required=True,
                        help="Path to the CSV file containing spot detections (e.g., 'Cy3_Best/spots.csv')")
    parser.add_argument("--video_out", type=str, required=True,
                        help="Path to save the output synthetic ground truth TIF video (e.g., 'Cy3_Best/synthetic_gt.tif')")
    parser.add_argument("--spots_out", type=str, required=True,
                        help="Path to save the output CSV file with GT spot info (e.g., 'Cy3_Best/synthetic_gt_spots.csv')")

    # --- Analysis Arguments ---
    parser.add_argument("--noise_regions", type=int, nargs='+', required=True,
                        help="List of noise region coordinates (x1 y1 x2 y2 x1 y1 x2 y2 ...)")
    parser.add_argument("--intensity_col", type=str, default="TOTAL_INTENSITY_CH1",
                        help="Name of the intensity column in the spots CSV file.")

    # --- Simulation Arguments ---
    parser.add_argument("--intensity_scale", type=float, default=0.1,
                        help="Global scaling factor to apply to spot amplitudes (Paper Sec 2.2.2 mentions 0.1).")
    parser.add_argument("--kernel_size", type=int, default=17,
                        help="Pixel size of the PSF kernel (must be odd).")
    parser.add_argument("--exposure", type=float, default=0.030,
                        help="Camera exposure time in seconds (for motion blur calc).")
    parser.add_argument("--diff_coeff", type=float, default=0.5,
                        help="Diffusion coefficient in um^2/s (for motion blur calc).")
    
    # --- Brightness-Size Coupling Arguments ---
    parser.add_argument("--blur_sigma_base", type=float, default=0.9,
                        help="Baseline Gaussian blur sigma (pixels).")
    parser.add_argument("--blur_std_min", type=float, default=0.2,
                        help="Min std dev for random component of blur.")
    parser.add_argument("--blur_std_scale", type=float, default=0.5,
                        help="Scaling factor for random component of blur std dev.")
    parser.add_argument("--blur_std_max", type=float, default=0.9,
                        help="Max std dev for random component of blur.")
    parser.add_argument("--blur_bright_scale", type=float, default=5.0,
                        help="Scaling factor for brightness-dependent deterministic blur.")
    parser.add_argument("--blur_max_add", type=float, default=12.0,
                        help="Maximum additional sigma from deterministic blur.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
