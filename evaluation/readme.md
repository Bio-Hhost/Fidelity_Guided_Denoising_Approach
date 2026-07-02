# Denoising Performance Evaluation Suite

## Table of Contents

- [Overview](#overview)
- [Evaluating Simulated Data](#evaluating-denoising-performance-on-simulated-data)
  - [Detection Threshold Scan](#detection-threshold-scan)
    - [Overview](#overview-1)
    - [Usage Example](#usage-example)
    - [Outputs](#outputs)
  - [AUC Plot Creation](#auc-plot-creation)
    - [Usage Example](#usage-example-1)
    - [Outputs](#outputs-1)
  - [Full Evaluation at Optimal Thresholds](#full-evaluation-at-optimal-thresholds)
    - [Overview](#overview-2)
    - [Usage Example](#usage-example-2)
    - [Outputs](#outputs-2)
- [Evaluating Experimental Data](#evaluation-on-experimental-data)
  - [Overview](#overview-3)
  - [Usage Example](#usage-example-3)
  - [Outputs](#outputs-3)

## Overview

This repository provides a suite of scripts for evaluating denoising methods on both simulated and experimental microscopy data. The evaluation pipeline includes:

- **For Simulated Data:** Detection threshold scanning, AUC analysis, and comprehensive performance metrics with ground truth comparison
- **For Experimental Data:** Real-world performance analysis using TrackMate detections as reference

---

# Evaluating Denoising Performance On Simulated Data

## Detection Threshold Scan 

After generating your datasets, you can use `evaluate_detection_threshold_scan.py` to quantitatively benchmark the performance of different denoising methods. This script automates the process described in **Section 2.4.1** of the paper.

### Overview

The script works by grouping your videos by noise level (e.g., `scale_1.0`, `scale_2.0`, etc.). For each group, it takes the noisy video and all its corresponding denoised variants and performs the following analysis:

1.  **Frame Sampling:** Randomly samples a subset of frames (e.g., 100) to speed up analysis. The *same* frames are used for all videos in a group for a fair comparison.
2.  **Blob Detection:** Uses a Laplacian of Gaussian (LoG) detector (`skimage.feature.blob_log`) to find spots in every sampled frame.
3.  **Threshold Scan:** Repeats the detection across a wide range of sensitivity thresholds.
4.  **Performance Matching:** Compares the detections at each threshold against the known ground truth spot list (`..._spot_info.csv`) to classify every detection as a **True Positive (TP)**, **False Positive (FP)**, or **False Negative (FN)**.
5.  **Metric Calculation:** Calculates **Precision**, **Recall**, and **F1-Score** for each method at each threshold.
6.  **Plotting:** Generates two key summary plots for each noise group:
    * A **Precision-Recall (PR) Curve** with the Area Under the Curve (AUC) for each method.
    * A plot of **Metrics (P, R, F1) vs. Threshold** to visualize detector sensitivity.

### Usage Example

Run this script on the output directory from Step 2.

```bash
python evaluate_detections.py \
    --gt_spots_csv "synthetic_gt_scaled_0.1_spot_info.csv" \
    --input_dir "Simulations_Output_Paper/Gauss_Poisson_Est_Paper" \
    --output_dir "Evaluation_Results" \
    --sample_frames 100 \
    --thresh_min 1.0 \
    --thresh_max 100.0 \
    --thresh_steps 20
```

### Outputs

This script will create a new sub-directory in your output_dir for each noise level it finds (e.g., Group_Scale_1.0, Group_Scale_2.0, etc.). Inside each folder, you will find:

* `detection_metrics_all_variants.csv`: A CSV file containing the raw TP, FP, FN, Precision, Recall, F1, and AUC data for every method at every threshold.
* `precision_recall_curve_comparison.png`: The summary PR curve plot, ideal for publication.
* `metrics_vs_threshold_comparison.png`: The plot of metrics vs. detector threshold.

> Important Note for Customization: The script uses the `get_method_style` function to automatically label and color the plots based on video filenames. If you use this script to evaluate your own denoising methods with different file-naming conventions, you must edit the `ALL_COLORS` dictionary and the `get_method_style` function at the top of `evaluate_detection_threshold_scan.py` to recognize your files.

---

## AUC PLOT CREATION

After running the evaluation (`evaluate_detection_threshold_scan.py`) on all your noise levels, you will have many separate `Group_Scale_...` folders. The `plot_auc_summary.py` script provides a step to aggregate all these results into a single summary plot.

This script recursively scans your evaluation output directory for all `detection_metrics_all_variants.csv` files. It reads each file, extracts the final Area Under the Curve (AUC) for every method, and plots this value against the noise scale.

The result is a single line plot showing how the performance (AUC) of each denoising method changes as the noise level increases.

### Usage Example

Run this script on the top-level directory where your evaluation results are stored.

```bash
python plot_auc_summary.py \
    --input_dir "Evaluation_Results" \
    --output_plot "Evaluation_Results/AUC_Summary_vs_Noise_Scale.png"
```

### Outputs

This will generate a single image file (AUC_Summary_vs_Noise_Scale.png in this example) showing the summary.

> This script also uses the `get_method_style` function and `ALL_COLORS` dictionary, so you must customize it if you are plotting your own methods with different naming conventions.

---

## Full Evaluation (at Optimal Thresholds)

The `evaluate_detection_threshold_scan.py` script (Step 3) is excellent for finding the optimal detection threshold for each method.

This script, `evaluate_detection_threshold_scan.py`, automates the entire in-depth analysis. It automatically combines the results from the threshold scan and then uses that information to perform the full, in-depth quantitative analysis described in **Section 2.4.1**.

### Overview

This script performs two major operations:

1.  Auto-Combination: It first scans the directory (`Evaluation_Results/`) for all the `detection_metrics_all_variants.csv` files. It combines them into a single master file (`combined_threshold_scan_results.csv`) that knows the optimal threshold (max F1-Score) for every method at every noise scale.

2.  In-Depth Analysis: It then re-loads the pristine GT video and all simulated/denoised videos. For each one, it:
    * Runs the blob detector only at that method's optimal threshold.
    * Performs 2D Gaussian fitting on all **True Positive** detections.
    * Calculates the full suite of performance metrics:
        * **Image Quality:** PSNR and SSIM.
        * **Detection:** F1, Precision, and Recall.
        * **Localization:** RMSE and Median Absolute Error (pixels).
        * **Photometry:** R-squared, Gain, and Median Absolute Error (ADU).

Then, it saves a master `evaluation_summary_all_methods.csv` file and generates summary plots for each of these key metrics vs. noise scale, giving you all the main figures for the paper.

### Usage Example

```bash
python evaluate_comprehensive.py \
    --gt_video "Cy3_Best/synthetic_gt_scaled_0.1.tif" \
    --gt_spots_csv "Cy3_Best/synthetic_gt_scaled_0.1_spot_info.csv" \
    --input_dir "Cy3_Best/Simulations_Output_Paper/Gauss_Poisson_Est_Paper" \
    --output_dir "Cy3_Best/Comprehensive_Evaluation_Results" \
    --threshold_scan_dir "Cy3_Best/Evaluation_Results" \
    --sample_frames 500 \
    --fit_region_size 7 \
    --opt_metric "F1"
```

### Outputs
This script generates the main quantitative results for your paper in the specified --output_dir:

* combined_threshold_scan_results.csv: The intermediate, combined CSV.
* evaluation_summary_all_methods.csv: The final, master CSV with all metrics (PSNR, F1, Loc_Error, etc.).
* Summary_F1_vs_Scale.png: Plot of F1-Score vs. Noise.
* Summary_PSNR_vs_Scale.png: Plot of PSNR vs. Noise.
* Summary_Loc_MedianAE_vs_Scale.png: Plot of Localization Error vs. Noise.
* Summary_Phot_MedianAE_vs_Scale.png: Plot of Photometry Error vs. Noise.
* Summary_Phot_R_squared_vs_Scale.png: Plot of Photometry R² vs. Noise.

---

---

# Evaluation on Experimental Data

This repository also includes the script `evaluate_experimental.py` to reproduce the quantitative analysis on real-world experimental data described in **Section 2.4.2** of the paper.

This script does not use a ground truth. Instead, it uses spot detections from the original noisy video (from TrackMate) as a reference to compare the quantitative properties of the noisy vs. denoised spots.

### Overview

This single, integrated script performs the entire experimental analysis and plotting pipeline.

1.  **Finds Experiments:** It scans a base directory for experiment subfolders. Each subfolder is expected to contain:
    * A noisy video (e.g., `Experiment_A.tif`)
    * A TrackMate spots list (e.g., `Experiment_A.csv`)
    * One or more denoised videos (e.g., `Experiment_A_n2v_denoised.tif`)
2.  **Per-Spot Analysis:** For every spot in the TrackMate CSV, it performs a 2D Gaussian fit on both the noisy and the denoised video at the reference coordinate.
3.  **Metric Calculation:** For each spot, it calculates the key metrics mentioned in the paper:
    * **Local Background Noise:** The robust standard deviation of the local background, calculated in an adaptive annulus around the spot.
    * **Local Background Brightness:** The median of the local background.
    * **Localization Error:** The Euclidean distance (in pixels) between the denoised fitted center and the original TrackMate coordinate.
    * **Photometry Error:** The absolute difference in fitted amplitude between the denoised spot and the original noisy spot.
4.  **Saves Detailed Results:** It saves a `_detailed_results.csv` file for each denoised video, containing the per-spot metrics.
5.  **Generates Summary Plots:** After processing all experiments, it automatically combines all results and generates the final summary bar plots, showing the median performance of each method for each metric across all experiments.

### Usage Example

Your data should be structured as follows:

```
/path/to/your/data/
├── Experiment_A/
│   ├── Experiment_A.tif
│   ├── Experiment_A.csv
│   ├── Experiment_A_n2v_denoised.tif
│   └── Experiment_A_deepcad_denoised.tif
├── Experiment_B/
│   ├── Experiment_B.tif
│   ├── Experiment_B.csv
│   └── Experiment_B_n2v_denoised.tif
...
```

Run the script by pointing it to the base directory:

```bash
python evaluate_experimental.py \
    --base_dir "/path/to/your/data/" \
    --output_dir_name "Experimental_Results" \
    --methods "N2V" "DeepCAD-RT" "λ = RL" "λ = 0.1 (T=1)" \
    --adaptive_radii \
    --spots_to_process 10000 \
    --exclude_training_data \
    --save_visuals
```

### Outputs

This script will create a new folder (e.g., Experimental_Results) inside your --base_dir. This folder will contain:

* Detailed CSVs: One ..._detailed_results.csv for each denoised video processed.
* Summary Plots: The final, publication-ready bar plots:
   * local_background_noise_reduction.png
   * local_background_mean_preservation.png
   * median_localization_error.png
   * median_photometry_error.png
* Visualizations (Optional): If --save_visuals is used, it will create a visualizations subfolder with debug images of the background annulus and Gaussian fits for a few spots.

> Important Note for Customization: 
> The script uses the METHOD_MAP dictionary at the top of evaluate_experimental.py to map filenames to display names (e.g., geo0.1 -> λ = 0.1 (T=1)). If you use this script to evaluate your own denoising methods with different file-naming conventions, you must edit this dictionary to recognize your files.
