
# Evaluating Denoising Performance

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

