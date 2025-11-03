This simulation pipeline is designed to create challenging, low-SNR single-molecule datasets with a known ground truth, as described in Section 2.2.2 of our paper. The process is split into two main scripts:

### 1. Ground Truth Generation (`create_ground_truth.py`)

This script generates a **pristine, noise-free** video based on spot locations and intensities from a real experimental dataset. It models fluorescent spots with high physical realism:

* **2D Airy Disk PSF:** Spots are modeled as diffraction-limited 2D Airy disks, not simple Gaussians, using the microscope's numerical aperture (NA) and emission wavelength.
* **Real Background:** A uniform background level is estimated from the experimental video and added to the pristine simulation.
* **Brightness-Size Coupling:** Brighter spots are convolved with a slightly larger Gaussian kernel to simulate optical and detector effects.
* **Motion Blur:** 2D Brownian diffusion during the camera exposure is modeled by convolving each spot with a small, randomly oriented line kernel.
* **Low-SNR Scaling:** The final amplitudes of all spots are scaled by a global factor (e.g., `0.1`) to deliberately create a challenging, low-signal scenario.

### 2. Noise Application (`add_noise_to_gt.py`)

This script corrupts the pristine ground truth video using the **mixed Poisson-Gaussian noise model** common to EMCCD/sCMOS cameras.

$$y = \frac{\text{Poisson}(\alpha \cdot x_{gt})}{\alpha} + \mathcal{N}(0, \sigma^2)$$

It first analyzes the *original experimental video* to estimate the key noise parameters:
* **Camera Gain ($\alpha$):** Estimated via Photon Transfer Curve (PTC) analysis (mean-variance plot).
* **Readout Noise ($\sigma$):** Estimated from the standard deviation of background regions and the intercept of the PTC.

It then applies this calibrated noise model to the pristine video. This script can also **scale the readout noise variance** (`--scales`) to generate a series of datasets with increasing noise levels for evaluation.

---

## Usage Workflow

To use this pipeline, you need two source files from your own experimental data:

1.  **A real video:** `my_experiment.tif`
2.  **A spot list:** `my_spots.csv` (must contain `POSITION_X`, `POSITION_Y`, `FRAME`, and an intensity column like `TOTAL_INTENSITY_CH1`).

### Step 1: Generate the Pristine Ground Truth

Use `create_ground_truth.py` to generate the clean simulation. You must provide paths for your source files and the desired outputs, as well as the noise regions for background estimation.

```bash
python create_ground_truth.py \
    --video_in "path/to/my_experiment.tif" \
    --spots_in "path/to/my_spots.csv" \
    --video_out "simulations/gt_pristine.tif" \
    --spots_out "simulations/gt_pristine_spots.csv" \
    --noise_regions [x1 y1 x2 y2 x3 y3 x4 y4 ...] \
    --intensity_col "TOTAL_INTENSITY_CH1" \
    --intensity_scale 0.1
```
> This creates gt_pristine.tif (the clean video) and gt_pristine_spots.csv (a file detailing the exact location and amplitude of every spot drawn).

### Step 2: Add Realistic Noise

Use `add_noise_to_gt.py` to apply the noise model. This script needs both the pristine GT video (from Step 1) and your original experimental video (to estimate the noise parameters).

```bash
python add_noise_to_gt.py \
    --gt_video_in "simulations/gt_pristine.tif" \
    --original_video_in "path/to/my_experiment.tif" \
    --output_dir "simulations/noisy_videos" \
    --noise_regions [x1 y1 x2 y2 x3 y3 x4 y4 ...] \
    --scenario_name "My_Simulation" \
    --scales 1.0 2.0 5.0
```

This will create a new folder simulations/noisy_videos/My_Simulation/ containing:
* sim_My_Simulation_scale_1.0.tif
* sim_My_Simulation_scale_2.0.tif
* sim_My_Simulation_scale_5.0.tif
* Validation plots (histograms, PSDs) comparing the noisy simulations to your original video.

---

## Example (Reproducing the Paper)

This example uses the parameters and file names from our paper to generate the scaled (0.1) ground truth and the 10 corresponding noisy datasets.

### Step 1: Create Ground Truth (Intensity Scale 0.1)

```bash
python create_ground_truth.py \
    --video_in "Cy3 best SN 30ms per frame.tif" \
    --spots_in "spots.csv" \
    --video_out "synthetic_gt_scaled_0.1.tif" \
    --spots_out "synthetic_gt_scaled_0.1_spot_info.csv" \
    --noise_regions 0 190 50 250 200 190 250 250 0 10 40 50 220 10 250 50 \
    --intensity_col "TOTAL_INTENSITY_CH1" \
    --intensity_scale 0.1 \
    --exposure 0.030 \
    --diff_coeff 0.5 \
    --blur_sigma_base 0.9
```

### Step 2: Add Poisson-Gaussian Noise (Scales 1x to 10x)

```bash
python add_noise_to_gt.py \
    --gt_video_in "synthetic_gt_scaled_0.1.tif" \
    --original_video_in "Cy3 best SN 30ms per frame.tif" \
    --output_dir "Simulations_Output_Paper" \
    --noise_regions 0 190 50 250 200 190 250 250 0 10 40 50 220 10 250 50 \
    --scenario_name "Gauss_Poisson_Est_Paper" \
    --noise_model "gp_estimated" \
    --scales 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
```

