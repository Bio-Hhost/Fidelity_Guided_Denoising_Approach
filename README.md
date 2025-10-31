# Fidelity_Guided_Denoising_Approach

This repository provides the official implementation for the paper **Enhanced Quantitative Accuracy in Single Molecule Imaging: A Fidelity-Guided Denoising Approach**. Our work introduces a novel self-supervised denoising framework for single-molecule TIRF microscopy data, centered on a 3D U-Net trained with a composite loss function.

This framework is presented in two distinct approaches:
1.  **Static $\lambda$:** A 3D U-Net trained with a manually-tuned, static fidelity weight ($\lambda$).
2.  **Adaptive (RL) $\lambda$:** A 3D U-Net trained jointly with a reinforcement learning (RL) agent that dynamically selects the optimal $\lambda$ for each training batch.

[![Paper](https://img.shields.io/badge/paper-link-b31b1b.svg)](https://link.to.paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of contents
- [Methodology Overview](#methodology-overview)
- [Installation](#installation)
- [Inference](#inference-using-a-pre-trained-model)

---
## Methodology Overview

Our denoising framework is built on three core components: a 3D U-Net architecture, a self-supervised training strategy, and a novel physics-informed loss function.

### 1. 3D U-Net Architecture
We use an adapted 3D U-Net to process spatiotemporal volumes of microscopy data (e.g., 256x256xT, where T is the number of frames). This allows the network to leverage both spatial and temporal context to reconstruct the central frame of the sequence.

(figures/unet_architecture.png)

### 2. Self-Supervised, Blind-Spot Training
Training is self-supervised, requiring no clean "ground truth" images. We adapt the blind-spot strategy by masking random pixels in the central frame of an input sequence. The network is then trained to predict the values of these masked pixels using only the surrounding spatial and temporal context.

(figures/training_process_diagram.png)

### 3. Composite Physics-Informed Loss
The core of our method is a composite loss function that balances denoising and data fidelity. It is "physics-informed" because it incorporates the Poisson-Gaussian noise statistics of the EMCCD camera:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL-masked}} + \lambda \cdot \mathcal{L}_{\text{MSE-unmasked}}
```

* **$\mathcal{L}_{\text{NLL-masked}}$**: A Poisson-Gaussian Negative Log-Likelihood loss calculated only at the masked pixels. This forces the network to learn a physically plausible reconstruction.
* **$\mathcal{L}_{\text{MSE-unmasked}}$**: A Mean Squared Error (MSE) fidelity term calculated only at the unmasked pixels. This penalizes the network for changing pixels it can see, preserving the original data structure.
* **$\lambda$**: A hyperparameter that balances the two loss terms. This repository provides code to train with a **static $\lambda$** (`train_static_lamda.py`) or an **adaptive $\lambda$** chosen by an RL agent (`train_lambda_RL.py`).

---

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.zhaw.ch/Bio-Hhost/Fidelity_Guided_Denoising_Approach.git
    cd Fidelity_Guided_Denoising_Approach
    ```

2.  Install the required Python packages.

    You can install the main dependencies manually:
    ```bash
    pip install numpy scipy opencv-python tifffile matplotlib pandas scikit-learn
    pip install tensorflow  # or tensorflow[and-cuda] depending on your setup
    ```
    > GPU highly recommended. If you use TensorFlow with GPU, install the matching CUDA/cuDNN per TensorFlow’s docs.
---


## Inference (Using a Pre-trained Model)

This is the fastest way to denoise your own data. You will first need to download our pre-trained models and associated files from [**link**].

### Approach 1: Static $\lambda$ Model

This model requires three files: the input video, the trained `.keras` model, and the `.npy` noise parameter file generated during training.

```bash
python inference_static_lambda.py \
    --input_file path_to_noisy_video.tif \
    --model_file path_to_pretrained_models.keras \
    --noise_params_file path_to_pretrained_models_noise_params.npy \
    --output_file path_to_denoised_video.tif \
    --sequence_length 1
```

  > --sequence_length: Must match the sequence length the model was trained with (e.g., 1, 3, or 5).

### Approach 2: Adaptive (RL) λ Model

This model only requires the input video and the folder containing the training run data (which includes the `config.json` and model weights). The RL agent is not used during inference; the U-Net is the final denoising model.

```bash
python inference_lambda_RL.py \
    --input_file path_to_noisy_video.tif \
    --model_folder path_to_pretrained_models/rl_run_folder/ \
    --output_file path_to_denoised_video.tif
```

---
