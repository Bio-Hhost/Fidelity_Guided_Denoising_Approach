# Example inference data

A small, self-contained example for running both denoising inference pipelines end to end. The commands below have been run and verified.

`demo_raw_video.tif` is a synthetic single-molecule TIRF stack (30 frames, 128×128, `uint16`) generated with a Poisson–Gaussian camera model matched to the denoiser's noise assumptions. It is a *runnable smoke-test input only* — **not** experimental data and **not** the paper's simulation pipeline. See `generate_demo_input.py` for how it was produced. Running the real models on it confirms the pipeline executes; it does not demonstrate denoising quality. For a quality check, use a real experimental or simulated data (see "Running on real data" below).

## Getting the trained models

The **RL demo runs straight from a clone** — its weights ship in `rl_demo_run/models/`, alongside a demo-specific `config.json` whose background regions are the four 128 px corners rather than the full-frame regions of the real run.

The **static model is not here in the repo**: at 62 MB it is over GitHub's upload limit. Download it from the Zenodo record (see `10.5281/zenodo.21925651`) and place it at:

```
example_data/static_models/unet_static_seq1_lambda0.001.keras
```

It is `best_model.keras` from `trained_models/static_models_new/static_T1_geo0.001_20260723-195633/`. For a different fidelity weight or temporal window, take the matching run from that same directory; `--sequence_length` must match the run's `T`.

## Run the static pipeline (verified)

```powershell
python inference_static_lambda.py `
  --input_file        example_data/demo_raw_video.tif `
  --model_file        example_data/static_models/unet_static_seq1_lambda0.001.keras `
  --noise_params_file example_data/demo_noise_params.npy `
  --output_file       example_data/output/demo_denoised_static.tif `
  --sequence_length   1 --batch_size 4
```

- `--sequence_length 1` must match the model (seq=1 here).
- The static model's spatial input is fixed and square (256×256 for this model). The 128×128 demo is reflection-padded up to 256 and cropped back.

## Run the RL pipeline on the demo (verified)

The RL script estimates background from `noise_analysis_regions` in `config.json`. The real `rl_models/config.json` regions are sized for full Cy3 frames (e.g. `[0, 190, 50, 250]`) and run off the edge of the 128×128 demo. So the demo uses `rl_demo_run/` — the same real weights, but a config whose regions are the four 128 px corners.

```powershell
python inference_lambda_RL.py `
  --input_file   example_data/demo_raw_video.tif `
  --model_folder example_data/rl_demo_run `
  --output_file  example_data/output/demo_denoised_RL.tif
```

`rl_demo_run/` must contain `config.json` (seq=5, channels=1, corner regions) and `models/unet_best.weights.h5`.

## Running on real Cy3 data

- **RL:** point `--model_folder` at `rl_models` (real `config.json`, real full-frame regions) — do not use `rl_demo_run` (its corner regions are demo-only).
- **Static:** use the real `noise_parameters.npy` from the model's run folder (not `demo_noise_params.npy`).
