# Figures

The scripts that draw the figures in the paper and the supplement, plus
`figure_style.py`, which assigns each method its display name and colour.

## These scripts only draw — run the pipeline first

Nothing here denoises a video, trains a model or computes a metric. Every script reads
finished analysis outputs and plots them, so the analysis has to exist first:

1. **Denoise the data.** `inference_static_lambda.py` and `inference_lambda_RL.py`
   produce the denoised videos that the figures compare against the raw ones.
2. **Run the evaluation.** See [`../evaluation/readme.md`](../evaluation/readme.md) for
   the detection threshold scan, the full evaluation and the experimental analysis.
   `../run_r11.py` produces the localization benchmark.
3. **Then** run any `make_figure*.py`. They take no arguments and resolve their paths
   relative to the repository root, so they can be run from any directory.

## Inputs

Three directories must be present at the repository root. None of them is in git —
unpack them from the archived record.

| directory | holds |
|---|---|
| `Data/` | the experimental recordings and the simulated datasets |
| `trained_models/` | the static-λ and RL models, and the comparison methods |
| `results/` | the evaluation outputs and the denoised videos |

`results/` is the one the figure scripts read most:

| path | used by |
|---|---|
| `COMPARISON_simulated_unified.csv` | figures 2, 3, S-cotraining, S-temporal |
| `figure_rescan/exp_data/<experiment>/` | figures 1, 5 — raw and denoised experimental videos |
| `figure_rescan/exp_results*/`, `identity_control/` | figure 4 |
| `rl_gain17689/`, `ablation_static_v2/` | S-cotraining, S-lambda_effect |
| `r11_evidence/` | S-crb, S-biasscatter — written by `../run_r11.py` |

`COMPARISON_simulated_unified.csv` is the merged simulated-data table. It is produced by
the aggregation step, which is not part of this repository; take it from the archived
record together with the rest of `results/`.

## Method labels

`figure_style.py` maps a filename to a display name and colour and is shared with the
evaluation scripts, so a method is labelled identically everywhere. It raises
`UnknownMethodError` on a filename it does not recognise rather than guessing, which
stops a run instead of letting a mislabelled series reach a figure.
