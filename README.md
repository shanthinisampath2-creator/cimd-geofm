# CIMD-GeoFM: Class-Conditional Interventional Modality Dependency for optical–SAR land-cover mapping

Code, data splits and result tables for the paper
**"[Paper title]"**, submitted to *Geodata and AI*.

CIMD measures how strongly each land-cover class depends on optical and SAR information by removing one sensor at a time and measuring the change in the model's class prediction. A lightweight head learns to predict this dependency from the full-input features in a single forward pass, and the prediction is used to forecast which classes and patches will lose accuracy if a sensor fails.

## Main findings (3 seeds each)

| Dataset | Model | Full input | Optical only | SAR only | Robust mean |
|---|---|---|---|---|---|
| WHU-OPT-SAR | A0F (full-modality baseline) | 0.416 | 0.021 | 0.071 | 0.169 |
| WHU-OPT-SAR | A1 (missing-modality training) | 0.419 | 0.379 | 0.364 | 0.387 |
| DFC2020 | A0F | 0.510 | 0.235 | 0.067 | 0.271 |
| DFC2020 | A1 | 0.524 | 0.510 | 0.396 | 0.477 |

Test mIoU. The predicted dependency detects patches whose class recall collapses after a sensor is lost better than the class-average dependency on both datasets (AUROC 0.73–0.74 when optical is removed). Full tables are in `results/`.

## Repository layout

```
notebooks/
  01_WHU-OPT-SAR_dual_stream_CNN.ipynb   # WHU-OPT-SAR, dual-stream CNN (GroupNorm), experiments A0–A3, A0F,
                                         # failure forecast and ablations (loss magnitudes, DWCR sweep,
                                         # one-way distillation, damage-target estimator)
  02_DFC2020_ResNet50_SSL4EO.ipynb       # DFC2020 (and WHU) with dual-stream ResNet-50 (SSL4EO-S12 or ImageNet),
                                         # A0F/A1/A2/A3, comparison methods (distillation, feature reconstruction,
                                         # gated fusion) and the uncertainty-baseline comparison
analysis/
  forecast_statistics.py                 # AUROC, bootstrap intervals and failure-capture rates (no GPU)
results/
  DFC2020/        WHU-OPT-SAR/           # result tables written by the notebooks
splits/
  dfc2020_official_split.csv             # train / val / test assignment of every DFC2020 patch
```

## Experiments

| ID | Description |
|---|---|
| A0F | Full-modality baseline, selected on validation full-input mIoU, no early stopping |
| A1 | Missing-modality supervised training (full + optical-only + SAR-only passes) |
| A2 | A1 + uniform prediction consistency |
| A3 | A1 + CIMD dependency estimator + dependency-weighted consistency (DWCR) |
| B_KD | A1 + one-way knowledge distillation from the full-input prediction |
| B_REC | Missing-modality training with feature reconstruction (the missing sensor's features are predicted from the available one) |
| B_GATE | A1 with learned per-pixel gated fusion of the two encoders |

## Data

- **WHU-OPT-SAR**: https://github.com/AmberHen/WHU-OPT-SAR-dataset (100 GaoFen optical/SAR scene pairs with 7-class labels). Extract to a folder named `WHU_OPT_SAR_EXTRACTED`.
- **DFC2020** (IEEE GRSS Data Fusion Contest 2020): https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest. Download `s1_validation`, `s2_validation`, `dfc_validation`, `s1_0`, `s2_0` and `dfc_0`, and extract them into one folder named `DFC2020`. Savanna and snow/ice are ignored, as in the contest protocol (8 classes remain).

DFC2020 patches are distributed without real geolocation, so the split follows the two official releases: the `validation` release (986 patches) is the test set, and the `test` release is divided into training (4,359) and validation (769) patches. The exact assignment is in `splits/dfc2020_official_split.csv`.

## Setup

```bash
conda create -n cimd python=3.11 -y && conda activate cimd
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
jupyter notebook
```

In Cell 1 of each notebook, set the data folder (`DATA_DIR_OVERRIDE` in notebook 01; `DFC_DIR_OVERRIDE` or `WHU_DIR_OVERRIDE` in notebook 02) if it is not found automatically. Results and checkpoints are written next to the data folder.

## Running

**Notebook 01 (WHU-OPT-SAR, small CNN):** Cells 1–9 train A0–A3, Cells 10–13 evaluate, Cell 21 trains and evaluates A0F, Cell 18 runs the failure forecast. Cells 14–17 and 19 reproduce the ablations.

**Notebook 02 (DFC2020, ResNet-50):** Cells 1–6 set up data and model, Cells 7–9 train A0F, A1 and A3, Cells 10–14 evaluate (test table, dependency, progressive degradation, failure forecast, results package). Part 2: Cell E0, then E1 (uncertainty baselines, no training), E2 (comparison methods), E3 (one table for all trained models), E4 (optional A2), E5 (package). Set `DATASET = "WHU"` and `BACKBONE = "imagenet_resnet50"` to run the same pipeline on WHU-OPT-SAR.

Training is resume-safe: rerunning a training cell skips finished runs and resumes interrupted ones from the last epoch. Mixed precision uses bfloat16; fp16 caused overflow with frozen BatchNorm in our experiments.

Statistics without a GPU:

```bash
python analysis/forecast_statistics.py results/DFC2020/failure_forecast_instances_A3.csv
```

## Hardware and runtime

NVIDIA RTX 4500 Ada (24 GB). DFC2020, batch size 8, 25 epochs: about 25 minutes per A0F run and about 50 minutes per A1/A3 run.

## Trained models

Checkpoints for all reported runs: **[Zenodo DOI — add after upload]**

## Citation

```
[Citation to be added after publication]
```

## License

Code: MIT (see `LICENSE`). The datasets are subject to their own licenses.
