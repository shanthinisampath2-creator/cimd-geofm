# CIMD: Class-Conditional Interventional Modality Dependency for optical–SAR land-cover mapping

Code for the paper **"Which classes fail when a sensor is lost? Class-conditional interventional modality dependency for optical–SAR land-cover mapping"** (submitted to *Geodata and AI*).

CIMD measures how much each land-cover class depends on optical and on SAR input by removing one sensor at inference and measuring the change in the class's predicted probability. A small dependency head learns to predict this from the full-input features in one forward pass, and the prediction is used to forecast which classes and patches will fail if a sensor is lost.

| Resource | Where |
|---|---|
| Code and notebooks | this repository |
| Full results (all CSVs, training histories, confusion matrices, figures) | `CIMD_results.zip` — [Zenodo DOI, add after upload] |
| Trained checkpoints (all reported runs) | [Zenodo DOI, add after upload] |
| DFC2020 split used in the paper | `splits/dfc2020_official_split.csv` |

## Main results (test set, mean of 3 seeds)

Robust mean = mean mIoU over full input, optical only (SAR removed) and SAR only (optical removed).

| Dataset | Model | Full input | Optical only | SAR only | Robust mean |
|---|---|---|---|---|---|
| WHU-OPT-SAR | A0F full-modality baseline | 0.416 | 0.021 | 0.071 | 0.169 |
| WHU-OPT-SAR | A1 missing-modality training | 0.419 | 0.379 | 0.364 | 0.387 |
| WHU-OPT-SAR | A3 A1 + CIMD | 0.414 | 0.377 | 0.361 | 0.384 |
| DFC2020 | A0F | 0.501 | 0.220 | 0.071 | 0.264 |
| DFC2020 | A1 | 0.521 | 0.503 | 0.390 | 0.471 |
| DFC2020 | A2 A1 + uniform consistency | 0.513 | 0.503 | 0.393 | 0.470 |
| DFC2020 | A3 | 0.520 | 0.502 | 0.390 | 0.471 |
| DFC2020 | B_KD distillation | 0.517 | 0.496 | 0.393 | 0.469 |
| DFC2020 | B_REC feature reconstruction | 0.516 | 0.525 | 0.384 | 0.475 |
| DFC2020 | B_GATE gated fusion | 0.500 | 0.492 | 0.377 | 0.457 |

Failure forecast (AUROC for class instances whose recall drops by more than 0.20 after a sensor is removed; instances pooled over 3 seeds):

| Dataset | Sensor removed | Predicted dependency | Class-average dependency | Gain (95% bootstrap CI) |
|---|---|---|---|---|
| WHU-OPT-SAR | optical | 0.730 | 0.647 | +0.083 (0.072, 0.094) |
| WHU-OPT-SAR | SAR | 0.709 | 0.598 | +0.111 (0.092, 0.130) |
| DFC2020 | optical | 0.698 | 0.599 | +0.099 (0.089, 0.109) |
| DFC2020 | SAR | 0.698 | 0.685 | +0.014 (−0.003, 0.033) |

On DFC2020 the best predictive-uncertainty score (1 − top-two margin) reaches 0.583 for optical loss and 0.690 for SAR loss. Every table of the paper is in `results/paper_tables/`.

## Repository layout

```
notebooks/
  01_WHU-OPT-SAR_dual_stream_CNN.ipynb   WHU-OPT-SAR, compact dual-stream CNN (GroupNorm): A0F, A1, A2, A3,
                                         failure forecast, ablations (loss-term magnitudes, DWCR strength sweep,
                                         one-way distillation, damage-target estimator A3D)
  02_DFC2020_ResNet50_SSL4EO.ipynb       DFC2020, dual-stream ResNet-50 (SSL4EO-S12 MoCo weights): A0F, A1, A2, A3,
                                         comparison methods B_KD / B_REC / B_GATE, uncertainty baselines (E1),
                                         all-methods table (E3)
scripts/                                 the same two notebooks exported as plain Python ("# %%" cells), for reading
analysis/
  forecast_statistics.py                 AUROC, bootstrap CI and failure-capture rates from a forecast-instances CSV
  make_paper_tables.py                   rebuilds Tables 3–6 of the paper from the results archive
results/
  paper_tables/                          Tables 2–6 of the paper as CSV
  forecast_instances/                    per-instance forecast files needed by analysis/ (no GPU needed)
splits/
  dfc2020_official_split.csv             train / val / test assignment of every DFC2020 patch
  whu_opt_sar_scene_split.csv            train / val / test assignment of every WHU-OPT-SAR scene
```

## Experiments

| ID | Training | Notebook |
|---|---|---|
| A0F | Full input only, selected on validation full-input mIoU | 01 (Cell 21), 02 (Cell 7) |
| A1 | Full + optical-only + SAR-only passes | 01, 02 (Cell 8) |
| A2 | A1 + uniform prediction consistency | 01, 02 (E4) |
| A3 | A1 + dependency head + dependency-weighted consistency (DWCR) | 01, 02 (Cell 9) |
| B_KD | A1 + class-wise KL distillation from the full-input prediction | 02 (E0, E2) |
| B_REC | Per-scale networks reconstruct the missing sensor's features | 02 (E0, E2) |
| B_GATE | A1 objective with learned per-pixel gates before fusion | 02 (E0, E2) |

Hyperparameters (both datasets): α = 0.5, β = 0.2, γ = 0.3, δ = 0.1, τ = 0.05; AdamW, learning rate 3e-4 (pretrained encoders 3e-5), weight decay 1e-4, cosine schedule, 25 epochs, batch size 8, seeds 42/43/44. A missing sensor is set to zero.

## Data

- **WHU-OPT-SAR**: https://github.com/AmberHen/WHU-OPT-SAR-dataset. Extract to a folder named `WHU_OPT_SAR_EXTRACTED`. Scenes are split 64/16/20 (train/val/test); the exact split used in the paper is `splits/whu_opt_sar_scene_split.csv`. To reuse it, copy it to `CIMD_GeoFM_WHU_RESULTS_GN/scene_split.csv` next to the data folder before running notebook 01, Cell 2 (otherwise Cell 2 creates a new seeded split).
- **DFC2020**: https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest. Download `s1_validation`, `s2_validation`, `dfc_validation`, `s1_0`, `s2_0`, `dfc_0` and extract them into one folder named `DFC2020`. Savanna and snow/ice are ignored (8 classes). Patches carry no geolocation, so the split follows the official releases: the `validation` release (986 patches) is the test set; the `test` release is split into 4,359 training and 769 validation patches (`splits/dfc2020_official_split.csv`).

## Setup

Tested on Windows 11 and Linux, Python 3.11, PyTorch 2.6 (CUDA 12.4), NVIDIA RTX 4500 Ada (24 GB).

```bash
conda env create -f environment.yml
conda activate cimd
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install timm torchgeo
python -m ipykernel install --user --name cimd --display-name "Python (cimd)"
jupyter notebook
```

Or with pip only: install PyTorch as above, then `pip install -r requirements.txt`.

In Cell 1 of a notebook, set `DATA_DIR_OVERRIDE` (notebook 01) or `DFC_DIR_OVERRIDE` / `WHU_DIR_OVERRIDE` (notebook 02) if the data folder is not found automatically. Results and checkpoints are written next to the data folder.

## Reproducing the paper

| Paper item | How |
|---|---|
| Table 2, WHU-OPT-SAR rows | Notebook 01: Cells 1–9, 21 (training), 10 (test) |
| Table 2, DFC2020 rows; Table 3 | Notebook 02: Cells 1–9, E0, E2, E4 (training), then E3 |
| Table 4, Figure 2 | Notebook 02: Cell 11 |
| Figure 3 (partial degradation) | Notebook 01 Cell 12; notebook 02 Cell 12 and E3 |
| Table 5, Figure 4 | Notebook 01 Cell 18; notebook 02 Cell 13; statistics: `analysis/forecast_statistics.py` |
| Table 6 | Notebook 02: E1 |

Training is resume-safe: rerunning a training cell skips finished runs and resumes interrupted ones. Approximate time per DFC2020 run (25 epochs): 25 min for A0F, 30–60 min for the other models.

Statistics without a GPU, from the files in this repository:

```bash
python analysis/forecast_statistics.py results/forecast_instances/DFC2020_failure_forecast_instances_A3.csv
python analysis/forecast_statistics.py results/forecast_instances/WHU-OPT-SAR_failure_forecast_instances_A3.csv
python analysis/make_paper_tables.py path/to/CIMD_results     # after downloading the results archive
```

## Notes

- Mixed precision uses bfloat16. float16 overflowed with frozen BatchNorm and produced non-finite losses; non-finite batches are skipped and counted in the training history (`skipped_batches`, 0 in all reported runs).
- On Windows, the kernel can die with `OMP: Error #15` because PyTorch and conda-forge packages load two OpenMP runtimes. Cell 1 sets `KMP_DUPLICATE_LIB_OK=TRUE` before any import; re-evaluating the same checkpoints with this setting gave identical metrics.
- B_KD, B_REC and B_GATE are re-implementations of these method families on the shared backbone, not the original authors' code.

## Citation

```bibtex
@article{cimd2026,
  title   = {Which classes fail when a sensor is lost? Class-conditional interventional modality dependency for optical--SAR land-cover mapping},
  author  = {[Authors]},
  journal = {Geodata and AI},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`). WHU-OPT-SAR and DFC2020 are subject to their own licences.
