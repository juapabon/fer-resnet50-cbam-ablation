# CBAM, focal loss and MixUp on ResNet-50 for facial expression recognition: multi-seed ablation and near-duplicate audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24357375.svg)](https://doi.org/10.5281/zenodo.24357375)

Code, results and per-image predictions accompanying the article

> Pabón Alajo, J. F., Curto Diego, M. B., & Moreno Rodilla, V. *Do CBAM attention, focal loss and MixUp improve facial expression recognition with ResNet-50? A multi-seed ablation with a validated near-duplicate audit, deduplicated retraining and cross-dataset evaluation.* Submitted to Frontiers.

## Contents

| Folder | What it contains |
|---|---|
| `code/` | The experiment notebook (`FER_ResNet50_CBAM_Frontiers.ipynb`) and its executed copy; `queues/` and `revision/` hold the scripts that ran the training runs in parallel on one RTX 3090 |
| `results/` | Every table and figure of the article, the CSV/JSON summaries, the near-duplicate lists (`test_con_duplicado_regla.csv`, `duplicados_*.csv`), the 360 manually reviewed pairs with their verdicts (`revision_duplicados_muestra.csv`) and the split-half validation of the decision rule |
| `runs/<group>/<variant>_s<seed>/` | For every training run: `meta.json` (configuration, epochs, time), `history.json` (per-epoch metrics) and `preds_test.csv` (per-image test probabilities, with and without TTA) |
| `splits/` | The official FER2013 split (Training / PublicTest / PrivateTest) reconstructed for this copy by pixel matching against FERPlus, with the FERPlus index of every image |

Run groups: `runs` (9 configurations × seeds 42–46), `runs_dedup` (deduplicated FER2013), `runs_rafdb` (RAF-DB), and the experiments added during review: `runs_oficial` (official split), `runs_ajuste` (baseline hyperparameter search), `runs_muestreo` (weighted sampling instead of stored augmented copies), `runs_semillas` (seeds 47–51) and `runs_ajuste_fino` (full fine-tuning without progressive unfreezing).

## Data

The images of FER2013 and RAF-DB are **not** redistributed. Paths in the CSV files are relative to the dataset root (`FER2013/<split>/<class>/<file>`, `RAF-DB/<split>/<code>/<file>`). FER2013 is available from the ICML 2013 challenge (Goodfellow et al., 2015); RAF-DB from its authors for non-commercial research (Li et al., 2017). The model weights (`best.pth` of every run) are deposited in the Zenodo record of this repository (https://doi.org/10.5281/zenodo.24357375).

## Reproducing

Python 3.8, PyTorch with CUDA; the exact package versions are recorded in `results/entorno_y_config.json`. Set `CFG["DATA_DIR"]` in the notebook and run it top to bottom; `FER_SMOKE=1` runs the whole pipeline on a tiny synthetic dataset in a few minutes. Finished runs (`meta.json`) are reused instead of retrained.

## License

Code: MIT. Results, lists and predictions: CC BY 4.0.
