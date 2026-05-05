# Minimal reproducible microbiome platform-bias pipeline

This repository provides a minimal, reviewer-friendly code package for manuscript submission. It contains the analysis scripts needed to reproduce the main machine-learning comparisons, platform stress tests, calibration analyses, ROC/PR plots, and SHAP-based feature interpretation.

The repository is intentionally focused on **analysis-ready feature tables** rather than raw sequencing processing. Representative QIIME2, PICRUSt2, and ANCOM-BC2 workflows can be documented separately in the manuscript or supplement.

## Pipeline order

The recommended manuscript-facing execution order is:

1. **Single train/test model comparison with recalibration**  
   Compares Taxa-Genus, Taxa-Species, Enterotype, MLP, Pathway, and Genus×Pathway models on the same stratified test split. Exports raw, Platt-calibrated, and isotonic-calibrated performance tables and probability predictions.

2. **Multi-platform LOPO analysis**  
   Performs leave-one-platform-out validation, platform/read-length confounding checks, ROC overlays, SHAP beeswarm plots, and SHAP top-feature CSV exports.

3. **V4 vs FL platform-held-out GroupKFold analysis**  
   Performs platform-held-out GroupKFold comparison between V4 and full-length platforms, with no-leakage recalibration and SHAP ranking on held-out test folds.

## Installation

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Required input files

Place analysis-ready files in `data/raw/` or modify `configs/config.yaml`.

Expected files:

```text
data/raw/gtdbr6_nobatch.csv
data/raw/gtdbrs6_nobatch.csv
data/raw/pathwayn_nobatch.csv
data/raw/gtdbrs6_MLP.csv
data/raw/gtdbr6_Enterotype.csv
data/raw/GenusFunction_nonzero.tsv.gz
```

At minimum, the reference metadata table must contain:

```text
Sample_ID, Group, Platform
```

where `Group` is encoded as `H` for healthy and `D` for disease.

## Usage

### 01. Model comparison with recalibration

```bash
python scripts/run_01_model_compare_recalibration.py --config configs/config.yaml
```

Main outputs:

```text
results/01_model_compare_recalibration/performance_summary_with_recalibration.csv
results/01_model_compare_recalibration/test_predictions_with_recalibration.csv
results/01_model_compare_recalibration/ROC_overlay_uncalibrated.png
results/01_model_compare_recalibration/PR_overlay_uncalibrated.png
results/01_model_compare_recalibration/Calibration_overlay_uncalibrated.png
results/01_model_compare_recalibration/Calibration_recalibration_selected.png
```

### 02. Multi-platform LOPO analysis

```bash
python scripts/run_platform_bias_analysis.py --config configs/config.yaml
```

Main outputs:

```text
results/PlatformBiasChecks_AllModels/within_platform_CV_all_models.csv
results/PlatformBiasChecks_AllModels/LOPO_results_all_models.csv
results/PlatformBiasChecks_AllModels/NegativeControl_ReadTypePred_all_models.csv
results/PlatformBiasChecks_AllModels/ROC_LOPO_holdout_<Platform>_overlay.png
results/PlatformBiasChecks_AllModels/SHAP_top_features_LOPO_holdout_<Platform>_<Model>_train.csv
results/PlatformBiasChecks_AllModels/SHAP_top_features_LOPO_holdout_<Platform>_<Model>_test.csv
```

### 03. V4 vs FL GroupKFold analysis

```bash
python scripts/run_v4fl_groupkfold_analysis.py --config configs/config.yaml
```

Main outputs:

```text
results/V4FL_GroupKFold_compare_SHAP/per_fold_metrics.csv
results/V4FL_GroupKFold_compare_SHAP/summary_mean_sd.csv
results/V4FL_GroupKFold_compare_SHAP/predictions_by_fold.csv
results/V4FL_GroupKFold_compare_SHAP/figures/
results/V4FL_GroupKFold_compare_SHAP/shap/
```

## Notes on calibration

The first pipeline evaluates probability calibration using raw XGBoost probabilities and post-hoc Platt/isotonic recalibration. ROC and PR plots are based on uncalibrated probabilities because discrimination metrics should not require recalibration. Calibration plots compare raw, Platt, and isotonic predictions for selected models.

The V4/FL GroupKFold pipeline uses training-fold-only recalibration to avoid information leakage before applying Platt or isotonic calibration to the held-out platform.

## Data availability

Raw sequencing data are not included in this repository. Processed feature tables and metadata necessary to reproduce the reported machine-learning analyses should be provided when permitted by data-sharing agreements. Public cohort accession or Qiita identifiers should be listed in the manuscript Data Availability section.
Processed data are available at:  
(https://doi.org/10.5281/zenodo.20041336)   
