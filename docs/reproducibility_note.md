# Reproducibility note for manuscript submission

This repository is organized as a minimal reproducible analysis package for the manuscript. The code is structured around analysis-ready taxonomic and functional feature tables rather than raw sequencing reads.

## Recommended analysis sequence

1. **Single train/test model comparison with recalibration**  
   This step provides the baseline model comparison across feature representations and evaluates post-hoc probability calibration using Platt scaling and isotonic regression.

2. **Leave-one-platform-out validation**  
   This step evaluates cross-platform generalizability and potential platform/read-length confounding across all available sequencing platforms.

3. **V4 vs full-length platform-held-out GroupKFold**  
   This step directly stress-tests transferability between V4 and full-length 16S profiles and applies no-leakage recalibration using only the training fold before evaluation on the held-out platform.

## Manuscript-ready wording

All custom scripts used for model comparison, recalibration, cross-platform validation, and SHAP-based feature interpretation are provided in this repository. The pipeline includes single train/test model comparison, leave-one-platform-out validation, V4/full-length platform-held-out GroupKFold analysis, ROC/PR plotting, probability calibration assessment, and SHAP feature ranking.

Probability calibration was evaluated using Platt scaling and isotonic regression. For the platform-held-out analyses, calibration models were trained exclusively within the training folds before application to the held-out platform to avoid information leakage.

## Input requirements

To reproduce the machine-learning analyses, users should provide analysis-ready feature tables with samples as rows and features as columns, together with metadata containing `Sample_ID`, `Group`, and `Platform`.

Representative preprocessing steps using QIIME2, PICRUSt2, and ANCOM-BC2 may be provided as supplementary workflow descriptions, while the core reproducible analyses in this repository begin from processed taxonomic and pathway abundance tables.

## Data-sharing caveat

Raw sequencing data from restricted institutional cohorts are not included. Processed feature tables and de-identified metadata should be shared when allowed by institutional review board and data ownership agreements. Public datasets should be cited using their accession numbers or Qiita study identifiers.
