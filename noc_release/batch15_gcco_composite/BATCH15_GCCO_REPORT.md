# Batch 15 GCC-O Spectrally Orthogonalized Candidate

Date: 2026-05-14

## Definition

GCC-O is a stricter candidate variant: R is mean wPLI-style lagged phase coupling, D_eff is a participation ratio of the normalized graph-Laplacian spectrum of the lagged-connectivity graph, and M_tau is the within-window variance of mean absolute lagged phase interaction. Bandpass amplitude does not enter the GCC-O observables; conventional spectral features are computed only as external covariates for residualization and baseline comparison.

## Paired Pi Effects

| dataset | band | target | n | baseline_mean | target_mean | mean_delta_baseline_minus_target | paired_d | wilcoxon_greater_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | moderate | 20 | 0.8276 | 0.7374 | 0.09023 | 1.024 | 0.0004454 |
| DS005620 | alpha | sed | 20 | 0.8276 | 0.7215 | 0.1061 | 0.6037 | 0.004154 |
| DS005620 | alpha | sed2 | 20 | 0.8276 | 0.7373 | 0.0903 | 0.5232 | 0.0138 |
| FarnesKetamine | alpha | ketamine | 10 | 0.8313 | 0.7852 | 0.04616 | 0.7964 | 0.02441 |
| Chennu | gamma | moderate | 20 | 0.8276 | 0.7652 | 0.06236 | 0.8488 | 0.0009233 |
| DS005620 | gamma | sed | 20 | 0.8276 | 0.6603 | 0.1673 | 1.729 | 9.537e-07 |
| DS005620 | gamma | sed2 | 20 | 0.8276 | 0.6975 | 0.1301 | 1.246 | 9.537e-07 |
| FarnesKetamine | gamma | ketamine | 10 | 0.8304 | 0.7724 | 0.05807 | 0.5954 | 0.02441 |


## Sleep-EDF Cross-Validated Metrics

| band | contrast | model | n | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| sigma | Wake_vs_NREM | spectral_all | 9291 | 0.9647 | 0.9259 |
| sigma | REM_vs_NREM | spectral_all | 9449 | 0.912 | 0.8527 |
| sigma | Wake_vs_NREM | gcco_pi | 9291 | 0.5471 | 0.5833 |
| sigma | REM_vs_NREM | gcco_pi | 9449 | 0.5522 | 0.5991 |
| sigma | Wake_vs_NREM | gcco_triad_plus_pi | 9291 | 0.8993 | 0.8376 |
| sigma | REM_vs_NREM | gcco_triad_plus_pi | 9449 | 0.7509 | 0.6981 |
| sigma | Wake_vs_NREM | spectral_all_plus_gcco | 9291 | 0.9707 | 0.9277 |
| sigma | REM_vs_NREM | spectral_all_plus_gcco | 9449 | 0.9093 | 0.8512 |


## Interpretation Rule

A bandpower-independent biomarker claim is allowed only if residualized GCC-O remains clearly above chance and/or spectral+GCC-O robustly improves over spectral-only baselines with positive bootstrap intervals. Otherwise GCC-O should be reported as phase-lagged and bandpower-controlled, but not generally bandpower-independent.

## Sleep-EDF Residualized Control

| band | contrast | model | n | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| sigma | Wake_vs_NREM | spectral_all | 9291 | 0.9647 | 0.9259 |
| sigma | Wake_vs_NREM | gcco_triad_plus_pi | 9291 | 0.8993 | 0.8376 |
| sigma | Wake_vs_NREM | spectral_all_plus_gcco | 9291 | 0.9707 | 0.9277 |
| sigma | Wake_vs_NREM | residual_gcco_after_spectral_all | 9291 | 0.6503 | 0.6248 |
| sigma | REM_vs_NREM | spectral_all | 9449 | 0.9120 | 0.8527 |
| sigma | REM_vs_NREM | gcco_triad_plus_pi | 9449 | 0.7509 | 0.6981 |
| sigma | REM_vs_NREM | spectral_all_plus_gcco | 9449 | 0.9093 | 0.8512 |
| sigma | REM_vs_NREM | residual_gcco_after_spectral_all | 9449 | 0.5250 | 0.5289 |
