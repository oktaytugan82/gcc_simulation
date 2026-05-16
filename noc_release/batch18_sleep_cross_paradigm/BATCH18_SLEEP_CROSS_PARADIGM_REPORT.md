# Batch 18 Sleep-EDF Cross-Paradigm GCC Geometry

Date: 2026-05-15

## Purpose

This analysis tests whether GCC observables capture state geometry outside anesthesia. The claim is deliberately not that Pi alone is a superior sleep-stage biomarker. The claim is that the GCC triad generalizes to sleep as an access-state geometry while conventional spectral features remain strong sleep-stage markers.

## Dataset Audit

| status | count |
| --- | --- |
| ok | 22 |

Analyzed subjects: 22

## Leave-Subject-Out Classification

| band | contrast | model | n | n_subjects | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| sigma | Wake_vs_NREM | spectral_all | 19136 | 22 | 0.9589 | 0.9308 |
| sigma | REM_vs_NREM | spectral_all | 19280 | 22 | 0.8919 | 0.8274 |
| sigma | Wake_vs_NREM | gcco_pi | 19136 | 22 | 0.5344 | 0.5723 |
| sigma | REM_vs_NREM | gcco_pi | 19280 | 22 | 0.5343 | 0.5825 |
| sigma | Wake_vs_NREM | gcco_triad_plus_pi | 19136 | 22 | 0.875 | 0.8047 |
| sigma | REM_vs_NREM | gcco_triad_plus_pi | 19280 | 22 | 0.7274 | 0.6642 |
| sigma | Wake_vs_NREM | spectral_all_plus_gcco | 19136 | 22 | 0.9671 | 0.9303 |
| sigma | REM_vs_NREM | spectral_all_plus_gcco | 19280 | 22 | 0.8938 | 0.8267 |

## Residual GCC After Spectral Regression

| band | contrast | model | n | n_subjects | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| sigma | Wake_vs_NREM | residual_gcco_after_spectral_all | 19136 | 22 | 0.6367 | 0.6122 |
| sigma | REM_vs_NREM | residual_gcco_after_spectral_all | 19280 | 22 | 0.5721 | 0.5589 |

## Subject-Level Paired Stage Contrasts

| band | feature | contrast | n_subjects | mean_delta | dz | wilcoxon_two_sided_p |
| --- | --- | --- | --- | --- | --- | --- |
| sigma | M_tau | REM_minus_NREM | 22 | 0.003323 | 0.855 | 0.001262 |
| sigma | M_tau | Wake_minus_NREM | 22 | 0.002349 | 0.5467 | 0.02754 |
| sigma | M_tau | Wake_minus_REM | 22 | -0.0009736 | -1.231 | 2.623e-05 |
| sigma | Pi | REM_minus_NREM | 22 | 0.05169 | 0.9485 | 0.0002556 |
| sigma | Pi | Wake_minus_NREM | 22 | 0.04269 | 0.7926 | 0.001455 |
| sigma | Pi | Wake_minus_REM | 22 | -0.008996 | -0.4998 | 0.02086 |
| sigma | R | REM_minus_NREM | 22 | -0.1492 | -3.208 | 4.768e-07 |
| sigma | R | Wake_minus_NREM | 22 | -0.2322 | -2.219 | 4.768e-07 |
| sigma | R | Wake_minus_REM | 22 | -0.08299 | -1.285 | 9.06e-06 |
| sigma | spectral_entropy | REM_minus_NREM | 22 | 0.08294 | 1.882 | 4.768e-07 |
| sigma | spectral_entropy | Wake_minus_NREM | 22 | 0.09033 | 1.177 | 4.196e-05 |
| sigma | spectral_entropy | Wake_minus_REM | 22 | 0.007391 | 0.1425 | 0.6789 |

## Interpretation

Sleep-EDF supports cross-paradigm GCC state geometry: GCC triad features separate Wake/NREM and REM/NREM above chance. However, spectral-only features are stronger for canonical sleep staging, and Pi alone is weak. Therefore this result should be reported as cross-paradigm geometric support for GCC, not as evidence that Pi is a standalone sleep biomarker.
