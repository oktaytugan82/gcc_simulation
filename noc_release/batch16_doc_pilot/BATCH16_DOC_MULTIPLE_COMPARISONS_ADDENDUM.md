# Batch 16 DoC Multiple-Comparisons and Permutation Addendum

Date: 2026-05-15

## Primary Endpoint

Pre-specified clinical anchor: alpha-band CV-GCC access Pi for MCS+ vs VS at calibration alpha = 0.10.

| alpha | band | contrast | score | n | auc | auc_ci_low | auc_ci_high | mannwhitney_p_greater | q_all_tests_bh | q_within_score_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | alpha | MCSplus_vs_VS | raw_cv_pi | 38 | 0.7542 | 0.5792 | 0.9042 | 0.01514 | 0.08367 | 0.07061 |
| 0.1 | alpha | MCSplus_vs_VS | spectral_residual_cv_pi | 38 | 0.7167 | 0.525 | 0.8833 | 0.03258 | 0.08367 | 0.06415 |
| 0.1 | alpha | MCSplus_vs_VS | spectral_plus_epoch_residual_cv_pi | 38 | 0.7208 | 0.5332 | 0.8834 | 0.02958 | 0.08367 | 0.09209 |

## Label-Permutation Tests

| endpoint | alpha | residual | n | observed_auc | observed_delta | permutation_auc_p_greater | permutation_delta_p_greater | null_auc_mean | null_auc_q95 | n_perm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_MCSplus_vs_VS | 0.1 | none | 38 | 0.7542 | 0.197 | 0.01698 | 0.01798 | 0.4918 | 0.6835 | 1000 |
| alpha_MCSplus_vs_VS | 0.1 | spectral | 38 | 0.7167 | 0.2003 | 0.02697 | 0.01598 | 0.4866 | 0.675 | 1000 |
| alpha_MCSplus_vs_VS | 0.1 | spectral_epoch | 38 | 0.7208 | 0.1957 | 0.01698 | 0.01598 | 0.4804 | 0.6627 | 1000 |

## Interpretation

The primary alpha MCS+ vs VS GCC-Pi endpoint remains positive under permutation testing. FDR values are reported transparently across all alpha-threshold/band/contrast/score tests and within each score family. This supports use as a clinical pilot anchor, not as a definitive diagnostic biomarker.
