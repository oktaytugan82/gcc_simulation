# Batch 9 Bandpower-Independence Stress Test

Date: 2026-05-14

## Aim

Test whether GCC can be defended as carrying information beyond conventional spectral features. This batch does not assume success; it explicitly reports when residualized GCC weakens.

## Tests

- Leave-one-subject-out models: spectral features, GCC, spectral+GCC, and residualized GCC.

- Cross-dataset transfer between Chennu and DS005620.

- Spectral-caliper subsets retaining positive samples with the smallest spectral shifts.


## Within-Dataset LOSO AUC

| dataset | band | model | auc | balanced_accuracy | log_loss | n |
| --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | gcc_pi | 0.800 | 0.875 | 0.464 | 40 |
| Chennu | alpha | gcc_triad_plus_pi | 0.820 | 0.875 | 0.454 | 40 |
| Chennu | alpha | residual_gcc_after_spectral_all | 0.700 | 0.850 | 0.927 | 40 |
| Chennu | alpha | residual_pi_after_spectral_all | 0.700 | 0.850 | 0.793 | 40 |
| Chennu | alpha | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | alpha | spectral_all_plus_gcc | 0.953 | 0.950 | 0.186 | 40 |
| Chennu | alpha | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | alpha | gcc_pi | 0.875 | 0.900 | 0.480 | 61 |
| DS005620 | alpha | gcc_triad_plus_pi | 0.751 | 0.863 | 0.591 | 61 |
| DS005620 | alpha | residual_gcc_after_spectral_all | 0.535 | 0.762 | 0.741 | 61 |
| DS005620 | alpha | residual_pi_after_spectral_all | 0.577 | 0.787 | 0.702 | 61 |
| DS005620 | alpha | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | alpha | spectral_all_plus_gcc | 0.975 | 0.963 | 0.187 | 61 |
| DS005620 | alpha | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |
| Chennu | gamma | gcc_pi | 0.950 | 0.950 | 0.214 | 40 |
| Chennu | gamma | gcc_triad_plus_pi | 0.950 | 0.950 | 0.228 | 40 |
| Chennu | gamma | residual_gcc_after_spectral_all | 0.700 | 0.850 | 0.888 | 40 |
| Chennu | gamma | residual_pi_after_spectral_all | 0.700 | 0.850 | 0.797 | 40 |
| Chennu | gamma | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | gamma | spectral_all_plus_gcc | 0.903 | 0.900 | 0.288 | 40 |
| Chennu | gamma | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | gamma | gcc_pi | 0.956 | 0.950 | 0.224 | 61 |
| DS005620 | gamma | gcc_triad_plus_pi | 0.975 | 0.938 | 0.238 | 61 |
| DS005620 | gamma | residual_gcc_after_spectral_all | 0.537 | 0.750 | 0.781 | 61 |
| DS005620 | gamma | residual_pi_after_spectral_all | 0.607 | 0.787 | 0.705 | 61 |
| DS005620 | gamma | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | gamma | spectral_all_plus_gcc | 1.000 | 0.963 | 0.111 | 61 |
| DS005620 | gamma | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |



## Cross-Dataset AUC

| band | direction | model | auc | balanced_accuracy | log_loss | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| alpha | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.600 | 0.775 | 1.108 | 40 | 61 |
| alpha | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.625 | 0.725 | 1.848 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.300 | 0.650 | 3.293 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.350 | 0.650 | 1.598 | 40 | 61 |
| alpha | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| alpha | DS005620_to_Chennu | gcc_triad_plus_pi | 0.800 | 0.875 | 0.417 | 61 | 40 |
| alpha | DS005620_to_Chennu | spectral_all_plus_gcc | 0.700 | 0.700 | 0.996 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.800 | 0.875 | 0.591 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.800 | 0.900 | 0.632 | 61 | 40 |
| gamma | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| gamma | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.825 | 0.900 | 0.634 | 40 | 61 |
| gamma | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.800 | 0.850 | 0.973 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.550 | 0.762 | 1.231 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.475 | 0.725 | 1.491 | 40 | 61 |
| gamma | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| gamma | DS005620_to_Chennu | gcc_triad_plus_pi | 0.950 | 0.975 | 0.165 | 61 | 40 |
| gamma | DS005620_to_Chennu | spectral_all_plus_gcc | 0.950 | 0.725 | 0.562 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.950 | 0.950 | 0.513 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.900 | 0.925 | 0.549 | 61 | 40 |



## Spectral-Caliper AUC

| dataset | band | caliper_quantile | n_pairs | model | auc | balanced_accuracy | mean_abs_spectral_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | 0.250 | 5 | pi_sign_rule_no_fit | 0.600 | 0.600 | 0.552 |
| Chennu | alpha | 0.250 | 5 | gcc_triad_plus_pi | 0.440 | 0.700 | 0.552 |
| Chennu | alpha | 0.250 | 5 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 0.552 |
| Chennu | alpha | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | alpha | 0.500 | 10 | pi_sign_rule_no_fit | 0.700 | 0.550 | 1.652 |
| Chennu | alpha | 0.500 | 10 | gcc_triad_plus_pi | 0.730 | 0.800 | 1.652 |
| Chennu | alpha | 0.500 | 10 | residual_gcc_after_spectral_all | 0.520 | 0.700 | 1.652 |
| Chennu | alpha | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | alpha | 0.750 | 15 | pi_sign_rule_no_fit | 0.800 | 0.567 | 3.282 |
| Chennu | alpha | 0.750 | 15 | gcc_triad_plus_pi | 0.867 | 0.867 | 3.282 |
| Chennu | alpha | 0.750 | 15 | residual_gcc_after_spectral_all | 0.600 | 0.733 | 3.282 |
| Chennu | alpha | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | alpha | 0.250 | 10 | pi_sign_rule_no_fit | 0.900 | 0.500 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | gcc_triad_plus_pi | 0.900 | 0.950 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | residual_gcc_after_spectral_all | 0.400 | 0.700 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | alpha | 0.500 | 20 | pi_sign_rule_no_fit | 0.850 | 0.550 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | gcc_triad_plus_pi | 0.750 | 0.850 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | residual_gcc_after_spectral_all | 0.453 | 0.725 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | alpha | 0.750 | 30 | pi_sign_rule_no_fit | 0.900 | 0.533 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | gcc_triad_plus_pi | 0.808 | 0.900 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | residual_gcc_after_spectral_all | 0.641 | 0.800 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |
| Chennu | gamma | 0.250 | 5 | pi_sign_rule_no_fit | 1.000 | 0.600 | 0.552 |
| Chennu | gamma | 0.250 | 5 | gcc_triad_plus_pi | 1.000 | 1.000 | 0.552 |
| Chennu | gamma | 0.250 | 5 | residual_gcc_after_spectral_all | 0.800 | 0.900 | 0.552 |
| Chennu | gamma | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | gamma | 0.500 | 10 | pi_sign_rule_no_fit | 1.000 | 0.650 | 1.652 |
| Chennu | gamma | 0.500 | 10 | gcc_triad_plus_pi | 1.000 | 0.950 | 1.652 |
| Chennu | gamma | 0.500 | 10 | residual_gcc_after_spectral_all | 0.500 | 0.750 | 1.652 |
| Chennu | gamma | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | gamma | 0.750 | 15 | pi_sign_rule_no_fit | 0.933 | 0.733 | 3.282 |
| Chennu | gamma | 0.750 | 15 | gcc_triad_plus_pi | 0.933 | 0.933 | 3.282 |
| Chennu | gamma | 0.750 | 15 | residual_gcc_after_spectral_all | 0.796 | 0.833 | 3.282 |
| Chennu | gamma | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | gamma | 0.250 | 10 | pi_sign_rule_no_fit | 0.900 | 0.600 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | gcc_triad_plus_pi | 0.830 | 0.850 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | residual_gcc_after_spectral_all | 0.400 | 0.700 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | gamma | 0.500 | 20 | pi_sign_rule_no_fit | 0.950 | 0.675 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | gcc_triad_plus_pi | 1.000 | 0.925 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | gamma | 0.750 | 30 | pi_sign_rule_no_fit | 0.967 | 0.667 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | gcc_triad_plus_pi | 1.000 | 0.967 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | residual_gcc_after_spectral_all | 0.590 | 0.783 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |



## Bootstrap Incremental AUC: LOSO

Cluster bootstrap by subject; values are AUC differences.

| cell | contrast | delta_auc_mean | ci_low | ci_high | p_delta_le_0 | n_boot |
| --- | --- | --- | --- | --- | --- | --- |
| Chennu_alpha | spectral_all_plus_gcc_minus_spectral_all | 0.004 | -0.128 | 0.150 | 0.457 | 600 |
| Chennu_alpha | residual_gcc_minus_spectral_all | -0.252 | -0.500 | -0.000 | 0.985 | 600 |
| Chennu_alpha | gcc_triad_minus_spectral_all | -0.131 | -0.350 | 0.050 | 0.913 | 600 |
| DS005620_alpha | spectral_all_plus_gcc_minus_spectral_all | -0.003 | -0.015 | 0.000 | 0.993 | 600 |
| DS005620_alpha | residual_gcc_minus_spectral_all | -0.442 | -0.636 | -0.248 | 1.000 | 600 |
| DS005620_alpha | gcc_triad_minus_spectral_all | -0.225 | -0.395 | -0.074 | 1.000 | 600 |
| Chennu_gamma | spectral_all_plus_gcc_minus_spectral_all | -0.050 | -0.150 | 0.000 | 0.970 | 600 |
| Chennu_gamma | residual_gcc_minus_spectral_all | -0.248 | -0.500 | 0.000 | 0.985 | 600 |
| Chennu_gamma | gcc_triad_minus_spectral_all | -0.001 | -0.150 | 0.150 | 0.638 | 600 |
| DS005620_gamma | spectral_all_plus_gcc_minus_spectral_all | 0.021 | 0.000 | 0.064 | 0.347 | 600 |
| DS005620_gamma | residual_gcc_minus_spectral_all | -0.441 | -0.634 | -0.224 | 1.000 | 600 |
| DS005620_gamma | gcc_triad_minus_spectral_all | -0.004 | -0.075 | 0.058 | 0.658 | 600 |



## Interpretation Rule

A credible bandpower-independent claim would require residualized GCC to remain clearly above chance and/or spectral+GCC to improve robustly over spectral-only models with positive bootstrap intervals. If this is not observed, the safer claim is incremental or bandpower-aware regime information, not independence.
