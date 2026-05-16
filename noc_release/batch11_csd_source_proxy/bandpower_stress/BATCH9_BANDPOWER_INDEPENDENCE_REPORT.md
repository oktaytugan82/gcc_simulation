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
| Chennu | alpha | gcc_pi | 0.700 | 0.825 | 0.563 | 40 |
| Chennu | alpha | gcc_triad_plus_pi | 0.722 | 0.825 | 0.570 | 40 |
| Chennu | alpha | residual_gcc_after_spectral_all | 0.605 | 0.800 | 0.927 | 40 |
| Chennu | alpha | residual_pi_after_spectral_all | 0.605 | 0.800 | 0.692 | 40 |
| Chennu | alpha | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | alpha | spectral_all_plus_gcc | 0.900 | 0.900 | 0.390 | 40 |
| Chennu | alpha | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | alpha | gcc_pi | 0.892 | 0.887 | 0.364 | 61 |
| DS005620 | alpha | gcc_triad_plus_pi | 0.888 | 0.850 | 0.411 | 61 |
| DS005620 | alpha | residual_gcc_after_spectral_all | 0.655 | 0.812 | 0.757 | 61 |
| DS005620 | alpha | residual_pi_after_spectral_all | 0.631 | 0.800 | 0.704 | 61 |
| DS005620 | alpha | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | alpha | spectral_all_plus_gcc | 1.000 | 0.963 | 0.113 | 61 |
| DS005620 | alpha | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |
| Chennu | gamma | gcc_pi | 0.802 | 0.825 | 0.475 | 40 |
| Chennu | gamma | gcc_triad_plus_pi | 0.950 | 0.950 | 0.249 | 40 |
| Chennu | gamma | residual_gcc_after_spectral_all | 0.693 | 0.825 | 0.674 | 40 |
| Chennu | gamma | residual_pi_after_spectral_all | 0.650 | 0.775 | 0.682 | 40 |
| Chennu | gamma | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | gamma | spectral_all_plus_gcc | 0.900 | 0.925 | 0.332 | 40 |
| Chennu | gamma | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | gamma | gcc_pi | 0.955 | 0.887 | 0.317 | 61 |
| DS005620 | gamma | gcc_triad_plus_pi | 0.935 | 0.887 | 0.336 | 61 |
| DS005620 | gamma | residual_gcc_after_spectral_all | 0.604 | 0.762 | 0.755 | 61 |
| DS005620 | gamma | residual_pi_after_spectral_all | 0.604 | 0.787 | 0.708 | 61 |
| DS005620 | gamma | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | gamma | spectral_all_plus_gcc | 1.000 | 0.988 | 0.099 | 61 |
| DS005620 | gamma | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |



## Cross-Dataset AUC

| band | direction | model | auc | balanced_accuracy | log_loss | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| alpha | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.875 | 0.925 | 0.355 | 40 | 61 |
| alpha | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.950 | 0.950 | 0.269 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.800 | 0.900 | 0.540 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.975 | 0.950 | 0.355 | 40 | 61 |
| alpha | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| alpha | DS005620_to_Chennu | gcc_triad_plus_pi | 0.700 | 0.700 | 0.627 | 61 | 40 |
| alpha | DS005620_to_Chennu | spectral_all_plus_gcc | 0.500 | 0.625 | 1.408 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.150 | 0.575 | 1.068 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.200 | 0.575 | 1.079 | 61 | 40 |
| gamma | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| gamma | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.650 | 0.750 | 1.185 | 40 | 61 |
| gamma | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.700 | 0.775 | 1.176 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.250 | 0.588 | 3.790 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.725 | 0.812 | 0.777 | 40 | 61 |
| gamma | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| gamma | DS005620_to_Chennu | gcc_triad_plus_pi | 0.800 | 0.800 | 0.494 | 61 | 40 |
| gamma | DS005620_to_Chennu | spectral_all_plus_gcc | 0.650 | 0.600 | 1.441 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.250 | 0.600 | 1.016 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.450 | 0.675 | 0.863 | 61 | 40 |



## Spectral-Caliper AUC

| dataset | band | caliper_quantile | n_pairs | model | auc | balanced_accuracy | mean_abs_spectral_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | 0.250 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.552 |
| Chennu | alpha | 0.250 | 5 | gcc_triad_plus_pi | 0.400 | 0.700 | 0.552 |
| Chennu | alpha | 0.250 | 5 | residual_gcc_after_spectral_all | 0.200 | 0.600 | 0.552 |
| Chennu | alpha | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | alpha | 0.500 | 10 | pi_sign_rule_no_fit | 0.800 | 0.500 | 1.652 |
| Chennu | alpha | 0.500 | 10 | gcc_triad_plus_pi | 0.810 | 0.850 | 1.652 |
| Chennu | alpha | 0.500 | 10 | residual_gcc_after_spectral_all | 0.200 | 0.600 | 1.652 |
| Chennu | alpha | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | alpha | 0.750 | 15 | pi_sign_rule_no_fit | 0.667 | 0.500 | 3.282 |
| Chennu | alpha | 0.750 | 15 | gcc_triad_plus_pi | 0.738 | 0.767 | 3.282 |
| Chennu | alpha | 0.750 | 15 | residual_gcc_after_spectral_all | 0.796 | 0.867 | 3.282 |
| Chennu | alpha | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | alpha | 0.250 | 10 | pi_sign_rule_no_fit | 0.900 | 0.600 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | gcc_triad_plus_pi | 0.580 | 0.750 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | residual_gcc_after_spectral_all | 0.300 | 0.650 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | alpha | 0.500 | 20 | pi_sign_rule_no_fit | 0.950 | 0.625 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | gcc_triad_plus_pi | 0.910 | 0.850 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | residual_gcc_after_spectral_all | 0.550 | 0.775 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | alpha | 0.750 | 30 | pi_sign_rule_no_fit | 0.967 | 0.600 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | gcc_triad_plus_pi | 0.946 | 0.900 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | residual_gcc_after_spectral_all | 0.542 | 0.767 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |
| Chennu | gamma | 0.250 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.552 |
| Chennu | gamma | 0.250 | 5 | gcc_triad_plus_pi | 0.800 | 0.800 | 0.552 |
| Chennu | gamma | 0.250 | 5 | residual_gcc_after_spectral_all | 0.800 | 0.900 | 0.552 |
| Chennu | gamma | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | gamma | 0.500 | 10 | pi_sign_rule_no_fit | 0.800 | 0.500 | 1.652 |
| Chennu | gamma | 0.500 | 10 | gcc_triad_plus_pi | 0.910 | 0.950 | 1.652 |
| Chennu | gamma | 0.500 | 10 | residual_gcc_after_spectral_all | 0.670 | 0.800 | 1.652 |
| Chennu | gamma | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | gamma | 0.750 | 15 | pi_sign_rule_no_fit | 0.733 | 0.500 | 3.282 |
| Chennu | gamma | 0.750 | 15 | gcc_triad_plus_pi | 0.893 | 0.933 | 3.282 |
| Chennu | gamma | 0.750 | 15 | residual_gcc_after_spectral_all | 0.662 | 0.800 | 3.282 |
| Chennu | gamma | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | gamma | 0.250 | 10 | pi_sign_rule_no_fit | 1.000 | 0.500 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | gcc_triad_plus_pi | 0.740 | 0.750 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | residual_gcc_after_spectral_all | 0.580 | 0.750 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | gamma | 0.500 | 20 | pi_sign_rule_no_fit | 0.950 | 0.625 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | gcc_triad_plus_pi | 0.900 | 0.850 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | residual_gcc_after_spectral_all | 0.597 | 0.775 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | gamma | 0.750 | 30 | pi_sign_rule_no_fit | 0.967 | 0.650 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | gcc_triad_plus_pi | 0.880 | 0.900 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | residual_gcc_after_spectral_all | 0.533 | 0.750 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |



## Bootstrap Incremental AUC: LOSO

Cluster bootstrap by subject; values are AUC differences.

| cell | contrast | delta_auc_mean | ci_low | ci_high | p_delta_le_0 | n_boot |
| --- | --- | --- | --- | --- | --- | --- |
| Chennu_alpha | spectral_all_plus_gcc_minus_spectral_all | -0.049 | -0.150 | 0.000 | 0.973 | 600 |
| Chennu_alpha | residual_gcc_minus_spectral_all | -0.344 | -0.550 | -0.150 | 1.000 | 600 |
| Chennu_alpha | gcc_triad_minus_spectral_all | -0.224 | -0.440 | 0.000 | 0.982 | 600 |
| DS005620_alpha | spectral_all_plus_gcc_minus_spectral_all | 0.021 | 0.000 | 0.064 | 0.347 | 600 |
| DS005620_alpha | residual_gcc_minus_spectral_all | -0.322 | -0.530 | -0.108 | 0.998 | 600 |
| DS005620_alpha | gcc_triad_minus_spectral_all | -0.087 | -0.211 | 0.008 | 0.962 | 600 |
| Chennu_gamma | spectral_all_plus_gcc_minus_spectral_all | -0.052 | -0.150 | 0.000 | 0.977 | 600 |
| Chennu_gamma | residual_gcc_minus_spectral_all | -0.253 | -0.500 | -0.050 | 0.990 | 600 |
| Chennu_gamma | gcc_triad_minus_spectral_all | -0.001 | -0.150 | 0.150 | 0.635 | 600 |
| DS005620_gamma | spectral_all_plus_gcc_minus_spectral_all | 0.021 | 0.000 | 0.064 | 0.347 | 600 |
| DS005620_gamma | residual_gcc_minus_spectral_all | -0.375 | -0.593 | -0.174 | 1.000 | 600 |
| DS005620_gamma | gcc_triad_minus_spectral_all | -0.040 | -0.111 | 0.024 | 0.877 | 600 |



## Interpretation Rule

A credible bandpower-independent claim would require residualized GCC to remain clearly above chance and/or spectral+GCC to improve robustly over spectral-only models with positive bootstrap intervals. If this is not observed, the safer claim is incremental or bandpower-aware regime information, not independence.
