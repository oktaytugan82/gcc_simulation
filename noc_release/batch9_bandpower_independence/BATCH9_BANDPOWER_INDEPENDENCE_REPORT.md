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
| Chennu | alpha | gcc_pi | 0.810 | 0.800 | 0.584 | 40 |
| Chennu | alpha | gcc_triad_plus_pi | 0.752 | 0.825 | 0.565 | 40 |
| Chennu | alpha | residual_gcc_after_spectral_all | 0.650 | 0.825 | 0.876 | 40 |
| Chennu | alpha | residual_pi_after_spectral_all | 0.576 | 0.750 | 0.710 | 40 |
| Chennu | alpha | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | alpha | spectral_all_plus_gcc | 1.000 | 0.925 | 0.199 | 40 |
| Chennu | alpha | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | alpha | gcc_pi | 0.850 | 0.875 | 0.483 | 61 |
| DS005620 | alpha | gcc_triad_plus_pi | 0.825 | 0.838 | 0.766 | 61 |
| DS005620 | alpha | residual_gcc_after_spectral_all | 0.555 | 0.750 | 0.940 | 61 |
| DS005620 | alpha | residual_pi_after_spectral_all | 0.475 | 0.738 | 0.698 | 61 |
| DS005620 | alpha | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | alpha | spectral_all_plus_gcc | 0.975 | 0.975 | 0.180 | 61 |
| DS005620 | alpha | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |
| Chennu | gamma | gcc_pi | 0.870 | 0.800 | 0.455 | 40 |
| Chennu | gamma | gcc_triad_plus_pi | 0.900 | 0.875 | 0.370 | 40 |
| Chennu | gamma | residual_gcc_after_spectral_all | 0.800 | 0.850 | 0.713 | 40 |
| Chennu | gamma | residual_pi_after_spectral_all | 0.750 | 0.825 | 0.707 | 40 |
| Chennu | gamma | spectral_all | 0.950 | 0.875 | 0.419 | 40 |
| Chennu | gamma | spectral_all_plus_gcc | 0.903 | 0.900 | 0.305 | 40 |
| Chennu | gamma | spectral_bandpower | 0.915 | 0.900 | 0.420 | 40 |
| DS005620 | gamma | gcc_pi | 1.000 | 0.963 | 0.182 | 61 |
| DS005620 | gamma | gcc_triad_plus_pi | 0.975 | 0.950 | 0.232 | 61 |
| DS005620 | gamma | residual_gcc_after_spectral_all | 0.693 | 0.775 | 0.731 | 61 |
| DS005620 | gamma | residual_pi_after_spectral_all | 0.588 | 0.775 | 0.693 | 61 |
| DS005620 | gamma | spectral_all | 0.977 | 0.975 | 0.136 | 61 |
| DS005620 | gamma | spectral_all_plus_gcc | 1.000 | 0.975 | 0.101 | 61 |
| DS005620 | gamma | spectral_bandpower | 0.975 | 0.887 | 0.312 | 61 |



## Cross-Dataset AUC

| band | direction | model | auc | balanced_accuracy | log_loss | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| alpha | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.625 | 0.750 | 1.169 | 40 | 61 |
| alpha | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.550 | 0.725 | 2.348 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.250 | 0.613 | 3.143 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.400 | 0.700 | 0.789 | 40 | 61 |
| alpha | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| alpha | DS005620_to_Chennu | gcc_triad_plus_pi | 0.750 | 0.725 | 0.583 | 61 | 40 |
| alpha | DS005620_to_Chennu | spectral_all_plus_gcc | 0.700 | 0.675 | 1.009 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.550 | 0.775 | 0.660 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.500 | 0.725 | 0.666 | 61 | 40 |
| gamma | Chennu_to_DS005620 | spectral_all | 0.775 | 0.775 | 0.930 | 40 | 61 |
| gamma | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.800 | 0.900 | 1.350 | 40 | 61 |
| gamma | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.775 | 0.887 | 1.489 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.550 | 0.775 | 2.475 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.600 | 0.787 | 1.475 | 40 | 61 |
| gamma | DS005620_to_Chennu | spectral_all | 0.750 | 0.675 | 1.051 | 61 | 40 |
| gamma | DS005620_to_Chennu | gcc_triad_plus_pi | 0.850 | 0.700 | 0.683 | 61 | 40 |
| gamma | DS005620_to_Chennu | spectral_all_plus_gcc | 0.750 | 0.650 | 1.029 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.400 | 0.700 | 0.707 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.350 | 0.650 | 0.716 | 61 | 40 |



## Spectral-Caliper AUC

| dataset | band | caliper_quantile | n_pairs | model | auc | balanced_accuracy | mean_abs_spectral_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | 0.250 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.552 |
| Chennu | alpha | 0.250 | 5 | gcc_triad_plus_pi | 0.480 | 0.700 | 0.552 |
| Chennu | alpha | 0.250 | 5 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 0.552 |
| Chennu | alpha | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | alpha | 0.500 | 10 | pi_sign_rule_no_fit | 0.800 | 0.500 | 1.652 |
| Chennu | alpha | 0.500 | 10 | gcc_triad_plus_pi | 0.730 | 0.800 | 1.652 |
| Chennu | alpha | 0.500 | 10 | residual_gcc_after_spectral_all | 0.700 | 0.850 | 1.652 |
| Chennu | alpha | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | alpha | 0.750 | 15 | pi_sign_rule_no_fit | 0.800 | 0.500 | 3.282 |
| Chennu | alpha | 0.750 | 15 | gcc_triad_plus_pi | 0.867 | 0.867 | 3.282 |
| Chennu | alpha | 0.750 | 15 | residual_gcc_after_spectral_all | 0.689 | 0.800 | 3.282 |
| Chennu | alpha | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | alpha | 0.250 | 10 | pi_sign_rule_no_fit | 0.700 | 0.500 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | gcc_triad_plus_pi | 0.810 | 0.800 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | residual_gcc_after_spectral_all | 0.500 | 0.750 | 3.576 |
| DS005620 | alpha | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | alpha | 0.500 | 20 | pi_sign_rule_no_fit | 0.800 | 0.500 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | gcc_triad_plus_pi | 0.702 | 0.750 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | residual_gcc_after_spectral_all | 0.555 | 0.775 | 4.878 |
| DS005620 | alpha | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | alpha | 0.750 | 30 | pi_sign_rule_no_fit | 0.867 | 0.500 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | gcc_triad_plus_pi | 0.804 | 0.850 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | residual_gcc_after_spectral_all | 0.591 | 0.767 | 4.913 |
| DS005620 | alpha | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |
| Chennu | gamma | 0.250 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.552 |
| Chennu | gamma | 0.250 | 5 | gcc_triad_plus_pi | 1.000 | 0.900 | 0.552 |
| Chennu | gamma | 0.250 | 5 | residual_gcc_after_spectral_all | 0.200 | 0.600 | 0.552 |
| Chennu | gamma | 0.250 | 5 | spectral_all | 1.000 | 0.900 | 0.552 |
| Chennu | gamma | 0.500 | 10 | pi_sign_rule_no_fit | 0.900 | 0.500 | 1.652 |
| Chennu | gamma | 0.500 | 10 | gcc_triad_plus_pi | 1.000 | 0.850 | 1.652 |
| Chennu | gamma | 0.500 | 10 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 1.652 |
| Chennu | gamma | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.652 |
| Chennu | gamma | 0.750 | 15 | pi_sign_rule_no_fit | 0.933 | 0.500 | 3.282 |
| Chennu | gamma | 0.750 | 15 | gcc_triad_plus_pi | 0.933 | 0.867 | 3.282 |
| Chennu | gamma | 0.750 | 15 | residual_gcc_after_spectral_all | 0.604 | 0.800 | 3.282 |
| Chennu | gamma | 0.750 | 15 | spectral_all | 1.000 | 0.900 | 3.282 |
| DS005620 | gamma | 0.250 | 10 | pi_sign_rule_no_fit | 1.000 | 0.600 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | gcc_triad_plus_pi | 0.900 | 0.850 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | residual_gcc_after_spectral_all | 0.200 | 0.600 | 3.576 |
| DS005620 | gamma | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 3.576 |
| DS005620 | gamma | 0.500 | 20 | pi_sign_rule_no_fit | 1.000 | 0.650 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | gcc_triad_plus_pi | 0.950 | 0.950 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | residual_gcc_after_spectral_all | 0.500 | 0.725 | 4.878 |
| DS005620 | gamma | 0.500 | 20 | spectral_all | 1.000 | 1.000 | 4.878 |
| DS005620 | gamma | 0.750 | 30 | pi_sign_rule_no_fit | 1.000 | 0.617 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | gcc_triad_plus_pi | 0.967 | 0.950 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | residual_gcc_after_spectral_all | 0.538 | 0.733 | 4.913 |
| DS005620 | gamma | 0.750 | 30 | spectral_all | 1.000 | 0.967 | 4.913 |



## Bootstrap Incremental AUC: LOSO

Cluster bootstrap by subject; values are AUC differences.

| cell | contrast | delta_auc_mean | ci_low | ci_high | p_delta_le_0 | n_boot |
| --- | --- | --- | --- | --- | --- | --- |
| Chennu_alpha | spectral_all_plus_gcc_minus_spectral_all | 0.051 | 0.000 | 0.150 | 0.337 | 600 |
| Chennu_alpha | residual_gcc_minus_spectral_all | -0.304 | -0.550 | -0.050 | 0.993 | 600 |
| Chennu_alpha | gcc_triad_minus_spectral_all | -0.208 | -0.400 | 0.000 | 0.982 | 600 |
| DS005620_alpha | spectral_all_plus_gcc_minus_spectral_all | -0.003 | -0.015 | 0.000 | 0.993 | 600 |
| DS005620_alpha | residual_gcc_minus_spectral_all | -0.416 | -0.601 | -0.237 | 1.000 | 600 |
| DS005620_alpha | gcc_triad_minus_spectral_all | -0.150 | -0.312 | -0.027 | 1.000 | 600 |
| Chennu_gamma | spectral_all_plus_gcc_minus_spectral_all | -0.050 | -0.150 | 0.000 | 0.975 | 600 |
| Chennu_gamma | residual_gcc_minus_spectral_all | -0.150 | -0.350 | 0.050 | 0.953 | 600 |
| Chennu_gamma | gcc_triad_minus_spectral_all | -0.049 | -0.200 | 0.100 | 0.798 | 600 |
| DS005620_gamma | spectral_all_plus_gcc_minus_spectral_all | 0.021 | 0.000 | 0.064 | 0.347 | 600 |
| DS005620_gamma | residual_gcc_minus_spectral_all | -0.280 | -0.461 | -0.115 | 1.000 | 600 |
| DS005620_gamma | gcc_triad_minus_spectral_all | -0.004 | -0.071 | 0.061 | 0.657 | 600 |



## Interpretation Rule

A credible bandpower-independent claim would require residualized GCC to remain clearly above chance and/or spectral+GCC to improve robustly over spectral-only models with positive bootstrap intervals. If this is not observed, the safer claim is incremental or bandpower-aware regime information, not independence.
