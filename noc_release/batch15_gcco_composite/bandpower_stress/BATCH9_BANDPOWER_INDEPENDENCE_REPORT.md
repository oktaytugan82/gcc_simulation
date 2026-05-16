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
| Chennu | alpha | gcc_pi | 0.762 | 0.825 | 0.477 | 40 |
| Chennu | alpha | gcc_triad_plus_pi | 0.750 | 0.800 | 0.516 | 40 |
| Chennu | alpha | residual_gcc_after_spectral_all | 0.655 | 0.800 | 0.780 | 40 |
| Chennu | alpha | residual_pi_after_spectral_all | 0.508 | 0.750 | 0.680 | 40 |
| Chennu | alpha | spectral_all | 0.953 | 0.875 | 0.284 | 40 |
| Chennu | alpha | spectral_all_plus_gcc | 1.000 | 0.850 | 0.263 | 40 |
| Chennu | alpha | spectral_bandpower | 0.910 | 0.900 | 0.289 | 40 |
| DS005620 | alpha | gcc_pi | 0.657 | 0.775 | 0.623 | 61 |
| DS005620 | alpha | gcc_triad_plus_pi | 0.648 | 0.738 | 0.699 | 61 |
| DS005620 | alpha | residual_gcc_after_spectral_all | 0.471 | 0.725 | 0.730 | 61 |
| DS005620 | alpha | residual_pi_after_spectral_all | 0.400 | 0.688 | 0.711 | 61 |
| DS005620 | alpha | spectral_all | 0.954 | 0.938 | 0.248 | 61 |
| DS005620 | alpha | spectral_all_plus_gcc | 0.926 | 0.938 | 0.298 | 61 |
| DS005620 | alpha | spectral_bandpower | 0.951 | 0.963 | 0.229 | 61 |
| FarnesKetamine | alpha | gcc_pi | 0.700 | 0.850 | 0.581 | 20 |
| FarnesKetamine | alpha | gcc_triad_plus_pi | 0.900 | 0.850 | 0.734 | 20 |
| FarnesKetamine | alpha | residual_gcc_after_spectral_all | 0.700 | 0.850 | 0.633 | 20 |
| FarnesKetamine | alpha | residual_pi_after_spectral_all | 0.500 | 0.750 | 0.671 | 20 |
| FarnesKetamine | alpha | spectral_all | 1.000 | 0.900 | 0.285 | 20 |
| FarnesKetamine | alpha | spectral_all_plus_gcc | 0.900 | 0.800 | 0.606 | 20 |
| FarnesKetamine | alpha | spectral_bandpower | 0.650 | 0.750 | 0.640 | 20 |
| Chennu | gamma | gcc_pi | 0.725 | 0.825 | 0.533 | 40 |
| Chennu | gamma | gcc_triad_plus_pi | 0.800 | 0.875 | 0.510 | 40 |
| Chennu | gamma | residual_gcc_after_spectral_all | 0.750 | 0.825 | 0.616 | 40 |
| Chennu | gamma | residual_pi_after_spectral_all | 0.650 | 0.800 | 0.685 | 40 |
| Chennu | gamma | spectral_all | 0.953 | 0.875 | 0.284 | 40 |
| Chennu | gamma | spectral_all_plus_gcc | 1.000 | 0.950 | 0.200 | 40 |
| Chennu | gamma | spectral_bandpower | 0.910 | 0.900 | 0.289 | 40 |
| DS005620 | gamma | gcc_pi | 1.000 | 0.963 | 0.221 | 61 |
| DS005620 | gamma | gcc_triad_plus_pi | 1.000 | 0.912 | 0.239 | 61 |
| DS005620 | gamma | residual_gcc_after_spectral_all | 0.585 | 0.775 | 0.694 | 61 |
| DS005620 | gamma | residual_pi_after_spectral_all | 0.504 | 0.750 | 0.680 | 61 |
| DS005620 | gamma | spectral_all | 0.954 | 0.938 | 0.248 | 61 |
| DS005620 | gamma | spectral_all_plus_gcc | 1.000 | 0.950 | 0.174 | 61 |
| DS005620 | gamma | spectral_bandpower | 0.951 | 0.963 | 0.229 | 61 |
| FarnesKetamine | gamma | gcc_pi | 0.800 | 0.750 | 0.603 | 20 |
| FarnesKetamine | gamma | gcc_triad_plus_pi | 0.800 | 0.800 | 0.587 | 20 |
| FarnesKetamine | gamma | residual_gcc_after_spectral_all | 0.520 | 0.750 | 0.647 | 20 |
| FarnesKetamine | gamma | residual_pi_after_spectral_all | 0.520 | 0.750 | 0.673 | 20 |
| FarnesKetamine | gamma | spectral_all | 1.000 | 0.900 | 0.285 | 20 |
| FarnesKetamine | gamma | spectral_all_plus_gcc | 0.940 | 0.900 | 0.283 | 20 |
| FarnesKetamine | gamma | spectral_bandpower | 0.650 | 0.750 | 0.640 | 20 |



## Cross-Dataset AUC

| band | direction | model | auc | balanced_accuracy | log_loss | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | Chennu_to_DS005620 | spectral_all | 0.875 | 0.875 | 0.730 | 40 | 61 |
| alpha | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.650 | 0.762 | 0.672 | 40 | 61 |
| alpha | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.875 | 0.925 | 0.573 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.700 | 0.825 | 0.782 | 40 | 61 |
| alpha | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.425 | 0.700 | 0.818 | 40 | 61 |
| alpha | DS005620_to_Chennu | spectral_all | 0.900 | 0.850 | 0.349 | 61 | 40 |
| alpha | DS005620_to_Chennu | gcc_triad_plus_pi | 0.600 | 0.750 | 0.588 | 61 | 40 |
| alpha | DS005620_to_Chennu | spectral_all_plus_gcc | 0.900 | 0.850 | 0.389 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.500 | 0.750 | 0.683 | 61 | 40 |
| alpha | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.600 | 0.775 | 0.687 | 61 | 40 |
| gamma | Chennu_to_DS005620 | spectral_all | 0.875 | 0.875 | 0.730 | 40 | 61 |
| gamma | Chennu_to_DS005620 | gcc_triad_plus_pi | 0.975 | 0.938 | 0.299 | 40 | 61 |
| gamma | Chennu_to_DS005620 | spectral_all_plus_gcc | 0.950 | 0.925 | 0.303 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_gcc_after_spectral_all | 0.525 | 0.750 | 1.789 | 40 | 61 |
| gamma | Chennu_to_DS005620 | residual_pi_after_spectral_all | 0.650 | 0.825 | 1.129 | 40 | 61 |
| gamma | DS005620_to_Chennu | spectral_all | 0.900 | 0.850 | 0.349 | 61 | 40 |
| gamma | DS005620_to_Chennu | gcc_triad_plus_pi | 0.650 | 0.725 | 0.724 | 61 | 40 |
| gamma | DS005620_to_Chennu | spectral_all_plus_gcc | 0.950 | 0.800 | 0.383 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_gcc_after_spectral_all | 0.450 | 0.725 | 0.860 | 61 | 40 |
| gamma | DS005620_to_Chennu | residual_pi_after_spectral_all | 0.450 | 0.725 | 0.700 | 61 | 40 |



## Spectral-Caliper AUC

| dataset | band | caliper_quantile | n_pairs | model | auc | balanced_accuracy | mean_abs_spectral_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | 0.250 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.486 |
| Chennu | alpha | 0.250 | 5 | gcc_triad_plus_pi | 0.400 | 0.700 | 0.486 |
| Chennu | alpha | 0.250 | 5 | residual_gcc_after_spectral_all | 1.000 | 1.000 | 0.486 |
| Chennu | alpha | 0.250 | 5 | spectral_all | 1.000 | 1.000 | 0.486 |
| Chennu | alpha | 0.500 | 10 | pi_sign_rule_no_fit | 0.600 | 0.500 | 1.275 |
| Chennu | alpha | 0.500 | 10 | gcc_triad_plus_pi | 0.430 | 0.700 | 1.275 |
| Chennu | alpha | 0.500 | 10 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 1.275 |
| Chennu | alpha | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.275 |
| Chennu | alpha | 0.750 | 15 | pi_sign_rule_no_fit | 0.733 | 0.500 | 2.242 |
| Chennu | alpha | 0.750 | 15 | gcc_triad_plus_pi | 0.556 | 0.733 | 2.242 |
| Chennu | alpha | 0.750 | 15 | residual_gcc_after_spectral_all | 0.733 | 0.800 | 2.242 |
| Chennu | alpha | 0.750 | 15 | spectral_all | 0.933 | 0.900 | 2.242 |
| DS005620 | alpha | 0.250 | 10 | pi_sign_rule_no_fit | 0.900 | 0.500 | 2.811 |
| DS005620 | alpha | 0.250 | 10 | gcc_triad_plus_pi | 0.810 | 0.850 | 2.811 |
| DS005620 | alpha | 0.250 | 10 | residual_gcc_after_spectral_all | 0.800 | 0.900 | 2.811 |
| DS005620 | alpha | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 2.811 |
| DS005620 | alpha | 0.500 | 20 | pi_sign_rule_no_fit | 0.700 | 0.550 | 2.960 |
| DS005620 | alpha | 0.500 | 20 | gcc_triad_plus_pi | 0.768 | 0.800 | 2.960 |
| DS005620 | alpha | 0.500 | 20 | residual_gcc_after_spectral_all | 0.522 | 0.750 | 2.960 |
| DS005620 | alpha | 0.500 | 20 | spectral_all | 1.000 | 0.975 | 2.960 |
| DS005620 | alpha | 0.750 | 30 | pi_sign_rule_no_fit | 0.667 | 0.550 | 3.262 |
| DS005620 | alpha | 0.750 | 30 | gcc_triad_plus_pi | 0.669 | 0.767 | 3.262 |
| DS005620 | alpha | 0.750 | 30 | residual_gcc_after_spectral_all | 0.500 | 0.733 | 3.262 |
| DS005620 | alpha | 0.750 | 30 | spectral_all | 1.000 | 0.983 | 3.262 |
| FarnesKetamine | alpha | 0.500 | 5 | pi_sign_rule_no_fit | 0.600 | 0.500 | 0.843 |
| FarnesKetamine | alpha | 0.500 | 5 | gcc_triad_plus_pi | 0.800 | 0.700 | 0.843 |
| FarnesKetamine | alpha | 0.500 | 5 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 0.843 |
| FarnesKetamine | alpha | 0.500 | 5 | spectral_all | 0.800 | 0.800 | 0.843 |
| FarnesKetamine | alpha | 0.750 | 7 | pi_sign_rule_no_fit | 0.714 | 0.500 | 1.136 |
| FarnesKetamine | alpha | 0.750 | 7 | gcc_triad_plus_pi | 0.857 | 0.786 | 1.136 |
| FarnesKetamine | alpha | 0.750 | 7 | residual_gcc_after_spectral_all | 0.429 | 0.714 | 1.136 |
| FarnesKetamine | alpha | 0.750 | 7 | spectral_all | 1.000 | 0.929 | 1.136 |
| Chennu | gamma | 0.250 | 5 | pi_sign_rule_no_fit | 0.700 | 0.500 | 0.486 |
| Chennu | gamma | 0.250 | 5 | gcc_triad_plus_pi | 0.320 | 0.600 | 0.486 |
| Chennu | gamma | 0.250 | 5 | residual_gcc_after_spectral_all | 0.800 | 0.900 | 0.486 |
| Chennu | gamma | 0.250 | 5 | spectral_all | 1.000 | 1.000 | 0.486 |
| Chennu | gamma | 0.500 | 10 | pi_sign_rule_no_fit | 0.750 | 0.500 | 1.275 |
| Chennu | gamma | 0.500 | 10 | gcc_triad_plus_pi | 0.600 | 0.750 | 1.275 |
| Chennu | gamma | 0.500 | 10 | residual_gcc_after_spectral_all | 0.400 | 0.700 | 1.275 |
| Chennu | gamma | 0.500 | 10 | spectral_all | 1.000 | 0.900 | 1.275 |
| Chennu | gamma | 0.750 | 15 | pi_sign_rule_no_fit | 0.833 | 0.500 | 2.242 |
| Chennu | gamma | 0.750 | 15 | gcc_triad_plus_pi | 0.804 | 0.867 | 2.242 |
| Chennu | gamma | 0.750 | 15 | residual_gcc_after_spectral_all | 0.733 | 0.833 | 2.242 |
| Chennu | gamma | 0.750 | 15 | spectral_all | 0.933 | 0.900 | 2.242 |
| DS005620 | gamma | 0.250 | 10 | pi_sign_rule_no_fit | 1.000 | 0.500 | 2.811 |
| DS005620 | gamma | 0.250 | 10 | gcc_triad_plus_pi | 1.000 | 0.900 | 2.811 |
| DS005620 | gamma | 0.250 | 10 | residual_gcc_after_spectral_all | 0.500 | 0.750 | 2.811 |
| DS005620 | gamma | 0.250 | 10 | spectral_all | 1.000 | 1.000 | 2.811 |
| DS005620 | gamma | 0.500 | 20 | pi_sign_rule_no_fit | 1.000 | 0.500 | 2.960 |
| DS005620 | gamma | 0.500 | 20 | gcc_triad_plus_pi | 0.950 | 0.900 | 2.960 |
| DS005620 | gamma | 0.500 | 20 | residual_gcc_after_spectral_all | 0.695 | 0.800 | 2.960 |
| DS005620 | gamma | 0.500 | 20 | spectral_all | 1.000 | 0.975 | 2.960 |
| DS005620 | gamma | 0.750 | 30 | pi_sign_rule_no_fit | 1.000 | 0.500 | 3.262 |
| DS005620 | gamma | 0.750 | 30 | gcc_triad_plus_pi | 1.000 | 0.900 | 3.262 |
| DS005620 | gamma | 0.750 | 30 | residual_gcc_after_spectral_all | 0.718 | 0.833 | 3.262 |
| DS005620 | gamma | 0.750 | 30 | spectral_all | 1.000 | 0.983 | 3.262 |
| FarnesKetamine | gamma | 0.500 | 5 | pi_sign_rule_no_fit | 0.800 | 0.500 | 0.843 |
| FarnesKetamine | gamma | 0.500 | 5 | gcc_triad_plus_pi | 0.640 | 0.600 | 0.843 |
| FarnesKetamine | gamma | 0.500 | 5 | residual_gcc_after_spectral_all | 0.600 | 0.800 | 0.843 |
| FarnesKetamine | gamma | 0.500 | 5 | spectral_all | 0.800 | 0.800 | 0.843 |
| FarnesKetamine | gamma | 0.750 | 7 | pi_sign_rule_no_fit | 0.714 | 0.500 | 1.136 |
| FarnesKetamine | gamma | 0.750 | 7 | gcc_triad_plus_pi | 0.510 | 0.643 | 1.136 |
| FarnesKetamine | gamma | 0.750 | 7 | residual_gcc_after_spectral_all | 0.857 | 0.929 | 1.136 |
| FarnesKetamine | gamma | 0.750 | 7 | spectral_all | 1.000 | 0.929 | 1.136 |



## Bootstrap Incremental AUC: LOSO

Cluster bootstrap by subject; values are AUC differences.

| cell | contrast | delta_auc_mean | ci_low | ci_high | p_delta_le_0 | n_boot |
| --- | --- | --- | --- | --- | --- | --- |
| Chennu_alpha | spectral_all_plus_gcc_minus_spectral_all | 0.049 | 0.000 | 0.150 | 0.333 | 600 |
| Chennu_alpha | residual_gcc_minus_spectral_all | -0.299 | -0.540 | -0.050 | 0.992 | 600 |
| Chennu_alpha | gcc_triad_minus_spectral_all | -0.199 | -0.450 | 0.001 | 0.973 | 600 |
| DS005620_alpha | spectral_all_plus_gcc_minus_spectral_all | -0.031 | -0.087 | 0.000 | 0.998 | 600 |
| DS005620_alpha | residual_gcc_minus_spectral_all | -0.492 | -0.676 | -0.321 | 1.000 | 600 |
| DS005620_alpha | gcc_triad_minus_spectral_all | -0.303 | -0.481 | -0.148 | 1.000 | 600 |
| FarnesKetamine_alpha | spectral_all_plus_gcc_minus_spectral_all | -0.103 | -0.300 | 0.000 | 0.998 | 600 |
| FarnesKetamine_alpha | residual_gcc_minus_spectral_all | -0.307 | -0.600 | -0.100 | 1.000 | 600 |
| FarnesKetamine_alpha | gcc_triad_minus_spectral_all | -0.103 | -0.300 | 0.000 | 0.998 | 600 |
| Chennu_gamma | spectral_all_plus_gcc_minus_spectral_all | 0.049 | 0.000 | 0.150 | 0.333 | 600 |
| Chennu_gamma | residual_gcc_minus_spectral_all | -0.196 | -0.400 | 0.000 | 0.980 | 600 |
| Chennu_gamma | gcc_triad_minus_spectral_all | -0.146 | -0.350 | 0.045 | 0.952 | 600 |
| DS005620_gamma | spectral_all_plus_gcc_minus_spectral_all | 0.045 | 0.000 | 0.139 | 0.347 | 600 |
| DS005620_gamma | residual_gcc_minus_spectral_all | -0.366 | -0.574 | -0.188 | 1.000 | 600 |
| DS005620_gamma | gcc_triad_minus_spectral_all | 0.045 | 0.000 | 0.139 | 0.345 | 600 |
| FarnesKetamine_gamma | spectral_all_plus_gcc_minus_spectral_all | -0.052 | -0.160 | 0.000 | 0.995 | 600 |
| FarnesKetamine_gamma | residual_gcc_minus_spectral_all | -0.482 | -0.800 | -0.180 | 1.000 | 600 |
| FarnesKetamine_gamma | gcc_triad_minus_spectral_all | -0.205 | -0.500 | 0.000 | 1.000 | 600 |



## Interpretation Rule

A credible bandpower-independent claim would require residualized GCC to remain clearly above chance and/or spectral+GCC to improve robustly over spectral-only models with positive bootstrap intervals. If this is not observed, the safer claim is incremental or bandpower-aware regime information, not independence.
