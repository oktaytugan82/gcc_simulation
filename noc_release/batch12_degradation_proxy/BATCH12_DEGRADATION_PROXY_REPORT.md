# Batch 12 ds004504 Degradation-Proxy GCC Analysis

Date: 2026-05-14

## Scope

This analysis treats ds004504 as a neurodegeneration/degradation proxy, not as a direct level-of-consciousness or terminal-lucidity dataset. It tests whether GCC observables carry clinically relevant structure across Alzheimer, frontotemporal dementia, and healthy control groups.

## Group Summary

| GroupName | n | MMSE_mean | alpha_Pi_mean | gamma_Pi_mean | alpha_R_mean | gamma_R_mean |
| --- | --- | --- | --- | --- | --- | --- |
| alzheimer | 36 | 17.75 | 0.4875 | 0.6445 | 0.1323 | 0.1786 |
| control | 29 | 30 | 0.5923 | 0.5804 | 0.1533 | 0.1584 |
| frontotemporal | 23 | 22.17 | 0.4844 | 0.5079 | 0.1419 | 0.1905 |


## Strongest Group Effects

| contrast | feature | group_mean | control_mean | mean_difference | cohen_d_group_minus_control | mannwhitney_p |
| --- | --- | --- | --- | --- | --- | --- |
| alzheimer_minus_control | alpha_R | 0.1323 | 0.1533 | -0.02099 | -0.954 | 0.0002634 |
| alzheimer_minus_control | alpha_D | 3.248 | 2.769 | 0.479 | 0.9061 | 0.0002376 |
| frontotemporal_minus_control | alpha_D | 3.25 | 2.769 | 0.4805 | 0.863 | 0.001263 |
| frontotemporal_minus_control | gamma_R | 0.1905 | 0.1584 | 0.03205 | 0.7078 | 0.01222 |
| alzheimer_minus_control | alpha_Pi | 0.4875 | 0.5923 | -0.1048 | -0.666 | 0.005471 |
| frontotemporal_minus_control | gamma_M | 0.008511 | 0.006681 | 0.00183 | 0.6117 | 0.01222 |
| frontotemporal_minus_control | alpha_Pi | 0.4844 | 0.5923 | -0.1079 | -0.5888 | 0.04661 |
| alzheimer_minus_control | gamma_R | 0.1786 | 0.1584 | 0.02018 | 0.4997 | 0.03199 |
| alzheimer_minus_control | gamma_D | 4.681 | 5.031 | -0.3492 | -0.4989 | 0.04557 |
| frontotemporal_minus_control | alpha_R | 0.1419 | 0.1533 | -0.01147 | -0.4797 | 0.1405 |
| alzheimer_minus_control | gamma_M | 0.007946 | 0.006681 | 0.001265 | 0.438 | 0.03199 |
| alzheimer_minus_control | gamma_Pi | 0.6445 | 0.5804 | 0.06413 | 0.3266 | 0.1031 |


## Strongest MMSE Associations

| subset | feature | n | spearman_rho | p |
| --- | --- | --- | --- | --- |
| all_subjects | alpha_R | 88 | 0.3615 | 0.0005399 |
| all_subjects | alpha_D | 88 | -0.3283 | 0.001792 |
| all_subjects | gamma_R | 88 | -0.2428 | 0.02265 |
| dementia_proxy_only | gamma_Pi | 59 | -0.2423 | 0.06447 |
| all_subjects | alpha_Pi | 88 | 0.2398 | 0.02442 |
| all_subjects | gamma_M | 88 | -0.2308 | 0.03052 |
| all_subjects | gamma_Pi | 88 | -0.1891 | 0.0777 |
| all_subjects | gamma_D | 88 | 0.1471 | 0.1714 |
| dementia_proxy_only | alpha_R | 59 | 0.1463 | 0.2688 |
| dementia_proxy_only | alpha_D | 59 | 0.07858 | 0.5541 |


## Best Leave-One-Subject-Out Classifiers

| task | feature_set | n | auc | balanced_accuracy | accuracy | p_perm_auc_ge_observed |
| --- | --- | --- | --- | --- | --- | --- |
| alzheimer_vs_control | alpha_triad | 65 | 0.7682 | 0.7682 | 0.7692 | 0.003322 |
| alzheimer_vs_control | combined_triad | 65 | 0.7672 | 0.8132 | 0.8154 | 0.006645 |
| alzheimer_vs_control | combined_all | 65 | 0.7548 | 0.7476 | 0.7538 | 0.003322 |
| alzheimer_vs_frontotemporal | combined_triad | 59 | 0.6039 | 0.6516 | 0.661 | 0.1362 |
| alzheimer_vs_frontotemporal | gamma_pi | 59 | 0.5942 | 0.6359 | 0.661 | 0.09967 |
| alzheimer_vs_frontotemporal | alpha_triad | 59 | 0.564 | 0.5664 | 0.5763 | 0.2027 |
| dementia_proxy_vs_control | combined_triad | 88 | 0.7691 | 0.7864 | 0.7841 | 0.003322 |
| dementia_proxy_vs_control | alpha_triad | 88 | 0.7528 | 0.7437 | 0.7386 | 0.003322 |
| dementia_proxy_vs_control | combined_all | 88 | 0.7282 | 0.7265 | 0.7273 | 0.003322 |
| frontotemporal_vs_control | alpha_triad | 52 | 0.6987 | 0.6837 | 0.6923 | 0.02658 |
| frontotemporal_vs_control | combined_triad | 52 | 0.6882 | 0.7054 | 0.7115 | 0.0299 |
| frontotemporal_vs_control | combined_all | 52 | 0.6327 | 0.6274 | 0.6346 | 0.1096 |


## Interpretation

The appropriate claim is limited: GCC features show degradation-proxy sensitivity and can be benchmarked against clinical group labels, but this does not validate the re-entry mechanism or conscious-access criterion. It strengthens empirical breadth by adding a non-anesthesia, clinical degradation axis.
