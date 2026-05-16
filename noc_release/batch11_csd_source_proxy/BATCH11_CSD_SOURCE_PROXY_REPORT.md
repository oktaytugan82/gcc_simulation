# Batch 11 CSD Source-Proxy Phase-Only GCC

Date: 2026-05-14

## Definition

This is a source-proxy validation, not individual MRI source reconstruction. Raw EEG is transformed with spherical current source density (surface Laplacian) after standard montage assignment and before phase-only GCC observables are computed.

## Paired CSD Phase-Only Pi Effects

| dataset | band | target | n | baseline_mean | target_mean | mean_delta_baseline_minus_target | delta_ci_low | delta_ci_high | paired_d | wilcoxon_greater_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | moderate | 20 | 0.8276 | 0.7118 | 0.1158 | 0.05114 | 0.1839 | 0.7589 | 0.001689 |
| DS005620 | alpha | sed | 20 | 0.8276 | 0.5227 | 0.3049 | 0.2149 | 0.3932 | 1.472 | 6.013e-05 |
| DS005620 | alpha | sed2 | 20 | 0.8276 | 0.5177 | 0.3098 | 0.211 | 0.4017 | 1.365 | 1.335e-05 |
| Chennu | gamma | moderate | 20 | 0.8276 | 0.6684 | 0.1592 | 0.09914 | 0.217 | 1.115 | 0.0001974 |
| DS005620 | gamma | sed | 20 | 0.8276 | 0.4349 | 0.3927 | 0.2938 | 0.4897 | 1.665 | 9.537e-07 |
| DS005620 | gamma | sed2 | 20 | 0.8276 | 0.4979 | 0.3297 | 0.2295 | 0.427 | 1.405 | 6.676e-06 |

## Delta Feature Table

Rows: 202; datasets: ['Chennu', 'DS005620']; bands: ['alpha', 'gamma'].

Use this as a source-proxy robustness layer, not as a replacement for future MRI-based source reconstruction.
