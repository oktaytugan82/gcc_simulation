# Batch 10 Phase-Only GCC Raw-Epoch Pipeline

Date: 2026-05-14

## Definition

The EEG signal is bandpassed and transformed to Hilbert phase. GCC observables are computed from unit phasors only: R from exp(i phi), D_eff from covariance of cos(phi)/sin(phi), and M_tau from temporal variance of R(t). Bandpass amplitudes do not enter the GCC observables.

## Paired Phase-Only Pi Effects

| dataset | band | target | n | baseline_mean | target_mean | mean_delta_baseline_minus_target | delta_ci_low | delta_ci_high | paired_d | wilcoxon_greater_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chennu | alpha | moderate | 20 | 0.8276 | 0.5968 | 0.2307 | 0.1379 | 0.3198 | 1.057 | 8.392e-05 |
| DS005620 | alpha | sed | 20 | 0.8276 | 0.594 | 0.2336 | 0.1562 | 0.3101 | 1.254 | 2.384e-05 |
| DS005620 | alpha | sed2 | 20 | 0.8276 | 0.624 | 0.2036 | 0.125 | 0.2837 | 1.039 | 0.0001307 |
| Chennu | gamma | moderate | 20 | 0.8276 | 0.3649 | 0.4626 | 0.3744 | 0.5411 | 2.366 | 1.907e-06 |
| DS005620 | gamma | sed | 20 | 0.8276 | 0.3888 | 0.4387 | 0.3457 | 0.5315 | 1.986 | 1.907e-06 |
| DS005620 | gamma | sed2 | 20 | 0.8276 | 0.4321 | 0.3955 | 0.3053 | 0.4819 | 1.884 | 9.537e-07 |

## Delta Feature Table

Rows: 202; datasets: ['Chennu', 'DS005620']; bands: ['alpha', 'gamma'].

The delta table can be passed directly to Batch 9 bandpower-independence stress tests.
