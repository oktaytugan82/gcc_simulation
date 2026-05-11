# Chennu 2016 Gamma Baseline-Holdout Analysis

This analysis uses the downloaded Cambridge Chennu et al. raw data bundle metadata
(`sedation-restingstate.zip`; MD5 verified separately) and window-resolved gamma
observables generated from the raw EEGLAB recordings in the prior pilot run.

Calibration is performed only on the first half of each participant's baseline
recording. The second half of baseline is then evaluated as an internal holdout
before applying the same bounds to mild sedation, moderate sedation, and recovery.

## Level Summary

| Level | N | Mean Pi | SD | Median |
|---|---:|---:|---:|---:|
| baseline holdout | 20 | 0.658 | 0.032 | 0.659 |
| mild | 20 | 0.599 | 0.085 | 0.626 |
| moderate | 20 | 0.530 | 0.116 | 0.543 |
| recovery | 20 | 0.572 | 0.153 | 0.648 |

## Paired Tests Against Baseline Holdout

| Comparison | N | Mean baseline minus level | Median baseline minus level | Wilcoxon p | Cohen dz | Declines |
|---|---:|---:|---:|---:|---:|---:|
| baseline holdout - mild | 20 | 0.059 | 0.036 | 0.0005856 | 0.75 | 16/20 |
| baseline holdout - moderate | 20 | 0.128 | 0.111 | 0.0001049 | 1.05 | 17/20 |
| baseline holdout - recovery | 20 | 0.086 | 0.029 | 0.04844 | 0.55 | 13/20 |
