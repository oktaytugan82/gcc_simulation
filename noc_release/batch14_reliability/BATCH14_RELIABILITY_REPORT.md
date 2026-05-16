# Batch 14 GCC Reliability

Date: 2026-05-14

## Scope

Reliability is estimated from first-half versus second-half windows within recordings and, where available, repeated DS005620 runs. This is a measurement-stability control, not an additional state-classification result.

## Pi Reliability Summary

| scope | source | band | metric | n | pearson_r | spearman_rho | icc_3_1 | mean_abs_delta | mean_signed_delta_second_minus_first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run1_vs_run2_sed | csd_phase_only | alpha | Pi | 20 | 0.775 | 0.774 | 0.7721 | 0.119 | -0.0006466 |
| run1_vs_run2_sed | csd_phase_only | gamma | Pi | 20 | 0.8676 | 0.8499 | 0.863 | 0.09953 | 0.01505 |
| run1_vs_run2_sed | phase_only | alpha | Pi | 20 | 0.7557 | 0.6697 | 0.7062 | 0.1032 | -0.06523 |
| run1_vs_run2_sed | phase_only | gamma | Pi | 20 | 0.6754 | 0.5408 | 0.6738 | 0.1276 | 0.0005747 |
| run1_vs_run2_sed2 | csd_phase_only | alpha | Pi | 18 | 0.7661 | 0.7259 | 0.7444 | 0.1321 | 0.04922 |
| run1_vs_run2_sed2 | csd_phase_only | gamma | Pi | 18 | 0.8958 | 0.8989 | 0.8721 | 0.09162 | 0.02437 |
| run1_vs_run2_sed2 | phase_only | alpha | Pi | 18 | 0.8288 | 0.7775 | 0.8253 | 0.09016 | 0.01901 |
| run1_vs_run2_sed2 | phase_only | gamma | Pi | 18 | 0.7853 | 0.8033 | 0.7758 | 0.1174 | -0.02875 |
| split_half | csd_phase_only | alpha | Pi | 206 | 0.9068 | 0.8265 | 0.9058 | 0.07569 | 0.00849 |
| split_half | csd_phase_only | gamma | Pi | 206 | 0.9258 | 0.8745 | 0.9257 | 0.07287 | -0.002169 |
| split_half | ketamine_phase_only | alpha | Pi | 39 | 0.6026 | 0.5098 | 0.5677 | 0.06166 | -0.01576 |
| split_half | ketamine_phase_only | gamma | Pi | 39 | 0.6011 | 0.4915 | 0.5837 | 0.09226 | -0.02323 |
| split_half | phase_only | alpha | Pi | 206 | 0.8571 | 0.7924 | 0.8555 | 0.08087 | 0.00344 |
| split_half | phase_only | gamma | Pi | 206 | 0.925 | 0.9071 | 0.9223 | 0.07618 | -0.01962 |


## Interpretation

High split-half reliability supports GCC as a stable recording-level measure. Repeated-run reliability is expected to be lower because physiological state, sedation depth, and acquisition runs can vary within nominal condition labels.
