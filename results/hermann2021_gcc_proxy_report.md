# Hermann 2021 GCC Proxy Analysis

This analysis uses the public Hermann et al. 2021 supplementary metadata table.
Raw EEG and FDG-PET recordings are not public, so this is not a raw-signal GCC
feature extraction. It is a conservative proxy analysis:

- PET MIBH is treated as metabolic capacity / preserved substrate.
- Out-of-sample EEG SVM p(MCS) is treated as dynamic access evidence.
- Local-global response is used only as a secondary ordinal task-response proxy.
- GCC PET+EEG proxy is the mean of z-scored PET MIBH and EEG SVM p(MCS).

## Cohort

- Total rows: 64
- Baseline: {'EMCS': 7, 'VS/UWS': 23, 'MCS-': 22, 'MCS+': 12}
- 6-month outcome: {'Conscious': 14, 'VS/UWS': 18, 'Dead': 9, 'MCS+': 10, 'MCS-': 13}

## Diagnostic Target: Baseline MCS-/MCS+ vs VS/UWS

| Score | N | AUC | 95% bootstrap CI | Mann-Whitney p |
|---|---:|---:|---:|---:|
| PET MIBH | 57 | 0.823 | 0.698-0.925 | 4.141e-05 |
| EEG SVM p(MCS) | 52 | 0.771 | 0.626-0.893 | 0.001022 |
| Local-global ordinal | 52 | 0.604 | 0.454-0.745 | 0.18 |
| GCC proxy PET+EEG | 52 | 0.834 | 0.707-0.939 | 5.19e-05 |
| GCC proxy PET+EEG+LG | 52 | 0.813 | 0.674-0.932 | 0.0001532 |

PET Youden threshold for this table: 3.090

## Prognostic Target: 6-Month MCS+/Conscious vs Other

| Score | N | AUC | 95% bootstrap CI | Mann-Whitney p |
|---|---:|---:|---:|---:|
| PET MIBH | 64 | 0.720 | 0.590-0.839 | 0.00343 |
| EEG SVM p(MCS) | 59 | 0.690 | 0.547-0.819 | 0.01462 |
| Local-global ordinal | 59 | 0.582 | 0.440-0.723 | 0.2644 |
| GCC proxy PET+EEG | 59 | 0.762 | 0.635-0.876 | 0.000766 |
| GCC proxy PET+EEG+LG | 59 | 0.736 | 0.601-0.851 | 0.002504 |

## Paired AUC-Delta Tests

The table below uses paired patient-level bootstrap resampling over complete
cases for both compared scores. It tests whether the descriptive AUC advantage
of the GCC PET+EEG proxy is stable against paired sampling uncertainty.

| Target | Comparison | N | Delta AUC | 95% paired bootstrap CI | Two-sided bootstrap p |
|---|---|---:|---:|---:|---:|
| Baseline MCS-/MCS+ vs VS/UWS | GCC proxy PET+EEG - PET MIBH | 52 | +0.017 | -0.062 to +0.098 | 0.6745 |
| Baseline MCS-/MCS+ vs VS/UWS | GCC proxy PET+EEG - EEG SVM p(MCS) | 52 | +0.063 | -0.002 to +0.137 | 0.05868 |
| 6-month MCS+/Conscious vs Other | GCC proxy PET+EEG - PET MIBH | 59 | +0.040 | -0.054 to +0.140 | 0.4096 |
| 6-month MCS+/Conscious vs Other | GCC proxy PET+EEG - EEG SVM p(MCS) | 59 | +0.072 | -0.007 to +0.160 | 0.07836 |

## GCC Gate Summary

PET is thresholded at the diagnostic Youden threshold above. EEG is thresholded
at the out-of-sample SVM decision boundary of 0.50.

| Gate | N | Command-following outcome rate | Patients |
|---|---:|---:|---|
| capacity_only | 16 | 0.500 | ['P2', 'P5', 'P8', 'P12', 'P23', 'P25', 'P26', 'P40', 'P46', 'P47', 'P49', 'P50', 'P55', 'P57', 'P59', 'P61'] |
| concordant_high | 23 | 0.565 | ['P1', 'P3', 'P4', 'P6', 'P9', 'P10', 'P11', 'P15', 'P20', 'P21', 'P27', 'P28', 'P29', 'P33', 'P36', 'P43', 'P45', 'P48', 'P51', 'P52', 'P58', 'P62', 'P64'] |
| concordant_low | 16 | 0.000 | ['P13', 'P14', 'P16', 'P18', 'P22', 'P31', 'P32', 'P34', 'P35', 'P37', 'P39', 'P41', 'P54', 'P56', 'P60', 'P63'] |
| dynamics_only | 4 | 0.500 | ['P7', 'P30', 'P38', 'P44'] |
| missing_eeg | 5 | 0.200 | ['P17', 'P19', 'P24', 'P42', 'P53'] |

Baseline VS/UWS patients with concordant PET+EEG high evidence:
['P9', 'P48']

Baseline VS/UWS patients with any PET or EEG positive evidence:
['P2', 'P9', 'P38', 'P44', 'P47', 'P48', 'P49']

## Outputs

- Features: `results/hermann2021_gcc_proxy_features.csv`
- Summary: `results/hermann2021_gcc_proxy_summary.json`
- Diagnostic figure: `figures/hermann2021_gcc_proxy_diagnostic.png`
- Outcome figure: `figures/hermann2021_gcc_proxy_outcome.png`
