# Batch 5: Bandpower Residualization and Gamma Artifact Control

Date: 2026-05-13

## Aim

This batch tests two reviewer-critical questions:

1. Does GCC regime occupancy carry condition information beyond conventional spectral bandpower?
2. Is the DS005620 gamma-band GCC effect plausibly reducible to high-frequency or muscle-related artifact?

The answer is deliberately split. The bandpower result is conservative and uncomfortable: much of the condition information overlaps with spectral power. The artifact control is stronger: the gamma GCC effect survives posterior and centro-posterior channel restrictions and is not explained by a 70-110 Hz high-frequency proxy.

## Data and Methods

### Residualization Against Bandpower

Input:

- `gcc_batch4_20260513/results/combined_gcc_spectral_normalized_features.csv`
- Datasets: Chennu sedation and DS005620 propofol subset
- Bands: alpha and gamma
- Features: subject-normalized GCC shifts and subject-normalized spectral shifts

Models tested:

- `spectral`: theta, alpha, beta, gamma relative bandpower shifts
- `pi_only`: GCC regime occupancy shift only
- `residual_pi`: Pi after linear residualization against spectral features
- `spectral_plus_pi`: spectral features plus Pi

Validation:

- Within-dataset leave-one-subject-out classification
- Cross-dataset transfer: Chennu -> DS005620 and DS005620 -> Chennu
- Two spectral feature sets:
  - bandpower-only: theta, alpha, beta, gamma power
  - all-spectral: bandpower plus alpha/gamma ratio and spectral entropy

### Gamma Artifact Control

Input:

- `gcc_batch3_20260513/data/ds005620_subset`
- 126 BrainVision EEG recordings
- 21 subjects
- Conditions: awake eyes closed, propofol sedation `sed`, propofol sedation `sed2`

Processing:

- Crop: first 120 s per recording
- Resampling: 250 Hz
- Average reference
- Per-channel z-scoring
- Gamma GCC band: 30-45 Hz
- High-frequency artifact proxy: relative 70-110 Hz power over 1-110 Hz total power
- Windowing: 2.0 s windows, 0.5 s stride
- Subject-wise awake calibration, alpha = 0.20

Channel sets:

- `all_eeg`: 62 channels
- `posterior`: 18 posterior/occipital/parietal channels
- `centro_posterior`: 32 central, centro-parietal, parietal and occipital channels
- `frontotemporal`: 30 frontal/frontocentral/frontotemporal/temporal channels

## Main Results

### 1. Residualization Against Bandpower

Bandpower-only residualization:

| Band | Scope | Spectral AUC | Pi AUC | Residual Pi AUC | Spectral + Pi AUC |
|---|---:|---:|---:|---:|---:|
| alpha | Chennu LOSO | 0.915 | 0.810 | 0.612 | 0.850 |
| alpha | DS005620 LOSO | 0.975 | 0.850 | 0.543 | 0.950 |
| alpha | Chennu -> DS005620 | 0.975 | 0.850 | 0.350 | 1.000 |
| alpha | DS005620 -> Chennu | 0.900 | 0.850 | 0.450 | 0.850 |
| gamma | Chennu LOSO | 0.915 | 0.870 | 0.605 | 0.900 |
| gamma | DS005620 LOSO | 0.975 | 1.000 | 0.684 | 1.000 |
| gamma | Chennu -> DS005620 | 0.975 | 1.000 | 0.925 | 1.000 |
| gamma | DS005620 -> Chennu | 0.900 | 0.900 | 0.300 | 0.750 |

All-spectral residualization:

| Band | Scope | Spectral AUC | Pi AUC | Residual Pi AUC | Spectral + Pi AUC |
|---|---:|---:|---:|---:|---:|
| alpha | Chennu LOSO | 0.950 | 0.810 | 0.576 | 0.920 |
| alpha | DS005620 LOSO | 0.977 | 0.850 | 0.475 | 0.975 |
| alpha | Chennu -> DS005620 | 0.775 | 0.850 | 0.400 | 0.800 |
| alpha | DS005620 -> Chennu | 0.750 | 0.850 | 0.500 | 0.750 |
| gamma | Chennu LOSO | 0.950 | 0.870 | 0.750 | 0.907 |
| gamma | DS005620 LOSO | 0.977 | 1.000 | 0.588 | 1.000 |
| gamma | Chennu -> DS005620 | 0.775 | 1.000 | 0.600 | 0.900 |
| gamma | DS005620 -> Chennu | 0.750 | 0.900 | 0.350 | 0.750 |

Interpretation:

- Raw Pi remains a strong state marker, especially in gamma.
- After residualization, Pi is often weakened substantially.
- Adding Pi to spectral features does not produce a consistent within-dataset gain.
- Therefore, the current evidence does not justify the claim that GCC is broadly independent of spectral bandpower.
- The defensible claim is narrower: GCC provides a regime-level composite that is reproducible across datasets and partially overlaps with classical spectral state markers.

### 2. Gamma Artifact Control

Gamma GCC Pi remains reduced under sedation across all channel restrictions:

| Channel set | Comparison | Awake Pi | Sedated Pi | Difference | Paired d | p | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| all_eeg | awake vs sed | 0.662 | 0.365 | 0.297 | 2.133 | 1.12e-08 | 1.000 |
| all_eeg | awake vs sed2 | 0.662 | 0.374 | 0.289 | 1.922 | 5.70e-08 | 1.000 |
| posterior | awake vs sed | 0.662 | 0.452 | 0.210 | 1.274 | 1.71e-05 | 1.000 |
| posterior | awake vs sed2 | 0.662 | 0.443 | 0.220 | 1.344 | 8.74e-06 | 1.000 |
| centro_posterior | awake vs sed | 0.662 | 0.454 | 0.209 | 1.156 | 5.44e-05 | 0.950 |
| centro_posterior | awake vs sed2 | 0.662 | 0.459 | 0.204 | 1.177 | 4.43e-05 | 1.000 |
| frontotemporal | awake vs sed | 0.662 | 0.454 | 0.208 | 1.470 | 2.71e-06 | 1.000 |
| frontotemporal | awake vs sed2 | 0.662 | 0.459 | 0.203 | 1.356 | 7.86e-06 | 0.950 |

High-frequency 70-110 Hz proxy:

| Channel set | Comparison | Awake HF | Sedated HF | Difference | Paired d | p | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| all_eeg | awake vs sed | 0.043 | 0.021 | 0.022 | 0.540 | 0.026 | 0.683 |
| all_eeg | awake vs sed2 | 0.043 | 0.020 | 0.023 | 0.625 | 0.012 | 0.728 |
| posterior | awake vs sed | 0.020 | 0.023 | -0.003 | -0.091 | 0.690 | 0.530 |
| posterior | awake vs sed2 | 0.020 | 0.022 | -0.002 | -0.071 | 0.755 | 0.550 |
| centro_posterior | awake vs sed | 0.032 | 0.022 | 0.010 | 0.276 | 0.232 | 0.598 |
| centro_posterior | awake vs sed2 | 0.032 | 0.024 | 0.008 | 0.220 | 0.338 | 0.598 |
| frontotemporal | awake vs sed | 0.053 | 0.019 | 0.034 | 0.732 | 0.004 | 0.760 |
| frontotemporal | awake vs sed2 | 0.053 | 0.017 | 0.036 | 0.824 | 0.002 | 0.803 |

Delta Pi vs delta HF correlations:

| Channel set | Comparison | Spearman rho | p |
|---|---|---:|---:|
| all_eeg | awake vs sed | -0.349 | 0.132 |
| all_eeg | awake vs sed2 | 0.140 | 0.556 |
| posterior | awake vs sed | 0.122 | 0.609 |
| posterior | awake vs sed2 | 0.167 | 0.482 |
| centro_posterior | awake vs sed | -0.080 | 0.736 |
| centro_posterior | awake vs sed2 | 0.217 | 0.359 |
| frontotemporal | awake vs sed | -0.041 | 0.865 |
| frontotemporal | awake vs sed2 | 0.081 | 0.734 |

Interpretation:

- The gamma GCC Pi effect persists when analysis is restricted to posterior channels.
- Posterior 70-110 Hz power does not distinguish awake from sedation.
- Subject-level changes in Pi do not correlate with changes in the high-frequency proxy.
- Frontotemporal HF power does change under sedation, which is expected and supports the usefulness of the artifact check.
- The gamma GCC effect is therefore unlikely to be only a frontotemporal muscle/high-frequency artifact.

## Consequences for the Paper

The evidence now supports this cautious statement:

> GCC regime occupancy is a reproducible regime-level marker of sedation-related state change across two independent datasets. Its signal overlaps substantially with conventional spectral power features, but the gamma-band DS005620 effect survives posterior topographic restriction and is not explained by a 70-110 Hz high-frequency artifact proxy.

The paper should not claim:

- GCC outperforms spectral bandpower.
- GCC is independent of spectral power in general.
- Gamma GCC is direct evidence for gamma-specific consciousness mechanisms.
- The gamma result is fully artifact-free.

The paper can claim:

- GCC is empirically measurable.
- GCC reproduces across independent sedation datasets.
- GCC has incremental conceptual value as a multi-observable regime marker.
- The gamma effect is not trivially reducible to frontotemporal high-frequency artifact in DS005620.

## Output Files

- `gcc_batch5_20260513/batch5_bandpower_residualization.py`
- `gcc_batch5_20260513/batch5_ds005620_gamma_artifact_control.py`
- `gcc_batch5_20260513/results/bandpower_residualization_summary.json`
- `gcc_batch5_20260513/results/ds005620_gamma_artifact_summary.json`
- `gcc_batch5_20260513/results/ds005620_gamma_artifact_paired_stats.csv`
- `gcc_batch5_20260513/results/ds005620_gamma_artifact_delta_correlations.csv`
- `gcc_batch5_20260513/results/ds005620_gamma_artifact_subject_condition.csv`
- `gcc_batch5_20260513/results/gamma_artifact_topography_awake_vs_sed.png`
- `gcc_batch5_20260513/results/gamma_artifact_topography_awake_vs_sed2.png`

## Bottom Line

Batch 5 strengthens the paper, but in a precise way. It does not show that GCC is spectrally independent. It shows that GCC is a reproducible regime marker whose gamma implementation is not obviously a high-frequency muscle artifact. This is scientifically stronger than an overclaim, because it closes one serious artifact objection while preserving the honest limitation that spectral power explains a large part of the state separation.
