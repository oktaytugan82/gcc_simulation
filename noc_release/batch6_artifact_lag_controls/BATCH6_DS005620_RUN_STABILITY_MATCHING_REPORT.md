# Batch 6 DS005620 Repeated-Run and Spectrally Closest-Run Control

Date: 2026-05-13

## Aim

This analysis strengthens DS005620 without additional downloads. It asks whether the GCC state effect survives a stricter local control:

1. repeated sedated runs within the same subject are summarized for stability;
2. for each subject, the sedated run that is spectrally closest to the awake recording is selected;
3. awake vs sedated Pi is re-tested on this spectrally closest subset.

This is not a full bandpower-independence proof. It is a conservative local stress test against the objection that the DS005620 effect is driven only by one extreme sedated run per subject.

## Local Data Status

The local DS005620 subset contains:

- 126 BrainVision recordings;
- 21 subjects;
- 21 awake recordings;
- 54 `sed` recordings;
- 51 `sed2` recordings;
- 20 subjects with awake + sed + sed2 available.

The public dataset is reported as 202 recordings and approximately 77.3 GB. The local subset is therefore not the full public dataset.

## Repeated-Run Stability

Within-subject Pi variability across repeated sedated runs:

| Band | Condition | Subjects with repeats | Mean within-subject SD | Median within-subject SD | Mean range |
|---|---:|---:|---:|---:|---:|
| alpha | sed | 20 | 0.073 | 0.057 | 0.132 |
| alpha | sed2 | 18 | 0.057 | 0.049 | 0.104 |
| gamma | sed | 20 | 0.113 | 0.072 | 0.197 |
| gamma | sed2 | 18 | 0.092 | 0.038 | 0.164 |

Interpretation:

- Repeated sedated runs are not identical; gamma is more run-variable than alpha.
- This supports the need for subject-level averaging or explicit repeated-run modeling.
- The DS005620 effect should not be described as a single-recording effect only.

## Spectrally Closest-Run Control

For each subject and target condition, the sedated run with the smallest Euclidean distance to the awake recording in standardized spectral feature space was selected. Spectral features:

- theta relative power;
- alpha relative power;
- beta relative power;
- gamma relative power;
- alpha/gamma ratio;
- spectral entropy.

Results:

| Band | Target | n | Awake Pi | Closest sedated Pi | Delta | 95% CI | Paired d | Wilcoxon p | AUC |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| alpha | sed | 20 | 0.831 | 0.655 | 0.176 | [0.104, 0.246] | 1.05 | 1.05e-4 | 0.80 |
| alpha | sed2 | 20 | 0.831 | 0.664 | 0.167 | [0.092, 0.241] | 0.96 | 3.54e-4 | 0.80 |
| gamma | sed | 20 | 0.831 | 0.472 | 0.360 | [0.263, 0.461] | 1.58 | 9.54e-7 | 1.00 |
| gamma | sed2 | 20 | 0.831 | 0.476 | 0.355 | [0.263, 0.449] | 1.63 | 9.54e-7 | 1.00 |

## Interpretation

The GCC Pi effect survives when each subject's sedated comparison is restricted to the locally most spectrally similar run. This does not overturn the Batch 5 residualization result: GCC still overlaps substantially with spectral power. But it adds a useful nuance:

> The DS005620 effect is not only an artifact of choosing extreme sedated runs; it remains visible even under a within-subject closest-spectral-run control.

This is paper-useful because it directly addresses the concern that the DS005620 AUC is inflated by repeated-run selection or an unusually separated sedated subset.

## Recommended Manuscript Sentence

In DS005620, the sedation effect remained when, for each subject, the sedated run closest to the awake recording in standardized spectral feature space was selected. Under this closest-run control, gamma Pi still declined strongly from awake to sedation (awake vs sed: Delta Pi = 0.360, 95% CI [0.263, 0.461], paired d = 1.58, Wilcoxon p = 9.5e-7). This does not establish spectral independence, but argues against the effect being driven solely by extreme sedated runs.

## Output Files

- `ds005620_recording_gcc_spectral_merged.csv`
- `ds005620_repeated_run_stability.csv`
- `ds005620_spectrally_closest_runs.csv`
- `ds005620_spectrally_closest_distances.csv`
- `ds005620_spectrally_closest_stats.csv`
- `ds005620_spectrally_closest_run_alpha.png`
- `ds005620_spectrally_closest_run_gamma.png`
- `ds005620_run_stability_matching_summary.json`
