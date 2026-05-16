# Batch 6 Phase-Surrogate Control

Date: 2026-05-13

## Aim

This control tests whether the DS005620 gamma GCC effect depends on temporal/cross-channel phase structure or whether it largely survives when each channel's power spectrum is preserved but its phase is randomized.

## Method

For each selected DS005620 subject:

- use the awake recording;
- use the spectrally closest `sed` and `sed2` recordings from the closest-run control;
- generate 3 phase-randomized surrogates per recording;
- preserve each channel's univariate Fourier amplitude spectrum;
- randomize Fourier phase independently per channel;
- compute gamma-band GCC Pi after surrogate awake calibration.

This destroys temporal phase structure and cross-channel phase relationships while preserving channel-wise spectral power.

## Results

| Kind | Target | n | Awake Pi | Target Pi | Delta | Paired d | p |
|---|---|---:|---:|---:|---:|---:|---:|
| original | sed | 20 | 0.831 | 0.554 | 0.277 | 1.31 | 7.0e-5 |
| original | sed2 | 20 | 0.831 | 0.507 | 0.324 | 1.47 | 6.7e-6 |
| phase surrogate | sed | 20 | 0.831 | 0.600 | 0.231 | 2.60 | 9.5e-7 |
| phase surrogate | sed2 | 20 | 0.831 | 0.561 | 0.270 | 2.51 | 4.4e-5 |

## Interpretation

The phase-randomized surrogate does not eliminate the sedation effect. It reduces the absolute delta relative to the original signal, but a large separation remains.

This means:

- the current channel-level gamma GCC Pi effect is not purely dependent on preserved cross-channel phase relationships;
- a substantial part of the effect is carried by spectral/univariate state changes that survive phase randomization;
- GCC should not be framed as a phase-only or bandpower-independent marker;
- the stronger defensible framing is that GCC provides a regime geometry that integrates spectral, coherence, dimensionality, and temporal-stability changes.

## Consequence for the Paper

This result should not be hidden. It strengthens the paper if reported honestly because it prevents reviewers from discovering the same limitation later.

Recommended wording:

> Phase-randomized surrogate controls preserved much of the DS005620 gamma Pi separation, indicating that the present channel-level implementation is strongly coupled to univariate spectral state structure. The GCC index should therefore be interpreted as a calibrated regime composite rather than a phase-structure-only marker. The original signal showed somewhat larger deltas than the surrogates, but the surrogate effect remained substantial.

## Next Technical Step

To make GCC more independent and stronger, the next analysis should move from channel-level Hilbert phase to volume-conduction-robust coupling:

- surface Laplacian / CSD;
- imaginary coherence or wPLI;
- source-level reconstruction where possible;
- matched-power bins with phase/coupling features tested inside bins.

## Output Files

- `ds005620_phase_surrogate_pi.csv`
- `ds005620_phase_surrogate_stats.csv`
- `ds005620_phase_surrogate_summary.json`
