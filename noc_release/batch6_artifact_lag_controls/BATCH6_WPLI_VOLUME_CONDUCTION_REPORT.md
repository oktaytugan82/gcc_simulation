# Batch 6 wPLI / Volume-Conduction Control

Date: 2026-05-13

## Aim

This analysis tests whether the DS005620 GCC effect survives when the coherence observable is made less sensitive to zero-lag channel synchrony and volume conduction.

Instead of the standard Kuramoto order parameter, this control uses a global weighted phase-lag index proxy:

- zero-lag phase alignment contributes little;
- only lagged phase consistency contributes strongly;
- dimensionality and temporal-stability constraints remain part of the GCC regime score.

This is a stress test, not a replacement of the main GCC definition.

## Data and Design

- Dataset: DS005620 local resting-state core.
- Subjects: 21 available; 20 paired awake-vs-sedation subjects for inference.
- Recordings: spectrally closest sedated runs from the Batch 6 closest-run control.
- Conditions: awake, sed, sed2.
- Preprocessing: average reference, channel z-scoring, downsampled to 125 Hz.
- Windowing: 3 s windows, 1.5 s stride, first 90 s.
- Pair sampling: 750 channel pairs per recording.
- Calibration: subject-wise awake calibration, alpha = 0.10.

## Results

### Gamma wPLI-GCC

| Contrast | n | Awake Pi | Target Pi | Delta | 95% CI | d | p | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awake vs sed | 20 | 0.828 | 0.577 | 0.251 | [0.180, 0.322] | 1.49 | 8.4e-5 | 0.913 |
| awake vs sed2 | 20 | 0.828 | 0.578 | 0.250 | [0.183, 0.318] | 1.59 | 7.0e-5 | 0.900 |

### Alpha wPLI-GCC

| Contrast | n | Awake Pi | Target Pi | Delta | 95% CI | d | p | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awake vs sed | 20 | 0.828 | 0.694 | 0.134 | [0.050, 0.226] | 0.64 | 0.014 | 0.700 |
| awake vs sed2 | 20 | 0.828 | 0.711 | 0.117 | [0.034, 0.211] | 0.56 | 0.028 | 0.600 |

## Interpretation

The gamma GCC effect survives a substantially more conservative coherence observable. This weakens the objection that the DS005620 effect is only a zero-lag volume-conduction artifact.

The alpha result remains directionally consistent but weaker. This supports a frequency-sensitive interpretation: in this dataset, the gamma-band regime score carries the stronger state-dependent signature.

This result should be presented as a robustness control:

> A wPLI-based GCC variant, replacing zero-lag-sensitive synchrony with a lagged phase-consistency observable, preserved the DS005620 sedation effect in the gamma band (awake-vs-sed d = 1.49, AUC = 0.91; awake-vs-sed2 d = 1.59, AUC = 0.90). Thus, the empirical GCC separation is not reducible to trivial zero-lag channel synchrony, although spectral residualization and phase-randomized surrogates show that the current channel-level implementation remains strongly coupled to spectral state structure.

## Consequence for the Paper

This is the strongest new methodological support added in Batch 6 so far. It allows the paper to make a narrower but more defensible claim:

- GCC is not a bandpower-independent biomarker.
- GCC is not merely zero-lag coherence.
- GCC is a calibrated regime geometry whose empirical state sensitivity survives several stress tests, including a volume-conduction-robust coupling proxy.

## Output Files

- `batch6_ds005620_wpli_control.py`
- `ds005620_gamma_wpli_gcc.csv`
- `ds005620_gamma_wpli_stats.csv`
- `ds005620_gamma_wpli_summary.json`
- `ds005620_gamma_wpli_gcc_paired.png`
- `ds005620_alpha_wpli_gcc.csv`
- `ds005620_alpha_wpli_stats.csv`
- `ds005620_alpha_wpli_summary.json`
- `ds005620_alpha_wpli_gcc_paired.png`
