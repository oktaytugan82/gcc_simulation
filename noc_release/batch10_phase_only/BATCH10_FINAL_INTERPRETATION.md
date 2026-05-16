# Batch 10 Final Interpretation: Phase-Only GCC

Date: 2026-05-14

## What Was Tested

This batch implemented an amplitude-normalized, phase-only GCC pipeline directly on raw EEG epochs.

Pipeline:

- bandpass EEG;
- extract Hilbert phase;
- replace each channel signal by the unit phasor `exp(i phi)`;
- compute `R` from unit phasors;
- compute `D_eff` from the covariance of `[cos(phi); sin(phi)]`;
- compute `M_tau` from temporal variance of `R(t)`;
- calibrate each participant against their own baseline/awake recording;
- evaluate Chennu and DS005620 in alpha and gamma bands.

This means bandpass amplitude does not enter GCC observables. Spectral features are used only later as external baselines.

## Main Phase-Only Result

The phase-only GCC access index shows strong state sensitivity.

| Dataset | Band | Comparison | Delta Pi | Cohen d | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| Chennu | alpha | baseline - moderate | 0.231 | 1.06 | 8.39e-05 |
| DS005620 | alpha | awake - sed | 0.234 | 1.25 | 2.38e-05 |
| DS005620 | alpha | awake - sed2 | 0.204 | 1.04 | 1.31e-04 |
| Chennu | gamma | baseline - moderate | 0.463 | 2.37 | 1.91e-06 |
| DS005620 | gamma | awake - sed | 0.439 | 1.99 | 1.91e-06 |
| DS005620 | gamma | awake - sed2 | 0.396 | 1.88 | 9.54e-07 |

Interpretation:

- The phase-only version is not dead after amplitude removal.
- The strongest signal is gamma.
- The effect replicates directionally across both propofol datasets.
- This is a meaningful improvement over the earlier critique that GCC may be mostly amplitude/bandpower driven.

## Bandpower-Stress Result

The stricter question is whether phase-only GCC is independent of spectral state information.

Within-dataset LOSO:

| Dataset | Band | GCC triad AUC | Residual GCC AUC | Spectral all AUC | Spectral + GCC AUC |
|---|---:|---:|---:|---:|---:|
| Chennu | alpha | 0.820 | 0.700 | 0.950 | 0.953 |
| DS005620 | alpha | 0.751 | 0.535 | 0.977 | 0.975 |
| Chennu | gamma | 0.950 | 0.700 | 0.950 | 0.903 |
| DS005620 | gamma | 0.975 | 0.537 | 0.977 | 1.000 |

Cross-dataset transfer:

| Band | Direction | GCC triad AUC | Residual GCC AUC | Spectral all AUC |
|---|---|---:|---:|---:|
| alpha | Chennu -> DS005620 | 0.600 | 0.300 | 0.775 |
| alpha | DS005620 -> Chennu | 0.800 | 0.800 | 0.750 |
| gamma | Chennu -> DS005620 | 0.825 | 0.550 | 0.775 |
| gamma | DS005620 -> Chennu | 0.950 | 0.950 | 0.750 |

## Scientific Interpretation

The phase-only pipeline supports a stronger claim than before:

> GCC state sensitivity is not solely dependent on bandpass amplitude entering the observables, because a strictly phase-only version still detects sedation-related state changes with large effects in two independent propofol datasets.

But it still does not justify the strongest claim:

> GCC is a generally bandpower-independent biomarker.

Why not:

- Spectral models remain extremely strong.
- Adding phase-only GCC to spectral features does not robustly improve AUC.
- Residualized phase-only GCC weakens substantially in most within-dataset tests.
- Cross-dataset residualized transfer is promising in one direction, especially gamma DS005620 -> Chennu, but asymmetric.

## Best Defensible Claim

The strongest accurate claim is:

> An amplitude-normalized, phase-only GCC implementation retains robust sedation sensitivity across two independent propofol EEG datasets. This shows that GCC is not merely an amplitude-defined score. However, the present evidence supports phase-based state sensitivity and partial incremental information, not a universally bandpower-independent biomarker.

## Reviewer-Safe Manuscript Wording

Use this:

> To test whether the GCC effect depended on bandpass amplitude entering the observables, we implemented an amplitude-normalized phase-only pipeline on raw EEG epochs. After bandpass filtering, each channel was represented only by its Hilbert phase via the unit phasor exp(i phi). The effective dimensionality observable was recomputed from the covariance of cos(phi) and sin(phi), so that no bandpass amplitude entered the GCC triad. The phase-only access index declined robustly under propofol sedation in both datasets, with the largest effects in gamma (Chennu baseline-vs-moderate Delta Pi = 0.463, d = 2.37; DS005620 awake-vs-sed Delta Pi = 0.439, d = 1.99). This demonstrates that GCC state sensitivity is not solely produced by amplitude entering the observables. However, spectral baselines remained strong, and residualized phase-only GCC did not improve robustly over spectral models in all tests. We therefore interpret the phase-only analysis as evidence for phase-based state sensitivity, not as proof of a generally bandpower-independent biomarker.

## Practical Consequence

This result should be added to the paper as a control analysis, not as a new central claim.

Recommended framing:

- call it "amplitude-normalized phase-only control";
- report the strong paired phase-only Pi effects;
- report that spectral baselines remain competitive;
- do not claim "bandpower-independent biomarker";
- claim "not reducible to amplitude entering GCC observables".

