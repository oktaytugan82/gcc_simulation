# Frozen GCC Validation Pipeline v1

Date frozen: 2026-05-14

## Purpose

This document freezes the GCC validation pipeline before additional external datasets are interpreted. Its function is to reduce researcher degrees of freedom: preprocessing, observables, thresholding, endpoints, baselines, and forbidden claims are fixed in advance.

## Primary Position

GCC is evaluated as a theory-grounded, state-sensitive regime composite. It is not claimed to be a generally bandpower-independent biomarker.

## Locked Primary Implementation

- EEG is average-referenced after non-EEG channels are removed.
- Recordings are resampled to 125 Hz.
- Each retained channel is demeaned and divided by its own standard deviation.
- The primary implementation is amplitude-normalized phase-only GCC.
- Primary bands are alpha (8-13 Hz) and gamma (35-45 Hz).
- Window length is 3.0 s and stride is 1.5 s.
- R is the mean Kuramoto order parameter from unit phase vectors.
- D_eff is the participation ratio of covariance eigenvalues from cos(phase) and sin(phase) channels.
- M_tau is the window-internal variance of instantaneous R.
- Pi is the mean of the three regime indicators R_ok, D_ok, and M_ok.

## Locked Calibration

All thresholds are calibrated from subject-level awake/baseline reference windows only:

- R_min = Q_0.10(R | reference)
- D_min = Q_0.10(D_eff | reference)
- D_max = Q_0.90(D_eff | reference)
- M_min = Q_0.10(M_tau | reference)
- M_max = Q_0.90(M_tau | reference)

No target-condition data may be used to set thresholds.

## Locked Endpoints

- For propofol or loss-of-consciousness datasets: paired baseline-minus-target Delta Pi, paired Cohen d, and one-sided Wilcoxon baseline > target.
- For altered but conscious pharmacological states such as subanaesthetic ketamine: component shifts in R, D_eff, M_tau, and Pi are reported without assuming loss of consciousness.
- For sleep datasets: Wake/REM/NREM separation in GCC feature space with subject-grouped cross-validation.
- For degradation proxies: group separation and clinical-score associations are reported as degradation sensitivity, not as conscious-access validation.

## Mandatory Baselines

Every applicable dataset must include:

- R-only, D-only, and M-only baselines.
- Conventional spectral-power baselines.
- Spectral plus GCC comparison.
- Residualized GCC after regression on bandpower.
- Source-proxy or phase-only control when raw channel data allow it.

## Interpretation Rules

GCC is strengthened if it retains reproducible state sensitivity across independent datasets and survives phase-only/source-proxy controls. GCC is weakened if effects vanish under those controls or do not exceed single-observable and spectral baselines.

Forbidden claims:

- GCC is a proven neural correlate of consciousness.
- GCC is generally independent of spectral power.
- GCC validates terminal lucidity without direct terminal-lucidity data.
- Gamma is uniquely necessary.

## Files

The machine-readable version is `gcc_frozen_pipeline_v1.json`.
