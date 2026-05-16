# Batch 17 Chennu fsaverage Source-Space GCC-O

Date: 2026-05-15

## Purpose

This batch tests whether the Chennu propofol GCC effect survives an explicit template-source reconstruction. EEG is mapped to fsaverage using a standard 10-20 montage, an fsaverage BEM/source model, an ad-hoc EEG noise covariance, and sLORETA/minimum-norm inversion. GCC-O is then computed on aparc ROI time series using lagged phase coupling.

## Parameters

```json
{
  "target_sfreq": 125.0,
  "crop_s": 90.0,
  "window_s": 3.0,
  "stride_s": 1.5,
  "alpha": 0.1,
  "spacing": "ico-5",
  "method": "sLORETA",
  "snr": 3.0,
  "loose": 0.2,
  "depth": 0.8,
  "subjects_requested": 1,
  "subjects_completed": 1,
  "failures": []
}
```

## Paired Source-Space Pi Effects

| band | n | baseline_mean | moderate_mean | mean_delta_baseline_minus_moderate | delta_ci_low | delta_ci_high | paired_dz | wilcoxon_greater_p | ttest_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | 1 | 0.8276 | 0.6897 | 0.1379 | 0.1379 | 0.1379 |  | 0.5 |  |
| gamma | 1 | 0.8276 | 0.7586 | 0.06897 | 0.06897 | 0.06897 |  | 0.5 |  |

## Interpretation

A positive same-direction effect supports source-space robustness. This is not individual-MRI source localization; it is a template-source robustness analysis intended to address sensor-level volume-conduction and reference concerns.
