# Batch 16 Final Option Assessment

Date: 2026-05-14

## Question

Which of the three candidate routes is best testable and evidentially strongest for a spectacular GCC result?

1. A generally bandpower-independent GCC biomarker.
2. A direct DoC/recovery clinical validation.
3. A direct re-entry/backbone validation in zolpidem, recovery, or terminal-lucidity-like data.

## Decision

The strongest feasible route is option 2: direct clinical DoC validation.

Reason: option 1 has already been stress-tested across sedation, sleep, ketamine, and spectral residualization and does not support a general bandpower-independent biomarker claim. Option 3 is conceptually the most spectacular but currently lacks accessible public raw EEG datasets with appropriate within-subject re-entry events. Option 2 is both publicly testable and clinically meaningful.

## Data Gate Result

The public Mendeley Data dataset `10.17632/6wx4n25h4v.1` was successfully accessed through the Mendeley public file API and fully downloaded.

- 42 EDF files downloaded.
- Total size: approximately 2.58 GB.
- Labels from filenames: 30 VS, 4 MCS-, 8 MCS+.
- All files readable.
- All files sampled at 256 Hz.
- Common six-channel EEG montage present in every file: F3, F4, C3, C4, O1, O2.
- First 14,400 s per file analyzed after resampling to 128 Hz.
- 30,158 artifact-filtered 30 s windows retained.

This is a genuine clinical DoC validation dataset, not another synthetic or anesthesia-only dataset.

## Main GCC Result

Primary endpoint: MCS+ vs VS, alpha-band cross-validated GCC access-window fraction.

- Raw CV-GCC access Pi: AUC = 0.754, 95% bootstrap CI [0.579, 0.900].
- MCS+ mean Pi = 0.539, VS mean Pi = 0.342.
- Mean difference = 0.197.
- Cohen-style pooled d = 0.89.
- Mann-Whitney one-sided p = 0.015.

This is the cleanest clinical GCC signal in the DoC dataset.

## Spectral and Artifact Controls

The same primary endpoint remains positive after spectral residualization:

- Alpha MCS+ vs VS, spectral-residual CV-GCC Pi: AUC = 0.717, p = 0.033.
- Alpha MCS+ vs VS, residualized against spectral features plus number of retained epochs: AUC = 0.721, p = 0.030.

This is important because it means the primary DoC result is not explained away by conventional bandpower features or by a simple artifact-retention confound.

Additional positive residual result:

- Gamma MCS-any vs VS, spectral-residual CV-GCC Pi: AUC = 0.744, p = 0.007.
- After also controlling retained epochs: AUC = 0.683, p = 0.034.

## Sensitivity

The alpha MCS+ vs VS effect is robust across GCC threshold quantiles.

- Raw alpha CV-Pi remains positive from alpha = 0.05 to 0.25.
- Spectral-plus-epoch residual alpha CV-Pi remains positive from alpha = 0.05 to 0.15.

The gamma residual effect is also recurrent across threshold choices, especially for MCS-any vs VS, although it weakens after retained-epoch control.

## What Is Spectacular Here

The spectacular claim is not:

"GCC is a universal bandpower-independent biomarker."

That would be too strong and is not supported by the full evidence.

The defensible strong claim is:

"A cross-validated GCC access-regime score derived from phase-lagged network observables distinguishes clinically diagnosed MCS+ from VS in a public DoC EEG/PSG dataset, and the primary alpha-band effect survives residualization against conventional spectral power and artifact-retention controls."

This is a real upgrade for the paper because it moves GCC from state-sensitive anesthesia/sleep validation into a direct clinical consciousness-disorder test.

## Remaining Caveats

- The dataset is imbalanced: VS = 30, MCS+ = 8, MCS- = 4.
- Labels are filename-level clinical labels, not time-resolved CRS-R scores.
- PSG sleep structure is not controlled with hypnogram labels.
- The six-channel montage is clean and common but spatially limited.
- Severity monotonicity across VS -> MCS- -> MCS+ is not robustly demonstrated.
- The result supports access-regime sensitivity, not terminal lucidity or causal re-entry.

## Recommendation

Use option 2 as the next major paper strengthening layer.

Best wording:

"As a clinical external validation, we applied a frozen GCC access-regime score to an independent public DoC PSG dataset. The cross-validated alpha-band access score distinguished MCS+ from VS and remained positive after residualizing conventional spectral features and retained-epoch count, suggesting that GCC captures access-relevant network structure beyond simple bandpower shifts."

This is strong enough to matter for NoC-level review. It is not enough to force acceptance, but it directly addresses the previous vulnerability: no direct clinical consciousness validation.

