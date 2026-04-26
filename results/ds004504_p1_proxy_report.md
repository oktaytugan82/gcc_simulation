# ds004504 P1-Lite Proxy Validation

## Design

- Population: AD+FTD patients only for MMSE prediction (n=59); controls (n=29) used only for control-calibrated access windows and normative backbone definition.
- Cross-validation: repeated stratified 5-fold CV, 100 repeats (500 folds).
- P1 features: alpha/low-gamma control-regime occupancy, normative backbone preservation, selective preservation, and random-backbone controls.

## Main CV Result

- Clinical baseline MAE: 3.014, Spearman rho: 0.435.
- Global-activity model MAE: 3.086, Spearman rho: 0.385.
- Standard EEG+FC model MAE: 3.194, Spearman rho: 0.334.
- P1 true-backbone core model MAE: 3.122, Spearman rho: 0.373.
- Standard EEG+FC plus P1 model MAE: 3.309, Spearman rho: 0.143.

## Incremental Tests

- Standard EEG+FC minus Standard+P1 MAE improvement: -0.107 (positive would favor P1; one-sided sign-flip p=0.840).
- Global activity minus P1-core MAE improvement: -0.157 (positive would favor P1; one-sided sign-flip p=0.862).
- Random-backbone minus P1-core MAE improvement: 0.046 (positive would favor true backbone; one-sided sign-flip p=0.374).

## Key Single-Feature Checks

- Low-gamma mean wPLI vs MMSE survives adjustment for clinical covariates and power: rho=0.410, p=0.0012.
- Low-gamma normative-backbone on-strength also survives adjustment: rho=0.291, p=0.0253.
- Low-gamma backbone ratio does not survive as a P1-selectivity marker: rho=-0.108, p=0.4175.
- P1 composite adjusted for clinical covariates and power: rho=0.018, p=0.8916.

## Interpretation

This analysis provides a useful negative boundary. ds004504 supports a weaker claim that preserved low-gamma functional connectivity is associated with cognitive preservation in AD/FTD, but it does not validate the stronger P1 selective-preservation mechanism. In the manuscript, this should be framed as a clinical proxy stress test that constrains P1 rather than confirms it.
