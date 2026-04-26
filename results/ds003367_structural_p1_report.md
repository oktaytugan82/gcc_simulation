# ds003367 Structural P1-Backbone Proxy Analysis

Input: `C:\Users\oktay\OneDrive\Dokumente\New project\data\ds003367\sub-all.dz`

## What This Tests

This analysis tests a structural prerequisite of P1: whether a predefined ascending-arousal / thalamo-cortical backbone is more preserved in recovered/healthy scans than in chronic post-traumatic DoC scans.

It does not test terminal lucidity, conscious report, or dynamic re-entry. It is a structural constraint layer.

## Cohorts

{
  "control": 16,
  "acute_only": 10,
  "recovery_early": 9,
  "recovery_late": 9,
  "chronic_doc": 6
}

Primary cross-sectional contrast: `recovery_late` vs `chronic_doc`.
Primary longitudinal contrast: paired `recovery_early` to `recovery_late` in the nine recovery subjects.

## Group Tests

| feature                 | group_a       | group_b     | n_a | n_b | mean_a    | mean_b  | median_a  | median_b | cohens_d_a_minus_b | mannwhitney_p_two_sided | welch_p_two_sided |
| ----------------------- | ------------- | ----------- | --- | --- | --------- | ------- | --------- | -------- | ------------------ | ----------------------- | ----------------- |
| p1_backbone_z           | recovery_late | chronic_doc | 9   | 6   | -0.2812   | -0.6756 | -0.2254   | -0.6918  | 1.035              | 0.1135                  | 0.04541           |
| p1_backbone_z           | control       | chronic_doc | 16  | 6   | 1.324e-07 | -0.6756 | 0.02225   | -0.6918  | 3.321              | 0.0001072               | 0.0002611         |
| p1_aan_core_z           | recovery_late | chronic_doc | 9   | 6   | -0.09603  | -0.362  | -0.116    | -0.4034  | 1.424              | 0.04955                 | 0.02767           |
| p1_aan_core_z           | control       | chronic_doc | 16  | 6   | 6.265e-08 | -0.362  | -0.04971  | -0.4034  | 1.587              | 0.00453                 | 0.005399          |
| p1_relay_tract_z        | recovery_late | chronic_doc | 9   | 6   | -0.4664   | -0.9892 | -0.06422  | -1.098   | 0.6845             | 0.181                   | 0.193             |
| p1_relay_tract_z        | control       | chronic_doc | 16  | 6   | 2.022e-07 | -0.9892 | -0.01382  | -1.098   | 2.394              | 0.001179                | 0.01097           |
| global_integrity_z      | recovery_late | chronic_doc | 9   | 6   | 0.5725    | -0.2345 | 0.08043   | -0.3722  | 0.7171             | 0.2721                  | 0.161             |
| global_integrity_z      | control       | chronic_doc | 16  | 6   | 1.673e-05 | -0.2345 | -0.1583   | -0.3722  | 0.291              | 0.4942                  | 0.5643            |
| p1_specificity_residual | recovery_late | chronic_doc | 9   | 6   | -0.04314  | -0.4565 | -0.003351 | -0.4602  | 1.094              | 0.08791                 | 0.03685           |
| p1_specificity_residual | control       | chronic_doc | 16  | 6   | 0.2246    | -0.4565 | 0.2637    | -0.4602  | 3.383              | 5.361e-05               | 0.0003422         |

## Longitudinal Tests

| feature                 | n_pairs | mean_delta_late_minus_early | median_delta_late_minus_early | wilcoxon_p_two_sided | paired_t_p_two_sided |
| ----------------------- | ------- | --------------------------- | ----------------------------- | -------------------- | -------------------- |
| p1_backbone_z           | 9       | 0.03337                     | 0.003768                      | 0.8203               | 0.608                |
| p1_aan_core_z           | 9       | 0.02404                     | 0.01279                       | 0.9102               | 0.8098               |
| p1_relay_tract_z        | 9       | 0.0427                      | 0.09604                       | 0.7344               | 0.5274               |
| global_integrity_z      | 9       | 0.1498                      | 0.1706                        | 0.9102               | 0.7112               |
| p1_specificity_residual | 9       | 0.03689                     | 0.007837                      | 0.8203               | 0.5918               |

## Leakage-Free Compact Model Comparison

Outcome: recovery_late vs chronic_doc. Model assessment uses leave-one-out CV with scaling fit inside each training fold and label permutations.

| model            | auc    | accuracy | permutation_p_upper | n_permutations | n  |
| ---------------- | ------ | -------- | ------------------- | -------------- | -- |
| global_integrity | 0.6111 | 0.5333   | 0.2547              | 1000           | 15 |
| p1_backbone      | 0.7222 | 0.6      | 0.08591             | 1000           | 15 |
| global_plus_p1   | 0.7963 | 0.6667   | 0.04396             | 1000           | 15 |
| p1_components    | 0.8704 | 0.7333   | 0.01898             | 1000           | 15 |

## Random-Backbone Specificity Controls

{
  "n_random_backbones": 500,
  "candidate_voxels": 127034,
  "observed_cross_d_recovery_late_minus_chronic": 1.035043410615459,
  "observed_longitudinal_mean_delta": 0.03336974364736778,
  "random_cross_p_upper": 0.04590818363273453,
  "random_longitudinal_p_upper": 0.3532934131736527,
  "random_cross_d_mean": 0.06818884168452338,
  "random_cross_d_sd": 0.6235262445491054,
  "random_longitudinal_delta_mean": 0.008551556729073814,
  "random_longitudinal_delta_sd": 0.07040113390020217
}

## Manuscript-Safe Interpretation

- Positive result: supports the structural plausibility of a residual backbone constraint in traumatic DoC/recovery data.
- Negative or mixed result: constrains P1 by showing that this public HARDI-only dataset does not isolate the predicted structural substrate.
- In either case, this should be framed as an independent structural proxy/constraint, not as empirical validation of terminal lucidity or re-entry dynamics.
