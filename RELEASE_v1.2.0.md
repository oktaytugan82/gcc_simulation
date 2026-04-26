# Release v1.2.0

Release date: 2026-04-26

This release extends the GCC simulation repository with the independent fMRI validation, the structural traumatic coma/DoC HARDI backbone constraint, and the neurodegenerative EEG P1-proxy constraint used in the focused NoC-oriented manuscript revision.

## New analyses

- Added OpenNeuro ds006623 fMRI pipeline:
  - minimal downloader and audit scripts
  - GCC-style fMRI feature extraction from XCP-D ROI time series
  - leakage-free subject-held-out model comparison
  - robustness grid over with-GSR vs without-GSR and 4S156 vs 4S256 atlas variants
  - subject-wise permutation controls
- Added OpenNeuro ds003367 structural P1-backbone analysis:
  - traumatic coma, recovery, chronic DoC, and healthy-control HARDI scans
  - predefined ascending-arousal/thalamocortical backbone score
  - group, longitudinal, leave-one-out, and random-backbone control analyses
- Added OpenNeuro ds004504 clinical proxy constraint:
  - control-calibrated access-window and normative-backbone features
  - repeated 5-fold cross-validation for MMSE prediction in AD+FTD patients
  - true-backbone versus random-backbone controls
  - explicit negative boundary for the strongest P1 claim

## Key derived results

- ds006623: GCC-derived dynamic-regime features generalize across subjects for loss/recovery-of-responsiveness classification; the strongest recovery-related model reached AUC 0.986 with 100-permutation p = 0.0099.
- ds003367: the predefined arousal/thalamocortical structural backbone is lower in chronic DoC than recovery-late scans (Cohen's d = 1.04; random-backbone p = 0.046), and P1 components classify recovery-late versus chronic DoC in leave-one-out testing (AUC = 0.870; permutation p = 0.019). The paired recovery early-to-late change is not significant.
- ds004504: low-gamma connectivity is associated with MMSE after clinical and power adjustment, but P1-specific selective-preservation features do not improve out-of-sample MMSE prediction.

## Main files added

- `ds006623/*.py`
- `ds003367/run_ds003367_structural_p1_backbone.py`
- `ds004504/run_ds004504_p1_proxy_validation.py`
- `results/ds006623_*`
- `results/ds003367_structural_p1_*`
- `results/ds004504_p1_proxy_*`
- `figures/ds006623_robustness_grid_*`
- `figures/ds003367_structural_p1_*`
- `figures/ds004504_p1_proxy_*`
- `manuscript/GCC_Paper_PlanB_v11_NoC_focused.tex`
- `manuscript/GCC_Paper_PlanB_v11_NoC_focused.pdf`

## Reproduction commands

```bash
python ds006623/fetch_ds006623_minimal.py
python ds006623/audit_ds006623_minimal.py
python ds006623/extract_ds006623_fmri_gcc_features.py
python ds006623/run_ds006623_leakage_free_model_comparison.py
python ds006623/summarize_ds006623_robustness_grid.py

python ds003367/run_ds003367_structural_p1_backbone.py

python ds004504/run_ds004504_p1_proxy_validation.py
```

The scripts expect the public OpenNeuro data to be available locally. The repository stores derived tables and figures used in the manuscript, not the full raw datasets.
