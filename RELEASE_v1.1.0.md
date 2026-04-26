# Release v1.1.0

Release date: 2026-04-26

This release extends the original GCC simulation archive with the real-connectome and external-dataset stress tests used for the NoC-oriented manuscript revision.

## Main additions

- HCP-derived Budapest Reference Connectome preparation pipeline.
- GCC selective-preservation simulation on empirical HCP-derived consensus connectome topologies.
- Degree-preserving and degree-plus-strength-preserving random graph controls.
- Full HCP-derived stress-test result tables, summaries, and figures.
- OpenNeuro ds004504 audit, feature extraction, and leakage-free model-comparison scripts.
- NoC manuscript version including the HCP-derived connectome stress-test section.

## Key HCP-derived connectome result

The real-connectome stress test is mixed but useful. Against simple degree-preserving controls, the empirical connectomes show larger event-driven backbone-order advantage in 21/24 variant-by-coupling comparisons, with 8/24 one-sided Mann-Whitney tests below p < 0.05. Against stricter degree-plus-strength-preserving nulls, 17/24 comparisons remain positive, but only 1/24 reaches p < 0.05 and 4/24 reach p < 0.10.

Best stricter-null comparison:

- Variant: Budapest Reference Connectome v3.0, 1m fiber-count, 50% consensus
- K = 2.4
- Real delta_A_R = 0.135
- Null delta_A_R = 0.043
- Real-minus-null = 0.091
- One-sided Mann-Whitney p = 0.020

Interpretation: the GCC re-entry signature can appear on empirical human structural topology, but it is not uniformly stronger than degree-plus-strength-preserving null models and should be presented as a topology-generalization stress test, not as clinical validation.

## Reproduction commands

Prepare the Budapest Reference Connectome matrices after downloading the public graph files:

```powershell
powershell -ExecutionPolicy Bypass -File connectome\download_budapest_connectome.ps1
python connectome\prepare_budapest_connectome.py
```

Run the HCP-derived topology stress test:

```powershell
python connectome\run_hcp_connectome_gcc_simulation.py --n-controls 12 --k-values "1.6,2.4,3.2,4.0"
```

Audit and analyze OpenNeuro ds004504 after local dataset download:

```powershell
python ds004504\audit_ds004504_eeg.py
python ds004504\extract_ds004504_gcc_features.py
python ds004504\run_ds004504_leakage_free_cv.py
python ds004504\run_ds004504_compact_theory_cv.py
```

## Data policy

This release includes derived result tables, figures, manifests, and lightweight prepared matrices for the Budapest Reference Connectome stress test. It does not redistribute large OpenNeuro raw EEG files. The ds004504 scripts expect the dataset to be downloaded locally from OpenNeuro.

## License

MIT License, matching the repository license.
