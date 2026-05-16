# NoC release validation package

This folder contains the author-generated analysis package for the NoC submission version of the GCC manuscript.

The folders are named by analysis batch. Each batch contains the scripts, derived result tables, figures, JSON summaries, and short reports used to support the final manuscript. Raw EEG/PSG datasets are not included.

## Recommended reading order

1. `frozen_validation_spec/`
2. `batch3_cross_dataset_validation/`
3. `batch5_spectral_controls/`
4. `batch6_artifact_lag_controls/`
5. `batch7_frozen_transfer_wpli/`
6. `batch10_phase_only/`
7. `batch11_csd_source_proxy/`
8. `batch17_chennu_source_proxy/`
9. `batch18_sleep_cross_paradigm/`
10. `batch16_doc_pilot/`
11. `phase4_evidence_summary/`

The remaining folders provide supporting robustness checks and boundary-state analyses.

## Raw data

Raw third-party datasets must be downloaded from their original repositories:

- Chennu et al. EEG: Cambridge Data Repository, DOI `10.17863/CAM.68959`;
- DS005620: OpenNeuro, DOI `10.18112/openneuro.ds005620.v1.0.0`;
- Sleep-EDF Sleep Cassette: PhysioNet;
- Mendeley DoC EEG/PSG: DOI `10.17632/6wx4n25h4v.1`;
- Farnes et al. ketamine EEG: public Dryad/Zenodo source described in the manuscript;
- DS004504: OpenNeuro, DOI `10.18112/openneuro.ds004504.v1.0.8`.

## Scope

This package supports the submitted manuscript. It is not a clinical diagnostic tool and should not be used for medical decision-making.
