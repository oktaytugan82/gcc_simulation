# GCC simulation and analysis code

This repository contains the reproducibility package for the manuscript:

> **Balanced-coherence regimes of conscious access: a computational model with public EEG validation**  
> Mehmet Oktay Tugan (2026)  
> Prepared for submission to *Neuroscience of Consciousness*.

The Gamma-Code Consensus (GCC) is a computational regime model of access-compatible neural dynamics. It combines synthetic stochastic phase-oscillator simulations with public EEG/PSG re-analyses to test whether conscious-access-related states occupy a bounded region of balanced coherence, effective dimensionality, and temporal stability.

## Current release

The NoC submission snapshot is designated:

```text
v2.0.0-noc
```

This release adds the final NoC manuscript files and the complete author-generated validation outputs used in the submission version:

- synthetic regime validation and matched-stress simulations;
- Chennu et al. propofol analysis with cross-validation;
- independent OpenNeuro DS005620 propofol replication;
- Sleep-EDF wake/REM/NREM state-geometry analysis;
- Mendeley disorders-of-consciousness pilot anchor;
- Farnes et al. subanaesthetic ketamine boundary-state analysis;
- OpenNeuro DS004504 degradation-proxy analysis;
- bandpower residualization and gamma-artifact controls;
- phase-surrogate, wPLI, phase-only, CSD/source-proxy, and template-source robustness checks;
- split-half and repeated-run reliability checks;
- frozen validation specification and figure-generation scripts.

Raw third-party datasets are **not** redistributed in this repository. They remain available from their original public repositories and are documented in the manuscript Data Availability statement.

## Repository structure

```text
gcc_simulation/
├── simulation/                  # Core synthetic GCC simulator and original ensemble tests
├── pilot/                       # Chennu EEG loader and original pilot analyses
├── connectome/                  # Earlier connectome stress-test scripts
├── ds003367/                    # Earlier structural backbone proxy scripts
├── ds004504/                    # Earlier ds004504 proxy scripts
├── ds006623/                    # Earlier fMRI validation scripts
├── clinical_doc/                # Earlier Hermann DoC proxy scripts and metadata notes
├── results/                     # Derived outputs from earlier releases
├── figures/                     # Figures from earlier releases
├── manuscript/
│   ├── GCC_Paper_NoC_revised.*  # Earlier NoC-oriented manuscript
│   └── noc_submission/          # Final NoC submission source/PDF and supplements
├── noc_release/                 # Final NoC validation package, Batch 3-18
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
└── RELEASE_v2.0.0-noc.md
```

## NoC validation package

The `noc_release/` folder is the primary reproduction entry point for the current manuscript.

```text
noc_release/
├── frozen_validation_spec/
├── batch3_cross_dataset_validation/
├── batch5_spectral_controls/
├── batch6_artifact_lag_controls/
├── batch7_frozen_transfer_wpli/
├── batch8_sleep_state_geometry/
├── batch9_bandpower_independence/
├── batch10_phase_only/
├── batch11_csd_source_proxy/
├── batch12_degradation_proxy/
├── batch13_ketamine/
├── batch14_reliability/
├── batch15_gcco_composite/
├── batch16_doc_pilot/
├── batch17_chennu_source_proxy/
├── batch18_sleep_cross_paradigm/
└── phase4_evidence_summary/
```

Each batch folder contains the relevant analysis script(s), derived result tables, figure files, JSON summaries, and short report files. Large raw EEG/PSG downloads are excluded.

## Installation

```bash
git clone https://github.com/oktaytugan82/gcc_simulation.git
cd gcc_simulation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Reproducing analyses

Most NoC release scripts are designed to be run from their own batch directory or with explicit input/output paths. The frozen specification is:

```bash
cd noc_release/frozen_validation_spec
python run_frozen_feature_audit.py
```

Examples:

```bash
cd noc_release/batch3_cross_dataset_validation
python batch3_synthetic_benchmark.py
python batch3_chennu_cv_summary.py
python batch3_ds005620_stats.py
python batch3_sleep_edf_validation.py
```

```bash
cd noc_release/batch10_phase_only
python batch10_phase_only_gcc.py --help
```

```bash
cd noc_release/batch16_doc_pilot
python run_doc_gcc_validation.py --help
```

Some empirical scripts require local copies of the public datasets. The repository provides manifests and derived outputs, but not the raw datasets.

## Public datasets

The current NoC manuscript re-analyses:

- Chennu et al. propofol EEG, Cambridge Data Repository, DOI `10.17863/CAM.68959`;
- OpenNeuro DS005620 propofol EEG, DOI `10.18112/openneuro.ds005620.v1.0.0`;
- Sleep-EDF Sleep Cassette via PhysioNet;
- Mendeley disorders-of-consciousness EEG/PSG dataset, DOI `10.17632/6wx4n25h4v.1`;
- Farnes et al. subanaesthetic ketamine EEG data;
- OpenNeuro DS004504 EEG dementia dataset, DOI `10.18112/openneuro.ds004504.v1.0.8`.

## Citation

If you use this repository, cite the archived Zenodo release and the manuscript:

```text
Tugan, M. O. (2026). GCC simulation and analysis code.
GitHub repository: https://github.com/oktaytugan82/gcc_simulation
NoC submission snapshot: v2.0.0-noc
```

The archived NoC submission snapshot DOI is:

```text
10.5281/zenodo.19798082
```

The repository-level Zenodo concept DOI, which resolves to the latest version, is:

```text
10.5281/zenodo.19744969
```

## License

Author-generated code is released under the MIT License. Third-party datasets retain their original licenses and terms of use.
