# Release v1.3.0

Release date: 2026-05-11

This release aligns the repository with the final Neuroscience of Consciousness-oriented manuscript version. It adds the Chennu 2016 gamma baseline-holdout check, the Hermann 2021 FDG-PET/EEG disorders-of-consciousness proxy analysis, and the final manuscript/supplement files that cite these outputs.

## New analyses

- Added Chennu 2016 gamma baseline-holdout validation:
  - first half of each participant's baseline recording is used only for within-subject calibration
  - second half of baseline is evaluated as the internal holdout
  - the same bounds are then applied to mild sedation, moderate sedation, and recovery
- Added Hermann et al. 2021 DoC proxy analysis:
  - public supplementary FDG-PET/EEG metadata are used because raw patient imaging data are not openly shareable
  - PET MIBH is treated as metabolic capacity, EEG SVM p(MCS) as dynamic access evidence, and their z-scored mean as a conservative GCC PET+EEG proxy
  - paired patient-level AUC-delta bootstrap tests quantify whether descriptive proxy gains exceed sampling uncertainty
- Added final manuscript and supplement matched to this repository state.

## Key derived results

- Chennu baseline-holdout: baseline holdout mean Pi = 0.658; mild mean Pi = 0.599; moderate mean Pi = 0.530; recovery mean Pi = 0.572. Paired baseline-holdout minus moderate difference = +0.128, Wilcoxon p = 0.0001049, Cohen dz = 1.05.
- Hermann diagnostic target: GCC PET+EEG proxy AUC = 0.834 versus PET MIBH AUC = 0.823 and EEG SVM p(MCS) AUC = 0.771. Paired AUC-delta tests show descriptive, not statistically secure, incremental gains.
- Hermann outcome target: GCC PET+EEG proxy AUC = 0.762 versus PET MIBH AUC = 0.720 and EEG SVM p(MCS) AUC = 0.690. Paired AUC-delta tests again support cautious descriptive wording.

## Main files added

- `manuscript/GCC_Paper_NoC_revised.tex`
- `manuscript/GCC_Paper_NoC_revised.pdf`
- `manuscript/GCC_Supplement_NoC_revised.tex`
- `manuscript/GCC_Supplement_NoC_revised.pdf`
- `pilot/run_chennu_baseline_holdout_from_observables.py`
- `data_manifests/chennu2016_datainfo.mat`
- `data_manifests/chennu_2016_raw_bundle_manifest.json`
- `results/chennu2016_gamma_baseline_holdout_*`
- `clinical_doc/*`
- `results/hermann2021_gcc_proxy_*`
- `figures/hermann2021_gcc_proxy_*`

## Reproduction commands

```bash
python pilot/run_chennu_baseline_holdout_from_observables.py \
    --observables-pkl results/pilot_results.pkl \
    --datainfo-mat data_manifests/chennu2016_datainfo.mat \
    --out-dir results

python clinical_doc/run_hermann2021_gcc_proxy_analysis.py
```

The Chennu raw EEG bundle is public but large, so GitHub stores a manifest and derived outputs rather than redistributing the ZIP. The Hermann analysis is a proxy analysis of public post-processed supplementary metadata, not raw EEG/PET feature extraction.
