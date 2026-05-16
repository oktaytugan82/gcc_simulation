# Batch 6 Status and Next Steps

Date: 2026-05-13

## Completed

### 1. DS005620 audit

The local DS005620 subset contains the full resting-state core needed for the present propofol replication:

- 21 eyes-closed awake recordings;
- 54 `sed` resting recordings;
- 51 `sed2` resting recordings;
- 20 paired awake-vs-sedation subjects for inference.

The public recordings not included locally are eyes-open awake and TMS acquisitions, not missing resting sedation files.

### 2. Spectrally closest-run control

For each subject, sedated runs closest to awake in spectral-feature space were selected. GCC effects survived this stricter run-selection rule.

Key gamma result:

- awake vs `sed`: Delta Pi = 0.360, d = 1.58, AUC = 1.00.

### 3. Phase-randomized surrogate control

Phase-randomized surrogates preserved much of the gamma Pi separation.

Interpretation:

- GCC is not a phase-structure-only marker;
- the channel-level implementation carries substantial univariate spectral state information;
- this limitation should be reported, not hidden.

### 4. wPLI / volume-conduction control

Replacing the zero-lag-sensitive coherence observable with a wPLI-style lagged-coupling proxy preserved the DS005620 gamma effect.

Key gamma results:

- awake vs `sed`: Delta Pi = 0.251, 95% CI [0.180, 0.322], d = 1.49, AUC = 0.91;
- awake vs `sed2`: Delta Pi = 0.250, 95% CI [0.183, 0.318], d = 1.59, AUC = 0.90.

Interpretation:

- GCC is not bandpower-independent;
- GCC is also not reducible to trivial zero-lag channel synchrony;
- the strongest defensible claim is a calibrated regime composite that survives spectral, artifact, run-selection, and lagged-coupling stress tests.

### 5. DoC data gate

The Mendeley DoC dataset metadata were extracted from the saved dataset page. The dataset is suitable as an exploratory degradation proxy but not yet usable as evidence until EDF files are downloaded and audited.

## Paper Updated

The following evidence was integrated into `GCC_Paper_Restructured_PLOS.tex`:

- DS005620 resting-core audit;
- spectrally closest-run control;
- phase-randomized surrogate limitation;
- wPLI lagged-coupling stress test;
- revised limitations, discussion, conclusion, data availability, and supporting-code description.

The PDF was rebuilt successfully:

- `GCC_Paper_Restructured_PLOS.pdf`

## Remaining High-Value Next Steps

1. Process a small Mendeley DoC EDF subset once file-list/download access is available.
2. Add source-level or CSD/Laplacian EEG controls if montage quality permits.
3. Add a true out-of-sample calibration experiment across participants/datasets.
4. Add a non-propofol anesthesia dataset if a public, usable dataset can be obtained.
5. Turn the analysis scripts into a clean reproducibility package with one command per figure/table.
