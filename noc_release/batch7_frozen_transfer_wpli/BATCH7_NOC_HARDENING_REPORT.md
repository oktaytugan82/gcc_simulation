# Batch 7 NoC Hardening Report

Date: 2026-05-14

## Aim

This batch upgrades the GCC empirical section toward a NoC-level evidential standard by adding:

1. a wPLI-GCC implementation for Chennu, so that lagged-coupling GCC is available in both independent propofol datasets;
2. a frozen cross-dataset benchmark against single-observable and spectral baselines;
3. an explicit no-fit sign rule: sedation is predicted whenever subject-calibrated Pi declines relative to baseline.

The goal is not to claim that GCC beats all spectral baselines. The goal is to show that GCC is a reproducible regime method whose core state effect survives frozen transfer, baseline comparison, and a zero-lag-robust coupling implementation.

## Data

### Chennu et al. 2016

- 20 subjects.
- Conditions used for binary transfer: baseline vs moderate propofol sedation.
- wPLI-GCC computed from the first 90 s of each recording.
- Bands: alpha and gamma.

### OpenNeuro DS005620

- 21 awake recordings, 20 paired sedation subjects.
- Conditions: awake vs `sed` and `sed2`.
- wPLI-GCC from Batch 6.
- Bands: alpha and gamma.

## wPLI-GCC Results in Chennu

| Band | Contrast | n | Baseline Pi | Target Pi | Delta | 95% CI | d | p | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma | baseline vs mild | 20 | 0.828 | 0.724 | 0.103 | [0.064, 0.151] | 1.01 | 2.5e-4 | 0.816 |
| gamma | baseline vs moderate | 20 | 0.828 | 0.655 | 0.172 | [0.111, 0.233] | 1.18 | 1.3e-4 | 0.766 |
| alpha | baseline vs moderate | 20 | 0.828 | 0.709 | 0.118 | [0.068, 0.177] | 0.92 | 4.1e-5 | 0.900 |

Important caveat: in the 90 s wPLI implementation, Chennu recovery remains below baseline. Therefore wPLI-GCC should be presented as a robust sedation-state stress test, not as a full recovery-trajectory model.

## Frozen Cross-Dataset Benchmark

Rules:

- For sign-rule models, no parameters are fitted.
- For logistic transfer models, scaling and weights are fitted only on the training dataset.
- Transfer directions are Chennu -> DS005620 and DS005620 -> Chennu.

### Standard GCC Pi

| Band | Model | Direction | AUC | Balanced accuracy |
|---|---|---|---:|---:|
| gamma | Pi sign rule | pooled no-fit | 0.967 | 0.983 |
| gamma | Pi delta | Chennu -> DS005620 | 1.000 | 1.000 |
| gamma | Pi delta | DS005620 -> Chennu | 0.900 | 0.725 |
| alpha | Pi sign rule | pooled no-fit | 0.850 | 0.925 |
| alpha | Pi delta | Chennu -> DS005620 | 0.850 | 0.913 |
| alpha | Pi delta | DS005620 -> Chennu | 0.850 | 0.725 |

### wPLI-GCC

| Band | Model | Direction | AUC | Balanced accuracy |
|---|---|---|---:|---:|
| gamma | wPLI Pi sign rule | pooled no-fit | 0.858 | 0.925 |
| gamma | wPLI Pi delta | Chennu -> DS005620 | 0.913 | 0.925 |
| gamma | wPLI Pi delta | DS005620 -> Chennu | 0.750 | 0.850 |
| gamma | wPLI triad + Pi | Chennu -> DS005620 | 0.950 | 0.950 |
| gamma | wPLI triad + Pi | DS005620 -> Chennu | 0.900 | 0.825 |
| alpha | wPLI Pi sign rule | pooled no-fit | 0.733 | 0.867 |
| alpha | wPLI triad + Pi | Chennu -> DS005620 | 0.650 | 0.788 |
| alpha | wPLI triad + Pi | DS005620 -> Chennu | 0.800 | 0.800 |

### Spectral Baselines

| Band | Model | Direction | AUC | Balanced accuracy |
|---|---|---|---:|---:|
| gamma | spectral bandpowers | Chennu -> DS005620 | 0.975 | 0.938 |
| gamma | spectral bandpowers | DS005620 -> Chennu | 0.900 | 0.775 |
| alpha | spectral bandpowers | Chennu -> DS005620 | 0.975 | 0.938 |
| alpha | spectral bandpowers | DS005620 -> Chennu | 0.900 | 0.775 |
| gamma | spectral all | Chennu -> DS005620 | 0.775 | 0.775 |
| gamma | spectral all | DS005620 -> Chennu | 0.750 | 0.675 |

## Interpretation

The result is scientifically strong but must be stated precisely.

What it supports:

- GCC Pi decline is highly reproducible under frozen, no-fit sign rules.
- Standard gamma GCC transfers strongly across two independent propofol datasets.
- wPLI-GCC, which suppresses trivial zero-lag synchrony, also transfers well, especially in gamma.
- wPLI triad + Pi achieves strong bidirectional gamma transfer (AUC 0.95 and 0.90), making the volume-conduction objection substantially weaker.

What it does not support:

- GCC is not shown to be independent of spectral power.
- Spectral bandpowers are highly competitive and sometimes equal or stronger.
- The present evidence remains propofol-centered and does not directly validate degradation/re-entry in clinical cohorts.

## Recommended Manuscript Claim

Use this wording:

> In a frozen cross-dataset benchmark, subject-calibrated GCC Pi decline generalized across two independent propofol EEG datasets. A no-fit sign rule reached pooled gamma AUC = 0.967, and train-on-Chennu/test-on-DS005620 gamma Pi transfer reached AUC = 1.00. Importantly, a wPLI-based GCC variant that replaces zero-lag-sensitive synchrony by lagged coupling also generalized bidirectionally in the gamma band when the full triad plus Pi was used (AUC = 0.95 and 0.90). Spectral bandpowers remained highly competitive, so these results do not establish GCC as a bandpower-independent biomarker; rather, they support GCC as a reproducible, theoretically constrained regime composite whose state sensitivity survives frozen transfer and lagged-coupling stress tests.

## Output Files

- `batch7_chennu_wpli_control.py`
- `batch7_frozen_cross_dataset_benchmark.py`
- `chennu_gamma_wpli_gcc.csv`
- `chennu_alpha_wpli_gcc.csv`
- `noc_standard_delta_features.csv`
- `noc_wpli_delta_features.csv`
- `noc_frozen_benchmark_results.csv`
- `noc_frozen_benchmark_summary.json`
- `noc_frozen_transfer_benchmark_auc.png`
