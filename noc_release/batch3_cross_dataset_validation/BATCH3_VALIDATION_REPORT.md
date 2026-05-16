# Batch 3 Validation Report: Synthetic Stress Tests, Chennu Raw EEG, Sleep-EDF

Date: 2026-05-13

## Executive Status

Batch 3 substantially strengthens the GCC manuscript against the PLOS criticism.
The central model is no longer supported only by theoretical arguments and
single-dataset proxy evidence. It now has three additional validation layers:

- Expanded synthetic validation with network-type variation, noise, lesion,
  gain, backbone size, and baseline comparisons.
- Full raw-data Chennu propofol re-analysis from the Cambridge repository.
- Sleep-EDF Wake/REM/NREM state-differentiation pilot using public raw EEG.

The planned independent propofol replication was completed with a selective
multi-subject DS005620 subset:

- OpenNeuro `ds005620`.
- 21 subjects.
- 126 BrainVision EEG recordings.
- Tasks: `awake`, `sed`, and `sed2`.
- Downloaded subset size: 30.4 GB.

## 1. Expanded Synthetic Validation

### Design

The matched synthetic stress test evaluates the model under controlled
conditions in which only the mechanistic manipulation differs while confounds
are matched across paired simulations.

Network types:

- Small-world
- Erdos-Renyi
- Barabasi-Albert
- Modular

Manipulations:

- Uniform degradation
- Selective residual-backbone re-entry
- Lesion severity
- Noise level
- Global-gain controls

Baselines:

- `R` only
- `D_eff` only
- `M_tau` only
- Criticality-only proxy
- Global-gain-only proxy
- GCC triad

### Main Matched Re-Entry Result

Backbone-scoped analysis:

| Score | Mean delta, re-entry minus uniform | p | Positive pairs |
|---|---:|---:|---:|
| GCC triad | +0.184 | 5.06e-12 | 86 / 120 |
| R-only | +0.343 | 7.18e-14 | 94 / 120 |
| D-only | +0.056 | 9.78e-4 | 72 / 120 |
| M-only | -0.034 | 5.52e-3 | 30 / 120 |
| Criticality-only | +0.042 | 1.91e-2 | 71 / 120 |
| Global-gain-only | 0.000 | n/a | 0 / 120 |

By network type, GCC triad mean delta remained positive:

| Network | Mean GCC delta |
|---|---:|
| Barabasi | +0.191 |
| Erdos | +0.198 |
| Modular | +0.144 |
| Small-world | +0.202 |

Full-network analysis:

- GCC triad mean delta = +0.041.
- p = 5.80e-5.
- Positive pairs = 27 / 120.

Interpretation:

The predicted effect is strongest when measured on the residual backbone,
which is exactly the theoretical claim. Full-network effects are smaller,
consistent with the model's statement that re-entry is not global healing.

Important limitation:

R-only is a strong baseline in the matched backbone test. The GCC triad should
therefore not be framed as simply "better than coherence" in every condition.
Its added value is the multi-constraint regime formulation and its behavior
under noise/lesion sweeps.

### Parameter Sweeps

Lesion sweep:

| Score | Spearman rho | p |
|---|---:|---:|
| GCC triad | -0.383 | 8.00e-10 |
| R-only | -0.648 | 5.36e-30 |
| D-only | +0.044 | 0.493 |
| M-only | +0.419 | 1.30e-11 |
| Criticality-only | +0.118 | 0.068 |

Noise sweep:

| Score | Spearman rho | p |
|---|---:|---:|
| GCC triad | -0.365 | 1.74e-10 |
| R-only | -0.018 | 0.765 |
| D-only | -0.366 | 1.40e-10 |
| M-only | +0.383 | 1.73e-11 |
| Criticality-only | -0.333 | 7.10e-9 |

Interpretation:

The GCC triad tracks degradation under lesion and noise. R-only is strong for
lesions but nearly blind to noise in this setup. This supports the argument for
a multi-observable regime rather than a single-coherence theory.

## 2. Chennu Propofol Raw-Data Validation

### Dataset

Source:

- Cambridge repository item: `sedation-restingstate.zip`
- Downloaded raw archive size: 3.69 GB.
- Extracted data: 80 EEGLAB `.set` files with corresponding `.fdt` files.
- Subjects: 20.
- Conditions per subject: baseline, mild, moderate, recovery.
- Channels: 91 EEG channels.

### Pipeline

For each file:

- Load EEGLAB raw EEG.
- Preprocess with 1-45 Hz filter, 50 Hz notch, average reference, 250 Hz
  resampling.
- Extract phase dynamics in alpha and gamma bands.
- Compute `R`, `D_eff`, `M_tau`.
- Calibrate GCC bounds within subject from baseline.
- Compute regime occupancy `Pi`.

Cross-validation:

- Leave-one-subject-out.
- Baselines: `R` only, `D_eff` only, `M_tau` only, `Pi` only, GCC triad, and
  GCC triad + `Pi`.

### Alpha-Band Results

Band: 8-15 Hz

| Condition | R mean | D mean | M mean | Pi mean |
|---|---:|---:|---:|---:|
| Baseline | 0.147 | 2.690 | 0.002865 | 0.659 |
| Mild | 0.142 | 2.722 | 0.002671 | 0.641 |
| Moderate | 0.117 | 2.918 | 0.002524 | 0.561 |
| Recovery | 0.149 | 2.619 | 0.002723 | 0.632 |

Paired baseline vs. moderate:

| Metric | Mean difference | Paired d | Wilcoxon p |
|---|---:|---:|---:|
| R | +0.030 | 0.964 | 3.22e-4 |
| D | -0.228 | -0.660 | 0.0136 |
| M | +0.000341 | 0.592 | 0.0240 |
| Pi | +0.098 | 0.688 | 0.00365 |

Leave-one-subject-out baseline vs. moderate AUC:

| Model | AUC |
|---|---:|
| R-only | 0.723 |
| D-only | 0.668 |
| M-only | 0.668 |
| Pi-only | 0.805 |
| GCC triad | 0.680 |
| GCC triad + Pi | 0.788 |

Interpretation:

Alpha-band GCC regime occupancy decreases from baseline to moderate propofol
sedation with a moderate paired effect. Pi-only is the strongest simple
baseline for baseline-vs-moderate in this aggregate-file analysis.

### Gamma-Band Results

Band: 30-45 Hz

| Condition | R mean | D mean | M mean | Pi mean |
|---|---:|---:|---:|---:|
| Baseline | 0.099 | 4.451 | 0.002396 | 0.661 |
| Mild | 0.094 | 4.209 | 0.002212 | 0.604 |
| Moderate | 0.087 | 3.987 | 0.001975 | 0.543 |
| Recovery | 0.102 | 4.358 | 0.002603 | 0.569 |

Paired baseline vs. moderate:

| Metric | Mean difference | Paired d | Wilcoxon p |
|---|---:|---:|---:|
| R | +0.011 | 0.562 | 0.0136 |
| D | +0.463 | 1.136 | 6.29e-5 |
| M | +0.000420 | 0.515 | 0.0362 |
| Pi | +0.117 | 0.992 | 6.29e-5 |

Leave-one-subject-out baseline vs. moderate AUC:

| Model | AUC |
|---|---:|
| R-only | 0.760 |
| D-only | 0.823 |
| M-only | 0.723 |
| Pi-only | 0.858 |
| GCC triad | 0.850 |
| GCC triad + Pi | 0.940 |

Interpretation:

Gamma-band results are strong for baseline vs. moderate propofol. The
multivariate GCC triad + Pi reaches AUC = 0.94 under leave-one-subject-out
cross-validation. This is the strongest empirical result in Batch 3.

Important limitation:

Four-level classification remains modest. The model is better supported as a
state-transition/regime-occupancy model than as a fine-grained classifier of all
sedation stages.

## 3. Sleep-EDF Wake/REM/NREM Validation

### Dataset

Subset downloaded with MNE Sleep PhysioNet loader:

- Subjects: `SC4001`, `SC4011`, `SC4021`, `SC4031`.
- Recordings: one full-night PSG per subject.
- EEG channels used: `EEG Fpz-Cz`, `EEG Pz-Oz`.
- Hypnogram labels: Wake, REM, NREM.
- Extracted epochs: 3921.

State counts:

| State | Epochs |
|---|---:|
| Wake | 613 |
| REM | 667 |
| NREM | 2641 |

### Alpha-Band Results

Band: 8-13 Hz

| State | R mean | D mean | M mean | Pi mean | Access-all mean |
|---|---:|---:|---:|---:|---:|
| Wake | 0.579 | 1.502 | 0.0951 | 0.831 | 0.595 |
| REM | 0.472 | 1.483 | 0.0940 | 0.766 | 0.426 |
| NREM | 0.482 | 1.553 | 0.0936 | 0.788 | 0.480 |

Kruskal-Wallis tests:

| Metric | H | p |
|---|---:|---:|
| R | 888.47 | 1.18e-193 |
| D_eff | 41.80 | 8.39e-10 |
| M_tau | 35.95 | 1.57e-8 |
| Pi | 35.53 | 1.93e-8 |

Leave-one-subject-out results:

| Model | Wake vs NREM AUC | REM vs NREM AUC | Multiclass accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| R-only | 0.864 | 0.528 | 0.448 | 0.446 |
| D-only | 0.451 | 0.388 | 0.477 | 0.277 |
| M-only | 0.395 | 0.524 | 0.243 | 0.207 |
| GCC triad | 0.814 | 0.407 | 0.542 | 0.452 |
| Pi-only | 0.417 | 0.504 | 0.154 | 0.162 |

Interpretation:

Sleep-EDF supports state differentiation in the GCC observable space, especially
Wake vs. NREM. However, REM is not cleanly separated from NREM in this
two-channel subset. This should be framed as a limited pilot, not as a complete
sleep-consciousness validation.

## 4. Integration with Batch 2 ds004504

Batch 2 already established a raw-data degradation proxy:

- Dataset: OpenNeuro `ds004504`.
- Subjects: 88.
- Groups: healthy controls, Alzheimer's disease, frontotemporal dementia.
- Alpha-band `Pi` separated controls from Alzheimer's disease:
  - Mean difference = 0.1048.
  - 95% bootstrap CI = [0.0255, 0.1803].
  - Cohen's d = 0.666.
  - Mann-Whitney p = 0.00547.
  - Spearman `Pi` vs. MMSE: rho = 0.240, p = 0.0244.

Scientific role:

This validates that GCC observables can be reconstructed from real public EEG
and that regime occupancy decreases in a clinical degradation proxy. It does
not validate terminal lucidity.

## 5. DS005620 Independent Propofol Replication

### Dataset

DS005620 is an independent public BIDS/BrainVision EEG dataset:

- OpenNeuro ID: `ds005620`, version `1.0.0`.
- DOI: `10.18112/openneuro.ds005620.v1.0.0`.
- Study: repeated awakening during propofol sedation.
- Downloaded subset: all available `awake-EC`, `sed-rest`, and `sed2-rest`
  recordings.
- Subjects in subset: 21.
- Recordings: 126.
- EEG channels used: 62, excluding VEOG/HEOG/EMG.
- Original sampling rate: 5000 Hz.
- Analysis sampling rate: 250 Hz.
- Analysis duration: first 120 seconds for 300-second recordings; full 60
  seconds for `sed2` recordings.

This is not the entire 77.3 GB dataset, but it is a large multi-subject
resting-state subset targeted to the consciousness-state question and excludes
TMS recordings.

### Alpha-Band Results

Band: 8-15 Hz

Subject-level condition means:

| Condition | R mean | D mean | M mean | Pi mean | Access-all mean |
|---|---:|---:|---:|---:|---:|
| Awake | 0.138 | 2.992 | 0.00366 | 0.831 | 0.595 |
| Sed | 0.129 | 3.845 | 0.00484 | 0.654 | 0.313 |
| Sed2 | 0.128 | 2.988 | 0.00528 | 0.668 | 0.313 |

Paired tests:

| Comparison | Metric | Mean difference | Paired d | Wilcoxon p |
|---|---|---:|---:|---:|
| Awake - Sed | Pi | +0.177 | 1.157 | 1.05e-4 |
| Awake - Sed | Access-all | +0.280 | 1.165 | 1.05e-4 |
| Awake - Sed2 | Pi | +0.164 | 1.081 | 1.34e-4 |
| Awake - Sed2 | Access-all | +0.281 | 1.127 | 1.68e-4 |

Leave-one-subject-out AUC:

| Model | Awake vs Sed | Awake vs Sed2 |
|---|---:|---:|
| Pi-only | 0.818 | 0.846 |
| GCC triad + Pi | 0.825 | 0.852 |
| GCC all | 0.830 | 0.854 |

Interpretation:

Alpha replicates the state-shift direction, but less strongly than gamma.

### Gamma-Band Results

Band: 30-45 Hz

Subject-level condition means:

| Condition | R mean | D mean | M mean | Pi mean | Access-all mean |
|---|---:|---:|---:|---:|---:|
| Awake | 0.126 | 5.873 | 0.00412 | 0.831 | 0.606 |
| Sed | 0.126 | 5.684 | 0.00512 | 0.444 | 0.112 |
| Sed2 | 0.147 | 5.240 | 0.00614 | 0.491 | 0.133 |

Paired tests:

| Comparison | Metric | Mean difference | Paired d | Wilcoxon p |
|---|---|---:|---:|---:|
| Awake - Sed | Pi | +0.387 | 1.910 | 1.91e-6 |
| Awake - Sed | Access-all | +0.496 | 4.419 | 1.91e-6 |
| Awake - Sed2 | Pi | +0.341 | 1.970 | 1.91e-6 |
| Awake - Sed2 | Access-all | +0.474 | 3.263 | 1.91e-6 |

Leave-one-subject-out AUC:

| Model | Awake vs Sed | Awake vs Sed2 |
|---|---:|---:|
| Pi-only | 1.000 | 1.000 |
| GCC triad + Pi | 0.981 | 0.979 |
| GCC all | 0.981 | 0.995 |

Interpretation:

This is now the strongest empirical replication result in the package. Gamma
GCC regime occupancy and strict access-regime occupancy sharply separate awake
from propofol-sedated recordings in an independent public dataset.

Important limitation:

`Pi` is calibrated from each subject's awake baseline. Therefore the result
supports within-subject regime shift, not universal cross-subject thresholds.
This is appropriate for the paper's current claim.

## 6. What Is Now Stronger

The manuscript can now make these empirically supported claims:

- GCC observables are reconstructable from real EEG.
- A regime occupancy score is computable from raw public EEG.
- Propofol sedation produces systematic shifts in GCC observables and Pi.
- Propofol effects replicate across two independent datasets: Chennu and
  DS005620.
- Neurodegenerative degradation produces lower alpha-band regime occupancy in
  ds004504.
- Synthetic matched simulations show selective backbone re-entry relative to
  uniform degradation.
- The multi-observable formulation is especially justified under noise and
  lesion perturbations, where single observables behave unevenly.

## 7. What Must Still Be Framed Carefully

Do not claim:

- That GCC is empirically validated as a theory of subjective experience.
- That terminal lucidity is empirically validated.
- That GCC always outperforms coherence-only baselines.
- That thresholds are universal.
- That Sleep-EDF two-channel results establish full large-scale network access.
- That the entire DS005620 77.3 GB dataset has been analyzed. The completed
  analysis uses a targeted 30.4 GB resting-state subset.

Best manuscript phrasing:

"The present results provide synthetic and public EEG support for the
operational GCC regime as a measurable marker of access-compatible dynamics.
They do not establish a universal consciousness criterion or empirically verify
terminal lucidity. Terminal lucidity remains a model-derived extension requiring
dedicated end-of-life or paradoxical-lucidity datasets."

## 8. Output Files

Synthetic:

- `results/synthetic_matched_stress_summary.json`
- `results/synthetic_matched_reentry_cases.csv`
- `results/synthetic_sweep_cases.csv`
- `results/synthetic_matched_reentry_delta.png`
- `results/synthetic_lesion_noise_sweeps.png`

Chennu:

- `data/sedation-restingstate.zip`
- `data/sedation_restingstate/Sedation-RestingState/`
- `data/chennu_labels.csv`
- `results/chennu_raw_alpha/pilot_summary_alpha.csv`
- `results/chennu_raw_gamma/pilot_summary_gamma.csv`
- `results/chennu_raw_cv_summary.json`
- `results/chennu_alpha_paired_summary.png`
- `results/chennu_gamma_paired_summary.png`

Sleep-EDF:

- `data/sleep_edf/physionet-sleep-data/`
- `results/sleep_edf_alpha_epoch_features.csv`
- `results/sleep_edf_alpha_summary.json`
- `results/sleep_edf_state_observables.png`
- `results/sleep_edf_R_D_plane.png`

DS005620:

- `data/ds005620_subset/`
- `results/ds005620/ds005620_alpha_recording_summary.csv`
- `results/ds005620/ds005620_gamma_recording_summary.csv`
- `results/ds005620/ds005620_alpha_subject_condition_means.csv`
- `results/ds005620/ds005620_gamma_subject_condition_means.csv`
- `results/ds005620/ds005620_subset_summary.json`
- `results/ds005620/ds005620_stats_summary.json`
- `results/ds005620/ds005620_alpha_subject_condition_means.png`
- `results/ds005620/ds005620_gamma_subject_condition_means.png`

Scripts:

- `batch3_synthetic_benchmark.py`
- `batch3_synthetic_matched_stress.py`
- `batch3_sleep_edf_validation.py`
- `batch3_chennu_cv_summary.py`
- `batch3_ds005620_subset_analysis.py`
- `batch3_ds005620_stats.py`

## 9. Recommended Manuscript Consequence

The paper should be reframed around evaluated predictions P2-P6:

- regime observables are measurable,
- sedation shifts the regime,
- degradation shifts the regime,
- noise/lesion simulations behave as predicted,
- backbone re-entry is demonstrated synthetically.

Terminal lucidity should remain a theoretically motivated extension, not the
central empirically evaluated claim.

This directly addresses the PLOS rejection better than the previous version,
because central claims are now evaluated in synthetic systems and public EEG
datasets.
