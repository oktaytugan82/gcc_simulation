# Batch 8 Sleep-EDF Third-State Validation

Date: 2026-05-14

## Aim

This batch adds a third, non-propofol state-transition dataset to reduce the main remaining NoC vulnerability: that the empirical evidence is propofol-specific.

The goal is conservative:

- not to claim full sleep-consciousness validation;
- not to claim dream-content access;
- only to test whether GCC observables differentiate annotated Wake/REM/NREM states in an independent public sleep EEG dataset.

## Dataset

Sleep-EDF Expanded / Sleep Cassette subset from PhysioNet.

Local expanded subset:

- 11 complete PSG + hypnogram pairs;
- 11,261 scored 30 s epochs;
- states: Wake, REM, NREM.

State counts:

- Wake: 1,812 epochs;
- REM: 1,970 epochs;
- NREM: 7,479 epochs.

Important limitation:

- only two EEG channels are available in this Sleep-EDF subset (`EEG Fpz-Cz`, `EEG Pz-Oz`);
- this is therefore a state-differentiation test of the observable triad, not a full large-scale network test.

## Method

The existing Sleep-EDF pipeline was rerun on the expanded subset.

Two bands were evaluated:

- alpha: 8-13 Hz;
- sigma: 12-16 Hz.

Regime boundaries were calibrated from Wake epochs using the same quantile rule (`alpha = 0.10`). Leave-one-subject-out classification was used for:

- Wake vs NREM;
- REM vs NREM;
- three-state multiclass classification.

Baseline models:

- R only;
- D only;
- M only;
- GCC triad `(R, D_eff, log M)`;
- Pi only.

## Key Results

### Alpha Band

Alpha was mixed and should not be overused.

| Model | Wake vs NREM AUC | REM vs NREM AUC | Multiclass accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| R only | 0.780 | 0.543 | 0.411 | 0.414 |
| GCC triad | 0.748 | 0.543 | 0.406 | 0.411 |
| Pi only | 0.547 | 0.497 | 0.572 | 0.340 |

Interpretation:

- alpha mainly separates Wake from NREM through coherence;
- it does not robustly separate REM from NREM;
- alpha Pi is not a strong sleep-state marker.

### Sigma Band

Sigma gives the useful third-dataset result.

| Model | Wake vs NREM AUC | REM vs NREM AUC | Multiclass accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| R only | 0.932 | 0.745 | 0.654 | 0.600 |
| D only | 0.656 | 0.862 | 0.618 | 0.501 |
| M only | 0.784 | 0.748 | 0.557 | 0.451 |
| GCC triad | 0.949 | 0.877 | 0.741 | 0.680 |
| Pi only | 0.800 | 0.631 | 0.525 | 0.472 |

State means in sigma:

| State | R | D_eff | M_tau | Pi | Access_all |
|---|---:|---:|---:|---:|---:|
| Wake | 0.584 | 1.671 | 0.096 | 0.833 | 0.571 |
| REM | 0.500 | 1.878 | 0.095 | 0.650 | 0.306 |
| NREM | 0.451 | 1.495 | 0.087 | 0.476 | 0.099 |

## Interpretation

The sigma-band result is valuable because it is not another propofol replication. It shows that the observable triad can separate sleep-state regimes in an independent public EEG dataset.

The key NoC-relevant result is:

> In Sleep-EDF, the sigma-band GCC triad differentiates Wake from NREM with AUC = 0.949 and REM from NREM with AUC = 0.877 under leave-one-subject-out validation, outperforming each single observable in the combined multiclass setting.

This supports the broad regime claim:

- GCC observables are not only propofol-sensitive;
- they also capture annotated sleep-state geometry;
- the strongest performance comes from the joint triad, not from Pi alone.

## Important Caveats

1. Sleep-EDF has only two EEG channels in this subset.
2. The analysis is sleep-state differentiation, not direct conscious-access measurement.
3. REM is not equivalent to wakeful access.
4. The sigma band is physiologically sleep-relevant; therefore this is not a direct replication of the gamma propofol result.
5. Pi alone is not the strongest Sleep-EDF marker; the triad is the relevant result.

## Recommended Manuscript Claim

Use this wording:

> As a non-propofol state-transition check, we applied the frozen GCC observable pipeline to an expanded Sleep-EDF subset (11 subjects, 11,261 scored 30 s epochs). In the sleep-relevant sigma band, the GCC triad separated Wake from NREM (AUC = 0.949) and REM from NREM (AUC = 0.877) under leave-one-subject-out validation, with multiclass accuracy = 0.741 and macro-F1 = 0.680. This does not validate conscious access directly, because Sleep-EDF contains only two EEG channels and sleep-stage labels are coarse, but it shows that GCC state geometry generalizes beyond propofol sedation to an independent sleep-state dataset.

## Output Files

- `sleep_edf_alpha_epoch_features.csv`
- `sleep_edf_alpha_summary.json`
- `sleep_edf_sigma_epoch_features.csv`
- `sleep_edf_sigma_summary.json`
- `sleep_edf_state_observables.png`
- `sleep_edf_R_D_plane.png`
