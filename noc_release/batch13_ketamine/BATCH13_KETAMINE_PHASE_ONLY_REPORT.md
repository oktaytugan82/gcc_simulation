# Batch 13 Ketamine Phase-Only GCC Extension

Date: 2026-05-14

## Scope

This is an other-anesthetics/altered-state extension using the Farnes et al. public ketamine EEG dataset. The dataset contains normal wakefulness and sub-anaesthetic ketamine, not deep loss of consciousness. The defensible GCC question is therefore whether access-compatible regime structure is preserved while observables, especially effective dimensionality, shift.

Source: https://zenodo.org/records/4245091

## Recording Counts

| band | condition | eyes | n_recordings | n_subjects | Pi_mean | D_mean |
| --- | --- | --- | --- | --- | --- | --- |
| alpha | awake | closed | 10 | 10 | 0.8385 | 6.418 |
| alpha | awake | open | 10 | 10 | 0.8219 | 8.007 |
| alpha | ketamine | closed | 10 | 10 | 0.7955 | 7.05 |
| alpha | ketamine | open | 9 | 9 | 0.705 | 8.863 |
| gamma | awake | closed | 10 | 10 | 0.816 | 14.39 |
| gamma | awake | open | 10 | 10 | 0.8436 | 16.48 |
| gamma | ketamine | closed | 10 | 10 | 0.6844 | 13.23 |
| gamma | ketamine | open | 9 | 9 | 0.7045 | 16.25 |


## Loader Exclusions

| subject | condition | eyes | filename | error |
| --- | --- | --- | --- | --- |
| 210 | ketamine | open | 210_20161207_0006eyesOpen_afterICA.set | buffer is too small for requested array |
| 210 | ketamine | open | 210_20161207_0006eyesOpen_afterICA.set | buffer is too small for requested array |


Excluded recordings: 2. These files could not be decoded by MNE/Scipy and were not used.



## Pooled Awake-vs-Ketamine Paired Effects

| band | eyes | metric | n | awake_mean | ketamine_mean | mean_delta_ketamine_minus_awake | paired_d_delta | wilcoxon_two_sided_p | wilcoxon_delta_greater_p | wilcoxon_delta_less_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | pooled_eyes | D_mean | 10 | 7.212 | 7.844 | 0.6311 | 0.7063 | 0.08398 | 0.04199 | 0.9678 |
| alpha | pooled_eyes | M_mean | 10 | 0.003019 | 0.003038 | 1.884e-05 | 0.0332 | 0.5566 | 0.2783 | 0.7539 |
| alpha | pooled_eyes | Pi | 10 | 0.8302 | 0.7577 | -0.07253 | -0.8541 | 0.08398 | 0.9678 | 0.04199 |
| alpha | pooled_eyes | R_mean | 10 | 0.1234 | 0.1264 | 0.003062 | 0.1212 | 0.9219 | 0.5771 | 0.4609 |
| gamma | pooled_eyes | D_mean | 10 | 15.44 | 14.65 | -0.7852 | -0.5042 | 0.1309 | 0.9473 | 0.06543 |
| gamma | pooled_eyes | M_mean | 10 | 0.003765 | 0.003735 | -2.94e-05 | -0.03676 | 0.7695 | 0.6523 | 0.3848 |
| gamma | pooled_eyes | Pi | 10 | 0.8298 | 0.6878 | -0.1419 | -1.093 | 0.009766 | 0.9971 | 0.004883 |
| gamma | pooled_eyes | R_mean | 10 | 0.1211 | 0.1231 | 0.001964 | 0.1514 | 0.7695 | 0.3848 | 0.6523 |


## Interpretation

This extension should not be framed as another propofol-style loss-of-consciousness replication. It is more valuable as a pharmacological boundary case: if Pi remains access-compatible while D_eff or coherence shifts, GCC gains specificity by distinguishing sedative loss from altered conscious content.
