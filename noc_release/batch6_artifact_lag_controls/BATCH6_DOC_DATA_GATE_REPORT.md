# Batch 6 DoC / Degradation-Proxy Data Gate

Date: 2026-05-13

## Candidate Dataset

Mendeley Data:

- Title: `Polysomnographic records of patients with disorders of consciousness`
- DOI: `10.17632/6wx4n25h4v.1`
- Version: 1
- Licence: CC BY 4.0
- Contributors: Julia Nekrasova, Mikhail Kanarskii, Dmitriy Yankevich, Ilya Borisov, Pranil Pradhan
- Categories: Electroencephalography, Sleep, Consciousness Studies

## Metadata Extracted From Saved Dataset Page

The locally saved Mendeley page reports:

- 42-43 polysomnographic records;
- approximately 6 hours per record;
- all records in EDF format;
- chronic disorders of consciousness;
- labels encoded in file names, e.g. `Patient_1_VS`, `Patient_2_MCS+`;
- 2 EOG channels, 1 EMG channel, and EEG channels over frontal, central and occipital zones;
- heterogeneous EEG references and channel counts depending on record quality.

## Scientific Use for GCC

This dataset is suitable only as an exploratory degradation proxy, not as a terminal-lucidity test.

Potential GCC use:

- VS vs MCS/MCS+ as a degradation-gradient proxy;
- test whether GCC regime scores are lower in VS than in MCS/MCS+;
- inspect whether residual-access windows occur in MCS/MCS+ more often than in VS;
- quantify robustness under heterogeneous montage/channel availability.

## Risks

- No healthy reference state in the same dataset.
- Long PSG recordings likely contain sleep-stage and artifact heterogeneity.
- Channel sets and references vary.
- Clinical labels are coarse and may not map cleanly to momentary conscious access.
- A negative result would not falsify GCC; it would mainly show poor transfer under heterogeneous clinical data.

## Current Gate Status

The live file-list API request was not executed because the desktop session rejected the network escalation due to the current usage limit. Therefore no download was started and no storage-heavy action was taken.

Current usable artifact:

- `mendeley_doc_metadata_from_page.json`

## Next Concrete Step

When network/API access is available again:

1. Query the public Mendeley dataset file list.
2. Record EDF filenames and sizes before downloading.
3. Download only a minimal stratified subset first: several VS, MCS-, and MCS+ records.
4. Run an EDF audit: channel names, sampling rates, duration, label parsing, missing channels.
5. Define a common EEG subset or region-averaged montage.
6. Compute GCC as exploratory clinical degradation proxy with explicit caveats.

## Paper Consequence

Until the EDF subset is actually processed, this dataset should not be presented as empirical evidence in the main paper. It can be listed as a planned external degradation-proxy validation or included in a preregistered analysis plan.
