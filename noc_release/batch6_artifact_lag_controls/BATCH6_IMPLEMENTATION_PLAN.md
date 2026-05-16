# Batch 6 Implementation Plan: Make GCC Evidentially Stronger

Date: 2026-05-13

## Strategic Goal

Strengthen GCC as a unique regime model, not as a generic EEG classifier. The target claim is:

> GCC explains sedation and degraded-state dynamics as movement through a calibrated regime geometry over coherence, dimensionality, and temporal stability, with spectral power treated as a related but non-identical state marker.

## Priority Order

### 1. DS005620 Full Audit and Full-Recording Extension

Rationale: This is the fastest route to stronger evidence because DS005620 is already partly local and already produced the strongest replication.

Tasks:

- Audit local DS005620 subset.
- Determine available subjects, tasks, acquisitions, runs, durations, and missing recordings.
- Check whether repeated-awakening/report metadata is locally available.
- If full dataset is missing, document exact gap and prepare targeted download.
- Extend current analysis from the 126-record subset to all locally available recordings first.

Stop/Go:

- If local data are only the 126-record subset, do not claim full DS005620.
- If full DS005620 is needed, request explicit download approval because the full dataset is approximately 77 GB.

### 2. Bandpower-Matched and Surrogate Controls

Rationale: The strongest current weakness is spectral overlap. We strengthen GCC by showing when regime information survives under matched spectral power.

Controls:

- Bandpower-matched awake/sedated pairs.
- Within-bin analysis over alpha/gamma power bins.
- Phase-randomized surrogate with preserved power spectrum.
- Channel-shuffled or phase-shuffled surrogate to destroy cross-channel phase structure.

Desired outcome:

- If GCC remains above chance under matched spectral power, this is a major strengthening.
- If not, we report GCC as a theoretical spectral-regime composite rather than independent marker.

### 3. DoC Degradation Proxy

Rationale: GCC's real unique claim is residual-backbone/re-entry under degradation. Propofol alone does not test that.

Candidate:

- Mendeley DoC polysomnography dataset, VS/MCS records, DOI 10.17632/6wx4n25h4v.1.

Tasks:

- Download only after approval.
- Parse EDF.
- Harmonize channel subsets.
- Compute GCC features over stable sleep/wake-like windows if annotations permit.
- Test VS vs MCS/MCS+ as degradation-gradient proxy.

Stop/Go:

- If channel heterogeneity is too high, report as exploratory only.
- Do not use DoC result as a terminal-lucidity proxy.

### 4. Volume-Conduction and Referencing Controls

Rationale: Channel-level coherence can be inflated by volume conduction.

Controls:

- Surface Laplacian/current-source-density transform when montage allows.
- Alternative references.
- Imaginary coherence or wPLI-style phase coupling proxy.
- Posterior-only and non-frontotemporal sensitivity already started in Batch 5.

### 5. Generative Parameter Fitting

Rationale: This would shift GCC from feature extraction to mechanistic explanation.

Tasks:

- Fit simplified Kuramoto parameters per state: effective coupling, noise, gain scale.
- Compare fitted parameter trajectories awake -> sedated -> recovery.
- Test whether fitted trajectories reproduce empirical shifts in R, D_eff, and M_tau.

## Expected Paper Impact

High impact:

- DS005620 full repeated-awakening analysis.
- DoC VS/MCS proxy.
- Bandpower-matched/surrogate controls.

Medium impact:

- More references.
- More conservative caveats.
- Additional plots without new tests.

Low impact:

- More synthetic simulations alone.

## Current Starting Point

Local DS005620 subset:

- 126 BrainVision `.vhdr` recordings found locally.
- Approximate local size: 30.4 GB.
- Full public dataset is reported as 202 recordings and approximately 77.3 GB.

Immediate next artifact:

- `ds005620_local_audit.csv`
- `ds005620_local_audit_summary.json`
- `BATCH6_DS005620_AUDIT_REPORT.md`
