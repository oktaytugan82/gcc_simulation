# Batch 6 DS005620 Local Audit

## Core Finding

Local DS005620 data contain 126 recordings from 21 subjects.
The public dataset is reported as 202 recordings and approximately 77.3 GB.
Estimated local gap: 76 recordings and about 46.9 GB.

## Local Task Counts

- awake: 21
- sed: 54
- sed2: 51

## Consequence

The current local analysis must be described as a DS005620 subset analysis, not a full-dataset analysis.
The next strengthening step is either to analyse all 126 local recordings more fully or to download the remaining public data before claiming full DS005620 coverage.

## Stop/Go

- GO now: repeated-run stability and within-subject robustness on the 126 local recordings.
- STOP before full-dataset claims: download/audit the missing recordings.