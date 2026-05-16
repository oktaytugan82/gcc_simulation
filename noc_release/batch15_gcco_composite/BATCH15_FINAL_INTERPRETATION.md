# Batch 15 Final Interpretation: Can GCC Be Claimed as Generally Bandpower-Independent?

Date: 2026-05-14

## Direct Answer

No. Even after defining a stricter GCC-O candidate with lagged phase coupling, graph-spectral effective dimensionality, and no amplitude entering the GCC-O observables, the available evidence does not support the claim that GCC is a generally bandpower-independent biomarker.

## What Worked

GCC-O remains state-sensitive:

| Dataset | Band | Main paired result |
| --- | --- | --- |
| Chennu | alpha | baseline > moderate, Delta Pi = 0.090, d = 1.02, p = 4.45e-4 |
| Chennu | gamma | baseline > moderate, Delta Pi = 0.062, d = 0.85, p = 9.23e-4 |
| DS005620 | alpha | awake > sed, Delta Pi = 0.106, d = 0.60, p = 0.0042 |
| DS005620 | gamma | awake > sed, Delta Pi = 0.167, d = 1.73, p = 9.54e-7 |
| Ketamine | gamma | awake > ketamine, Delta Pi = 0.058, d = 0.60, p = 0.024 |

Sleep-EDF also shows state geometry: sigma GCC-O triad separates Wake from NREM with AUC = 0.899.

## What Failed

The strict bandpower-independence criterion failed.

Within-dataset residualized GCC-O after spectral regression:

| Dataset | Band | Residual GCC-O AUC |
| --- | --- | --- |
| Chennu | alpha | 0.655 |
| Chennu | gamma | 0.750 |
| DS005620 | alpha | 0.471 |
| DS005620 | gamma | 0.585 |
| Ketamine | alpha | 0.700 |
| Ketamine | gamma | 0.520 |
| Sleep-EDF Wake vs NREM | sigma | 0.650 |
| Sleep-EDF REM vs NREM | sigma | 0.525 |

Incremental AUC over spectral-only baselines was not robust. Bootstrap intervals generally touched zero or were negative; residualized GCC-O was usually below spectral-only models.

## Scientific Claim Allowed

The strongest defensible wording is:

> GCC-O is a lagged phase-regime variant whose observables do not directly use bandpass amplitude and which remains state-sensitive across propofol, ketamine, and sleep datasets. However, after stringent control for conventional spectral features, residual GCC-O does not yet support a general bandpower-independent biomarker claim.

## Scientific Claim Not Allowed

The following is not supported:

> GCC is a generally bandpower-independent biomarker of consciousness.

This would be vulnerable in peer review because spectral-only models remain highly competitive and residualized GCC-O weakens substantially in multiple datasets.

## Best Strategic Framing

Use GCC as a theory-grounded regime model and bandpower-aware composite, not as a bandpower-independent biomarker. If a future paper wants the independence claim, it needs a new prospective dataset or benchmark where residualized GCC-O shows robust above-chance performance and positive incremental value over spectral-only baselines.
