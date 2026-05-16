# Batch 16 Public DoC GCC Validation

Date: 2026-05-14

## Dataset

Public Mendeley Data dataset 10.17632/6wx4n25h4v.1, 42 polysomnographic EDF recordings from chronic disorders of consciousness. Labels are parsed from filenames: VS, MCS-, MCS+. The frozen common montage uses F3, F4, C3, C4, O1, and O2, present in all files.

## Audit

- Files analyzed: 42

- Label counts: {'VS': 30, 'MCS+': 8, 'MCS-': 4}

- Sampling rate: {'256_raw_to_128': 42}

- Analysis duration per file: first 14400.0 s

## Cross-Validated Clinical Endpoints

| band | contrast | model | n | auc | auc_ci_low | auc_ci_high | balanced_accuracy_median_cut | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | MCSany_vs_VS | spectral_plus_GCC | 42 | 0.6889 | 0.5111 | 0.85 | 0.6667 |  |  |
| alpha | MCSany_vs_VS | CV_GCC_access_Pi | 42 | 0.6861 | 0.4778 | 0.8722 | 0.5583 |  |  |
| alpha | MCSany_vs_VS | spectral_all | 42 | 0.6694 | 0.4833 | 0.8417 | 0.5833 |  |  |
| alpha | MCSany_vs_VS | M_only | 42 | 0.5361 | 0.3528 | 0.7167 | 0.5583 |  |  |
| alpha | MCSany_vs_VS | R_only | 42 | 0.4833 | 0.2667 | 0.7083 | 0.5333 |  |  |
| alpha | MCSany_vs_VS | GCC_triad | 42 | 0.4139 | 0.2389 | 0.6 | 0.5167 |  |  |
| alpha | MCSany_vs_VS | GCC_triad_quantiles | 42 | 0.3944 | 0.2222 | 0.575 | 0.5167 |  |  |
| alpha | MCSany_vs_VS | D_only | 42 | 0.2 | 0.07222 | 0.3611 | 0.3667 |  |  |
| gamma | MCSany_vs_VS | spectral_plus_GCC | 42 | 0.6917 | 0.5139 | 0.85 | 0.6583 |  |  |
| gamma | MCSany_vs_VS | spectral_all | 42 | 0.6694 | 0.4833 | 0.8417 | 0.5833 |  |  |
| gamma | MCSany_vs_VS | M_only | 42 | 0.6528 | 0.4667 | 0.8222 | 0.6583 |  |  |
| gamma | MCSany_vs_VS | D_only | 42 | 0.6472 | 0.4417 | 0.8444 | 0.7083 |  |  |
| gamma | MCSany_vs_VS | CV_GCC_access_Pi | 42 | 0.6028 | 0.4083 | 0.7806 | 0.5 |  |  |
| gamma | MCSany_vs_VS | GCC_triad | 42 | 0.5528 | 0.3556 | 0.7472 | 0.55 |  |  |
| gamma | MCSany_vs_VS | GCC_triad_quantiles | 42 | 0.5389 | 0.3333 | 0.7362 | 0.4917 |  |  |
| gamma | MCSany_vs_VS | R_only | 42 | 0.3611 | 0.1778 | 0.5611 | 0.4 |  |  |
| alpha | MCSplus_vs_VS | CV_GCC_access_Pi | 38 | 0.7542 | 0.5792 | 0.9 | 0.6583 |  |  |
| alpha | MCSplus_vs_VS | spectral_plus_GCC | 38 | 0.6792 | 0.4625 | 0.8668 | 0.5542 |  |  |
| alpha | MCSplus_vs_VS | M_only | 38 | 0.6417 | 0.4625 | 0.8083 | 0.5958 |  |  |
| alpha | MCSplus_vs_VS | spectral_all | 38 | 0.575 | 0.3417 | 0.7958 | 0.5042 |  |  |
| alpha | MCSplus_vs_VS | GCC_triad_quantiles | 38 | 0.55 | 0.3667 | 0.725 | 0.4708 |  |  |
| alpha | MCSplus_vs_VS | GCC_triad | 38 | 0.5458 | 0.3667 | 0.7125 | 0.55 |  |  |
| alpha | MCSplus_vs_VS | R_only | 38 | 0.475 | 0.2 | 0.7542 | 0.5708 |  |  |
| alpha | MCSplus_vs_VS | D_only | 38 | 0.3667 | 0.15 | 0.6042 | 0.4833 |  |  |
| gamma | MCSplus_vs_VS | M_only | 38 | 0.7458 | 0.5292 | 0.9208 | 0.7083 |  |  |
| gamma | MCSplus_vs_VS | CV_GCC_access_Pi | 38 | 0.6417 | 0.4042 | 0.8542 | 0.6583 |  |  |
| gamma | MCSplus_vs_VS | D_only | 38 | 0.6375 | 0.3958 | 0.8542 | 0.7083 |  |  |
| gamma | MCSplus_vs_VS | spectral_plus_GCC | 38 | 0.5875 | 0.3792 | 0.7792 | 0.5375 |  |  |
| gamma | MCSplus_vs_VS | spectral_all | 38 | 0.575 | 0.3417 | 0.7958 | 0.5042 |  |  |
| gamma | MCSplus_vs_VS | GCC_triad_quantiles | 38 | 0.5708 | 0.3167 | 0.8125 | 0.6333 |  |  |
| gamma | MCSplus_vs_VS | R_only | 38 | 0.5333 | 0.2792 | 0.8 | 0.55 |  |  |
| gamma | MCSplus_vs_VS | GCC_triad | 38 | 0.5333 | 0.2833 | 0.7667 | 0.5667 |  |  |
| alpha | VS_to_MCSminus_to_MCSplus | severity_spearman_R_mean | 42 |  |  |  |  | 0.1474 | 0.3515 |
| alpha | VS_to_MCSminus_to_MCSplus | severity_spearman_D_eff_mean | 42 |  |  |  |  | -0.02188 | 0.8906 |
| alpha | VS_to_MCSminus_to_MCSplus | severity_spearman_log_M_mean | 42 |  |  |  |  | -0.1654 | 0.2951 |
| alpha | VS_to_MCSminus_to_MCSplus | severity_spearman_R_q75 | 42 |  |  |  |  | 0.1241 | 0.4335 |
| alpha | VS_to_MCSminus_to_MCSplus | severity_spearman_D_eff_q25 | 42 |  |  |  |  | -0.00409 | 0.9795 |
| gamma | VS_to_MCSminus_to_MCSplus | severity_spearman_R_mean | 42 |  |  |  |  | 0.0182 | 0.9089 |
| gamma | VS_to_MCSminus_to_MCSplus | severity_spearman_D_eff_mean | 42 |  |  |  |  | -0.2671 | 0.08732 |
| gamma | VS_to_MCSminus_to_MCSplus | severity_spearman_log_M_mean | 42 |  |  |  |  | -0.319 | 0.03948 |
| gamma | VS_to_MCSminus_to_MCSplus | severity_spearman_R_q75 | 42 |  |  |  |  | -0.07914 | 0.6184 |
| gamma | VS_to_MCSminus_to_MCSplus | severity_spearman_D_eff_q25 | 42 |  |  |  |  | -0.1808 | 0.252 |

## Label Means

| band | label | R_mean | D_eff_mean | M_tau_mean | n_epochs |
| --- | --- | --- | --- | --- | --- |
| alpha | MCS+ | 0.1039 | 4.766 | 0.009829 | 389.9 |
| alpha | MCS- | 0.08414 | 4.776 | 0.01229 | 404.8 |
| alpha | VS | 0.08729 | 4.771 | 0.01206 | 344.7 |
| gamma | MCS+ | 0.09454 | 4.741 | 0.0076 | 389.9 |
| gamma | MCS- | 0.102 | 4.743 | 0.008471 | 404.8 |
| gamma | VS | 0.09924 | 4.748 | 0.009224 | 344.7 |

## Interpretation Rule

A spectacular GCC result would require GCC_triad or CV_GCC_access_Pi to outperform spectral_all on VS/MCS classification with non-overlapping or clearly shifted bootstrap CIs, and preferably a monotonic severity trend VS < MCS- < MCS+. If spectral_all dominates, the DoC dataset still supports state sensitivity but not a unique GCC clinical biomarker claim.
