# Hermann et al. 2021 FDG-PET + EEG DoC Supplement

Source paper:

Bertrand Hermann et al. (2021). "Multimodal FDG-PET and EEG assessment improves diagnosis and prognostication of disorders of consciousness." NeuroImage: Clinical 30:102601. https://doi.org/10.1016/j.nicl.2021.102601

Downloaded files:

- `Hermann2021_supplementary_methods_mmc1.docx`
  - Supplementary methods / software details.
  - Source: https://ars.els-cdn.com/content/image/1-s2.0-S2213158221000450-mmc1.docx
- `Hermann2021_supplementary_metadata_mmc2.csv`
  - Anonymized post-processed patient metadata table.
  - Source: https://ars.els-cdn.com/content/image/1-s2.0-S2213158221000450-mmc2.csv

Note:

The article states that raw patient brain-imaging data cannot be openly shared because of ethics restrictions. The available public dataset is the anonymized/post-processed supplementary metadata CSV, not raw EEG or FDG-PET recordings.

GCC proxy analysis:

- Script: `run_hermann2021_gcc_proxy_analysis.py`
- Feature table: `../results/hermann2021_gcc_proxy_features.csv`
- Summary: `../results/hermann2021_gcc_proxy_summary.json`
- Short report: `../results/hermann2021_gcc_proxy_report.md`
- Figures:
  - `../figures/hermann2021_gcc_proxy_diagnostic.png`
  - `../figures/hermann2021_gcc_proxy_outcome.png`

Interpretation guardrail: this is a proxy analysis of post-processed biomarkers. PET MIBH is treated as metabolic capacity, out-of-sample EEG SVM p(MCS) as dynamic access evidence, and their z-scored mean as a conservative GCC PET+EEG proxy. It is not raw EEG/PET GCC feature extraction.
