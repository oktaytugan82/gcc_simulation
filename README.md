# GCC — Gamma-Code Consensus Simulation Code

This repository contains the complete simulation and analysis code accompanying the manuscript:

> **The Gamma-Code Consensus: a dynamical regime model for level of consciousness and global access**
> Mehmet Oktay Tugan (2026)
> Submitted to PLOS Computational Biology.

The Gamma-Code Consensus (GCC) is a dynamical regime model for level of consciousness and global access in a network of coupled neural populations. This code reproduces all numerical simulations (synthetic Kuramoto networks) and the empirical pilot analysis on the publicly available Chennu et al. 2016 EEG dataset.

## Repository structure

```
gcc_simulation/
├── simulation/              # Synthetic Kuramoto network experiments (predictions P2–P6)
│   ├── gcc_simulator.py          # Core stochastic simulator + network/observables
│   ├── calibrate.py              # Access-region calibration from healthy ensemble
│   ├── test_v2_v3_ensemble.py    # P3/P4: threshold shift & bounded interval (10 seeds × 4 lesion levels × 9 K)
│   ├── test_v4_ensemble.py       # P5: anaesthesia trajectory (20 seeds)
│   ├── test_v5_ensemble.py       # P6/P1: gain-mediated re-entry, baseline K=1.6 (20 seeds)
│   ├── test_v5_retuning_control.py  # P6 retuning control sweep: K∈{1.6, 2.0, 2.5, 3.0}, 20 seeds each
│   ├── test_v5_mixed_effects.py  # Primary analysis: mixed regression with seed clustering
│   └── test_v5_v2.py             # Helper module used by retuning-control
│
├── pilot/                   # Empirical EEG pilot analysis (Chennu 2016 dataset)
│   ├── load_chennu.py            # Loader for Chennu 2016 dataset structure
│   ├── load_eeglab_hdf5.py       # EEGLAB .set / HDF5 loader
│   ├── preprocessing.py          # Band-pass filter, Hilbert transform → phase extraction
│   ├── observables.py            # Computes R, D_eff, M_tau and the access indicator Π
│   ├── run_pilot_v2.py           # Main pilot run: per-subject Π per sedation level, both bands
│   └── run_sensitivity_sweep.py  # 72-config parameter sensitivity sweep (α × τ_D × τ_M)
│
├── results/                 # Pre-computed outputs cited in the paper
│   ├── calibration.pkl                 # Access-region bounds (R_min, D_min, D_max, M_max)
│   ├── v2_v3_ensemble_results.pkl      # K_c^op shifts and max Π per lesion level
│   ├── v4_ensemble_results.pkl         # Anaesthesia trajectory ensemble
│   ├── v5_ensemble_results.pkl         # Re-entry results at baseline K=1.6
│   ├── v5_retuning_control.pkl         # Full retuning-control sweep (4 K × 20 seeds)
│   ├── mixed_effects_summary.pkl       # Mixed-regression primary analysis output
│   ├── pilot_summary_gamma.csv         # Per-subject Π, gamma band
│   └── pilot_summary_alpha.csv         # Per-subject Π, alpha band
│
├── figures/                 # All figures used in the manuscript
│   ├── fig_v2_v3_ensemble.png            # Figure for P3/P4
│   ├── fig_v4_ensemble.png               # Figure for P5 (anaesthesia trajectory)
│   ├── fig_v5_ensemble.png               # Figure for P6/P1 main result
│   ├── fig_v5_retuning_control.png       # Figure for the retuning-control sweep
│   ├── pilot_boxplot_gamma.png           # Pilot Π distribution, gamma
│   ├── pilot_boxplot_alpha.png           # Pilot Π distribution, alpha
│   ├── pilot_trajectories_gamma.png      # Per-subject Π trajectories, gamma
│   ├── pilot_trajectories_alpha.png      # Per-subject Π trajectories, alpha
│   └── pilot_robustness.png              # Sensitivity sweep heat map
│
├── requirements.txt
├── LICENSE                  # MIT
├── .gitignore
└── README.md                # This file
```

## Installation

```bash
git clone https://github.com/oktaytugan82/gcc_simulation.git
cd gcc_simulation
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python ≥ 3.10 recommended. Tested on 3.11.

## Reproducing the paper's numerical results

All scripts are self-contained and produce the outputs already available in `results/` and `figures/`. To re-run end-to-end:

### 1. Calibrate the access region

```bash
cd simulation/
python calibrate.py
```

This creates `calibration.pkl` (~30 s).

### 2. Run predictions P3/P4 ensemble (threshold shift + bounded interval)

```bash
python test_v2_v3_ensemble.py
```

10 seeds × 4 lesion levels × 9 K-values. Runtime ~2 min on a modern laptop. Outputs: `v2_v3_ensemble_results.pkl` and `fig_v2_v3_ensemble.png`.

### 3. Run prediction P5 ensemble (anaesthesia trajectory)

```bash
python test_v4_ensemble.py
```

20 seeds. Runtime ~1 min. Outputs: `v4_ensemble_results.pkl`, `fig_v4_ensemble.png`.

### 4. Run prediction P6/P1 ensemble (gain-mediated re-entry at K=1.6)

```bash
python test_v5_ensemble.py
```

20 seeds, paired design (condition A uniform vs. B selective). Runtime ~2 min. Outputs: `v5_ensemble_results.pkl`, `fig_v5_ensemble.png`.

### 5. Retuning-control sweep (primary V6 result)

```bash
python test_v5_retuning_control.py
```

K ∈ {1.6, 2.0, 2.5, 3.0} × 20 seeds, full paired design. Runtime ~7 min. Outputs: `v5_retuning_control.pkl`, `fig_v5_retuning_control.png`.

### 6. Mixed-effects statistical analysis over the sweep

```bash
python test_v5_mixed_effects.py
```

Computes the paper's primary statistic:
- **Main effect of Condition (B vs A):** mixed regression with seed clustering, cluster-robust SE
- **Condition × K interaction test**
- **Secondary pooled Wilcoxon** (reported as descriptive plausibility check only)

Runtime ~5 s. Outputs: `mixed_effects_summary.pkl`.

## Reproducing the empirical pilot (Chennu 2016 EEG)

The pilot requires you to first download the Chennu 2016 dataset:

**Dataset**: [Chennu et al. 2016, University of Cambridge Data Repository, DOI: 10.17863/CAM.68959](https://doi.org/10.17863/CAM.68959) (CC BY 2.0 UK license)

Download the `.set` files (EEGLAB format) and place them under `data/chennu2016/`. The expected structure is one `.set` file per subject per condition (baseline, mild, moderate, recovery).

### Run the main pilot analysis

```bash
cd pilot/
python run_pilot_v2.py \
    --data_dir ../data/chennu2016 \
    --out_dir ../results \
    --band gamma
```

Repeat for `--band alpha`. Outputs: `pilot_summary_{band}.csv` with per-subject Π per sedation level.

### Run the 72-configuration sensitivity sweep

```bash
python run_sensitivity_sweep.py \
    --data_dir ../data/chennu2016 \
    --out_dir ../results
```

Tests robustness across 72 combinations of quantile width α and window lengths (τ_D, τ_M). Output: `pilot_robustness.png`.

## Paper reference

Tugan, M.O. (2026). The Gamma-Code Consensus: a dynamical regime model for level of consciousness and global access. *PLOS Computational Biology*, submitted.

## Chennu 2016 dataset reference

Chennu, S., O'Connor, S., Adapa, R., Menon, D.K., Bekinschtein, T.A. (2016). Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousness. *PLOS Computational Biology*, 12(1): e1004669. [DOI: 10.1371/journal.pcbi.1004669](https://doi.org/10.1371/journal.pcbi.1004669)

Dataset: [DOI: 10.17863/CAM.68959](https://doi.org/10.17863/CAM.68959)

## License

MIT License for all code in this repository (see `LICENSE`).

The Chennu 2016 EEG dataset is licensed separately under CC BY 2.0 UK — please follow that license's terms when using the data.

## Contact

Mehmet Oktay Tugan — oktaytugan82@gmail.com
ORCID: [0009-0005-8665-3583](https://orcid.org/0009-0005-8665-3583)
