"""
GCC Pilot v3 — Full sensitivity sweep over calibration hyperparameters.

Re-runs the Chennu 2016 analysis across a grid of (alpha, tau_D, tau_M)
parameter values and reports how the main finding (baseline-moderate Pi
decline, Wilcoxon p-value) varies across the grid.

This script EXISTS TO BE RUN ONCE, as supplementary material to the
Pilot Demonstration section. Expected runtime: 6-10 hours on a modern
laptop (processes 80 files x ~15-20 parameter combinations).

Usage:
    python run_sensitivity_sweep.py <data_dir> --labels labels.csv \\
                                    --band gamma --outdir ./sweep_results

Output:
    sweep_results.csv — one row per parameter combination with
                        p-value, mean-decline, Cohen's d, N-decliners
    sweep_heatmap.png — visualization of the grid
"""

import argparse
import csv
from pathlib import Path
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from collections import defaultdict

from preprocessing import preprocess_raw, extract_gamma_phases
from observables import (compute_observables, access_region_indicator,
                          calibrate_from_baseline)
from load_eeglab_hdf5 import smart_load_set
import re


# Default sweep grid — edit as needed
ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]
TAU_D_GRID = [0.1, 0.2, 0.4]      # seconds
TAU_M_GRID = [0.3, 0.5, 1.0]      # seconds


def extract_subject(fn):
    m = re.match(r'^(\d{2})-', fn)
    return m.group(1) if m else 'XX'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--labels', required=True)
    ap.add_argument('--band', choices=['gamma', 'alpha'], default='gamma')
    ap.add_argument('--outdir', default='./sweep_results')
    args = ap.parse_args()

    flow, fhigh = (30.0, 45.0) if args.band == 'gamma' else (8.0, 15.0)
    print(f"Band: {args.band} ({flow}-{fhigh} Hz)")
    print(f"Grid: alpha={ALPHA_GRID}, tau_D={TAU_D_GRID}, tau_M={TAU_M_GRID}")
    total = len(ALPHA_GRID) * len(TAU_D_GRID) * len(TAU_M_GRID)
    print(f"Total parameter combinations: {total}")

    # Load labels
    labels = {}
    with open(args.labels) as f:
        for row in csv.DictReader(f):
            labels[row['filename'].strip()] = row['level'].strip().lower()

    # Find and group files
    data_dir = Path(args.data_dir)
    set_files = sorted(data_dir.glob('*.set'))
    by_subject = defaultdict(list)
    for f in set_files:
        by_subject[extract_subject(f.name)].append(
            (f, labels.get(f.name, 'unknown')))
    print(f"Loaded {len(set_files)} files across {len(by_subject)} subjects")

    # STEP 1: load+preprocess+phases ONCE per file (expensive), then
    # re-run observables and calibration for each (tau_D, tau_M) combination
    print("\n=== Stage 1: loading and extracting phases (once per file) ===")
    phases_cache = {}  # (subject, filename, level) -> phases, fs
    for subj in sorted(by_subject.keys()):
        print(f"  Subject {subj}...")
        for fpath, level in by_subject[subj]:
            try:
                raw = smart_load_set(str(fpath), preload=True)
                raw_pp = preprocess_raw(raw, l_freq=1.0, h_freq=45.0,
                                         notch_freq=50.0, resample_to=250.0,
                                         rereference=True, verbose=False)
                phases, fs, _ = extract_gamma_phases(raw_pp, flow, fhigh)
                phases_cache[(subj, fpath.name, level)] = (phases, fs)
            except Exception as e:
                print(f"    ERROR {fpath.name}: {e}")

    print(f"\nCached phases for {len(phases_cache)} files")

    # STEP 2: sweep the grid
    print("\n=== Stage 2: sweeping parameter grid ===")
    sweep_results = []
    for idx, (alpha, tau_D, tau_M) in enumerate(
            product(ALPHA_GRID, TAU_D_GRID, TAU_M_GRID)):
        print(f"\n[{idx+1}/{total}] alpha={alpha}, tau_D={tau_D}, tau_M={tau_M}")

        # Compute observables for each subject under this (tau_D, tau_M)
        # then calibrate per subject with this alpha
        base_Pi = {}
        mod_Pi = {}
        for subj in sorted(by_subject.keys()):
            subj_results = {}
            for fpath, level in by_subject[subj]:
                key = (subj, fpath.name, level)
                if key not in phases_cache:
                    continue
                phases, fs = phases_cache[key]
                obs = compute_observables(phases, fs,
                                            tau_D_s=tau_D, tau_M_s=tau_M,
                                            ridge_lambda=1e-3, stride_s=0.05)
                subj_results[level] = obs

            if 'baseline' not in subj_results:
                continue
            bounds = calibrate_from_baseline(subj_results['baseline'],
                                               alpha=alpha)
            for lvl, obs in subj_results.items():
                Pi = access_region_indicator(obs, bounds)['fraction']
                if lvl == 'baseline':
                    base_Pi[subj] = Pi
                elif lvl == 'moderate':
                    mod_Pi[subj] = Pi

        # Wilcoxon on paired subjects
        paired = sorted(set(base_Pi.keys()) & set(mod_Pi.keys()))
        if len(paired) < 5:
            print(f"  Too few paired subjects ({len(paired)})")
            continue
        diffs = np.array([base_Pi[s] - mod_Pi[s] for s in paired])
        try:
            stat, p = wilcoxon(diffs)
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")
            continue
        d_cohen = diffs.mean() / diffs.std() if diffs.std() > 0 else 0
        n_decline = int(np.sum(diffs > 0))
        print(f"  N={len(paired)}, mean_diff={diffs.mean():+.4f}, "
              f"W={stat:.1f}, p={p:.4g}, d={d_cohen:.3f}, "
              f"decliners={n_decline}/{len(paired)}")

        sweep_results.append({
            'alpha': alpha, 'tau_D': tau_D, 'tau_M': tau_M,
            'N': len(paired), 'mean_diff': diffs.mean(),
            'p_value': p, 'cohen_d': d_cohen, 'n_decline': n_decline,
        })

    # Save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f'sweep_results_{args.band}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
        w.writeheader()
        w.writerows(sweep_results)
    print(f"\nSaved {out_csv}")

    # Heatmap: one subplot per alpha, rows=tau_D, cols=tau_M, cell=log10(p)
    fig, axes = plt.subplots(1, len(ALPHA_GRID),
                              figsize=(5*len(ALPHA_GRID), 4), sharey=True)
    if len(ALPHA_GRID) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, ALPHA_GRID):
        data = np.full((len(TAU_D_GRID), len(TAU_M_GRID)), np.nan)
        for r in sweep_results:
            if r['alpha'] != alpha:
                continue
            i = TAU_D_GRID.index(r['tau_D'])
            j = TAU_M_GRID.index(r['tau_M'])
            data[i, j] = -np.log10(r['p_value'])
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(TAU_M_GRID)))
        ax.set_xticklabels([f'{t}' for t in TAU_M_GRID])
        ax.set_yticks(range(len(TAU_D_GRID)))
        ax.set_yticklabels([f'{t}' for t in TAU_D_GRID])
        ax.set_xlabel(r'$\tau_M$ (s)')
        ax.set_ylabel(r'$\tau_D$ (s)')
        ax.set_title(f'alpha = {alpha}')
        for i in range(len(TAU_D_GRID)):
            for j in range(len(TAU_M_GRID)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center',
                             color='white' if data[i,j] > 3 else 'black', fontsize=9)
        plt.colorbar(im, ax=ax, label=r'$-\log_{10}(p)$')
    fig.suptitle(f'Sensitivity sweep, {args.band} band — $-\\log_{{10}}(p)$ '
                  f'of baseline-moderate Wilcoxon test', fontsize=13)
    plt.tight_layout()
    out_png = outdir / f'sweep_heatmap_{args.band}.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")
    print("\n=== Sweep complete ===")


if __name__ == "__main__":
    main()
