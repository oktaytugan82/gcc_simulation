"""
GCC Pilot v2 — Per-subject calibration + configurable frequency band.

Key improvements over run_pilot_simple.py:

  1. Per-subject calibration: For each subject, the bounds R_min, D_min,
     D_max, M_max are calibrated from THAT subject's own baseline file.
     This removes inter-subject baseline variability and isolates the
     within-subject sedation effect.

  2. Configurable frequency band: --band gamma|alpha|custom allows running
     the same pipeline on different bands for direct comparison.

  3. Subject ID extraction from filename (first two digits).

  4. Structured output: pilot_summary_<band>.csv with columns subject,
     level, R_mean, D_mean, M_mean, Pi; one row per file.

Usage:
    python run_pilot_v2.py <data_dir> --labels labels.csv --band gamma
    python run_pilot_v2.py <data_dir> --labels labels.csv --band alpha
    python run_pilot_v2.py <data_dir> --labels labels.csv \\
                          --band custom --flow 4 --fhigh 8   # theta
"""

import argparse
import csv
import pickle
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import preprocess_raw, extract_gamma_phases
from observables import (compute_observables, access_region_indicator,
                          calibrate_from_baseline)
from load_eeglab_hdf5 import smart_load_set


SEDATION_ORDER = ['baseline', 'mild', 'moderate', 'recovery', 'unknown']


def extract_subject(filename):
    """Extract subject ID from filename like '02-2010-anest ...' -> '02'."""
    m = re.match(r'^(\d{2})-', filename)
    return m.group(1) if m else 'XX'


def load_labels(labels_csv):
    mapping = {}
    with open(labels_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['filename'].strip()] = row['level'].strip().lower()
    return mapping


def process_file(set_path, flow, fhigh, verbose=True):
    """Load .set file, preprocess, extract phases in [flow, fhigh], compute observables."""
    if verbose:
        print(f"  Loading {set_path.name} ...")
    raw = smart_load_set(str(set_path), preload=True)
    raw_pp = preprocess_raw(raw, l_freq=1.0, h_freq=45.0, notch_freq=50.0,
                             resample_to=250.0, rereference=True, verbose=False)
    phases, fs, ch_names = extract_gamma_phases(raw_pp,
                                                  gamma_low=flow,
                                                  gamma_high=fhigh)
    if verbose:
        print(f"    Phases: {phases.shape} at {fs} Hz "
              f"({flow}-{fhigh} Hz band)")
    obs = compute_observables(phases, fs, tau_D_s=0.2, tau_M_s=0.5,
                                ridge_lambda=1e-3, stride_s=0.05)
    return obs, len(ch_names), phases.shape[1] / fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--labels', required=True,
                     help='CSV mapping filename to sedation level')
    ap.add_argument('--band', choices=['gamma', 'alpha', 'custom'],
                     default='gamma')
    ap.add_argument('--flow', type=float, default=None,
                     help='Lower freq for custom band')
    ap.add_argument('--fhigh', type=float, default=None,
                     help='Upper freq for custom band')
    ap.add_argument('--outdir', default='./results')
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()

    # Determine band frequencies
    if args.band == 'gamma':
        flow, fhigh = 30.0, 45.0
    elif args.band == 'alpha':
        flow, fhigh = 8.0, 15.0
    else:
        if args.flow is None or args.fhigh is None:
            print("ERROR: --band custom requires --flow and --fhigh")
            return 1
        flow, fhigh = args.flow, args.fhigh

    band_tag = args.band if args.band != 'custom' else f'{flow:g}-{fhigh:g}'
    print(f"\n=== GCC Pilot v2, band = {band_tag} ({flow}-{fhigh} Hz) ===\n")

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels from {args.labels}")

    # Find .set files
    set_files = sorted(data_dir.glob('*.set'))
    if not set_files:
        print(f"ERROR: no .set files in {data_dir}")
        return 1
    print(f"Found {len(set_files)} .set files")
    if args.max_files:
        set_files = set_files[:args.max_files]
        print(f"Processing first {len(set_files)} (--max-files)")

    # Group by subject and attach labels
    from collections import defaultdict
    by_subject = defaultdict(list)
    for f in set_files:
        subj = extract_subject(f.name)
        level = labels.get(f.name, 'unknown')
        by_subject[subj].append((f, level))

    print(f"\nSubjects found: {len(by_subject)}")

    # Process each subject: compute observables, then calibrate from baseline
    all_rows = []
    for subj in sorted(by_subject.keys()):
        print(f"\n=== Subject {subj} ({len(by_subject[subj])} files) ===")
        subj_results = []
        for fpath, level in by_subject[subj]:
            try:
                obs, n_ch, duration = process_file(fpath, flow, fhigh, verbose=True)
                subj_results.append({
                    'filename': fpath.name, 'level': level,
                    'subject': subj, 'obs': obs,
                    'n_channels': n_ch, 'duration': duration
                })
                print(f"    [{level:>9}] R={obs['R'].mean():.3f}  "
                      f"D={obs['D'].mean():.2f}  "
                      f"M={obs['M'].mean():.2e}")
            except Exception as e:
                print(f"  ERROR: {fpath.name}: {e}")

        if not subj_results:
            print(f"  No results for subject {subj}")
            continue

        # Find baseline file for this subject
        baselines = [r for r in subj_results if r['level'] == 'baseline']
        if not baselines:
            print(f"  WARNING: no baseline for {subj} — using first file")
            calib_src = subj_results[0]
        else:
            calib_src = baselines[0]

        bounds = calibrate_from_baseline(calib_src['obs'], alpha=0.1)
        print(f"  Bounds from baseline ({calib_src['filename']}):")
        for k, v in bounds.items():
            print(f"    {k}: {v:.4f}")

        # Compute Pi for each file under this subject's own calibration
        for r in subj_results:
            acc = access_region_indicator(r['obs'], bounds)
            r['access'] = acc
            r['bounds'] = bounds
            all_rows.append({
                'subject': r['subject'],
                'level': r['level'],
                'filename': r['filename'],
                'R_mean': float(r['obs']['R'].mean()),
                'D_mean': float(r['obs']['D'].mean()),
                'M_mean': float(r['obs']['M'].mean()),
                'Pi': float(acc['fraction']),
            })

    # Save CSV
    out_csv = outdir / f'pilot_summary_{band_tag}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'level', 'filename',
                                             'R_mean', 'D_mean',
                                             'M_mean', 'Pi'])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nSaved {out_csv}")

    # Aggregate table per level
    print(f"\n=== Aggregate per sedation level (band={band_tag}) ===")
    print(f"{'Level':<12} {'N':>4} {'<R>':>8} {'<D>':>8} {'<Pi>':>8} "
          f"{'SD(Pi)':>8}")
    print("-" * 52)
    level_stats = {}
    for lvl in SEDATION_ORDER:
        vals_R = [r['R_mean'] for r in all_rows if r['level'] == lvl]
        vals_D = [r['D_mean'] for r in all_rows if r['level'] == lvl]
        vals_Pi = [r['Pi'] for r in all_rows if r['level'] == lvl]
        if not vals_Pi:
            continue
        level_stats[lvl] = {
            'N': len(vals_Pi),
            'R_mean': np.mean(vals_R), 'D_mean': np.mean(vals_D),
            'Pi_mean': np.mean(vals_Pi), 'Pi_std': np.std(vals_Pi),
        }
        print(f"{lvl:<12} {len(vals_Pi):>4} {np.mean(vals_R):>8.3f} "
              f"{np.mean(vals_D):>8.2f} {np.mean(vals_Pi):>8.3f} "
              f"{np.std(vals_Pi):>8.3f}")

    # Paired test: baseline vs moderate (if both available)
    base_dict = {r['subject']: r['Pi'] for r in all_rows if r['level'] == 'baseline'}
    mod_dict = {r['subject']: r['Pi'] for r in all_rows if r['level'] == 'moderate'}
    paired_subjects = sorted(set(base_dict.keys()) & set(mod_dict.keys()))
    if len(paired_subjects) >= 5:
        from scipy.stats import wilcoxon
        diffs = [base_dict[s] - mod_dict[s] for s in paired_subjects]
        try:
            stat, p = wilcoxon(diffs)
            print(f"\nWilcoxon signed-rank test (baseline - moderate Π), "
                  f"N={len(paired_subjects)}:")
            print(f"  Mean difference = {np.mean(diffs):+.4f}")
            print(f"  Median difference = {np.median(diffs):+.4f}")
            print(f"  W = {stat:.2f}, p = {p:.4g}")
        except Exception as e:
            print(f"Wilcoxon failed: {e}")

    # Plot: boxplots per level for R, D, Pi
    levels_present = [l for l in SEDATION_ORDER if l in level_stats]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics = [('Pi', r'$\Pi$ (access fraction)'),
                ('R_mean', r'$\langle R \rangle$'),
                ('D_mean', r'$\langle D_{eff} \rangle$')]
    for ax, (key, ylabel) in zip(axes, metrics):
        data = [[r[key] for r in all_rows if r['level'] == lvl]
                 for lvl in levels_present]
        ax.boxplot(data, tick_labels=levels_present)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(f'{key} by level')
    fig.suptitle(f'GCC Pilot v2 — {band_tag} band, per-subject calibration',
                  fontsize=13)
    plt.tight_layout()
    out_png = outdir / f'pilot_boxplot_{band_tag}.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_png}")

    # Individual trajectory plot: Pi per subject across levels
    fig, ax = plt.subplots(figsize=(10, 6))
    for subj in sorted(by_subject.keys()):
        subj_rows = sorted([r for r in all_rows if r['subject'] == subj],
                            key=lambda r: SEDATION_ORDER.index(r['level'])
                            if r['level'] in SEDATION_ORDER else 99)
        if len(subj_rows) < 2:
            continue
        xs = [SEDATION_ORDER.index(r['level']) for r in subj_rows]
        ys = [r['Pi'] for r in subj_rows]
        ax.plot(xs, ys, 'o-', alpha=0.5, label=f'Subj {subj}')
    ax.set_xticks(range(len(levels_present)))
    ax.set_xticklabels(levels_present)
    ax.set_ylabel(r'$\Pi$ (access fraction)')
    ax.set_title(f'Per-subject Π trajectories across sedation levels '
                  f'({band_tag} band)')
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.legend(ncol=4, fontsize=8, loc='lower left')
    plt.tight_layout()
    out_traj = outdir / f'pilot_trajectories_{band_tag}.png'
    plt.savefig(out_traj, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_traj}")

    # Save full results as pickle
    out_pkl = outdir / f'pilot_results_{band_tag}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'rows': all_rows, 'level_stats': level_stats,
                      'band': band_tag, 'flow': flow, 'fhigh': fhigh}, f)
    print(f"Saved {out_pkl}")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
