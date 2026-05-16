#!/usr/bin/env python
"""CSD/source-proxy phase-only GCC validation.

This is not individual MRI source reconstruction. It is a conservative
sensor-space source-proxy: the EEG is transformed by spherical current source
density (surface Laplacian) before phase-only GCC observables are computed.

Purpose:
- reduce reference and broad volume-conduction effects;
- test whether phase-only GCC survives a source-proxy transform;
- keep the same frozen parameters as Batch 10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT / "gcc_batch10_phase_only_20260514"))

from batch10_phase_only_gcc import (  # noqa: E402
    BANDS,
    SPECTRAL_COLS,
    aggregate_subject_condition,
    analyze_chennu,
    analyze_ds,
    calibrate_bounds,
    find_file,
    load_chennu_recording,
    load_ds_recording,
    load_spectral_features,
    make_delta_features,
    markdown_table,
    phase_only_observable_series,
    score_recording,
)


def valid_position_channels(raw: mne.io.BaseRaw) -> list[str]:
    good = []
    for ch in raw.info["chs"]:
        loc = ch["loc"][:3]
        if np.all(np.isfinite(loc)) and np.linalg.norm(loc) > 1e-6:
            good.append(ch["ch_name"])
    return good


def load_csd_data(loader, path: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    # Recreate a Raw object because Batch 10 loaders return arrays. This keeps
    # the CSD step explicit and auditable.
    if loader == "chennu":
        from batch10_phase_only_gcc import smart_load_set

        raw = smart_load_set(path)
    elif loader == "ds":
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    else:
        raise ValueError(loader)

    if crop_s is not None:
        raw.crop(tmin=0.0, tmax=min(float(crop_s), float(raw.times[-1])))
    exclude_tokens = ("EOG", "EMG", "ECG", "EXG", "M1", "M2", "A1", "A2", "VEOG", "HEOG")
    keep = [ch for ch in raw.ch_names if not any(tok.lower() in ch.lower() for tok in exclude_tokens)]
    raw.pick(keep)
    raw.load_data(verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="ERROR")
    good = valid_position_channels(raw)
    if len(good) < 8:
        raise RuntimeError(f"Too few valid-position channels for CSD in {path.name}: {len(good)}")
    raw.pick(good)
    raw = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    data = raw.get_data().astype(np.float64, copy=False)
    data = data - np.nanmean(data, axis=1, keepdims=True)
    scale = np.nanstd(data, axis=1, keepdims=True)
    data = data / np.where(scale > 1e-12, scale, 1.0)
    return data, float(raw.info["sfreq"]), list(raw.ch_names)


def analyze_chennu_csd(args: argparse.Namespace, band_name: str, band: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(args.chennu_manifest)
    rows = []
    windows = []
    for subject, sub in manifest.groupby("subject", sort=True):
        baseline = sub[sub["level"] == "baseline"]
        if baseline.empty:
            continue
        base_path = find_file(args.chennu_root, str(baseline.iloc[0]["filename"]))
        base_data, sfreq, channels = load_csd_data("chennu", base_path, args.target_sfreq, args.crop_s)
        base_series = phase_only_observable_series(base_data, sfreq, band, args.window_s, args.stride_s)
        bounds = calibrate_bounds(base_series, args.alpha)
        for _, rec in sub.iterrows():
            path = find_file(args.chennu_root, str(rec["filename"]))
            data, sfreq, channels = load_csd_data("chennu", path, args.target_sfreq, args.crop_s)
            series, summary = score_recording(data, sfreq, band, args.window_s, args.stride_s, bounds)
            series["dataset"] = "Chennu"
            series["subject"] = str(subject)
            series["condition"] = str(rec["level"])
            series["filename"] = str(rec["filename"])
            series["band"] = band_name
            windows.append(series)
            rows.append(
                {
                    "dataset": "Chennu",
                    "subject": str(subject),
                    "condition": str(rec["level"]),
                    "filename": str(rec["filename"]),
                    "band": band_name,
                    "n_channels": len(channels),
                    "duration_s": float(data.shape[1] / sfreq),
                    **summary,
                }
            )
        print(f"CSD Chennu {band_name}: subject {subject} ({len(channels)} positioned channels)", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def parse_ds_recordings(ds_root: Path) -> pd.DataFrame:
    from batch10_phase_only_gcc import parse_ds_recording

    return pd.DataFrame([parse_ds_recording(p) for p in sorted(ds_root.rglob("*_eeg.vhdr"))])


def analyze_ds_csd(args: argparse.Namespace, band_name: str, band: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = parse_ds_recordings(args.ds_root)
    rows = []
    windows = []
    for subject, sub in recs.groupby("subject", sort=True):
        awake = sub[(sub["condition"] == "awake") & (sub["acq"] == "EC")]
        if awake.empty:
            continue
        awake_path = Path(awake.iloc[0]["path"])
        awake_data, sfreq, channels = load_csd_data("ds", awake_path, args.target_sfreq, args.crop_s)
        awake_series = phase_only_observable_series(awake_data, sfreq, band, args.window_s, args.stride_s)
        bounds = calibrate_bounds(awake_series, args.alpha)
        for _, rec in sub.iterrows():
            condition = str(rec["condition"])
            if condition not in {"awake", "sed", "sed2"}:
                continue
            path = Path(rec["path"])
            data, sfreq, channels = load_csd_data("ds", path, args.target_sfreq, args.crop_s)
            series, summary = score_recording(data, sfreq, band, args.window_s, args.stride_s, bounds)
            series["dataset"] = "DS005620"
            series["subject"] = str(subject)
            series["condition"] = condition
            series["filename"] = str(rec["filename"])
            series["run"] = rec["run"]
            series["band"] = band_name
            windows.append(series)
            rows.append(
                {
                    "dataset": "DS005620",
                    "subject": str(subject),
                    "condition": condition,
                    "run": rec["run"],
                    "filename": str(rec["filename"]),
                    "band": band_name,
                    "n_channels": len(channels),
                    "duration_s": float(data.shape[1] / sfreq),
                    **summary,
                }
            )
        print(f"CSD DS005620 {band_name}: subject {subject} ({len(channels)} positioned channels)", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def paired_summary(df: pd.DataFrame, dataset: str, band: str, baseline: str, target: str) -> dict[str, float]:
    sub = df[(df["dataset"] == dataset) & (df["band"] == band) & (df["condition"].isin([baseline, target]))]
    wide = sub.pivot(index="subject", columns="condition", values="Pi").dropna()
    if wide.empty:
        return {"dataset": dataset, "band": band, "target": target, "n": 0}
    base = wide[baseline].to_numpy()
    tgt = wide[target].to_numpy()
    diff = base - tgt
    sd = np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    p_t = stats.ttest_rel(base, tgt).pvalue if len(diff) > 1 else np.nan
    try:
        p_w = stats.wilcoxon(diff, alternative="greater").pvalue if len(diff) > 4 else np.nan
    except ValueError:
        p_w = np.nan
    rng = np.random.default_rng(20260514)
    boot = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(2000)] if len(diff) else [np.nan]
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return {
        "dataset": dataset,
        "band": band,
        "target": target,
        "n": int(len(diff)),
        "baseline_mean": float(base.mean()),
        "target_mean": float(tgt.mean()),
        "mean_delta_baseline_minus_target": float(diff.mean()),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "paired_d": float(diff.mean() / sd) if sd and sd > 0 else np.nan,
        "ttest_p": float(p_t) if np.isfinite(p_t) else np.nan,
        "wilcoxon_greater_p": float(p_w) if np.isfinite(p_w) else np.nan,
    }


def write_report(outdir: Path, stats_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Batch 11 CSD Source-Proxy Phase-Only GCC\n")
    lines.append("Date: 2026-05-14\n")
    lines.append("## Definition\n")
    lines.append(
        "This is a source-proxy validation, not individual MRI source reconstruction. "
        "Raw EEG is transformed with spherical current source density (surface Laplacian) "
        "after standard montage assignment and before phase-only GCC observables are computed.\n"
    )
    lines.append("## Paired CSD Phase-Only Pi Effects\n")
    lines.append(markdown_table(stats_df, ["dataset", "band", "target", "n", "baseline_mean", "target_mean", "mean_delta_baseline_minus_target", "delta_ci_low", "delta_ci_high", "paired_d", "wilcoxon_greater_p"]))
    lines.append("\n## Delta Feature Table\n")
    lines.append(f"Rows: {len(delta_df)}; datasets: {sorted(delta_df['dataset'].unique().tolist())}; bands: {sorted(delta_df['band'].unique().tolist())}.\n")
    lines.append("Use this as a source-proxy robustness layer, not as a replacement for future MRI-based source reconstruction.\n")
    (outdir / "BATCH11_CSD_SOURCE_PROXY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chennu-root", type=Path, required=True)
    parser.add_argument("--chennu-manifest", type=Path, required=True)
    parser.add_argument("--chennu-spectral", type=Path, required=True)
    parser.add_argument("--ds-root", type=Path, required=True)
    parser.add_argument("--ds-spectral", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=90.0)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    recordings = []
    windows = []
    for band_name, band in BANDS.items():
        ch_rec, ch_win = analyze_chennu_csd(args, band_name, band)
        ds_rec, ds_win = analyze_ds_csd(args, band_name, band)
        recordings.extend([ch_rec, ds_rec])
        windows.extend([ch_win, ds_win])

    rec_df = pd.concat(recordings, ignore_index=True)
    win_df = pd.concat(windows, ignore_index=True)
    subj_df = aggregate_subject_condition(rec_df)
    spectral = load_spectral_features(args.chennu_spectral, args.ds_spectral)
    delta_df = make_delta_features(subj_df, spectral)

    stats_rows = []
    for band_name in BANDS:
        stats_rows.append(paired_summary(subj_df, "Chennu", band_name, "baseline", "moderate"))
        stats_rows.append(paired_summary(subj_df, "DS005620", band_name, "awake", "sed"))
        stats_rows.append(paired_summary(subj_df, "DS005620", band_name, "awake", "sed2"))
    stats_df = pd.DataFrame(stats_rows)

    rec_df.to_csv(args.outdir / "csd_phase_only_recording_summary.csv", index=False)
    win_df.to_csv(args.outdir / "csd_phase_only_window_features.csv", index=False)
    subj_df.to_csv(args.outdir / "csd_phase_only_subject_condition_means.csv", index=False)
    delta_df.to_csv(args.outdir / "csd_phase_only_delta_features.csv", index=False)
    stats_df.to_csv(args.outdir / "csd_phase_only_paired_stats.csv", index=False)
    meta = {
        "parameters": {
            "target_sfreq": args.target_sfreq,
            "crop_s": args.crop_s,
            "window_s": args.window_s,
            "stride_s": args.stride_s,
            "alpha": args.alpha,
            "transform": "MNE spherical current source density/source-proxy",
        },
        "outputs": [
            "csd_phase_only_recording_summary.csv",
            "csd_phase_only_subject_condition_means.csv",
            "csd_phase_only_delta_features.csv",
            "csd_phase_only_paired_stats.csv",
        ],
    }
    (args.outdir / "csd_phase_only_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.outdir, stats_df, delta_df)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
