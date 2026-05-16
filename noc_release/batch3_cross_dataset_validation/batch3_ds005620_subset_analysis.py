#!/usr/bin/env python
"""DS005620 subset analysis for GCC observables.

The full dataset is large (~77 GB). This script is designed to work on any
selectively downloaded BIDS/BrainVision subset and scales subject-by-subject.
It uses awake eyes-closed recordings for within-subject calibration when
available, then evaluates sedation recordings against those bounds.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats


EXCLUDE_CHANNELS = {"VEOG", "HEOG", "EMG"}


def parse_recording(vhdr: Path) -> dict[str, str | int | None | Path]:
    name = vhdr.name
    subject = re.search(r"sub-(\d+)", name)
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    acq = re.search(r"acq-([A-Za-z0-9]+)", name)
    run = re.search(r"run-(\d+)", name)
    return {
        "subject": subject.group(1) if subject else "unknown",
        "task": task.group(1) if task else "unknown",
        "acq": acq.group(1) if acq else None,
        "run": int(run.group(1)) if run else None,
        "path": vhdr,
        "filename": name,
    }


def list_recordings(root: Path) -> pd.DataFrame:
    rows = [parse_recording(vhdr) for vhdr in sorted(root.rglob("*_eeg.vhdr"))]
    return pd.DataFrame(rows)


def load_preprocessed(vhdr: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    if crop_s is not None:
        raw.crop(tmin=0, tmax=min(crop_s, raw.times[-1]))
    keep = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
    raw.pick(keep)
    raw.load_data(verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    return data, float(raw.info["sfreq"]), raw.ch_names


def bandpass_phase(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    phase = np.angle(signal.hilbert(filtered, axis=-1))
    return filtered, phase


def effective_dimensionality(x: np.ndarray) -> float:
    cov = np.cov(x)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def compute_observable_series(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    window_s: float,
    stride_s: float,
) -> pd.DataFrame:
    filtered, phase = bandpass_phase(data, sfreq, band)
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    rows = []
    for start in range(0, data.shape[1] - win + 1, stride):
        stop = start + win
        ph = phase[:, start:stop]
        f = filtered[:, start:stop]
        order_t = np.abs(np.mean(np.exp(1j * ph), axis=0))
        rows.append(
            {
                "t": float((start + win / 2) / sfreq),
                "R": float(np.mean(order_t)),
                "D_eff": effective_dimensionality(f),
                "M_tau": float(np.var(order_t)),
            }
        )
    return pd.DataFrame(rows)


def calibrate_bounds(df: pd.DataFrame, alpha: float) -> dict[str, float]:
    return {
        "R_min": float(df["R"].quantile(alpha)),
        "D_min": float(df["D_eff"].quantile(alpha)),
        "D_max": float(df["D_eff"].quantile(1.0 - alpha)),
        "M_min": float(df["M_tau"].quantile(alpha)),
        "M_max": float(df["M_tau"].quantile(1.0 - alpha)),
    }


def add_pi(series: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    out = series.copy()
    out["R_ok"] = out["R"] >= bounds["R_min"]
    out["D_ok"] = (out["D_eff"] >= bounds["D_min"]) & (out["D_eff"] <= bounds["D_max"])
    out["M_ok"] = (out["M_tau"] >= bounds["M_min"]) & (out["M_tau"] <= bounds["M_max"])
    out["Pi_window"] = out[["R_ok", "D_ok", "M_ok"]].mean(axis=1)
    out["Access_all"] = out[["R_ok", "D_ok", "M_ok"]].all(axis=1).astype(int)
    return out


def recording_summary(series: pd.DataFrame) -> dict[str, float]:
    return {
        "R_mean": float(series["R"].mean()),
        "D_mean": float(series["D_eff"].mean()),
        "M_mean": float(series["M_tau"].mean()),
        "Pi": float(series["Pi_window"].mean()),
        "Access_all": float(series["Access_all"].mean()),
        "n_windows": int(len(series)),
    }


def analyze_band(
    root: Path,
    outdir: Path,
    band_name: str,
    band: tuple[float, float],
    alpha: float,
    target_sfreq: float,
    crop_s: float | None,
    window_s: float,
    stride_s: float,
) -> dict:
    recs = list_recordings(root)
    if recs.empty:
        raise RuntimeError(f"No BrainVision .vhdr files found under {root}")

    all_series = []
    summaries = []
    bounds_by_subject: dict[str, dict[str, float]] = {}

    for subject, sub_recs in recs.groupby("subject"):
        awake = sub_recs[(sub_recs["task"] == "awake") & (sub_recs["acq"] == "EC")]
        if awake.empty:
            continue
        awake_path = Path(awake.iloc[0]["path"])
        data, sfreq, channels = load_preprocessed(awake_path, target_sfreq=target_sfreq, crop_s=crop_s)
        awake_series = compute_observable_series(data, sfreq, band, window_s, stride_s)
        bounds = calibrate_bounds(awake_series, alpha)
        bounds_by_subject[subject] = bounds

        for _, row in sub_recs.iterrows():
            vhdr = Path(row["path"])
            data, sfreq, channels = load_preprocessed(vhdr, target_sfreq=target_sfreq, crop_s=crop_s)
            series = compute_observable_series(data, sfreq, band, window_s, stride_s)
            series = add_pi(series, bounds)
            for key in ["subject", "task", "acq", "run", "filename"]:
                series[key] = row[key]
            series["band"] = band_name
            all_series.append(series)
            summary = {
                "subject": subject,
                "task": row["task"],
                "acq": row["acq"],
                "run": row["run"],
                "filename": row["filename"],
                "band": band_name,
                "n_channels": len(channels),
                "duration_s": float(data.shape[1] / sfreq),
                **recording_summary(series),
            }
            summaries.append(summary)

    series_df = pd.concat(all_series, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    series_df.to_csv(outdir / f"ds005620_{band_name}_window_features.csv", index=False)
    summary_df.to_csv(outdir / f"ds005620_{band_name}_recording_summary.csv", index=False)

    group_summary = (
        summary_df.groupby(["task", "acq"])[["R_mean", "D_mean", "M_mean", "Pi", "Access_all"]]
        .agg(["mean", "std", "count"])
        .round(6)
    )
    group_summary.columns = [f"{a}_{b}" for a, b in group_summary.columns]

    # With one subject this is descriptive; once more subjects are downloaded,
    # the same script produces group summaries without changing the code.
    group_records = group_summary.reset_index().to_dict(orient="records")
    out = {
        "band": band_name,
        "band_hz": band,
        "n_recordings": int(len(summary_df)),
        "subjects": sorted(summary_df["subject"].unique().tolist()),
        "bounds_by_subject": bounds_by_subject,
        "recordings": summary_df.round(6).to_dict(orient="records"),
        "group_summary": group_records,
    }
    return out


def plot_results(outdir: Path, summaries: dict[str, dict]) -> None:
    rows = []
    for band, summary in summaries.items():
        rows.extend(summary["recordings"])
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["condition"] = df.apply(
        lambda r: f"{r['task']}-{r['acq']}" + (f"-run{int(r['run'])}" if pd.notna(r["run"]) else ""),
        axis=1,
    )
    order = df.sort_values(["band", "task", "run"])["condition"].unique().tolist()
    for band, sub in df.groupby("band"):
        fig, axes = plt.subplots(1, 5, figsize=(17, 4))
        for ax, metric in zip(axes, ["R_mean", "D_mean", "M_mean", "Pi", "Access_all"]):
            vals = [sub.loc[sub["condition"] == cond, metric].to_numpy() for cond in order if cond in sub["condition"].values]
            labels = [cond for cond in order if cond in sub["condition"].values]
            ax.bar(range(len(vals)), [v.mean() for v in vals], color="#315C72")
            ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right")
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"DS005620 subset GCC observables: {band}")
        fig.tight_layout()
        fig.savefig(outdir / f"ds005620_{band}_subset_summary.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--crop-s", type=float, default=None)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    bands = {
        "alpha": (8.0, 15.0),
        "gamma": (30.0, 45.0),
    }
    summaries = {}
    for band_name, band in bands.items():
        summaries[band_name] = analyze_band(
            root=args.root,
            outdir=args.outdir,
            band_name=band_name,
            band=band,
            alpha=args.alpha,
            target_sfreq=args.target_sfreq,
            crop_s=args.crop_s,
            window_s=args.window_s,
            stride_s=args.stride_s,
        )

    with open(args.outdir / "ds005620_subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    plot_results(args.outdir, summaries)
    print(json.dumps(summaries, indent=2)[:8000])


if __name__ == "__main__":
    main()
