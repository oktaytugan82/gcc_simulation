#!/usr/bin/env python
"""Amplitude-normalized / phase-only GCC pipeline on raw EEG recordings.

This batch removes amplitude information before GCC observables are computed.
For each bandpassed recording we extract Hilbert phase and replace the signal
by unit phasors z(t)=exp(i phi(t)). The observables are then:

R      : Kuramoto order parameter of unit phasors.
D_eff  : participation ratio of the covariance of [cos(phi); sin(phi)].
M_tau  : temporal variance of R(t) within the window.

Thus no bandpass amplitude enters the GCC observables. Spectral features are
merged only later as external baselines/controls.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats


BANDS = {
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 45.0),
}
EXCLUDE_TOKENS = ("EOG", "EMG", "ECG", "EXG", "M1", "M2", "A1", "A2", "VEOG", "HEOG")
SPECTRAL_COLS = ["theta_power", "alpha_power", "beta_power", "gamma_power", "alpha_gamma_ratio", "spectral_entropy"]


def _h5_read_string(f: h5py.File, dataset) -> str:
    arr = np.array(dataset).flatten()
    if arr.dtype.kind in ("i", "u"):
        try:
            return "".join(chr(int(c)) for c in arr if c > 0)
        except (ValueError, OverflowError):
            return ""
    if arr.dtype.kind in ("S", "O"):
        try:
            return b"".join(arr.tolist()).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def load_eeglab_v73(set_path: Path):
    with h5py.File(str(set_path), "r") as f:
        eeg = f["EEG"]
        srate = float(np.array(eeg["srate"]).flatten()[0])
        nbchan = int(np.array(eeg["nbchan"]).flatten()[0])
        pnts = int(np.array(eeg["pnts"]).flatten()[0])
        trials = int(np.array(eeg["trials"]).flatten()[0])
        data = None
        data_field = eeg["data"]
        if data_field.dtype.kind == "f":
            data = np.array(data_field)
        else:
            data_fname = _h5_read_string(f, data_field)
            if data_fname and not data_fname.endswith(".set"):
                fdt_path = set_path.parent / data_fname
                if not fdt_path.exists():
                    fdt_path = set_path.with_suffix(".fdt")
                if fdt_path.exists():
                    raw_bytes = np.fromfile(str(fdt_path), dtype=np.float32)
                    data = raw_bytes.reshape(pnts * trials, nbchan).T
        if data is None:
            raise ValueError(f"Could not extract data from {set_path.name}")
        if data.shape[0] != nbchan:
            if data.shape[1] == nbchan:
                data = data.T
            else:
                raise ValueError(f"Data shape {data.shape} does not match nbchan={nbchan}")
        ch_names = []
        if "chanlocs" in eeg and "labels" in eeg["chanlocs"]:
            labels_refs = np.array(eeg["chanlocs"]["labels"]).flatten()
            for ref in labels_refs:
                try:
                    ch_names.append(_h5_read_string(f, f[ref]))
                except Exception:
                    ch_names.append("")
        if not ch_names or len(ch_names) != nbchan or all(name == "" for name in ch_names):
            ch_names = [f"EEG{i:03d}" for i in range(nbchan)]
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose="ERROR")


def smart_load_set(set_path: Path):
    try:
        return mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    except (TypeError, NotImplementedError, ValueError) as exc:
        msg = str(exc)
        if "v7.3" in msg or "HDF" in msg or "h5py" in msg:
            return load_eeglab_v73(set_path)
        raise


def load_chennu_recording(path: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    raw = smart_load_set(path)
    return preprocess_raw(raw, target_sfreq, crop_s)


def load_ds_recording(path: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    return preprocess_raw(raw, target_sfreq, crop_s)


def preprocess_raw(raw: mne.io.BaseRaw, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    if crop_s is not None:
        raw.crop(tmin=0.0, tmax=min(float(crop_s), float(raw.times[-1])))
    keep = [ch for ch in raw.ch_names if not any(tok.lower() in ch.lower() for tok in EXCLUDE_TOKENS)]
    raw.pick(keep)
    raw.load_data(verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    data = raw.get_data().astype(np.float64, copy=False)
    data = data - np.nanmean(data, axis=1, keepdims=True)
    scale = np.nanstd(data, axis=1, keepdims=True)
    data = data / np.where(scale > 1e-12, scale, 1.0)
    return data, float(raw.info["sfreq"]), list(raw.ch_names)


def parse_ds_recording(vhdr: Path) -> dict:
    name = vhdr.name
    subject = re.search(r"sub-(\d+)", name)
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    acq = re.search(r"acq-([A-Za-z0-9]+)", name)
    run = re.search(r"run-(\d+)", name)
    return {
        "subject": subject.group(1) if subject else "unknown",
        "condition": task.group(1) if task else "unknown",
        "acq": acq.group(1) if acq else None,
        "run": int(run.group(1)) if run else None,
        "filename": name,
        "path": vhdr,
    }


def phase_from_band(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return np.angle(signal.hilbert(filtered, axis=-1))


def effective_dimensionality_phase(phase_window: np.ndarray) -> float:
    x = np.vstack([np.cos(phase_window), np.sin(phase_window)])
    x = x - x.mean(axis=1, keepdims=True)
    cov = np.cov(x)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def phase_only_observable_series(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    window_s: float,
    stride_s: float,
) -> pd.DataFrame:
    phase = phase_from_band(data, sfreq, band)
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    rows = []
    for start in range(0, phase.shape[1] - win + 1, stride):
        stop = start + win
        ph = phase[:, start:stop]
        order_t = np.abs(np.exp(1j * ph).mean(axis=0))
        rows.append(
            {
                "t": float((start + win / 2) / sfreq),
                "R": float(order_t.mean()),
                "D_eff": effective_dimensionality_phase(ph),
                "M_tau": float(order_t.var()),
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


def score_recording(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    window_s: float,
    stride_s: float,
    bounds: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    series = phase_only_observable_series(data, sfreq, band, window_s, stride_s)
    scored = add_pi(series, bounds)
    return scored, recording_summary(scored)


def analyze_chennu(args: argparse.Namespace, band_name: str, band: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(args.chennu_manifest)
    rows = []
    windows = []
    for subject, sub in manifest.groupby("subject", sort=True):
        baseline = sub[sub["level"] == "baseline"]
        if baseline.empty:
            continue
        base_filename = str(baseline.iloc[0]["filename"])
        base_path = find_file(args.chennu_root, base_filename)
        base_data, sfreq, channels = load_chennu_recording(base_path, args.target_sfreq, args.crop_s)
        base_series = phase_only_observable_series(base_data, sfreq, band, args.window_s, args.stride_s)
        bounds = calibrate_bounds(base_series, args.alpha)
        for _, rec in sub.iterrows():
            filename = str(rec["filename"])
            path = find_file(args.chennu_root, filename)
            data, sfreq, channels = load_chennu_recording(path, args.target_sfreq, args.crop_s)
            series, summary = score_recording(data, sfreq, band, args.window_s, args.stride_s, bounds)
            series["dataset"] = "Chennu"
            series["subject"] = str(subject)
            series["condition"] = str(rec["level"])
            series["filename"] = filename
            series["band"] = band_name
            windows.append(series)
            rows.append(
                {
                    "dataset": "Chennu",
                    "subject": str(subject),
                    "condition": str(rec["level"]),
                    "filename": filename,
                    "band": band_name,
                    "n_channels": len(channels),
                    "duration_s": float(data.shape[1] / sfreq),
                    **summary,
                }
            )
        print(f"Chennu {band_name}: subject {subject}", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def analyze_ds(args: argparse.Namespace, band_name: str, band: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = pd.DataFrame([parse_ds_recording(p) for p in sorted(args.ds_root.rglob("*_eeg.vhdr"))])
    rows = []
    windows = []
    for subject, sub in recs.groupby("subject", sort=True):
        awake = sub[(sub["condition"] == "awake") & (sub["acq"] == "EC")]
        if awake.empty:
            continue
        awake_path = Path(awake.iloc[0]["path"])
        awake_data, sfreq, channels = load_ds_recording(awake_path, args.target_sfreq, args.crop_s)
        awake_series = phase_only_observable_series(awake_data, sfreq, band, args.window_s, args.stride_s)
        bounds = calibrate_bounds(awake_series, args.alpha)
        for _, rec in sub.iterrows():
            condition = str(rec["condition"])
            if condition not in {"awake", "sed", "sed2"}:
                continue
            path = Path(rec["path"])
            data, sfreq, channels = load_ds_recording(path, args.target_sfreq, args.crop_s)
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
        print(f"DS005620 {band_name}: subject {subject}", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def aggregate_subject_condition(recordings: pd.DataFrame) -> pd.DataFrame:
    numeric = ["R_mean", "D_mean", "M_mean", "Pi", "Access_all", "n_channels", "duration_s", "n_windows"]
    out = (
        recordings.groupby(["dataset", "band", "subject", "condition"], as_index=False)[numeric]
        .mean(numeric_only=True)
    )
    return out


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


def load_spectral_features(chennu_spectral: Path, ds_spectral: Path) -> pd.DataFrame:
    ch = pd.read_csv(chennu_spectral).copy()
    ch["subject"] = ch["subject"].astype(str)
    ch = ch[["dataset", "subject", "condition"] + SPECTRAL_COLS]
    ds = pd.read_csv(ds_spectral).copy()
    ds["subject"] = ds["subject"].astype(str)
    ds = ds[["dataset", "subject", "condition"] + SPECTRAL_COLS]
    ds = ds.groupby(["dataset", "subject", "condition"], as_index=False)[SPECTRAL_COLS].mean()
    return pd.concat([ch, ds], ignore_index=True)


def make_delta_features(subject_condition: pd.DataFrame, spectral: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = subject_condition.merge(spectral, on=["dataset", "subject", "condition"], how="left")
    for (dataset, band), sub in merged.groupby(["dataset", "band"], sort=True):
        baseline = "baseline" if dataset == "Chennu" else "awake"
        targets = ["moderate"] if dataset == "Chennu" else ["sed", "sed2"]
        for subject, ssub in sub.groupby("subject"):
            base = ssub[ssub["condition"] == baseline]
            if base.empty:
                continue
            base = base.iloc[0]
            rows.append(delta_row(dataset, band, subject, baseline, 0, base, base))
            for target in targets:
                hit = ssub[ssub["condition"] == target]
                if hit.empty:
                    continue
                rows.append(delta_row(dataset, band, subject, target, 1, base, hit.iloc[0]))
    return pd.DataFrame(rows)


def delta_row(dataset: str, band: str, subject: str, condition: str, y: int, base: pd.Series, row: pd.Series) -> dict:
    out = {
        "dataset": dataset,
        "band": band,
        "subject": str(subject),
        "condition": condition,
        "y": int(y),
        "dR": float(row["R_mean"] - base["R_mean"]),
        "dD": float(row["D_mean"] - base["D_mean"]),
        "dlogM": float(np.log10(row["M_mean"] + 1e-12) - np.log10(base["M_mean"] + 1e-12)),
        "dPi": float(row["Pi"] - base["Pi"]),
    }
    for col in SPECTRAL_COLS:
        out[f"d_{col}"] = float(row[col] - base[col]) if pd.notna(row.get(col, np.nan)) and pd.notna(base.get(col, np.nan)) else np.nan
    return out


def write_report(outdir: Path, stats_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Batch 10 Phase-Only GCC Raw-Epoch Pipeline\n")
    lines.append("Date: 2026-05-14\n")
    lines.append("## Definition\n")
    lines.append(
        "The EEG signal is bandpassed and transformed to Hilbert phase. GCC observables are computed from "
        "unit phasors only: R from exp(i phi), D_eff from covariance of cos(phi)/sin(phi), and M_tau from "
        "temporal variance of R(t). Bandpass amplitudes do not enter the GCC observables.\n"
    )
    lines.append("## Paired Phase-Only Pi Effects\n")
    lines.append(markdown_table(stats_df, ["dataset", "band", "target", "n", "baseline_mean", "target_mean", "mean_delta_baseline_minus_target", "delta_ci_low", "delta_ci_high", "paired_d", "wilcoxon_greater_p"]))
    lines.append("\n## Delta Feature Table\n")
    lines.append(f"Rows: {len(delta_df)}; datasets: {sorted(delta_df['dataset'].unique().tolist())}; bands: {sorted(delta_df['band'].unique().tolist())}.\n")
    lines.append("The delta table can be passed directly to Batch 9 bandpower-independence stress tests.\n")
    (outdir / "BATCH10_PHASE_ONLY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep] + body)


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
        ch_rec, ch_win = analyze_chennu(args, band_name, band)
        ds_rec, ds_win = analyze_ds(args, band_name, band)
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

    rec_df.to_csv(args.outdir / "phase_only_recording_summary.csv", index=False)
    # Window table is useful but large; keep it compressed enough by not writing duplicate raw arrays.
    win_df.to_csv(args.outdir / "phase_only_window_features.csv", index=False)
    subj_df.to_csv(args.outdir / "phase_only_subject_condition_means.csv", index=False)
    delta_df.to_csv(args.outdir / "phase_only_delta_features.csv", index=False)
    stats_df.to_csv(args.outdir / "phase_only_paired_stats.csv", index=False)

    meta = {
        "parameters": {
            "target_sfreq": args.target_sfreq,
            "crop_s": args.crop_s,
            "window_s": args.window_s,
            "stride_s": args.stride_s,
            "alpha": args.alpha,
        },
        "outputs": [
            "phase_only_recording_summary.csv",
            "phase_only_subject_condition_means.csv",
            "phase_only_delta_features.csv",
            "phase_only_paired_stats.csv",
        ],
    }
    (args.outdir / "phase_only_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.outdir, stats_df, delta_df)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
