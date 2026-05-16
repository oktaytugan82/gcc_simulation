from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gcc_batch10_phase_only_20260514"))
sys.path.insert(0, str(ROOT / "gcc_batch13_ketamine_20260514"))

from batch10_phase_only_gcc import find_file, load_chennu_recording, load_ds_recording, parse_ds_recording  # noqa: E402
from batch13_ketamine_phase_only import load_recording as load_ketamine_recording, parse_spontaneous_file  # noqa: E402


BANDS = {
    "alpha": (8.0, 13.0),
    "gamma": (35.0, 45.0),
}
SLEEP_BANDS = {
    "sigma": (12.0, 16.0),
}
SPECTRAL_COLS = ["theta_power", "alpha_power", "beta_power", "gamma_power", "alpha_gamma_ratio", "spectral_entropy"]
STATE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage R": "REM",
    "Sleep stage 1": "NREM",
    "Sleep stage 2": "NREM",
    "Sleep stage 3": "NREM",
    "Sleep stage 4": "NREM",
}


def bandpass_phase(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return np.angle(signal.hilbert(filtered, axis=-1))


def sampled_pairs(n_channels: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(n_channels, k=1)
    if max_pairs <= 0 or max_pairs >= len(ii):
        return ii, jj
    rng = np.random.default_rng(seed + n_channels)
    idx = rng.choice(len(ii), size=max_pairs, replace=False)
    return ii[idx], jj[idx]


def graph_participation_from_weights(n_channels: int, pair_i: np.ndarray, pair_j: np.ndarray, weights: np.ndarray) -> float:
    adj = np.zeros((n_channels, n_channels), dtype=np.float64)
    valid = np.isfinite(weights)
    adj[pair_i[valid], pair_j[valid]] = weights[valid]
    adj[pair_j[valid], pair_i[valid]] = weights[valid]
    deg = adj.sum(axis=1)
    keep = deg > 1e-10
    if int(np.sum(keep)) < 2:
        return 1.0
    a = adj[np.ix_(keep, keep)]
    d = deg[keep]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    norm_adj = a * inv_sqrt[:, None] * inv_sqrt[None, :]
    lap = np.eye(norm_adj.shape[0]) - norm_adj
    eig = np.linalg.eigvalsh(lap)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def gcco_observable_series(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    window_s: float,
    stride_s: float,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    phase = bandpass_phase(data, sfreq, band)
    pair_i, pair_j = sampled_pairs(phase.shape[0], max_pairs=max_pairs, seed=seed)
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    rows = []
    for start in range(0, phase.shape[1] - win + 1, stride):
        stop = start + win
        ph = phase[:, start:stop]
        imag = np.sin(ph[pair_i] - ph[pair_j])
        denom = np.mean(np.abs(imag), axis=1)
        w = np.abs(np.mean(imag, axis=1)) / np.where(denom > 1e-12, denom, np.nan)
        lagged_t = np.nanmean(np.abs(imag), axis=0)
        rows.append(
            {
                "t": float((start + stop) / (2.0 * sfreq)),
                "R": float(np.nanmean(w)),
                "D_eff": graph_participation_from_weights(phase.shape[0], pair_i, pair_j, w),
                "M_tau": float(np.nanvar(lagged_t)),
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


def recording_summary(scored: pd.DataFrame) -> dict[str, float]:
    return {
        "R_mean": float(scored["R"].mean()),
        "D_mean": float(scored["D_eff"].mean()),
        "M_mean": float(scored["M_tau"].mean()),
        "Pi": float(scored["Pi_window"].mean()),
        "Access_all": float(scored["Access_all"].mean()),
        "n_windows": int(len(scored)),
    }


def spectral_features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    nperseg = min(data.shape[1], int(round(4.0 * sfreq)))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_psd = np.nanmean(psd, axis=0)
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total = float(np.trapezoid(mean_psd[total_mask], freqs[total_mask])) if np.any(total_mask) else np.nan

    def rel_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask) or not np.isfinite(total) or total <= 0:
            return np.nan
        return float(np.trapezoid(mean_psd[mask], freqs[mask]) / total)

    theta = rel_power(4.0, 8.0)
    alpha = rel_power(8.0, 13.0)
    beta = rel_power(13.0, 30.0)
    gamma = rel_power(30.0, 45.0)
    p = mean_psd[total_mask].astype(float)
    p = p / np.nansum(p) if np.nansum(p) > 0 else np.full_like(p, np.nan)
    entropy = float(-np.nansum(p * np.log(p + 1e-12)) / np.log(len(p))) if len(p) > 1 else np.nan
    slope_mask = (freqs >= 2.0) & (freqs <= 45.0) & (mean_psd > 0)
    if np.sum(slope_mask) >= 5:
        slope = float(np.polyfit(np.log10(freqs[slope_mask]), np.log10(mean_psd[slope_mask]), 1)[0])
    else:
        slope = np.nan
    return {
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "gamma_power": gamma,
        "alpha_gamma_ratio": float(alpha / (gamma + 1e-12)) if np.isfinite(alpha) and np.isfinite(gamma) else np.nan,
        "spectral_entropy": entropy,
        "aperiodic_slope": slope,
    }


def score_recording(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    bounds: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    series = gcco_observable_series(data, sfreq, band, args.window_s, args.stride_s, args.max_pairs, args.seed)
    scored = add_pi(series, bounds)
    return scored, {**recording_summary(scored), **spectral_features(data, sfreq)}


def aggregate_subject_condition(rec_df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "R_mean",
        "D_mean",
        "M_mean",
        "Pi",
        "Access_all",
        "n_windows",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "alpha_gamma_ratio",
        "spectral_entropy",
        "aperiodic_slope",
    ]
    group_cols = ["dataset", "subject", "condition", "band"]
    return rec_df.groupby(group_cols, as_index=False)[numeric].mean()


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
        out[f"d_{col}"] = float(row[col] - base[col])
    return out


def make_delta_features(subject_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets_by_dataset = {
        "Chennu": ("baseline", ["moderate"]),
        "DS005620": ("awake", ["sed", "sed2"]),
        "FarnesKetamine": ("awake", ["ketamine"]),
    }
    for (dataset, band), sub in subject_condition.groupby(["dataset", "band"], sort=True):
        if dataset not in targets_by_dataset:
            continue
        baseline, targets = targets_by_dataset[dataset]
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


def analyze_chennu(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(args.chennu_manifest)
    rows = []
    windows = []
    for band_name, band in BANDS.items():
        for subject, sub in manifest.groupby("subject", sort=True):
            sub = sub[sub["level"].isin(["baseline", "moderate"])].copy()
            baseline = sub[sub["level"] == "baseline"]
            if baseline.empty:
                continue
            base_path = find_file(args.chennu_root, str(baseline.iloc[0]["filename"]))
            base_data, sfreq, channels = load_chennu_recording(base_path, args.target_sfreq, args.crop_s)
            base_series = gcco_observable_series(base_data, sfreq, band, args.window_s, args.stride_s, args.max_pairs, args.seed)
            bounds = calibrate_bounds(base_series, args.alpha)
            for _, rec in sub.iterrows():
                filename = str(rec["filename"])
                path = find_file(args.chennu_root, filename)
                data, sfreq, channels = load_chennu_recording(path, args.target_sfreq, args.crop_s)
                series, summary = score_recording(data, sfreq, band, bounds, args)
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
                        "run": np.nan,
                        "n_channels": len(channels),
                        "duration_s": float(data.shape[1] / sfreq),
                        **summary,
                    }
                )
            print(f"GCC-O Chennu {band_name}: subject {subject}", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def analyze_ds(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [parse_ds_recording(p) for p in sorted(args.ds_root.rglob("*.vhdr"))]
    records = [
        r
        for r in records
        if r["condition"] in {"awake", "sed", "sed2"}
        and ((r["condition"] == "awake" and r["acq"] in {None, "EC"}) or (r["condition"] != "awake" and r["acq"] in {None, "rest"}))
    ]
    rows = []
    windows = []
    for band_name, band in BANDS.items():
        for subject in sorted({r["subject"] for r in records}):
            subj_records = [r for r in records if r["subject"] == subject]
            awake = [r for r in subj_records if r["condition"] == "awake"]
            if not awake:
                continue
            base_data, sfreq, channels = load_ds_recording(awake[0]["path"], args.target_sfreq, args.crop_s)
            base_series = gcco_observable_series(base_data, sfreq, band, args.window_s, args.stride_s, args.max_pairs, args.seed)
            bounds = calibrate_bounds(base_series, args.alpha)
            for rec in subj_records:
                data, sfreq, channels = load_ds_recording(rec["path"], args.target_sfreq, args.crop_s)
                series, summary = score_recording(data, sfreq, band, bounds, args)
                series["dataset"] = "DS005620"
                series["subject"] = subject
                series["condition"] = rec["condition"]
                series["filename"] = rec["filename"]
                series["band"] = band_name
                series["run"] = rec["run"]
                windows.append(series)
                rows.append(
                    {
                        "dataset": "DS005620",
                        "subject": subject,
                        "condition": rec["condition"],
                        "filename": rec["filename"],
                        "band": band_name,
                        "run": rec["run"],
                        "n_channels": len(channels),
                        "duration_s": float(data.shape[1] / sfreq),
                        **summary,
                    }
                )
            print(f"GCC-O DS005620 {band_name}: subject {subject}", flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def analyze_ketamine(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [parse_spontaneous_file(p) for p in sorted(args.ketamine_spontaneous_dir.glob("*.set"))]
    records = [r for r in records if r["eyes"] in {"open", "closed"} and r["condition"] in {"awake", "ketamine"}]
    rows = []
    windows = []
    exclusions = []
    for band_name, band in BANDS.items():
        for subject in sorted({r["subject"] for r in records}):
            subj_records = [r for r in records if r["subject"] == subject]
            loaded = {}
            awake_series = []
            for rec in subj_records:
                try:
                    data, sfreq, channels = load_ketamine_recording(rec["path"], args.target_sfreq, None)
                    series = gcco_observable_series(data, sfreq, band, args.window_s, args.stride_s, args.max_pairs, args.seed)
                except Exception as exc:
                    exclusions.append({"subject": subject, "filename": rec["filename"], "band": band_name, "error": str(exc)[:240]})
                    continue
                loaded[rec["filename"]] = (data, sfreq, channels, series)
                if rec["condition"] == "awake":
                    awake_series.append(series)
            if not awake_series:
                continue
            bounds = calibrate_bounds(pd.concat(awake_series, ignore_index=True), args.alpha)
            for rec in subj_records:
                if rec["filename"] not in loaded:
                    continue
                data, sfreq, channels, _ = loaded[rec["filename"]]
                series, summary = score_recording(data, sfreq, band, bounds, args)
                series["dataset"] = "FarnesKetamine"
                series["subject"] = subject
                series["condition"] = rec["condition"]
                series["eyes"] = rec["eyes"]
                series["filename"] = rec["filename"]
                series["band"] = band_name
                windows.append(series)
                rows.append(
                    {
                        "dataset": "FarnesKetamine",
                        "subject": subject,
                        "condition": rec["condition"],
                        "eyes": rec["eyes"],
                        "filename": rec["filename"],
                        "band": band_name,
                        "run": rec["recording"],
                        "n_channels": len(channels),
                        "duration_s": float(data.shape[1] / sfreq),
                        **summary,
                    }
                )
            print(f"GCC-O Ketamine {band_name}: subject {subject}", flush=True)
    excl = pd.DataFrame(exclusions)
    if not excl.empty:
        excl.to_csv(args.outdir / "gcco_ketamine_loader_exclusions.csv", index=False)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def pair_sleep_files(root: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for psg in sorted(root.glob("*-PSG.edf")):
        subject = psg.name[:6]
        hypos = sorted(root.glob(subject + "*Hypnogram.edf"))
        if hypos:
            pairs.append((psg, hypos[0], subject))
    return pairs


def crop_to_sleep_window(raw: mne.io.BaseRaw, annotations: mne.Annotations, margin_s: float = 1800.0) -> mne.io.BaseRaw:
    raw.set_annotations(annotations)
    sleep_rows = [
        (float(onset), float(onset + duration))
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description)
        if desc in STATE_MAP and STATE_MAP[desc] != "Wake"
    ]
    if not sleep_rows:
        return raw
    return raw.crop(tmin=max(0.0, min(x[0] for x in sleep_rows) - margin_s), tmax=min(raw.times[-1], max(x[1] for x in sleep_rows) + margin_s))


def sleep_epoch_features(epoch: np.ndarray, sfreq: float, band: tuple[float, float], args: argparse.Namespace) -> dict[str, float]:
    series = gcco_observable_series(epoch, sfreq, band, window_s=30.0, stride_s=30.0, max_pairs=args.max_pairs, seed=args.seed)
    out = {
        "R": float(series["R"].iloc[0]),
        "D_eff": float(series["D_eff"].iloc[0]),
        "M_tau": float(series["M_tau"].iloc[0]),
    }
    out.update(spectral_features(epoch, sfreq))
    return out


def analyze_sleep(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    event_id = {k: i + 1 for i, k in enumerate(STATE_MAP)}
    for psg, hypno, subject in pair_sleep_files(args.sleep_root):
        annotations = mne.read_annotations(hypno)
        raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
        raw = crop_to_sleep_window(raw, annotations)
        raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])
        raw.load_data(verbose="ERROR")
        events, id_map = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.0, verbose="ERROR")
        if len(events) == 0:
            continue
        epochs = mne.Epochs(raw, events, event_id=id_map, tmin=0, tmax=30.0 - 1.0 / raw.info["sfreq"], baseline=None, preload=True, verbose="ERROR")
        inv_id = {v: k for k, v in id_map.items()}
        data = epochs.get_data(copy=False)
        for idx, ev in enumerate(events):
            desc = inv_id[int(ev[2])]
            state = STATE_MAP[desc]
            for band_name, band in SLEEP_BANDS.items():
                feat = sleep_epoch_features(data[idx], raw.info["sfreq"], band, args)
                rows.append({"dataset": "SleepEDF", "subject": subject, "state": state, "annotation": desc, "epoch_index": idx, "band": band_name, **feat})
        print(f"GCC-O SleepEDF: subject {subject}", flush=True)
    df = pd.DataFrame(rows)
    scored = []
    for (subject, band), sub in df.groupby(["subject", "band"], sort=True):
        wake = sub[sub["state"] == "Wake"]
        if len(wake) < 10:
            continue
        bounds = calibrate_bounds(wake.rename(columns={"D_eff": "D_eff"}), args.alpha)
        tmp = add_pi(sub.rename(columns={"Pi": "Pi_window"}) if "Pi" in sub else sub.copy(), bounds)
        tmp["Pi"] = tmp["Pi_window"]
        scored.append(tmp.drop(columns=[c for c in ["Pi_window", "R_ok", "D_ok", "M_ok", "Access_all"] if c in tmp.columns]))
    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()


def paired_summary(subject_condition: pd.DataFrame, dataset: str, band: str, baseline: str, target: str) -> dict[str, float]:
    sub = subject_condition[(subject_condition["dataset"] == dataset) & (subject_condition["band"] == band)]
    wide = sub[sub["condition"].isin([baseline, target])].pivot(index="subject", columns="condition", values="Pi").dropna()
    if len(wide) < 4:
        return {"dataset": dataset, "band": band, "target": target, "n": int(len(wide))}
    diff = wide[baseline].to_numpy(float) - wide[target].to_numpy(float)
    return {
        "dataset": dataset,
        "band": band,
        "target": target,
        "n": int(len(diff)),
        "baseline_mean": float(wide[baseline].mean()),
        "target_mean": float(wide[target].mean()),
        "mean_delta_baseline_minus_target": float(np.mean(diff)),
        "paired_d": float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan,
        "wilcoxon_greater_p": float(stats.wilcoxon(diff, alternative="greater").pvalue),
    }


def sleep_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["log_M"] = np.log10(work["M_tau"] + 1e-12)
    feature_sets = {
        "spectral_all": SPECTRAL_COLS,
        "gcco_pi": ["Pi"],
        "gcco_triad_plus_pi": ["R", "D_eff", "log_M", "Pi"],
        "spectral_all_plus_gcco": SPECTRAL_COLS + ["R", "D_eff", "log_M", "Pi"],
    }
    for band, bdf in work.groupby("band"):
        for model, cols in feature_sets.items():
            for positive, negative in [("Wake", "NREM"), ("REM", "NREM")]:
                sub = bdf[bdf["state"].isin([positive, negative])].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
                y = (sub["state"] == positive).astype(int).to_numpy()
                groups = sub["subject"].to_numpy()
                scores = np.full(len(sub), np.nan)
                logo = LeaveOneGroupOut()
                for train, test in logo.split(sub[cols], y, groups):
                    if len(np.unique(y[train])) < 2:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260514))
                    clf.fit(sub.iloc[train][cols], y[train])
                    scores[test] = clf.predict_proba(sub.iloc[test][cols])[:, 1]
                mask = np.isfinite(scores)
                auc = float(roc_auc_score(y[mask], scores[mask])) if np.sum(mask) and len(np.unique(y[mask])) == 2 else np.nan
                pred = (scores[mask] >= 0.5).astype(int)
                rows.append(
                    {
                        "band": band,
                        "contrast": f"{positive}_vs_{negative}",
                        "model": model,
                        "n": int(np.sum(mask)),
                        "auc": auc,
                        "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)) if np.sum(mask) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_report(outdir: Path, paired: pd.DataFrame, sleep_metrics: pd.DataFrame) -> None:
    lines = []
    lines.append("# Batch 15 GCC-O Spectrally Orthogonalized Candidate\n")
    lines.append("Date: 2026-05-14\n")
    lines.append("## Definition\n")
    lines.append(
        "GCC-O is a stricter candidate variant: R is mean wPLI-style lagged phase coupling, D_eff is a participation ratio of the normalized graph-Laplacian spectrum of the lagged-connectivity graph, and M_tau is the within-window variance of mean absolute lagged phase interaction. Bandpass amplitude does not enter the GCC-O observables; conventional spectral features are computed only as external covariates for residualization and baseline comparison.\n"
    )
    lines.append("## Paired Pi Effects\n")
    lines.append(md_table(paired))
    if not sleep_metrics.empty:
        lines.append("\n\n## Sleep-EDF Cross-Validated Metrics\n")
        lines.append(md_table(sleep_metrics))
    lines.append("\n\n## Interpretation Rule\n")
    lines.append(
        "A bandpower-independent biomarker claim is allowed only if residualized GCC-O remains clearly above chance and/or spectral+GCC-O robustly improves over spectral-only baselines with positive bootstrap intervals. Otherwise GCC-O should be reported as phase-lagged and bandpower-controlled, but not generally bandpower-independent.\n"
    )
    (outdir / "BATCH15_GCCO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if not np.isfinite(v) else format(float(v), floatfmt))
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chennu-root", type=Path, required=True)
    parser.add_argument("--chennu-manifest", type=Path, required=True)
    parser.add_argument("--ds-root", type=Path, required=True)
    parser.add_argument("--ketamine-spontaneous-dir", type=Path, required=True)
    parser.add_argument("--sleep-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=90.0)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--skip-sleep", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ch_rec, ch_win = analyze_chennu(args)
    ds_rec, ds_win = analyze_ds(args)
    ke_rec, ke_win = analyze_ketamine(args)
    rec_df = pd.concat([ch_rec, ds_rec, ke_rec], ignore_index=True)
    win_df = pd.concat([ch_win, ds_win, ke_win], ignore_index=True)
    subj = aggregate_subject_condition(rec_df)
    delta = make_delta_features(subj)

    paired_rows = []
    for band in BANDS:
        paired_rows.append(paired_summary(subj, "Chennu", band, "baseline", "moderate"))
        paired_rows.append(paired_summary(subj, "DS005620", band, "awake", "sed"))
        paired_rows.append(paired_summary(subj, "DS005620", band, "awake", "sed2"))
        paired_rows.append(paired_summary(subj, "FarnesKetamine", band, "awake", "ketamine"))
    paired = pd.DataFrame(paired_rows)

    sleep_df = pd.DataFrame()
    sleep_metrics = pd.DataFrame()
    if not args.skip_sleep:
        sleep_df = analyze_sleep(args)
        sleep_metrics = sleep_cv(sleep_df)

    rec_df.to_csv(args.outdir / "gcco_recording_summary.csv", index=False)
    win_df.to_csv(args.outdir / "gcco_window_features.csv", index=False)
    subj.to_csv(args.outdir / "gcco_subject_condition_means.csv", index=False)
    delta.to_csv(args.outdir / "gcco_delta_features.csv", index=False)
    paired.to_csv(args.outdir / "gcco_paired_stats.csv", index=False)
    if not sleep_df.empty:
        sleep_df.to_csv(args.outdir / "gcco_sleep_epoch_features.csv", index=False)
    if not sleep_metrics.empty:
        sleep_metrics.to_csv(args.outdir / "gcco_sleep_cv_metrics.csv", index=False)
    write_report(args.outdir, paired, sleep_metrics)

    print(
        json.dumps(
            {
                "recordings": int(len(rec_df)),
                "delta_rows": int(len(delta)),
                "sleep_epochs": int(len(sleep_df)) if not sleep_df.empty else 0,
                "outputs": [
                    "gcco_recording_summary.csv",
                    "gcco_subject_condition_means.csv",
                    "gcco_delta_features.csv",
                    "gcco_paired_stats.csv",
                    "gcco_sleep_epoch_features.csv",
                    "gcco_sleep_cv_metrics.csv",
                    "BATCH15_GCCO_REPORT.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
