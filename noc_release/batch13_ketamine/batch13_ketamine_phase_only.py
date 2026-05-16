from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gcc_batch10_phase_only_20260514"))

from batch10_phase_only_gcc import (  # noqa: E402
    add_pi,
    calibrate_bounds,
    phase_only_observable_series,
    preprocess_raw,
    smart_load_set,
)


BANDS = {
    "alpha": (8.0, 13.0),
    "gamma": (35.0, 45.0),
}


def parse_spontaneous_file(path: Path) -> dict:
    name = path.name
    subject = name.split("_", 1)[0]
    rec_matches = re.findall(r"_(\d{4})(?=[A-Za-z_])", name)
    recording = int(rec_matches[-1]) if rec_matches else -1
    condition = "ketamine" if recording >= 5 else "awake"
    low = name.lower()
    if "open" in low or "eyespen" in low or "eyesopen" in low:
        eyes = "open"
    elif "closed" in low or "cloed" in low or "clo" in low:
        eyes = "closed"
    else:
        eyes = "unknown"
    return {
        "subject": subject,
        "recording": recording,
        "condition": condition,
        "eyes": eyes,
        "filename": name,
        "path": path,
    }


def load_recording(path: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    try:
        epochs = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    except Exception:
        raw = smart_load_set(path)
        return preprocess_raw(raw, target_sfreq, crop_s)
    keep = [ch for ch in epochs.ch_names if not any(tok.lower() in ch.lower() for tok in ["eog", "emg", "ecg", "trigger", "stim"])]
    epochs.pick(keep)
    epochs.load_data()
    epochs.resample(target_sfreq, npad="auto", verbose="ERROR")
    epochs.set_eeg_reference("average", projection=False, verbose="ERROR")
    data3 = epochs.get_data().astype(np.float64, copy=False)
    if crop_s is not None:
        keep_t = int(round(float(crop_s) * float(epochs.info["sfreq"])))
        flat = np.transpose(data3, (1, 0, 2)).reshape(data3.shape[1], -1)[:, :keep_t]
    else:
        flat = np.transpose(data3, (1, 0, 2)).reshape(data3.shape[1], -1)
    flat = flat - np.nanmean(flat, axis=1, keepdims=True)
    scale = np.nanstd(flat, axis=1, keepdims=True)
    flat = flat / np.where(scale > 1e-12, scale, 1.0)
    return flat, float(epochs.info["sfreq"]), list(epochs.ch_names)


def cohen_d_paired(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2 or np.nanstd(delta, ddof=1) == 0:
        return np.nan
    return float(np.nanmean(delta) / np.nanstd(delta, ddof=1))


def paired_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["Pi", "Access_all", "R_mean", "D_mean", "M_mean"]
    for band, bdf in summary.groupby("band"):
        for eyes_label, edf in list(bdf.groupby("eyes")) + [("pooled_eyes", bdf)]:
            pairs = []
            if eyes_label == "pooled_eyes":
                awake = edf[edf["condition"] == "awake"].groupby("subject")[metrics].mean()
                ket = edf[edf["condition"] == "ketamine"].groupby("subject")[metrics].mean()
            else:
                awake = edf[edf["condition"] == "awake"].set_index("subject")[metrics]
                ket = edf[edf["condition"] == "ketamine"].set_index("subject")[metrics]
            common = sorted(set(awake.index).intersection(set(ket.index)))
            if not common:
                continue
            for metric in metrics:
                a = awake.loc[common, metric].to_numpy(dtype=float)
                k = ket.loc[common, metric].to_numpy(dtype=float)
                delta = k - a
                try:
                    p_two = stats.wilcoxon(delta).pvalue if len(delta) >= 4 else np.nan
                except ValueError:
                    p_two = np.nan
                try:
                    p_greater = stats.wilcoxon(delta, alternative="greater").pvalue if len(delta) >= 4 else np.nan
                except ValueError:
                    p_greater = np.nan
                try:
                    p_less = stats.wilcoxon(delta, alternative="less").pvalue if len(delta) >= 4 else np.nan
                except ValueError:
                    p_less = np.nan
                rows.append(
                    {
                        "band": band,
                        "eyes": eyes_label,
                        "metric": metric,
                        "n": int(len(common)),
                        "awake_mean": float(np.nanmean(a)),
                        "ketamine_mean": float(np.nanmean(k)),
                        "mean_delta_ketamine_minus_awake": float(np.nanmean(delta)),
                        "paired_d_delta": cohen_d_paired(delta),
                        "wilcoxon_two_sided_p": float(p_two) if np.isfinite(p_two) else np.nan,
                        "wilcoxon_delta_greater_p": float(p_greater) if np.isfinite(p_greater) else np.nan,
                        "wilcoxon_delta_less_p": float(p_less) if np.isfinite(p_less) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


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


def write_report(outdir: Path, summary: pd.DataFrame, paired: pd.DataFrame, source_url: str, exclusions: pd.DataFrame) -> None:
    key = paired[
        (paired["eyes"] == "pooled_eyes")
        & (paired["metric"].isin(["Pi", "R_mean", "D_mean", "M_mean"]))
    ].copy()
    key = key.sort_values(["band", "metric"])
    record_counts = (
        summary.groupby(["band", "condition", "eyes"])
        .agg(n_recordings=("filename", "count"), n_subjects=("subject", "nunique"), Pi_mean=("Pi", "mean"), D_mean=("D_mean", "mean"))
        .reset_index()
    )
    report = []
    report.append("# Batch 13 Ketamine Phase-Only GCC Extension\n")
    report.append("Date: 2026-05-14\n")
    report.append("## Scope\n")
    report.append(
        "This is an other-anesthetics/altered-state extension using the Farnes et al. public ketamine EEG dataset. "
        "The dataset contains normal wakefulness and sub-anaesthetic ketamine, not deep loss of consciousness. "
        "The defensible GCC question is therefore whether access-compatible regime structure is preserved while observables, especially effective dimensionality, shift.\n"
    )
    report.append(f"Source: {source_url}\n")
    report.append("## Recording Counts\n")
    report.append(md_table(record_counts))
    if not exclusions.empty:
        report.append("\n\n## Loader Exclusions\n")
        report.append(md_table(exclusions[["subject", "condition", "eyes", "filename", "error"]].head(20)))
        report.append(f"\n\nExcluded recordings: {len(exclusions)}. These files could not be decoded by MNE/Scipy and were not used.\n")
    report.append("\n\n## Pooled Awake-vs-Ketamine Paired Effects\n")
    report.append(md_table(key))
    report.append("\n\n## Interpretation\n")
    report.append(
        "This extension should not be framed as another propofol-style loss-of-consciousness replication. "
        "It is more valuable as a pharmacological boundary case: if Pi remains access-compatible while D_eff or coherence shifts, GCC gains specificity by distinguishing sedative loss from altered conscious content.\n"
    )
    (outdir / "BATCH13_KETAMINE_PHASE_ONLY_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spontaneous-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=None)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--source-url", default="https://zenodo.org/records/4245091")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    records = [parse_spontaneous_file(p) for p in sorted(args.spontaneous_dir.glob("*.set"))]
    records = [r for r in records if r["eyes"] in {"open", "closed"} and r["condition"] in {"awake", "ketamine"}]
    if not records:
        raise RuntimeError(f"No spontaneous EEGLAB .set files found in {args.spontaneous_dir}")
    manifest = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in records])
    manifest.to_csv(args.outdir / "ketamine_manifest.csv", index=False)

    all_windows = []
    all_rows = []
    exclusions = []
    for band_name, band in BANDS.items():
        for subject, sub_records in manifest.groupby("subject", sort=True):
            subject_records = [r for r in records if r["subject"] == str(subject)]
            awake_windows_for_bounds = []
            loaded = {}
            for rec in subject_records:
                print(f"Ketamine {band_name}: subject {subject} {rec['condition']} {rec['eyes']}", flush=True)
                try:
                    data, sfreq, channels = load_recording(rec["path"], args.target_sfreq, args.crop_s)
                except Exception as exc:
                    exclusions.append(
                        {
                            "subject": rec["subject"],
                            "condition": rec["condition"],
                            "eyes": rec["eyes"],
                            "recording": rec["recording"],
                            "filename": rec["filename"],
                            "error": str(exc).splitlines()[0][:240],
                        }
                    )
                    print(f"Skipping {rec['filename']}: {exc}", flush=True)
                    continue
                series = phase_only_observable_series(data, sfreq, band, args.window_s, args.stride_s)
                loaded[rec["filename"]] = (series, data.shape[1] / sfreq, len(channels))
                if rec["condition"] == "awake":
                    awake_windows_for_bounds.append(series)
            if not awake_windows_for_bounds:
                continue
            bounds = calibrate_bounds(pd.concat(awake_windows_for_bounds, ignore_index=True), args.alpha)
            for rec in subject_records:
                if rec["filename"] not in loaded:
                    continue
                series, duration_s, n_channels = loaded[rec["filename"]]
                scored = add_pi(series, bounds)
                scored["dataset"] = "FarnesKetamine"
                scored["subject"] = rec["subject"]
                scored["condition"] = rec["condition"]
                scored["eyes"] = rec["eyes"]
                scored["recording"] = rec["recording"]
                scored["filename"] = rec["filename"]
                scored["band"] = band_name
                all_windows.append(scored)
                all_rows.append(
                    {
                        "dataset": "FarnesKetamine",
                        "subject": rec["subject"],
                        "condition": rec["condition"],
                        "eyes": rec["eyes"],
                        "recording": rec["recording"],
                        "filename": rec["filename"],
                        "band": band_name,
                        "n_channels": n_channels,
                        "duration_s": float(duration_s),
                        "R_mean": float(scored["R"].mean()),
                        "D_mean": float(scored["D_eff"].mean()),
                        "M_mean": float(scored["M_tau"].mean()),
                        "Pi": float(scored["Pi_window"].mean()),
                        "Access_all": float(scored["Access_all"].mean()),
                        "n_windows": int(len(scored)),
                    }
                )

    windows = pd.concat(all_windows, ignore_index=True)
    summary = pd.DataFrame(all_rows)
    paired = paired_summary(summary)
    exclusions_df = pd.DataFrame(exclusions)
    windows.to_csv(args.outdir / "ketamine_phase_only_window_features.csv", index=False)
    summary.to_csv(args.outdir / "ketamine_phase_only_recording_summary.csv", index=False)
    paired.to_csv(args.outdir / "ketamine_phase_only_paired_stats.csv", index=False)
    exclusions_df.to_csv(args.outdir / "ketamine_loader_exclusions.csv", index=False)
    write_report(args.outdir, summary, paired, args.source_url, exclusions_df)

    print(
        json.dumps(
            {
                "records": int(len(summary)),
                "subjects": int(summary["subject"].nunique()),
                "outputs": [
                    "ketamine_manifest.csv",
                    "ketamine_phase_only_window_features.csv",
                    "ketamine_phase_only_recording_summary.csv",
                    "ketamine_phase_only_paired_stats.csv",
                    "BATCH13_KETAMINE_PHASE_ONLY_REPORT.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
