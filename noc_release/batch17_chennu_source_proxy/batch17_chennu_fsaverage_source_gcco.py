from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gcc_batch10_phase_only_20260514"))
sys.path.insert(0, str(ROOT / "gcc_batch15_gcco_20260514"))

from batch10_phase_only_gcc import find_file, smart_load_set  # noqa: E402
from batch15_gcco_pipeline import add_pi, calibrate_bounds, gcco_observable_series, recording_summary  # noqa: E402


BANDS = {"alpha": (8.0, 13.0), "gamma": (35.0, 45.0)}
EXCLUDE_TOKENS = ("EOG", "EMG", "ECG", "EXG", "M1", "M2", "A1", "A2", "VEOG", "HEOG")


def subject_from_filename(filename: str) -> str:
    m = re.match(r"(\d+)-2010-anest", filename)
    if not m:
        return "unknown"
    return m.group(1).lstrip("0") or m.group(1)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "subject" not in df.columns:
        df["subject"] = df["filename"].map(subject_from_filename)
    return df[df["level"].isin(["baseline", "moderate"])].copy()


def positioned_eeg_channels(raw: mne.io.BaseRaw) -> list[str]:
    out = []
    for ch in raw.info["chs"]:
        loc = ch["loc"][:3]
        if np.all(np.isfinite(loc)) and np.linalg.norm(loc) > 1e-6:
            out.append(ch["ch_name"])
    return out


def set_combined_chennu_montage(raw: mne.io.BaseRaw) -> None:
    """Assign positions for Chennu's mixed EGI/10-20 channel naming."""
    hydro = mne.channels.make_standard_montage("GSN-HydroCel-128").get_positions()["ch_pos"]
    ten20 = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    ten20_lower = {k.lower(): v for k, v in ten20.items()}
    ch_pos = {}
    for ch in raw.ch_names:
        if ch in hydro:
            ch_pos[ch] = hydro[ch]
        elif ch.lower() in ten20_lower:
            ch_pos[ch] = ten20_lower[ch.lower()]
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")


def prepare_raw(path: Path, target_sfreq: float, crop_s: float) -> mne.io.BaseRaw:
    raw = smart_load_set(path)
    raw.crop(tmin=0.0, tmax=min(float(crop_s), float(raw.times[-1])), include_tmax=False)
    keep = [ch for ch in raw.ch_names if not any(tok.lower() in ch.lower() for tok in EXCLUDE_TOKENS)]
    raw.pick(keep)
    raw.load_data(verbose="ERROR")
    set_combined_chennu_montage(raw)
    good = positioned_eeg_channels(raw)
    if len(good) < 32:
        raise RuntimeError(f"Too few positioned EEG channels in {path.name}: {len(good)}")
    raw.pick(good)
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference("average", projection=True, verbose="ERROR")
    raw.apply_proj(verbose="ERROR")
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin", verbose="ERROR")
    return raw


def make_inverse(raw: mne.io.BaseRaw, subjects_dir: Path, spacing: str, loose: float, depth: float):
    subject = "fsaverage"
    src = subjects_dir / subject / "bem" / f"{subject}-{spacing}-src.fif"
    bem = subjects_dir / subject / "bem" / f"{subject}-5120-5120-5120-bem-sol.fif"
    trans = subjects_dir / subject / "bem" / f"{subject}-trans.fif"
    fwd = mne.make_forward_solution(
        raw.info,
        trans=str(trans),
        src=str(src),
        bem=str(bem),
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=1,
        verbose="ERROR",
    )
    cov = mne.make_ad_hoc_cov(raw.info, std=dict(eeg=20e-6))
    inv = mne.minimum_norm.make_inverse_operator(
        raw.info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
        fixed=False,
        rank="info",
        verbose="ERROR",
    )
    return inv, fwd["src"]


def labels(subjects_dir: Path) -> list[mne.Label]:
    labs = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=str(subjects_dir), verbose="ERROR")
    bad = ("unknown", "corpuscallosum")
    return [lab for lab in labs if not any(tok in lab.name.lower() for tok in bad)]


def source_roi_data(raw: mne.io.BaseRaw, inv, src, labs: list[mne.Label], lambda2: float, method: str) -> tuple[np.ndarray, float, list[str]]:
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=lambda2, method=method, pick_ori=None, verbose="ERROR")
    tc = mne.extract_label_time_course(stc, labs, src, mode="mean_flip", return_generator=False, verbose="ERROR")
    data = np.asarray(tc, dtype=np.float64)
    data = data - np.nanmean(data, axis=1, keepdims=True)
    scale = np.nanstd(data, axis=1, keepdims=True)
    data = data / np.where(scale > 1e-12, scale, 1.0)
    names = [lab.name for lab in labs]
    return data, float(raw.info["sfreq"]), names


def score_source_recording(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    bounds: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    series = gcco_observable_series(
        data,
        sfreq,
        band,
        window_s=args.window_s,
        stride_s=args.stride_s,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    scored = add_pi(series, bounds)
    return scored, recording_summary(scored)


def paired_summary(subject_condition: pd.DataFrame, band: str) -> dict[str, float]:
    sub = subject_condition[(subject_condition["band"] == band) & (subject_condition["condition"].isin(["baseline", "moderate"]))]
    wide = sub.pivot(index="subject", columns="condition", values="Pi").dropna()
    if wide.empty:
        return {"band": band, "n": 0}
    diff = wide["baseline"].to_numpy(float) - wide["moderate"].to_numpy(float)
    rng = np.random.default_rng(20260515)
    boot = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(5000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    try:
        p_w = stats.wilcoxon(diff, alternative="greater").pvalue
    except ValueError:
        p_w = np.nan
    return {
        "band": band,
        "n": int(len(diff)),
        "baseline_mean": float(wide["baseline"].mean()),
        "moderate_mean": float(wide["moderate"].mean()),
        "mean_delta_baseline_minus_moderate": float(np.mean(diff)),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "paired_dz": float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else np.nan,
        "wilcoxon_greater_p": float(p_w) if np.isfinite(p_w) else np.nan,
        "ttest_p": float(stats.ttest_rel(wide["baseline"], wide["moderate"]).pvalue) if len(diff) > 1 else np.nan,
    }


def write_report(outdir: Path, stats_df: pd.DataFrame, meta: dict) -> None:
    lines = []
    lines.append("# Batch 17 Chennu fsaverage Source-Space GCC-O\n")
    lines.append("Date: 2026-05-15\n")
    lines.append("## Purpose\n")
    lines.append(
        "This batch tests whether the Chennu propofol GCC effect survives an explicit template-source reconstruction. "
        "EEG is mapped to fsaverage using a standard 10-20 montage, an fsaverage BEM/source model, an ad-hoc EEG noise covariance, "
        "and sLORETA/minimum-norm inversion. GCC-O is then computed on aparc ROI time series using lagged phase coupling.\n"
    )
    lines.append("## Parameters\n")
    lines.append("```json\n" + json.dumps(meta, indent=2) + "\n```\n")
    lines.append("## Paired Source-Space Pi Effects\n")
    lines.append(md_table(stats_df))
    lines.append("\n## Interpretation\n")
    lines.append(
        "A positive same-direction effect supports source-space robustness. This is not individual-MRI source localization; it is a template-source robustness analysis intended to address sensor-level volume-conduction and reference concerns.\n"
    )
    (outdir / "BATCH17_CHENNU_FSAVERAGE_SOURCE_GCCO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(show.columns)
    return "\n".join(
        ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        + ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in show.iterrows()]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chennu-root", type=Path, required=True)
    parser.add_argument("--chennu-manifest", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, default=ROOT / "gcc_external_data" / "mne_subjects")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=90.0)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--spacing", default="ico-5")
    parser.add_argument("--method", default="sLORETA", choices=["MNE", "dSPM", "sLORETA", "eLORETA"])
    parser.add_argument("--snr", type=float, default=3.0)
    parser.add_argument("--loose", type=float, default=0.2)
    parser.add_argument("--depth", type=float, default=0.8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--max-subjects", type=int, default=0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.chennu_manifest)
    subjects = sorted([s for s in manifest["subject"].unique() if s != "unknown"], key=lambda x: int(x) if str(x).isdigit() else str(x))
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    labs = labels(args.subjects_dir)
    lambda2 = 1.0 / (args.snr**2)

    inverse_cache = {}
    roi_cache: dict[tuple[str, str], tuple[np.ndarray, float, list[str]]] = {}
    rows = []
    windows = []
    failures = []
    for subject in subjects:
        sub = manifest[manifest["subject"] == subject]
        if set(sub["level"]) < {"baseline", "moderate"}:
            continue
        try:
            base_file = str(sub[sub["level"] == "baseline"].iloc[0]["filename"])
            base_path = find_file(args.chennu_root, base_file)
            base_raw = prepare_raw(base_path, args.target_sfreq, args.crop_s)
            key = tuple(base_raw.ch_names)
            if key not in inverse_cache:
                inv, src = make_inverse(base_raw, args.subjects_dir, args.spacing, args.loose, args.depth)
                inverse_cache[key] = (inv, src)
            inv, src = inverse_cache[key]
            for _, rec in sub.iterrows():
                filename = str(rec["filename"])
                path = find_file(args.chennu_root, filename)
                raw = prepare_raw(path, args.target_sfreq, args.crop_s)
                if tuple(raw.ch_names) != key:
                    inv, src = make_inverse(raw, args.subjects_dir, args.spacing, args.loose, args.depth)
                data, sfreq, roi_names = source_roi_data(raw, inv, src, labs, lambda2, args.method)
                roi_cache[(subject, str(rec["level"]))] = (data, sfreq, roi_names)
            for band_name, band in BANDS.items():
                base_data, sfreq, roi_names = roi_cache[(subject, "baseline")]
                base_series = gcco_observable_series(
                    base_data,
                    sfreq,
                    band,
                    window_s=args.window_s,
                    stride_s=args.stride_s,
                    max_pairs=args.max_pairs,
                    seed=args.seed,
                )
                bounds = calibrate_bounds(base_series, args.alpha)
                for _, rec in sub.iterrows():
                    condition = str(rec["level"])
                    filename = str(rec["filename"])
                    data, sfreq, roi_names = roi_cache[(subject, condition)]
                    scored, summary = score_source_recording(data, sfreq, band, bounds, args)
                    scored["subject"] = subject
                    scored["condition"] = condition
                    scored["filename"] = filename
                    scored["band"] = band_name
                    windows.append(scored)
                    rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "filename": filename,
                            "band": band_name,
                            "n_rois": len(roi_names),
                            "duration_s": float(data.shape[1] / sfreq),
                            **summary,
                        }
                    )
            print(f"Source GCC-O subject {subject}: ok ({len(roi_names)} ROIs)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"subject": subject, "error": str(exc)})
            print(f"Source GCC-O subject {subject}: FAILED {exc}", flush=True)

    rec_df = pd.DataFrame(rows)
    win_df = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    subj = rec_df.groupby(["subject", "condition", "band"], as_index=False)[
        ["R_mean", "D_mean", "M_mean", "Pi", "Access_all", "n_windows", "n_rois", "duration_s"]
    ].mean(numeric_only=True) if not rec_df.empty else pd.DataFrame()
    stats_df = pd.DataFrame([paired_summary(subj, band) for band in BANDS]) if not subj.empty else pd.DataFrame()

    rec_df.to_csv(args.outdir / "source_gcco_recording_summary.csv", index=False)
    win_df.to_csv(args.outdir / "source_gcco_window_features.csv", index=False)
    subj.to_csv(args.outdir / "source_gcco_subject_condition_means.csv", index=False)
    stats_df.to_csv(args.outdir / "source_gcco_paired_stats.csv", index=False)
    meta = {
        "target_sfreq": args.target_sfreq,
        "crop_s": args.crop_s,
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "alpha": args.alpha,
        "spacing": args.spacing,
        "method": args.method,
        "snr": args.snr,
        "loose": args.loose,
        "depth": args.depth,
        "subjects_requested": len(subjects),
        "subjects_completed": int(rec_df["subject"].nunique()) if not rec_df.empty else 0,
        "failures": failures,
    }
    (args.outdir / "source_gcco_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.outdir, stats_df, meta)
    print(json.dumps({"record_rows": len(rec_df), "window_rows": len(win_df), "stats_rows": len(stats_df), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
