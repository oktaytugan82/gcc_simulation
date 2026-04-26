"""Extract GCC-style fMRI phase features from the minimal ds006623 subset.

Primary design:
* calibrate each participant's access-compatible range on awake task baseline
  (imagery run 1)
* compute sliding-window Hilbert-phase coherence (R_phase), effective
  dimensionality (D_eff), and metastability (M_tau) for Base1, PreLOR, LOR,
  ROR, and Base2 phases
* report Pi_fMRI as the fraction of windows inside the subject-specific
  baseline-calibrated regime

This is deliberately a feature-extraction stage, not yet the final statistical
model comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert


RUN_RE = re.compile(r"sub-(?P<subject>\d+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)")


@dataclass(frozen=True)
class Segment:
    subject: str
    run: int
    phase: str
    start: int
    stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path("data") / "ds006623-minimal"))
    parser.add_argument("--pipeline", default="xcp_d_without_GSR_bandpass_output")
    parser.add_argument("--atlas", default="4S156Parcels")
    parser.add_argument("--tr", type=float, default=0.8)
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--fd-threshold", type=float, default=0.8)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--output-prefix", default="ds006623_fmri_gcc")
    return parser.parse_args()


def subject_from_name(path: Path) -> str:
    match = RUN_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse subject/run from {path.name}")
    return match.group("subject")


def read_lor_ror(root: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(root / "derivatives" / "LOR_ROR_Timing.csv")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        sub = str(row["Subject"]).replace("sub-", "")
        lor = pd.to_numeric(row["LOR time (TR in task2)"], errors="coerce")
        ror = pd.to_numeric(row["ROR time (TR in task3)"], errors="coerce")
        out[sub] = {
            "lor_tr_task2": float(lor) if pd.notna(lor) else math.nan,
            "ror_tr_task3": float(ror) if pd.notna(ror) else math.nan,
        }
    return out


def load_timeseries(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    good_cols = data.columns[data.notna().mean(axis=0) >= 0.95]
    return data.loc[:, good_cols].interpolate(limit_direction="both").fillna(0.0)


def load_motion(path: Path, n_rows: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"framewise_displacement": np.zeros(n_rows)})
    motion = pd.read_csv(path, sep="\t")
    if "framewise_displacement" not in motion:
        motion["framewise_displacement"] = 0.0
    motion["framewise_displacement"] = pd.to_numeric(motion["framewise_displacement"], errors="coerce").fillna(0.0)
    if len(motion) < n_rows:
        padding = pd.DataFrame({"framewise_displacement": np.zeros(n_rows - len(motion))})
        motion = pd.concat([motion, padding], ignore_index=True)
    return motion.iloc[:n_rows].copy()


def robust_zscore(values: np.ndarray) -> np.ndarray:
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
    scale[scale < 1e-8] = np.nanstd(values[:, scale < 1e-8], axis=0)
    scale[scale < 1e-8] = 1.0
    return (values - center) / scale


def effective_dimensionality(values: np.ndarray) -> float:
    if values.shape[0] < 4 or values.shape[1] < 2:
        return math.nan
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    denom = float(np.sum(eigvals**2))
    if denom <= 1e-12:
        return math.nan
    return float((np.sum(eigvals) ** 2 / denom) / len(eigvals))


def window_metrics(values: np.ndarray) -> dict[str, float]:
    z = robust_zscore(values)
    phase = np.angle(hilbert(z, axis=0))
    order = np.abs(np.mean(np.exp(1j * phase), axis=1))
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    tri = corr[np.triu_indices_from(corr, k=1)]
    return {
        "R_phase": float(np.mean(order)),
        "D_eff": effective_dimensionality(z),
        "M_tau": float(np.std(order, ddof=1)) if len(order) > 1 else math.nan,
        "mean_fc": float(np.mean(tri)),
        "mean_abs_fc": float(np.mean(np.abs(tri))),
    }


def iter_windows(start: int, stop: int, window: int, step: int) -> list[tuple[int, int]]:
    if stop - start < window:
        return []
    return [(idx, idx + window) for idx in range(start, stop - window + 1, step)]


def build_segments(subject: str, n_by_run: dict[int, int], lor_ror: dict[str, dict[str, float]]) -> list[Segment]:
    timing = lor_ror.get(subject, {})
    lor = timing.get("lor_tr_task2", math.nan)
    ror = timing.get("ror_tr_task3", math.nan)
    segments: list[Segment] = []
    if 1 in n_by_run:
        segments.append(Segment(subject, 1, "Base1", 0, n_by_run[1]))
    if 2 in n_by_run and not math.isnan(lor):
        lor_i = max(0, min(int(round(lor)), n_by_run[2]))
        segments.append(Segment(subject, 2, "PreLOR", 0, lor_i))
        segments.append(Segment(subject, 2, "LOR_task2", lor_i, n_by_run[2]))
    if 3 in n_by_run:
        if math.isnan(ror):
            segments.append(Segment(subject, 3, "LOR_task3", 0, n_by_run[3]))
        else:
            ror_i = max(0, min(int(round(ror)), n_by_run[3]))
            segments.append(Segment(subject, 3, "LOR_task3", 0, ror_i))
            segments.append(Segment(subject, 3, "ROR", ror_i, n_by_run[3]))
    if 4 in n_by_run:
        segments.append(Segment(subject, 4, "Base2", 0, n_by_run[4]))
    return [seg for seg in segments if seg.stop > seg.start]


def access_bounds(baseline_windows: pd.DataFrame, alpha: float) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in ["R_phase", "D_eff", "M_tau"]:
        lower = float(baseline_windows[col].quantile(alpha / 2))
        upper = float(baseline_windows[col].quantile(1 - alpha / 2))
        bounds[col] = (lower, upper)
    return bounds


def inside_bounds(row: pd.Series, bounds: dict[str, tuple[float, float]]) -> bool:
    return all(bounds[col][0] <= float(row[col]) <= bounds[col][1] for col in bounds)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path("results")
    fig_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    window = int(round(args.window_sec / args.tr))
    step = int(round(args.step_sec / args.tr))
    lor_ror = read_lor_ror(root)

    ts_files = sorted(
        (root / "derivatives" / args.pipeline).glob(
            f"sub-*/func/*_task-imagery_run-*_space-MNI152NLin2009cAsym_seg-{args.atlas}_stat-mean_timeseries.tsv"
        )
    )
    by_subject: dict[str, dict[int, Path]] = {}
    for path in ts_files:
        match = RUN_RE.search(path.name)
        if not match:
            continue
        subject = match.group("subject")
        run = int(match.group("run"))
        by_subject.setdefault(subject, {})[run] = path

    phase_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for subject, run_paths in sorted(by_subject.items()):
        data_by_run: dict[int, pd.DataFrame] = {}
        motion_by_run: dict[int, pd.DataFrame] = {}
        for run, ts_path in sorted(run_paths.items()):
            data = load_timeseries(ts_path)
            motion_path = ts_path.with_name(
                f"sub-{subject}_task-imagery_run-{run}_motion.tsv"
            )
            data_by_run[run] = data
            motion_by_run[run] = load_motion(motion_path, len(data))

        segments = build_segments(subject, {run: len(df) for run, df in data_by_run.items()}, lor_ror)
        base1 = next((seg for seg in segments if seg.phase == "Base1"), None)
        if base1 is None:
            skipped.append({"subject": subject, "reason": "missing Base1"})
            continue

        base_window_rows: list[dict[str, object]] = []
        for w_start, w_stop in iter_windows(base1.start, base1.stop, window, step):
            fd = motion_by_run[1]["framewise_displacement"].iloc[w_start:w_stop].to_numpy()
            valid_fraction = float(np.mean(fd <= args.fd_threshold))
            if valid_fraction < args.min_valid_fraction:
                continue
            metrics = window_metrics(data_by_run[1].iloc[w_start:w_stop].to_numpy())
            base_window_rows.append(metrics)
        if len(base_window_rows) < 5:
            skipped.append({"subject": subject, "reason": "too few baseline windows", "n": len(base_window_rows)})
            continue
        bounds = access_bounds(pd.DataFrame(base_window_rows), args.alpha)

        for seg in segments:
            rows: list[dict[str, object]] = []
            for w_start, w_stop in iter_windows(seg.start, seg.stop, window, step):
                fd = motion_by_run[seg.run]["framewise_displacement"].iloc[w_start:w_stop].to_numpy()
                valid_fraction = float(np.mean(fd <= args.fd_threshold))
                if valid_fraction < args.min_valid_fraction:
                    continue
                metrics = window_metrics(data_by_run[seg.run].iloc[w_start:w_stop].to_numpy())
                row = {
                    "subject": f"sub-{subject}",
                    "run": seg.run,
                    "phase": seg.phase,
                    "window_start_tr": w_start,
                    "window_stop_tr": w_stop,
                    "window_start_sec": w_start * args.tr,
                    "window_stop_sec": w_stop * args.tr,
                    "fd_mean": float(np.mean(fd)),
                    "fd_max": float(np.max(fd)),
                    "valid_fraction": valid_fraction,
                    **metrics,
                }
                row["inside_baseline_regime"] = inside_bounds(pd.Series(row), bounds)
                rows.append(row)
                window_rows.append(row)

            if rows:
                phase_df = pd.DataFrame(rows)
                phase_rows.append(
                    {
                        "subject": f"sub-{subject}",
                        "run": seg.run,
                        "phase": seg.phase,
                        "n_windows": int(len(phase_df)),
                        "Pi_fMRI": float(phase_df["inside_baseline_regime"].mean()),
                        "R_phase_mean": float(phase_df["R_phase"].mean()),
                        "D_eff_mean": float(phase_df["D_eff"].mean()),
                        "M_tau_mean": float(phase_df["M_tau"].mean()),
                        "mean_fc": float(phase_df["mean_fc"].mean()),
                        "mean_abs_fc": float(phase_df["mean_abs_fc"].mean()),
                        "fd_mean": float(phase_df["fd_mean"].mean()),
                        "fd_max": float(phase_df["fd_max"].max()),
                        "baseline_R_lower": bounds["R_phase"][0],
                        "baseline_R_upper": bounds["R_phase"][1],
                        "baseline_D_lower": bounds["D_eff"][0],
                        "baseline_D_upper": bounds["D_eff"][1],
                        "baseline_M_lower": bounds["M_tau"][0],
                        "baseline_M_upper": bounds["M_tau"][1],
                    }
                )
            else:
                skipped.append({"subject": subject, "phase": seg.phase, "reason": "no valid windows"})

    phase_df = pd.DataFrame(phase_rows)
    window_df = pd.DataFrame(window_rows)
    phase_path = out_dir / f"{args.output_prefix}_phase_features.csv"
    window_path = out_dir / f"{args.output_prefix}_window_features.csv"
    phase_df.to_csv(phase_path, index=False)
    window_df.to_csv(window_path, index=False)

    summary = {
        "root": str(root.resolve()),
        "pipeline": args.pipeline,
        "atlas": args.atlas,
        "tr": args.tr,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "alpha": args.alpha,
        "fd_threshold": args.fd_threshold,
        "min_valid_fraction": args.min_valid_fraction,
        "subjects_seen": len(by_subject),
        "subjects_analyzed": int(phase_df["subject"].nunique()) if not phase_df.empty else 0,
        "phase_rows": int(len(phase_df)),
        "window_rows": int(len(window_df)),
        "phase_counts": phase_df.groupby("phase").size().astype(int).to_dict() if not phase_df.empty else {},
        "skipped": skipped,
        "phase_features": str(phase_path.resolve()),
        "window_features": str(window_path.resolve()),
    }
    summary_path = out_dir / f"{args.output_prefix}_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
