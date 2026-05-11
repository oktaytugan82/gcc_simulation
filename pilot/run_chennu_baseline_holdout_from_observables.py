from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon

try:
    from observables import access_region_indicator, calibrate_from_baseline
except ImportError:
    from pilot.observables import access_region_indicator, calibrate_from_baseline


ROOT = Path(__file__).resolve().parents[1]

LEVELS = {
    1: "baseline",
    2: "mild",
    3: "moderate",
    4: "recovery",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chennu 2016 gamma baseline-holdout validation from saved window-resolved observables."
    )
    parser.add_argument(
        "--observables-pkl",
        type=Path,
        default=ROOT / "results" / "pilot_results.pkl",
        help="Pickle produced by the Chennu pilot pipeline containing per-file observables.",
    )
    parser.add_argument(
        "--datainfo-mat",
        type=Path,
        default=ROOT / "data_manifests" / "chennu2016_datainfo.mat",
        help="Cambridge datainfo.mat metadata table from the Chennu raw bundle.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for CSV, JSON, and Markdown outputs.",
    )
    return parser.parse_args()


def subject_from_filename(name: str) -> str:
    match = re.match(r"^(\d{2})-", name)
    return match.group(1) if match else "XX"


def load_datainfo(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    datainfo = mat["datainfo"]
    label_by_file = {}
    subject_by_file = {}
    for row in datainfo:
        stem = str(row[0])
        level = LEVELS[int(row[1])]
        filename = f"{stem}.set"
        label_by_file[filename] = level
        subject_by_file[filename] = subject_from_filename(filename)
    return label_by_file, subject_by_file


def restrict_obs(obs: dict[str, np.ndarray], t_start: float, t_stop: float) -> dict[str, np.ndarray]:
    dense_mask = (obs["t"] >= t_start) & (obs["t"] < t_stop)
    sparse_mask = (obs["D_times"] >= t_start) & (obs["D_times"] < t_stop)
    return {
        "t": obs["t"][dense_mask] - t_start,
        "R": obs["R"][dense_mask],
        "M": obs["M"][dense_mask],
        "D_times": obs["D_times"][sparse_mask] - t_start,
        "D": obs["D"][sparse_mask],
    }


def cohens_d_paired(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd) if sd > 0 else float("nan")


def require_input(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}\n{message}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "chennu2016_gamma_baseline_holdout.csv"
    out_json = out_dir / "chennu2016_gamma_baseline_holdout_summary.json"
    out_report = out_dir / "chennu2016_gamma_baseline_holdout_report.md"

    require_input(
        args.observables_pkl,
        "Re-run the Chennu pilot pipeline to generate the window-resolved observables pickle. "
        "The derived CSV/JSON outputs are tracked, but the large pickle is not stored in GitHub.",
    )
    require_input(
        args.datainfo_mat,
        "Download the Chennu 2016 Cambridge raw bundle and extract Sedation-RestingState/datainfo.mat, "
        "or use data_manifests/chennu2016_datainfo.mat from this repository.",
    )

    label_by_file, subject_by_file = load_datainfo(args.datainfo_mat)

    with args.observables_pkl.open("rb") as f:
        pilot = pickle.load(f)

    by_subject = defaultdict(dict)
    for item in pilot["results"]:
        name = item["filename"]
        if name not in label_by_file:
            continue
        subject = subject_by_file[name]
        level = label_by_file[name]
        by_subject[subject][level] = item

    rows = []
    for subject, levels in sorted(by_subject.items()):
        if "baseline" not in levels:
            continue
        base = levels["baseline"]
        duration = float(base["duration_s"])
        split = duration / 2.0
        calib_obs = restrict_obs(base["obs"], 0.0, split)
        holdout_obs = restrict_obs(base["obs"], split, duration)
        bounds = calibrate_from_baseline(calib_obs, alpha=0.1)
        holdout_acc = access_region_indicator(holdout_obs, bounds)
        rows.append(
            {
                "subject": subject,
                "level": "baseline_holdout",
                "filename": base["filename"],
                "calibration": "first_half_baseline",
                "eval_window": "second_half_baseline",
                "duration_s": duration - split,
                "Pi": holdout_acc["fraction"],
                **bounds,
            }
        )
        for level in ["mild", "moderate", "recovery"]:
            if level not in levels:
                continue
            item = levels[level]
            acc = access_region_indicator(item["obs"], bounds)
            rows.append(
                {
                    "subject": subject,
                    "level": level,
                    "filename": item["filename"],
                    "calibration": "first_half_baseline",
                    "eval_window": "full_recording",
                    "duration_s": float(item["duration_s"]),
                    "Pi": acc["fraction"],
                    **bounds,
                }
            )

    fieldnames = [
        "subject",
        "level",
        "filename",
        "calibration",
        "eval_window",
        "duration_s",
        "Pi",
        "R_min",
        "D_min",
        "D_max",
        "M_min",
        "M_max",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    level_values = defaultdict(list)
    for row in rows:
        level_values[row["level"]].append(float(row["Pi"]))

    base = {row["subject"]: float(row["Pi"]) for row in rows if row["level"] == "baseline_holdout"}
    summary = {
        "source_observables": str(args.observables_pkl),
        "raw_data_source": "Chennu et al. 2016 Cambridge Data Repository, DOI 10.17863/CAM.68959",
        "raw_zip_md5": "C38293499CDFE090FE2C1AAFA687785C",
        "metadata_source": str(args.datainfo_mat),
        "metadata_note": "Sedation labels are read from the Cambridge datainfo.mat table: 1=baseline, 2=mild, 3=moderate, 4=recovery.",
        "band": "gamma",
        "calibration": "first half of each subject's baseline recording",
        "holdout": "second half of the same baseline recording",
        "n_subjects": len(base),
        "level_summary": {},
        "paired_tests_vs_baseline_holdout": {},
    }
    for level, values in sorted(level_values.items()):
        arr = np.asarray(values, dtype=float)
        summary["level_summary"][level] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
            "median": float(np.median(arr)),
        }

    for level in ["mild", "moderate", "recovery"]:
        comp = {row["subject"]: float(row["Pi"]) for row in rows if row["level"] == level}
        subjects = sorted(set(base) & set(comp))
        diffs = np.asarray([base[s] - comp[s] for s in subjects], dtype=float)
        if len(diffs) >= 2:
            stat, p = wilcoxon(diffs)
            summary["paired_tests_vs_baseline_holdout"][level] = {
                "n": int(len(diffs)),
                "mean_baseline_minus_level": float(np.mean(diffs)),
                "median_baseline_minus_level": float(np.median(diffs)),
                "wilcoxon_W": float(stat),
                "wilcoxon_p": float(p),
                "cohens_dz": cohens_d_paired(diffs),
                "n_declines": int(np.sum(diffs > 0)),
            }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    moderate = summary["paired_tests_vs_baseline_holdout"]["moderate"]
    mild = summary["paired_tests_vs_baseline_holdout"]["mild"]
    recovery = summary["paired_tests_vs_baseline_holdout"]["recovery"]
    report = f"""# Chennu 2016 Gamma Baseline-Holdout Analysis

Calibration is performed only on the first half of each participant's baseline
recording. The second half of baseline is then evaluated as an internal holdout
before applying the same bounds to mild sedation, moderate sedation, and recovery.

## Level Summary

| Level | N | Mean Pi | SD | Median |
|---|---:|---:|---:|---:|
| baseline holdout | {summary['level_summary']['baseline_holdout']['n']} | {summary['level_summary']['baseline_holdout']['mean']:.3f} | {summary['level_summary']['baseline_holdout']['sd']:.3f} | {summary['level_summary']['baseline_holdout']['median']:.3f} |
| mild | {summary['level_summary']['mild']['n']} | {summary['level_summary']['mild']['mean']:.3f} | {summary['level_summary']['mild']['sd']:.3f} | {summary['level_summary']['mild']['median']:.3f} |
| moderate | {summary['level_summary']['moderate']['n']} | {summary['level_summary']['moderate']['mean']:.3f} | {summary['level_summary']['moderate']['sd']:.3f} | {summary['level_summary']['moderate']['median']:.3f} |
| recovery | {summary['level_summary']['recovery']['n']} | {summary['level_summary']['recovery']['mean']:.3f} | {summary['level_summary']['recovery']['sd']:.3f} | {summary['level_summary']['recovery']['median']:.3f} |

## Paired Tests Against Baseline Holdout

| Comparison | N | Mean baseline minus level | Median baseline minus level | Wilcoxon p | Cohen dz | Declines |
|---|---:|---:|---:|---:|---:|---:|
| baseline holdout - mild | {mild['n']} | {mild['mean_baseline_minus_level']:.3f} | {mild['median_baseline_minus_level']:.3f} | {mild['wilcoxon_p']:.4g} | {mild['cohens_dz']:.2f} | {mild['n_declines']}/20 |
| baseline holdout - moderate | {moderate['n']} | {moderate['mean_baseline_minus_level']:.3f} | {moderate['median_baseline_minus_level']:.3f} | {moderate['wilcoxon_p']:.4g} | {moderate['cohens_dz']:.2f} | {moderate['n_declines']}/20 |
| baseline holdout - recovery | {recovery['n']} | {recovery['mean_baseline_minus_level']:.3f} | {recovery['median_baseline_minus_level']:.3f} | {recovery['wilcoxon_p']:.4g} | {recovery['cohens_dz']:.2f} | {recovery['n_declines']}/20 |
"""
    out_report.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
