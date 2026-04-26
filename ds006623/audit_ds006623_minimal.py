"""Audit the local minimal ds006623 subset before GCC fMRI validation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(r"sub-(?P<subject>\d+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {path}")


def parse_run(path: Path) -> dict[str, str]:
    match = RUN_RE.search(path.name)
    if not match:
        return {"subject": "unknown", "task": "unknown", "run": "unknown"}
    return match.groupdict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path("data") / "ds006623-minimal"),
        help="Local minimal ds006623 subset root.",
    )
    parser.add_argument("--atlas", default="4S156Parcels")
    parser.add_argument("--pipeline", default="xcp_d_without_GSR_bandpass_output")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"missing root: {root}")

    manifest = root / "ds006623_minimal_manifest.csv"
    manifest_df = read_table(manifest) if manifest.exists() else pd.DataFrame()

    participant_info = root / "derivatives" / "Participant_Info.csv"
    lor_ror = root / "derivatives" / "LOR_ROR_Timing.csv"
    participant_df = read_table(participant_info) if participant_info.exists() else pd.DataFrame()
    lor_df = read_table(lor_ror) if lor_ror.exists() else pd.DataFrame()

    ts_pattern = f"derivatives/{args.pipeline}/sub-*/func/*_seg-{args.atlas}_stat-mean_timeseries.tsv"
    motion_pattern = f"derivatives/{args.pipeline}/sub-*/func/*_motion.tsv"
    coverage_pattern = f"derivatives/{args.pipeline}/sub-*/func/*_seg-{args.atlas}_stat-coverage_bold.tsv"
    ts_files = sorted(root.glob(ts_pattern))
    motion_files = sorted(root.glob(motion_pattern))
    coverage_files = sorted(root.glob(coverage_pattern))

    run_rows: list[dict[str, object]] = []
    for path in ts_files:
        parsed = parse_run(path)
        rel = path.relative_to(root).as_posix()
        try:
            preview = pd.read_csv(path, sep="\t", nrows=5)
            n_cols = len(preview.columns)
        except Exception:
            n_cols = None
        run_rows.append({**parsed, "path": rel, "size_bytes": path.stat().st_size, "columns": n_cols})
    run_df = pd.DataFrame(run_rows)

    if not run_df.empty:
        runs_by_task = {
            f"{task}_run-{run}": int(count)
            for (task, run), count in run_df.groupby(["task", "run"]).size().items()
        }
    else:
        runs_by_task = {}

    summary = {
        "root": str(root.resolve()),
        "manifest_rows": int(len(manifest_df)),
        "manifest_size_mb": float(manifest_df["size"].sum() / 1024 / 1024) if "size" in manifest_df else 0.0,
        "participant_rows": int(len(participant_df)),
        "participant_columns": list(participant_df.columns),
        "lor_ror_rows": int(len(lor_df)),
        "lor_ror_columns": list(lor_df.columns),
        "timeseries_files": int(len(ts_files)),
        "motion_files": int(len(motion_files)),
        "coverage_files": int(len(coverage_files)),
        "subjects_with_timeseries": sorted(run_df["subject"].unique().tolist()) if not run_df.empty else [],
        "runs_by_task": runs_by_task,
    }

    output_dir = root / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ds006623_minimal_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not run_df.empty:
        run_df.to_csv(output_dir / "ds006623_timeseries_inventory.csv", index=False)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
