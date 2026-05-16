#!/usr/bin/env python
"""Compare local DS005620 files against the OpenNeuro manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openneuro._download import _get_download_metadata


def parse(path: str) -> dict:
    p = Path(path)
    name = p.name
    subject = re.search(r"sub-([A-Za-z0-9]+)", path)
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    acq = re.search(r"acq-([A-Za-z0-9]+)", name)
    run = re.search(r"run-(\d+)", name)
    suffix = ""
    if "_eeg." in name:
        suffix = "eeg"
    elif "_events." in name:
        suffix = "events"
    elif "_channels." in name:
        suffix = "channels"
    return {
        "relative_path": path.replace("\\", "/"),
        "subject": subject.group(1) if subject else "",
        "task": task.group(1) if task else "",
        "acq": acq.group(1) if acq else "",
        "run": int(run.group(1)) if run else 0,
        "extension": p.suffix,
        "suffix": suffix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ds005620")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    snap = _get_download_metadata(dataset_id=args.dataset, max_retries=3, metadata_timeout=120)
    rows = []
    for file in snap.files:
        row = parse(file.filename)
        row["size"] = int(file.size or 0)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.outdir / "ds005620_openneuro_manifest.csv", index=False)

    local_paths = {
        str(path.relative_to(args.local_root)).replace("\\", "/")
        for path in args.local_root.rglob("*")
        if path.is_file()
    }
    manifest["local_present"] = manifest["relative_path"].isin(local_paths)
    manifest.to_csv(args.outdir / "ds005620_openneuro_vs_local_manifest.csv", index=False)

    vhdr = manifest[(manifest["extension"] == ".vhdr") & (manifest["suffix"] == "eeg")].copy()
    summary = {
        "dataset": args.dataset,
        "openneuro_total_files": int(len(manifest)),
        "openneuro_total_size_gb": float(manifest["size"].sum() / 1024**3),
        "openneuro_vhdr_recordings": int(len(vhdr)),
        "local_present_files": int(manifest["local_present"].sum()),
        "local_present_vhdr_recordings": int(vhdr["local_present"].sum()),
        "missing_vhdr_recordings": int((~vhdr["local_present"]).sum()),
        "openneuro_recordings_by_task_acq": vhdr.groupby(["task", "acq"]).size().reset_index(name="n").to_dict(orient="records"),
        "local_recordings_by_task_acq": vhdr[vhdr["local_present"]].groupby(["task", "acq"]).size().reset_index(name="n").to_dict(orient="records"),
        "missing_recordings_by_task_acq": vhdr[~vhdr["local_present"]].groupby(["task", "acq"]).size().reset_index(name="n").to_dict(orient="records"),
        "missing_eeg_size_gb": float(
            manifest[
                (~manifest["local_present"])
                & (manifest["suffix"] == "eeg")
                & (manifest["extension"].isin([".eeg", ".vhdr", ".vmrk", ".json"]))
            ]["size"].sum()
            / 1024**3
        ),
    }
    with open(args.outdir / "ds005620_openneuro_manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# DS005620 OpenNeuro Manifest Comparison",
        "",
        f"OpenNeuro manifest files: {summary['openneuro_total_files']}",
        f"OpenNeuro manifest size: {summary['openneuro_total_size_gb']:.1f} GB",
        f"OpenNeuro EEG .vhdr recordings: {summary['openneuro_vhdr_recordings']}",
        f"Local EEG .vhdr recordings present: {summary['local_present_vhdr_recordings']}",
        f"Missing EEG .vhdr recordings: {summary['missing_vhdr_recordings']}",
        f"Estimated missing EEG-related size: {summary['missing_eeg_size_gb']:.1f} GB",
        "",
        "## Missing Recordings by Task/Acquisition",
        "",
    ]
    for row in summary["missing_recordings_by_task_acq"]:
        lines.append(f"- task={row['task']}, acq={row['acq']}: {row['n']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The local subset contains the awake eyes-closed calibration and resting sed/sed2 recordings used in the current analysis.",
            "Most missing recordings are additional acquisition types such as awake eyes-open and TMS, not necessarily missing resting-sedation runs.",
        ]
    )
    (args.outdir / "BATCH6_DS005620_OPENNEURO_MANIFEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
