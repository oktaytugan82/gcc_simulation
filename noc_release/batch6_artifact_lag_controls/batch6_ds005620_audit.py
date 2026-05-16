#!/usr/bin/env python
"""Audit the local DS005620 subset before extending GCC validation.

The goal is to avoid overclaiming "full DS005620" when only a local subset is
available. The script reads BrainVision headers only and produces a manifest
with subject/task/acquisition/run/duration/channel metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mne
import pandas as pd


PUBLIC_REPORTED_RECORDINGS = 202
PUBLIC_REPORTED_SIZE_GB = 77.3


def parse_bids_name(path: Path) -> dict[str, str | int]:
    name = path.name
    subject = re.search(r"sub-([A-Za-z0-9]+)", name)
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    acq = re.search(r"acq-([A-Za-z0-9]+)", name)
    run = re.search(r"run-(\d+)", name)
    return {
        "subject": subject.group(1) if subject else "",
        "task": task.group(1) if task else "",
        "condition": "awake" if task and task.group(1) == "awake" else (task.group(1) if task else ""),
        "acq": acq.group(1) if acq else "",
        "run": int(run.group(1)) if run else 0,
        "filename": name,
        "path": str(path),
    }


def audit_recording(path: Path) -> dict:
    row = parse_bids_name(path)
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    row.update(
        {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
            "channels": "|".join(raw.ch_names),
            "has_veog": "VEOG" in raw.ch_names,
            "has_heog": "HEOG" in raw.ch_names,
            "has_emg": "EMG" in raw.ch_names,
        }
    )
    eeg_file = path.with_suffix(".eeg")
    row["eeg_size_mb"] = eeg_file.stat().st_size / 1024**2 if eeg_file.exists() else 0.0
    return row


def summarize(df: pd.DataFrame, root: Path) -> dict:
    local_size_gb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1024**3
    by_task = df.groupby("task").size().to_dict()
    by_subject_task = (
        df.pivot_table(index="subject", columns="task", values="filename", aggfunc="count", fill_value=0)
        .astype(int)
        .reset_index()
        .to_dict(orient="records")
    )
    subjects_without_awake = sorted(
        set(df["subject"].unique()) - set(df.loc[df["task"] == "awake", "subject"].unique())
    )
    complete_local_subjects = []
    for subject, sub in df.groupby("subject"):
        tasks = set(sub["task"])
        if {"awake", "sed", "sed2"}.issubset(tasks):
            complete_local_subjects.append(subject)
    return {
        "root": str(root),
        "local_recordings": int(len(df)),
        "local_subjects": int(df["subject"].nunique()),
        "local_size_gb": local_size_gb,
        "public_reported_recordings": PUBLIC_REPORTED_RECORDINGS,
        "public_reported_size_gb": PUBLIC_REPORTED_SIZE_GB,
        "estimated_missing_recordings_vs_public": int(PUBLIC_REPORTED_RECORDINGS - len(df)),
        "estimated_missing_size_gb_vs_public": float(PUBLIC_REPORTED_SIZE_GB - local_size_gb),
        "tasks": by_task,
        "acquisitions": df.groupby(["task", "acq"]).size().reset_index(name="n").to_dict(orient="records"),
        "subjects_without_awake": subjects_without_awake,
        "subjects_with_awake_sed_sed2": sorted(complete_local_subjects),
        "n_subjects_with_awake_sed_sed2": int(len(complete_local_subjects)),
        "duration_by_task": df.groupby("task")["duration_s"].agg(["count", "mean", "min", "max"]).reset_index().to_dict(orient="records"),
        "sfreq_values": sorted(df["sfreq"].unique().tolist()),
        "channel_count_values": sorted(df["n_channels"].unique().tolist()),
        "by_subject_task": by_subject_task,
    }


def write_report(summary: dict, outdir: Path) -> None:
    lines = [
        "# Batch 6 DS005620 Local Audit",
        "",
        "## Core Finding",
        "",
        f"Local DS005620 data contain {summary['local_recordings']} recordings from {summary['local_subjects']} subjects.",
        f"The public dataset is reported as {summary['public_reported_recordings']} recordings and approximately {summary['public_reported_size_gb']} GB.",
        f"Estimated local gap: {summary['estimated_missing_recordings_vs_public']} recordings and about {summary['estimated_missing_size_gb_vs_public']:.1f} GB.",
        "",
        "## Local Task Counts",
        "",
    ]
    for task, count in summary["tasks"].items():
        lines.append(f"- {task}: {count}")
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "The current local analysis must be described as a DS005620 subset analysis, not a full-dataset analysis.",
            "The next strengthening step is either to analyse all 126 local recordings more fully or to download the remaining public data before claiming full DS005620 coverage.",
            "",
            "## Stop/Go",
            "",
            "- GO now: repeated-run stability and within-subject robustness on the 126 local recordings.",
            "- STOP before full-dataset claims: download/audit the missing recordings.",
        ]
    )
    (outdir / "BATCH6_DS005620_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = [audit_recording(path) for path in sorted(args.root.rglob("*_eeg.vhdr"))]
    if not rows:
        raise RuntimeError(f"No BrainVision .vhdr recordings found in {args.root}")
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "ds005620_local_audit.csv", index=False)
    summary = summarize(df, args.root)
    with open(args.outdir / "ds005620_local_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(summary, args.outdir)
    print(json.dumps(summary, indent=2)[:12000])


if __name__ == "__main__":
    main()
