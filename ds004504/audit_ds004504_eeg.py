from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import mne
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = Path(os.environ.get("DS004504_ROOT", REPO_ROOT / "data" / "ds004504-main"))
RESULTS_DIR = REPO_ROOT / "results"
OUT_CSV = RESULTS_DIR / "ds004504_eeg_audit.csv"
OUT_JSON = RESULTS_DIR / "ds004504_eeg_audit_summary.json"


def subject_file(subject_id: str) -> Path:
    return ROOT / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"


def main() -> None:
    participants = pd.read_csv(ROOT / "participants.tsv", sep="\t")
    rows = []
    failures = []

    for _, participant in participants.iterrows():
        subject_id = participant["participant_id"]
        eeg_path = subject_file(subject_id)
        record = {
            "participant_id": subject_id,
            "group": participant["Group"],
            "gender": participant["Gender"],
            "age": int(participant["Age"]),
            "mmse": int(participant["MMSE"]),
            "file": str(eeg_path),
            "file_exists": eeg_path.exists(),
            "file_bytes": eeg_path.stat().st_size if eeg_path.exists() else 0,
            "read_ok": False,
            "n_channels": None,
            "sfreq_hz": None,
            "n_samples": None,
            "duration_s": None,
            "channel_names": None,
            "error": "",
        }

        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose="ERROR")
            record.update(
                {
                    "read_ok": True,
                    "n_channels": len(raw.ch_names),
                    "sfreq_hz": float(raw.info["sfreq"]),
                    "n_samples": int(raw.n_times),
                    "duration_s": float(raw.n_times / raw.info["sfreq"]),
                    "channel_names": "|".join(raw.ch_names),
                }
            )
        except Exception as exc:  # keep audit running even if one file fails
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(record)

        rows.append(record)
        print(
            f"{subject_id}: "
            f"{'OK' if record['read_ok'] else 'FAIL'} "
            f"group={record['group']} "
            f"duration={record['duration_s']}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["read_ok"]]
    channel_sets = Counter(row["channel_names"] for row in ok_rows)
    sfreqs = Counter(row["sfreq_hz"] for row in ok_rows)
    n_channels = Counter(row["n_channels"] for row in ok_rows)
    by_group = Counter(row["group"] for row in rows)
    ok_by_group = Counter(row["group"] for row in ok_rows)
    durations = [row["duration_s"] for row in ok_rows]
    sizes = [row["file_bytes"] for row in rows]

    summary = {
        "dataset_root": str(ROOT),
        "n_participants": len(rows),
        "n_read_ok": len(ok_rows),
        "n_failed": len(failures),
        "groups": dict(sorted(by_group.items())),
        "read_ok_by_group": dict(sorted(ok_by_group.items())),
        "sfreqs_hz": {str(k): v for k, v in sorted(sfreqs.items())},
        "n_channels": {str(k): v for k, v in sorted(n_channels.items())},
        "unique_channel_sets": len(channel_sets),
        "dominant_channel_set_count": channel_sets.most_common(1)[0][1] if channel_sets else 0,
        "duration_s_min": min(durations) if durations else None,
        "duration_s_max": max(durations) if durations else None,
        "duration_s_mean": sum(durations) / len(durations) if durations else None,
        "file_bytes_min": min(sizes) if sizes else None,
        "file_bytes_max": max(sizes) if sizes else None,
        "file_bytes_total": sum(sizes),
        "failures": [
            {
                "participant_id": row["participant_id"],
                "file": row["file"],
                "error": row["error"],
            }
            for row in failures
        ],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
