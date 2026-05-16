from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mne
import pandas as pd


BAD_TOKENS = (
    "EOG",
    "EMG",
    "ECG",
    "EKG",
    "RESP",
    "FLOW",
    "SNORE",
    "SPO",
    "PLETH",
    "PULSE",
    "PRESS",
    "LIGHT",
    "BODY",
    "POSITION",
    "MARK",
    "EVENT",
)


def label_from_name(name: str) -> str:
    if re.search(r"MCS\+", name):
        return "MCS+"
    if re.search(r"MCS-", name):
        return "MCS-"
    if re.search(r"VS", name):
        return "VS"
    return "UNK"


def is_eeg_like(ch: str) -> bool:
    up = ch.upper()
    if any(tok in up for tok in BAD_TOKENS):
        return False
    # Keep common EEG derivations and monopolar labels, including vendor variants.
    return bool(
        re.search(r"\b(FP|F|C|P|O|T|A)\s*Z?\d?", up)
        or re.search(r"(FP1|FP2|F3|F4|F7|F8|FZ|CZ|C3|C4|PZ|P3|P4|O1|O2|T3|T4|T5|T6)", up)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    channel_rows = []
    failures = []
    for edf in sorted(args.root.glob("*.edf")):
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times / sfreq)
            eeg_like = [ch for ch in raw.ch_names if is_eeg_like(ch)]
            rows.append(
                {
                    "filename": edf.name,
                    "label": label_from_name(edf.name),
                    "sfreq": sfreq,
                    "duration_s": duration,
                    "n_channels": len(raw.ch_names),
                    "n_eeg_like": len(eeg_like),
                    "eeg_like_channels": ";".join(eeg_like),
                    "all_channels": ";".join(raw.ch_names),
                }
            )
            for ch in raw.ch_names:
                channel_rows.append({"filename": edf.name, "label": label_from_name(edf.name), "channel": ch, "eeg_like": is_eeg_like(ch)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"filename": edf.name, "error": str(exc)})

    audit = pd.DataFrame(rows)
    chan = pd.DataFrame(channel_rows)
    audit.to_csv(args.outdir / "doc_edf_audit.csv", index=False)
    chan.to_csv(args.outdir / "doc_channel_inventory.csv", index=False)
    common = {}
    if not chan.empty:
        common = (
            chan[chan["eeg_like"]]
            .groupby("channel")["filename"]
            .nunique()
            .sort_values(ascending=False)
            .head(50)
            .to_dict()
        )
    summary = {
        "n_files": int(len(audit)),
        "failures": failures,
        "labels": audit["label"].value_counts().to_dict() if not audit.empty else {},
        "sfreq_counts": audit["sfreq"].value_counts().sort_index().to_dict() if not audit.empty else {},
        "duration_s_min": float(audit["duration_s"].min()) if not audit.empty else None,
        "duration_s_max": float(audit["duration_s"].max()) if not audit.empty else None,
        "n_eeg_like_min": int(audit["n_eeg_like"].min()) if not audit.empty else None,
        "n_eeg_like_max": int(audit["n_eeg_like"].max()) if not audit.empty else None,
        "most_common_eeg_like_channels": common,
    }
    (args.outdir / "doc_edf_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
