from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-subjects", type=int, default=20)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    subjects = list(range(args.n_subjects))
    records = [1, 2]
    files = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=subjects,
        recording=records,
        path=args.outdir,
        on_missing="warn",
        verbose=True,
    )
    flat = [{"psg": str(pair[0]), "hypnogram": str(pair[1])} for pair in files]
    (args.outdir / "sleep_edf_fetch_manifest.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
    print(json.dumps({"requested_subjects": subjects, "recordings": len(flat), "outdir": str(args.outdir)}, indent=2))


if __name__ == "__main__":
    main()
