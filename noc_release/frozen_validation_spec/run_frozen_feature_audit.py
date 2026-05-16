from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_feature_file(path: Path) -> dict:
    df = pd.read_csv(path)
    summary: dict[str, object] = {
        "file": str(path),
        "sha256": sha256(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    for col in ["dataset", "band", "condition", "target", "subject"]:
        if col in df.columns:
            summary[f"{col}_values"] = sorted([str(v) for v in df[col].dropna().unique()])
    numeric_cols = [c for c in ["Pi", "delta_Pi", "R_mean", "D_mean", "M_mean", "delta_R", "delta_D", "delta_M"] if c in df.columns]
    summary["numeric_summary"] = {
        col: {
            "mean": float(np.nanmean(df[col])),
            "sd": float(np.nanstd(df[col], ddof=1)),
            "min": float(np.nanmin(df[col])),
            "max": float(np.nanmax(df[col])),
        }
        for col in numeric_cols
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feature CSVs against the frozen GCC validation spec.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    audit = {
        "spec_file": str(args.spec),
        "spec_sha256": sha256(args.spec),
        "spec_version": spec.get("version"),
        "audited_files": [summarize_feature_file(p) for p in args.features],
        "locked_primary_variant": spec["gcc_observables"]["primary_variant"],
        "locked_window_s": spec["gcc_observables"]["window_s"],
        "locked_stride_s": spec["gcc_observables"]["stride_s"],
        "locked_alpha_quantile": spec["thresholding"]["alpha_quantile"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_files": len(args.features)}, indent=2))


if __name__ == "__main__":
    main()
