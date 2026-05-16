from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS_WINDOWS = {
    "Pi": "Pi_window",
    "R": "R",
    "D_eff": "D_eff",
    "M_tau": "M_tau",
}

METRICS_SUMMARY = {
    "Pi": "Pi",
    "R": "R_mean",
    "D_eff": "D_mean",
    "M_tau": "M_mean",
}


def icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    mat = np.column_stack([x, y]).astype(float)
    mat = mat[np.all(np.isfinite(mat), axis=1)]
    n, k = mat.shape
    if n < 3:
        return np.nan
    grand = np.mean(mat)
    row_means = mat.mean(axis=1)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_err = np.sum((mat - row_means[:, None]) ** 2)
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    if denom == 0:
        return np.nan
    return float((ms_rows - ms_err) / denom)


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 4:
        return np.nan
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan
    if kind == "pearson":
        return float(stats.pearsonr(x[mask], y[mask]).statistic)
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def reliability_row(scope: str, source: str, band: str, metric: str, x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    delta = y - x
    return {
        "scope": scope,
        "source": source,
        "band": band,
        "metric": metric,
        "n": int(len(x)),
        "pearson_r": corr(x, y, "pearson"),
        "spearman_rho": corr(x, y, "spearman"),
        "icc_3_1": icc_3_1(x, y),
        "mean_abs_delta": float(np.nanmean(np.abs(delta))) if len(delta) else np.nan,
        "mean_signed_delta_second_minus_first": float(np.nanmean(delta)) if len(delta) else np.nan,
    }


def split_half_from_windows(path: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    group_cols = [c for c in ["dataset", "subject", "condition", "band", "filename", "run", "eyes", "recording"] if c in df.columns]
    rows = []
    pairs = []
    for key, g in df.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        if len(g) < 6:
            continue
        half = len(g) // 2
        first = g.iloc[:half]
        second = g.iloc[half:]
        base = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        base["source"] = source
        for metric, col in METRICS_WINDOWS.items():
            rows.append({**base, "metric": metric, "split": "first_half", "value": float(first[col].mean())})
            rows.append({**base, "metric": metric, "split": "second_half", "value": float(second[col].mean())})
            pairs.append({**base, "metric": metric, "first": float(first[col].mean()), "second": float(second[col].mean())})
    pairs_df = pd.DataFrame(pairs)
    rel_rows = []
    if not pairs_df.empty:
        for (band, metric), sub in pairs_df.groupby(["band", "metric"]):
            rel_rows.append(reliability_row("split_half", source, str(band), str(metric), sub["first"].to_numpy(float), sub["second"].to_numpy(float)))
    return pd.DataFrame(rel_rows), pairs_df


def repeated_run_reliability(path: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if "run" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df = df[pd.to_numeric(df["run"], errors="coerce").notna()].copy()
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype(int)
    rows = []
    pairs = []
    for (dataset, band, condition), sub in df.groupby(["dataset", "band", "condition"], dropna=False):
        run1 = sub[sub["run"] == 1].set_index("subject")
        run2 = sub[sub["run"] == 2].set_index("subject")
        common = sorted(set(run1.index).intersection(run2.index))
        if len(common) < 4:
            continue
        for metric, col in METRICS_SUMMARY.items():
            x = run1.loc[common, col].to_numpy(float)
            y = run2.loc[common, col].to_numpy(float)
            rows.append(reliability_row(f"run1_vs_run2_{condition}", source, str(band), metric, x, y))
            for subj, xv, yv in zip(common, x, y):
                pairs.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "band": band,
                        "condition": condition,
                        "subject": subj,
                        "metric": metric,
                        "run1": float(xv),
                        "run2": float(yv),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if not np.isfinite(v) else format(float(v), floatfmt))
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--window-file", action="append", nargs=2, metavar=("SOURCE", "PATH"), default=[])
    parser.add_argument("--summary-file", action="append", nargs=2, metavar=("SOURCE", "PATH"), default=[])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rel_tables = []
    split_pairs = []
    repeated_pairs = []
    for source, path_str in args.window_file:
        rel, pairs = split_half_from_windows(Path(path_str), source)
        rel_tables.append(rel)
        split_pairs.append(pairs)
    for source, path_str in args.summary_file:
        rel, pairs = repeated_run_reliability(Path(path_str), source)
        rel_tables.append(rel)
        repeated_pairs.append(pairs)

    reliability = pd.concat([x for x in rel_tables if not x.empty], ignore_index=True)
    split_pairs_df = pd.concat([x for x in split_pairs if not x.empty], ignore_index=True)
    repeated_pairs_df = pd.concat([x for x in repeated_pairs if not x.empty], ignore_index=True) if repeated_pairs else pd.DataFrame()

    reliability.to_csv(args.outdir / "gcc_reliability_summary.csv", index=False)
    split_pairs_df.to_csv(args.outdir / "gcc_split_half_pairs.csv", index=False)
    repeated_pairs_df.to_csv(args.outdir / "gcc_repeated_run_pairs.csv", index=False)

    key = reliability[(reliability["metric"] == "Pi")].copy()
    key = key.sort_values(["scope", "source", "band"])
    report = []
    report.append("# Batch 14 GCC Reliability\n")
    report.append("Date: 2026-05-14\n")
    report.append("## Scope\n")
    report.append(
        "Reliability is estimated from first-half versus second-half windows within recordings and, where available, repeated DS005620 runs. "
        "This is a measurement-stability control, not an additional state-classification result.\n"
    )
    report.append("## Pi Reliability Summary\n")
    report.append(md_table(key))
    report.append("\n\n## Interpretation\n")
    report.append(
        "High split-half reliability supports GCC as a stable recording-level measure. Repeated-run reliability is expected to be lower because physiological state, sedation depth, and acquisition runs can vary within nominal condition labels.\n"
    )
    (args.outdir / "BATCH14_RELIABILITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"rows": int(len(reliability)), "outdir": str(args.outdir)}, indent=2))


if __name__ == "__main__":
    main()
