#!/usr/bin/env python
"""DS005620 repeated-run stability and spectrally closest-run controls.

This analysis uses already computed GCC recording summaries and spectral
recording features. It does not reprocess raw EEG.

Main questions:
1. How stable is Pi across repeated sed/sed2 runs within the same subject?
2. Does awake-vs-sedated Pi separation remain when selecting, for each subject,
   the sedated run that is spectrally closest to the awake recording?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


SPECTRAL_COLS = [
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "alpha_gamma_ratio",
    "spectral_entropy",
]


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_band_summary(path: Path, band: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"task": "condition"})
    df["subject"] = df["subject"].astype(str)
    df["condition"] = df["condition"].replace({"awake": "awake"})
    df["band"] = band
    return coerce_numeric(df, ["run", "Pi", "Access_all", "R_mean", "D_mean", "M_mean"])


def merge_with_spectral(gcc: pd.DataFrame, spectral_path: Path) -> pd.DataFrame:
    spec = pd.read_csv(spectral_path)
    spec["subject"] = spec["subject"].astype(str)
    spec = coerce_numeric(spec, SPECTRAL_COLS)
    cols = ["subject", "condition", "filename", *SPECTRAL_COLS]
    merged = gcc.merge(spec[cols], on=["subject", "condition", "filename"], how="inner")
    return merged


def paired_stats(wide: pd.DataFrame, target: str, metric: str = "Pi") -> dict:
    sub = wide[wide["condition"].isin(["awake", target])]
    pivot = sub.pivot(index="subject", columns="condition", values=metric).dropna()
    if len(pivot) < 5:
        return {"n": int(len(pivot))}
    awake = pivot["awake"].to_numpy()
    sed = pivot[target].to_numpy()
    diff = awake - sed
    _, p_t = stats.ttest_rel(awake, sed)
    p_w = stats.wilcoxon(diff, alternative="greater").pvalue
    sd = float(np.std(diff, ddof=1))
    auc = float(roc_auc_score(np.r_[np.ones_like(awake), np.zeros_like(sed)], np.r_[awake, sed]))
    return {
        "n": int(len(pivot)),
        "awake_mean": float(np.mean(awake)),
        "target_mean": float(np.mean(sed)),
        "mean_delta_awake_minus_target": float(np.mean(diff)),
        "ci95_low": float(np.quantile([np.mean(np.random.default_rng(20260513 + i).choice(diff, size=len(diff), replace=True)) for i in range(5000)], 0.025)),
        "ci95_high": float(np.quantile([np.mean(np.random.default_rng(20270513 + i).choice(diff, size=len(diff), replace=True)) for i in range(5000)], 0.975)),
        "paired_d": float(np.mean(diff) / sd) if sd > 0 else np.nan,
        "ttest_p": float(p_t),
        "wilcoxon_greater_p": float(p_w),
        "auc": auc,
    }


def run_stability(df: pd.DataFrame) -> list[dict]:
    rows = []
    for band, band_df in df.groupby("band"):
        for condition in ["sed", "sed2"]:
            sub = band_df[band_df["condition"] == condition].copy()
            run_counts = sub.groupby("subject")["Pi"].count()
            repeat_sub = sub[sub["subject"].isin(run_counts[run_counts >= 2].index)]
            per_subject = repeat_sub.groupby("subject")["Pi"].agg(["count", "mean", "std", "min", "max"]).reset_index()
            if per_subject.empty:
                continue
            rows.append(
                {
                    "band": band,
                    "condition": condition,
                    "n_subjects_with_repeats": int(len(per_subject)),
                    "mean_within_subject_sd": float(per_subject["std"].mean()),
                    "median_within_subject_sd": float(per_subject["std"].median()),
                    "mean_within_subject_range": float((per_subject["max"] - per_subject["min"]).mean()),
                    "median_within_subject_range": float((per_subject["max"] - per_subject["min"]).median()),
                }
            )
    return rows


def select_spectrally_closest(df: pd.DataFrame, spectral_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    distances = []
    complete = df.dropna(subset=spectral_cols).copy()
    scaler = StandardScaler()
    complete[[f"z_{c}" for c in spectral_cols]] = scaler.fit_transform(complete[spectral_cols])
    zcols = [f"z_{c}" for c in spectral_cols]
    for (band, subject), sub in complete.groupby(["band", "subject"]):
        awake = sub[sub["condition"] == "awake"]
        if awake.empty:
            continue
        awake_row = awake.iloc[0]
        rows.append(awake_row)
        awake_vec = awake_row[zcols].to_numpy(dtype=float)
        for target in ["sed", "sed2"]:
            cand = sub[sub["condition"] == target].copy()
            if cand.empty:
                continue
            dist = np.linalg.norm(cand[zcols].to_numpy(dtype=float) - awake_vec, axis=1)
            idx = int(np.argmin(dist))
            chosen = cand.iloc[idx].copy()
            chosen["spectral_distance_to_awake"] = float(dist[idx])
            rows.append(chosen)
            distances.append(
                {
                    "band": band,
                    "subject": subject,
                    "target": target,
                    "chosen_filename": chosen["filename"],
                    "spectral_distance_to_awake": float(dist[idx]),
                    "n_candidate_runs": int(len(cand)),
                }
            )
    matched = pd.DataFrame(rows)
    return matched, pd.DataFrame(distances)


def plot_matching(matched: pd.DataFrame, outdir: Path) -> None:
    for band, sub in matched.groupby("band"):
        fig, ax = plt.subplots(figsize=(7, 4))
        order = ["awake", "sed", "sed2"]
        vals = [sub[sub["condition"] == c]["Pi"].to_numpy() for c in order if c in set(sub["condition"])]
        labels = [c for c in order if c in set(sub["condition"])]
        ax.boxplot(vals, labels=labels, showmeans=True)
        ax.set_title(f"DS005620 spectrally closest-run control ({band})")
        ax.set_ylabel("Pi")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"ds005620_spectrally_closest_run_{band}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-summary", type=Path, required=True)
    parser.add_argument("--gamma-summary", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    alpha = load_band_summary(args.alpha_summary, "alpha")
    gamma = load_band_summary(args.gamma_summary, "gamma")
    df = pd.concat([alpha, gamma], ignore_index=True)
    merged = merge_with_spectral(df, args.spectral)
    merged.to_csv(args.outdir / "ds005620_recording_gcc_spectral_merged.csv", index=False)

    stability = run_stability(merged)
    pd.DataFrame(stability).to_csv(args.outdir / "ds005620_repeated_run_stability.csv", index=False)

    matched, distances = select_spectrally_closest(merged, SPECTRAL_COLS)
    matched.to_csv(args.outdir / "ds005620_spectrally_closest_runs.csv", index=False)
    distances.to_csv(args.outdir / "ds005620_spectrally_closest_distances.csv", index=False)
    plot_matching(matched, args.outdir)

    stats_rows = []
    for band, sub in matched.groupby("band"):
        for target in ["sed", "sed2"]:
            res = paired_stats(sub, target, "Pi")
            stats_rows.append({"band": band, "target": target, **res})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(args.outdir / "ds005620_spectrally_closest_stats.csv", index=False)

    summary = {
        "run_stability": stability,
        "spectrally_closest_stats": stats_df.to_dict(orient="records"),
        "interpretation": (
            "This control minimizes spectral distance between each subject's awake "
            "recording and available sedated runs. It does not prove spectral "
            "independence, but tests whether the Pi effect survives the most "
            "spectrally similar local sedated recording."
        ),
    }
    with open(args.outdir / "ds005620_run_stability_matching_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:12000])


if __name__ == "__main__":
    main()
