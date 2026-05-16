#!/usr/bin/env python
"""Statistical summaries and baselines for DS005620 GCC subset results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METRICS = ["R_mean", "D_mean", "M_mean", "Pi", "Access_all"]


def paired_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def condition_label(row: pd.Series) -> str:
    if row["task"] == "awake":
        return "awake"
    if row["task"] == "sed":
        return "sed"
    if row["task"] == "sed2":
        return "sed2"
    return str(row["task"])


def subject_condition_means(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["condition"] = work.apply(condition_label, axis=1)
    return work.groupby(["subject", "condition"])[METRICS].mean().reset_index()


def paired_tests(sc: pd.DataFrame) -> dict:
    out = {}
    for target in ["sed", "sed2"]:
        wide = sc[sc["condition"].isin(["awake", target])].pivot(index="subject", columns="condition", values=METRICS)
        target_out = {}
        for metric in METRICS:
            sub = wide[metric].dropna()
            if len(sub) < 3:
                continue
            awake = sub["awake"].to_numpy()
            sed = sub[target].to_numpy()
            test = stats.wilcoxon(awake, sed, zero_method="wilcox", alternative="two-sided")
            target_out[metric] = {
                "n_subjects": int(len(sub)),
                "awake_mean": float(awake.mean()),
                f"{target}_mean": float(sed.mean()),
                f"awake_minus_{target}_mean": float((awake - sed).mean()),
                f"awake_minus_{target}_median": float(np.median(awake - sed)),
                "paired_d": paired_d(awake, sed),
                "wilcoxon_W": float(test.statistic),
                "wilcoxon_p": float(test.pvalue),
            }
        out[f"awake_vs_{target}"] = target_out
    return out


def logo_auc(df: pd.DataFrame, target: str, features: list[str]) -> float | None:
    work = df.copy()
    work["condition"] = work.apply(condition_label, axis=1)
    sub = work[work["condition"].isin(["awake", target])].copy()
    if sub["subject"].nunique() < 3:
        return None
    y = (sub["condition"] == "awake").astype(int).to_numpy()
    x = sub[features].to_numpy()
    groups = sub["subject"].astype(str).to_numpy()
    scores = np.full(len(sub), np.nan)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=31),
        )
        clf.fit(x[train], y[train])
        scores[test] = clf.predict_proba(x[test])[:, 1]
    mask = np.isfinite(scores)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], scores[mask]))


def logo_multiclass(df: pd.DataFrame, features: list[str]) -> dict[str, float | None]:
    work = df.copy()
    work["condition"] = work.apply(condition_label, axis=1)
    sub = work[work["condition"].isin(["awake", "sed", "sed2"])].copy()
    labels = ["awake", "sed", "sed2"]
    y = pd.Categorical(sub["condition"], categories=labels).codes
    x = sub[features].to_numpy()
    groups = sub["subject"].astype(str).to_numpy()
    pred = np.full(len(sub), -1)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 3:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=31),
        )
        clf.fit(x[train], y[train])
        pred[test] = clf.predict(x[test])
    mask = pred >= 0
    if mask.sum() < 20:
        return {"accuracy": None, "macro_f1": None}
    return {
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }


def cv_baselines(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["log_M"] = np.log10(work["M_mean"] + 1e-12)
    feature_sets = {
        "R_only": ["R_mean"],
        "D_only": ["D_mean"],
        "M_only": ["log_M"],
        "Pi_only": ["Pi"],
        "Access_all_only": ["Access_all"],
        "GCC_triad": ["R_mean", "D_mean", "log_M"],
        "GCC_triad_plus_Pi": ["R_mean", "D_mean", "log_M", "Pi"],
        "GCC_all": ["R_mean", "D_mean", "log_M", "Pi", "Access_all"],
    }
    out = {}
    for name, features in feature_sets.items():
        out[name] = {
            "awake_vs_sed_auc": logo_auc(work, "sed", features),
            "awake_vs_sed2_auc": logo_auc(work, "sed2", features),
            "three_condition_multiclass": logo_multiclass(work, features),
        }
    return out


def plot_subject_means(sc: pd.DataFrame, band: str, outdir: Path) -> None:
    order = ["awake", "sed", "sed2"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4))
    for ax, metric in zip(axes, METRICS):
        data = [sc.loc[sc["condition"] == cond, metric].to_numpy() for cond in order]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=order)
        for box, color in zip(bp["boxes"], ["#26547C", "#EF476F", "#F4A261"]):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        for subject, sub in sc.groupby("subject"):
            sub = sub.set_index("condition").reindex(order)
            ax.plot(range(1, len(order) + 1), sub[metric], color="black", alpha=0.18, linewidth=0.8)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"DS005620 subject-level condition means: {band}")
    fig.tight_layout()
    fig.savefig(outdir / f"ds005620_{band}_subject_condition_means.png", dpi=180)
    plt.close(fig)


def process(csv_path: Path, band: str, outdir: Path) -> dict:
    df = pd.read_csv(csv_path)
    sc = subject_condition_means(df)
    sc.to_csv(outdir / f"ds005620_{band}_subject_condition_means.csv", index=False)
    plot_subject_means(sc, band, outdir)
    return {
        "band": band,
        "n_recordings": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "recording_counts": df.assign(condition=df.apply(condition_label, axis=1)).groupby("condition").size().astype(int).to_dict(),
        "condition_means_subject_level": sc.groupby("condition")[METRICS].mean().round(6).to_dict(orient="index"),
        "paired_tests": paired_tests(sc),
        "cross_validated_baselines_recording_level": cv_baselines(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha-csv", type=Path, required=True)
    parser.add_argument("--gamma-csv", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "alpha": process(args.alpha_csv, "alpha", args.outdir),
        "gamma": process(args.gamma_csv, "gamma", args.outdir),
    }
    with open(args.outdir / "ds005620_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:10000])


if __name__ == "__main__":
    main()
