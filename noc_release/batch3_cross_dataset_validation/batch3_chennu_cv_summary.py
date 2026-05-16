#!/usr/bin/env python
"""Cross-validated summary for Chennu propofol raw-data GCC outputs."""

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


LEVEL_ORDER = ["baseline", "mild", "moderate", "recovery"]


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def paired_stats(df: pd.DataFrame) -> dict:
    wide = df.pivot(index="subject", columns="level", values=["R_mean", "D_mean", "M_mean", "Pi"])
    out: dict[str, dict] = {}
    for metric in ["R_mean", "D_mean", "M_mean", "Pi"]:
        metric_wide = wide[metric].dropna()
        values = [metric_wide[level].to_numpy() for level in LEVEL_ORDER if level in metric_wide]
        friedman = stats.friedmanchisquare(*values) if len(values) == 4 else (np.nan, np.nan)
        base = metric_wide["baseline"].to_numpy()
        moderate = metric_wide["moderate"].to_numpy()
        recovery = metric_wide["recovery"].to_numpy()
        w_bm = stats.wilcoxon(base, moderate, zero_method="wilcox", alternative="two-sided")
        w_mr = stats.wilcoxon(moderate, recovery, zero_method="wilcox", alternative="two-sided")
        out[metric] = {
            "n_subjects": int(len(metric_wide)),
            "baseline_mean": float(np.mean(base)),
            "moderate_mean": float(np.mean(moderate)),
            "recovery_mean": float(np.mean(recovery)),
            "baseline_minus_moderate_mean": float(np.mean(base - moderate)),
            "baseline_minus_moderate_median": float(np.median(base - moderate)),
            "paired_d_baseline_minus_moderate": _cohens_d_paired(base, moderate),
            "wilcoxon_baseline_vs_moderate_W": float(w_bm.statistic),
            "wilcoxon_baseline_vs_moderate_p": float(w_bm.pvalue),
            "wilcoxon_moderate_vs_recovery_W": float(w_mr.statistic),
            "wilcoxon_moderate_vs_recovery_p": float(w_mr.pvalue),
            "friedman_chi2": float(friedman.statistic if hasattr(friedman, "statistic") else friedman[0]),
            "friedman_p": float(friedman.pvalue if hasattr(friedman, "pvalue") else friedman[1]),
        }
    return out


def _logo_auc(df: pd.DataFrame, feature_cols: list[str], positive: str = "baseline", negative: str = "moderate") -> float | None:
    sub = df[df["level"].isin([positive, negative])].copy()
    y = (sub["level"] == positive).astype(int).to_numpy()
    x = sub[feature_cols].to_numpy()
    groups = sub["subject"].astype(str).to_numpy()
    scores = np.full(len(sub), np.nan)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=23),
        )
        clf.fit(x[train], y[train])
        scores[test] = clf.predict_proba(x[test])[:, 1]
    mask = np.isfinite(scores)
    if mask.sum() < 10:
        return None
    return float(roc_auc_score(y[mask], scores[mask]))


def _logo_multiclass(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float | None]:
    sub = df[df["level"].isin(LEVEL_ORDER)].copy()
    y = pd.Categorical(sub["level"], categories=LEVEL_ORDER).codes
    x = sub[feature_cols].to_numpy()
    groups = sub["subject"].astype(str).to_numpy()
    pred = np.full(len(sub), -1)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < len(LEVEL_ORDER):
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=23),
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
        "GCC_triad": ["R_mean", "D_mean", "log_M"],
        "GCC_triad_plus_Pi": ["R_mean", "D_mean", "log_M", "Pi"],
    }
    out = {}
    for name, cols in feature_sets.items():
        out[name] = {
            "baseline_vs_moderate_auc": _logo_auc(work, cols, "baseline", "moderate"),
            "baseline_vs_mild_auc": _logo_auc(work, cols, "baseline", "mild"),
            "moderate_vs_recovery_auc": _logo_auc(work, cols, "recovery", "moderate"),
            "four_level_multiclass": _logo_multiclass(work, cols),
        }
    return out


def plot_band(df: pd.DataFrame, band: str, outdir: Path) -> None:
    order = LEVEL_ORDER
    colors = ["#1F5A7A", "#56A3A6", "#D66A3A", "#577B55"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, metric in zip(axes, ["R_mean", "D_mean", "M_mean", "Pi"]):
        data = [df.loc[df["level"] == lvl, metric].to_numpy() for lvl in order]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=order)
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        for subj, sub in df.groupby("subject"):
            sub = sub.set_index("level").reindex(order)
            ax.plot(range(1, len(order) + 1), sub[metric], color="black", alpha=0.18, linewidth=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Chennu propofol raw EEG: {band} band")
    fig.tight_layout()
    fig.savefig(outdir / f"chennu_{band}_paired_summary.png", dpi=180)
    plt.close(fig)


def process(csv_path: Path, band: str, outdir: Path) -> dict:
    df = pd.read_csv(csv_path)
    out = {
        "band": band,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "level_counts": {k: int(v) for k, v in df["level"].value_counts().to_dict().items()},
        "group_means": df.groupby("level")[["R_mean", "D_mean", "M_mean", "Pi"]].mean().round(6).to_dict(orient="index"),
        "paired_stats": paired_stats(df),
        "cross_validated_baselines": cv_baselines(df),
    }
    plot_band(df, band, outdir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-csv", type=Path, required=True)
    parser.add_argument("--gamma-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "alpha": process(args.alpha_csv, "alpha", args.outdir),
        "gamma": process(args.gamma_csv, "gamma", args.outdir),
    }
    with open(args.outdir / "chennu_raw_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
