#!/usr/bin/env python
"""Bandpower-independence stress tests for GCC.

This batch asks a deliberately hard question:

Does GCC retain state information when conventional spectral features are
controlled, residualized, or minimized by spectral matching?

The script uses the already generated subject-normalized delta feature table
from Batch 7. It does not try to tune a new biomarker. It compares:

1. spectral features alone;
2. GCC features alone;
3. spectral + GCC features;
4. GCC features residualized against spectral features inside each training
   fold;
5. a no-fit Pi sign rule;
6. spectral-caliper subsets where positive-state samples with the smallest
   spectral shifts are retained.

The intended interpretation is conservative. A robust positive result supports
"incremental information beyond bandpower"; a weak result argues against the
stronger claim "bandpower-independent biomarker".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDPOWER = ["d_theta_power", "d_alpha_power", "d_beta_power", "d_gamma_power"]
SPECTRAL_ALL = BANDPOWER + ["d_alpha_gamma_ratio", "d_spectral_entropy"]
GCC = ["dR", "dD", "dlogM", "dPi"]
PI = ["dPi"]

MODEL_FEATURES = {
    "spectral_bandpower": BANDPOWER,
    "spectral_all": SPECTRAL_ALL,
    "gcc_pi": PI,
    "gcc_triad_plus_pi": GCC,
    "spectral_all_plus_gcc": SPECTRAL_ALL + GCC,
}


def finite_df(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["y", "subject"])
    return out.copy()


def make_clf() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260514),
    )


def auc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def metric_dict(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0.5).astype(int)
    out = {
        "auc": auc_safe(y, score),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
    }
    try:
        clipped = np.clip(score, 1e-6, 1 - 1e-6)
        out["log_loss"] = float(log_loss(y, clipped, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def residualize_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_cols: list[str],
    covariate_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train_cov = train[covariate_cols].to_numpy(dtype=float)
    x_test_cov = test[covariate_cols].to_numpy(dtype=float)
    train_resids = []
    test_resids = []
    for col in target_cols:
        reg = make_pipeline(StandardScaler(), LinearRegression())
        reg.fit(x_train_cov, train[col].to_numpy(dtype=float))
        train_resids.append(train[col].to_numpy(dtype=float) - reg.predict(x_train_cov))
        test_resids.append(test[col].to_numpy(dtype=float) - reg.predict(x_test_cov))
    return np.column_stack(train_resids), np.column_stack(test_resids)


def fit_scores(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    if model == "residual_gcc_after_spectral_all":
        x_train, x_test = residualize_features(train, test, GCC, SPECTRAL_ALL)
    elif model == "residual_pi_after_spectral_all":
        x_train, x_test = residualize_features(train, test, PI, SPECTRAL_ALL)
    else:
        cols = MODEL_FEATURES[model]
        x_train = train[cols].to_numpy(dtype=float)
        x_test = test[cols].to_numpy(dtype=float)
    y_train = train["y"].astype(int).to_numpy()
    clf = make_clf()
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def loso_scores(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    groups = df["subject"].astype(str).to_numpy()
    y = df["y"].astype(int).to_numpy()
    logo = LeaveOneGroupOut()
    for model in models:
        scores = np.full(len(df), np.nan)
        for train_idx, test_idx in logo.split(df, y, groups):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            if train["y"].nunique() < 2:
                continue
            scores[test_idx] = fit_scores(train, test, model)
        for i, score in enumerate(scores):
            if np.isfinite(score):
                rows.append(
                    {
                        "row_index": int(df.index[i]),
                        "subject": str(df.iloc[i]["subject"]),
                        "condition": str(df.iloc[i]["condition"]),
                        "y": int(df.iloc[i]["y"]),
                        "model": model,
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def summarize_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in score_df.groupby("model"):
        rows.append({"model": model, **metric_dict(sub["y"].to_numpy(), sub["score"].to_numpy())})
    return pd.DataFrame(rows)


def bootstrap_auc_delta(
    score_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_boot: int = 600,
    seed: int = 20260514,
) -> dict[str, float]:
    """Cluster bootstrap by subject for AUC(model_a)-AUC(model_b)."""
    wide = score_df.pivot_table(index=["subject", "condition", "y"], columns="model", values="score").reset_index()
    wide = wide.dropna(subset=[model_a, model_b])
    subjects = np.array(sorted(wide["subject"].unique()))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for k, subj in enumerate(sampled):
            part = wide[wide["subject"] == subj].copy()
            part["boot_id"] = k
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        if boot["y"].nunique() < 2:
            continue
        deltas.append(auc_safe(boot["y"], boot[model_a]) - auc_safe(boot["y"], boot[model_b]))
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"delta_auc_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_delta_le_0": float("nan")}
    return {
        "delta_auc_mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_delta_le_0": float(np.mean(arr <= 0)),
        "n_boot": int(arr.size),
    }


def run_loso_and_incremental(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    models = [
        "spectral_bandpower",
        "spectral_all",
        "gcc_pi",
        "gcc_triad_plus_pi",
        "spectral_all_plus_gcc",
        "residual_pi_after_spectral_all",
        "residual_gcc_after_spectral_all",
    ]
    metric_rows = []
    all_scores = []
    bootstrap = {}
    for band in sorted(df["band"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            sub = finite_df(df[(df["band"] == band) & (df["dataset"] == dataset)], SPECTRAL_ALL + GCC)
            if sub.empty:
                continue
            score_df = loso_scores(sub, models)
            score_df.insert(0, "dataset", dataset)
            score_df.insert(1, "band", band)
            all_scores.append(score_df)
            summary = summarize_scores(score_df)
            summary.insert(0, "dataset", dataset)
            summary.insert(1, "band", band)
            summary.insert(2, "scope", "within_dataset_LOSO")
            metric_rows.append(summary)
            key = f"{dataset}_{band}"
            bootstrap[key] = {
                "spectral_all_plus_gcc_minus_spectral_all": bootstrap_auc_delta(
                    score_df, "spectral_all_plus_gcc", "spectral_all"
                ),
                "residual_gcc_minus_spectral_all": bootstrap_auc_delta(
                    score_df, "residual_gcc_after_spectral_all", "spectral_all"
                ),
                "gcc_triad_minus_spectral_all": bootstrap_auc_delta(
                    score_df, "gcc_triad_plus_pi", "spectral_all"
                ),
            }
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    scores.to_csv(outdir / "batch9_loso_scores.csv", index=False)
    metrics.to_csv(outdir / "batch9_loso_metrics.csv", index=False)
    return scores, metrics, bootstrap


def run_cross_dataset(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    models = [
        "spectral_all",
        "gcc_triad_plus_pi",
        "spectral_all_plus_gcc",
        "residual_gcc_after_spectral_all",
        "residual_pi_after_spectral_all",
    ]
    rows = []
    for band in sorted(df["band"].unique()):
        band_df = finite_df(df[df["band"] == band], SPECTRAL_ALL + GCC)
        for train_dataset, test_dataset in [("Chennu", "DS005620"), ("DS005620", "Chennu")]:
            train = band_df[band_df["dataset"] == train_dataset].copy()
            test = band_df[band_df["dataset"] == test_dataset].copy()
            if train.empty or test.empty:
                continue
            for model in models:
                score = fit_scores(train, test, model)
                metrics = metric_dict(test["y"].to_numpy(), score)
                rows.append(
                    {
                        "band": band,
                        "direction": f"{train_dataset}_to_{test_dataset}",
                        "model": model,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "batch9_cross_dataset_metrics.csv", index=False)
    return out


def add_spectral_norms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_norm_all"] = np.nan
    out["spectral_norm_bandpower"] = np.nan
    for (dataset, band), sub_idx in out.groupby(["dataset", "band"]).groups.items():
        sub = out.loc[sub_idx]
        pos = sub[sub["y"] == 1]
        if len(pos) < 2:
            continue
        for cols, norm_col in [(SPECTRAL_ALL, "spectral_norm_all"), (BANDPOWER, "spectral_norm_bandpower")]:
            scaler = StandardScaler()
            scaler.fit(pos[cols].to_numpy(dtype=float))
            z = scaler.transform(sub[cols].to_numpy(dtype=float))
            out.loc[sub_idx, norm_col] = np.linalg.norm(z, axis=1)
    return out


def pi_sign_rule(sub: pd.DataFrame) -> dict[str, float]:
    # Positive state is predicted when Pi declined relative to baseline.
    score = -sub["dPi"].to_numpy(dtype=float)
    return metric_dict(sub["y"].to_numpy(), score)


def spectral_caliper_tests(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Keep positives with smallest spectral shift and add their baselines."""
    df = add_spectral_norms(finite_df(df, SPECTRAL_ALL + GCC))
    rows = []
    models = ["spectral_all", "gcc_triad_plus_pi", "residual_gcc_after_spectral_all"]
    for band in sorted(df["band"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            sub = df[(df["band"] == band) & (df["dataset"] == dataset)].copy()
            pos = sub[sub["y"] == 1].dropna(subset=["spectral_norm_all"]).copy()
            if pos.empty:
                continue
            for q in [0.25, 0.50, 0.75]:
                cutoff = float(pos["spectral_norm_all"].quantile(q))
                keep_pos = pos[pos["spectral_norm_all"] <= cutoff].copy()
                if keep_pos.empty:
                    continue
                parts = []
                # Duplicate each subject baseline for each retained positive sample.
                for _, row in keep_pos.iterrows():
                    base = sub[(sub["subject"].astype(str) == str(row["subject"])) & (sub["y"] == 0)]
                    if base.empty:
                        continue
                    b = base.iloc[[0]].copy()
                    p = row.to_frame().T.copy()
                    pair_id = f"{dataset}_{band}_{row['subject']}_{row['condition']}"
                    b["pair_id"] = pair_id
                    p["pair_id"] = pair_id
                    parts.extend([b, p])
                if not parts:
                    continue
                matched = pd.concat(parts, ignore_index=True)
                if matched["subject"].nunique() < 4 or matched["y"].nunique() < 2:
                    continue
                score_df = loso_scores(matched, models)
                score_summary = summarize_scores(score_df)
                sign = pi_sign_rule(matched)
                rows.append(
                    {
                        "dataset": dataset,
                        "band": band,
                        "caliper_quantile": q,
                        "n_pairs": int(matched["pair_id"].nunique()),
                        "n_subjects": int(matched["subject"].nunique()),
                        "spectral_norm_cutoff": cutoff,
                        "mean_abs_spectral_delta": float(keep_pos[SPECTRAL_ALL].abs().mean().mean()),
                        "model": "pi_sign_rule_no_fit",
                        **sign,
                    }
                )
                for _, mrow in score_summary.iterrows():
                    rows.append(
                        {
                            "dataset": dataset,
                            "band": band,
                            "caliper_quantile": q,
                            "n_pairs": int(matched["pair_id"].nunique()),
                            "n_subjects": int(matched["subject"].nunique()),
                            "spectral_norm_cutoff": cutoff,
                            "mean_abs_spectral_delta": float(keep_pos[SPECTRAL_ALL].abs().mean().mean()),
                            **mrow.to_dict(),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "batch9_spectral_caliper_metrics.csv", index=False)
    return out


def plot_metrics(metrics: pd.DataFrame, outdir: Path) -> None:
    if metrics.empty:
        return
    for (dataset, band), sub in metrics.groupby(["dataset", "band"]):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        order = [
            "spectral_all",
            "spectral_all_plus_gcc",
            "gcc_triad_plus_pi",
            "residual_gcc_after_spectral_all",
            "residual_pi_after_spectral_all",
        ]
        vals = []
        labels = []
        for model in order:
            hit = sub[sub["model"] == model]
            if hit.empty:
                continue
            vals.append(float(hit["auc"].iloc[0]))
            labels.append(model.replace("_", "\n"))
        ax.bar(np.arange(len(vals)), vals, color=["#6B7280", "#DC7F37", "#2563EB", "#0891B2", "#7C3AED"][: len(vals)])
        ax.axhline(0.5, color="black", lw=1, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC")
        ax.set_xticks(np.arange(len(vals)), labels, rotation=0, fontsize=8)
        ax.set_title(f"Batch 9 LOSO bandpower stress: {dataset} {band}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"batch9_loso_auc_{dataset}_{band}.png")
        plt.close(fig)


def write_report(
    outdir: Path,
    loso_metrics: pd.DataFrame,
    cross_metrics: pd.DataFrame,
    caliper_metrics: pd.DataFrame,
    bootstrap: dict,
) -> None:
    lines = []
    lines.append("# Batch 9 Bandpower-Independence Stress Test\n")
    lines.append("Date: 2026-05-14\n")
    lines.append("## Aim\n")
    lines.append(
        "Test whether GCC can be defended as carrying information beyond conventional spectral features. "
        "This batch does not assume success; it explicitly reports when residualized GCC weakens.\n"
    )
    lines.append("## Tests\n")
    lines.append("- Leave-one-subject-out models: spectral features, GCC, spectral+GCC, and residualized GCC.\n")
    lines.append("- Cross-dataset transfer between Chennu and DS005620.\n")
    lines.append("- Spectral-caliper subsets retaining positive samples with the smallest spectral shifts.\n")

    def simple_markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> str:
        show = df[cols].copy().head(max_rows)
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        header = "| " + " | ".join(show.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in show.to_numpy()]
        return "\n".join([header, sep] + body)

    def add_table(title: str, df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> None:
        lines.append(f"\n## {title}\n")
        if df.empty:
            lines.append("No rows.\n")
            return
        lines.append(simple_markdown_table(df, cols, max_rows=max_rows))
        lines.append("\n")

    add_table(
        "Within-Dataset LOSO AUC",
        loso_metrics,
        ["dataset", "band", "model", "auc", "balanced_accuracy", "log_loss", "n"],
    )
    add_table(
        "Cross-Dataset AUC",
        cross_metrics,
        ["band", "direction", "model", "auc", "balanced_accuracy", "log_loss", "n_train", "n_test"],
    )
    add_table(
        "Spectral-Caliper AUC",
        caliper_metrics,
        ["dataset", "band", "caliper_quantile", "n_pairs", "model", "auc", "balanced_accuracy", "mean_abs_spectral_delta"],
    )

    lines.append("\n## Bootstrap Incremental AUC: LOSO\n")
    lines.append("Cluster bootstrap by subject; values are AUC differences.\n")
    boot_rows = []
    for key, vals in bootstrap.items():
        for contrast, res in vals.items():
            boot_rows.append({"cell": key, "contrast": contrast, **res})
    boot_df = pd.DataFrame(boot_rows)
    if not boot_df.empty:
        lines.append(simple_markdown_table(
            boot_df,
            ["cell", "contrast", "delta_auc_mean", "ci_low", "ci_high", "p_delta_le_0", "n_boot"],
            max_rows=80,
        ))
        lines.append("\n")

    lines.append("\n## Interpretation Rule\n")
    lines.append(
        "A credible bandpower-independent claim would require residualized GCC to remain clearly above chance "
        "and/or spectral+GCC to improve robustly over spectral-only models with positive bootstrap intervals. "
        "If this is not observed, the safer claim is incremental or bandpower-aware regime information, not independence.\n"
    )
    (outdir / "BATCH9_BANDPOWER_INDEPENDENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df["subject"] = df["subject"].astype(str)
    loso_scores_df, loso_metrics, bootstrap = run_loso_and_incremental(df, args.outdir)
    cross_metrics = run_cross_dataset(df, args.outdir)
    caliper_metrics = spectral_caliper_tests(df, args.outdir)
    plot_metrics(loso_metrics, args.outdir)
    summary = {
        "inputs": {"features": str(args.features)},
        "bootstrap_incremental_auc": bootstrap,
        "key_outputs": [
            "batch9_loso_metrics.csv",
            "batch9_cross_dataset_metrics.csv",
            "batch9_spectral_caliper_metrics.csv",
            "BATCH9_BANDPOWER_INDEPENDENCE_REPORT.md",
        ],
    }
    (args.outdir / "batch9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.outdir, loso_metrics, cross_metrics, caliper_metrics, bootstrap)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
