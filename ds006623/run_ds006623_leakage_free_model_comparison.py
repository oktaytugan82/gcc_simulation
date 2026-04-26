"""Leakage-free ds006623 model comparison for fMRI GCC validation.

Models are evaluated with leave-one-subject-out cross-validation. Scaling,
imputation, and logistic-regression fitting happen inside each training fold.

Feature blocks:
* confounds: motion + propofol effect-site/plasma estimates
* standard_fc: confounds + mean FC / mean absolute FC
* gcc_core: confounds + R_phase, D_eff, M_tau
* gcc_regime: confounds + R_phase, D_eff, M_tau, Pi_fMRI
* full: confounds + standard FC + GCC regime features

Primary reviewer-relevant tasks:
* prelor_vs_lor_task2: within the same sedation run, responsive pre-LOR vs LOR
* lor_vs_ror_task3: within the recovery run, LOR vs ROR

Base1 comparisons are reported as sanity checks, not as the strongest evidence,
because Pi_fMRI is calibrated on Base1 by design.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_BLOCKS: dict[str, list[str]] = {
    "confounds_motion_propofol": ["fd_mean", "fd_max", "esc_mean", "esc_max", "plas_mean", "plas_max"],
    "standard_fc_plus_confounds": [
        "fd_mean",
        "fd_max",
        "esc_mean",
        "esc_max",
        "plas_mean",
        "plas_max",
        "mean_fc",
        "mean_abs_fc",
    ],
    "gcc_core_plus_confounds": [
        "fd_mean",
        "fd_max",
        "esc_mean",
        "esc_max",
        "plas_mean",
        "plas_max",
        "R_phase_mean",
        "D_eff_mean",
        "M_tau_mean",
    ],
    "gcc_regime_plus_confounds": [
        "fd_mean",
        "fd_max",
        "esc_mean",
        "esc_max",
        "plas_mean",
        "plas_max",
        "R_phase_mean",
        "D_eff_mean",
        "M_tau_mean",
        "Pi_fMRI",
    ],
    "full_standard_plus_gcc": [
        "fd_mean",
        "fd_max",
        "esc_mean",
        "esc_max",
        "plas_mean",
        "plas_max",
        "mean_fc",
        "mean_abs_fc",
        "R_phase_mean",
        "D_eff_mean",
        "M_tau_mean",
        "Pi_fMRI",
    ],
}


TASKS: dict[str, dict[str, object]] = {
    "awake_base1_vs_lor": {
        "phases": ["Base1", "LOR_task2", "LOR_task3"],
        "positive": ["LOR_task2", "LOR_task3"],
        "paired_required": False,
        "note": "sanity check; Base1 is calibration source",
    },
    "responsive_vs_lor_all_phases": {
        "phases": ["Base1", "PreLOR", "LOR_task2", "LOR_task3", "ROR", "Base2"],
        "positive": ["LOR_task2", "LOR_task3"],
        "paired_required": False,
        "note": "broad responsiveness contrast across all phases",
    },
    "prelor_vs_lor_task2": {
        "phases": ["PreLOR", "LOR_task2"],
        "positive": ["LOR_task2"],
        "paired_required": True,
        "note": "primary: same run, pre-loss versus loss of responsiveness",
    },
    "lor_vs_ror_task3": {
        "phases": ["LOR_task3", "ROR"],
        "positive": ["LOR_task3"],
        "paired_required": True,
        "note": "primary: same run, loss versus recovery of responsiveness",
    },
    "base1_vs_base2_control": {
        "phases": ["Base1", "Base2"],
        "positive": ["Base2"],
        "paired_required": True,
        "note": "control: non-primary post-session/run-order contrast",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path("data") / "ds006623-minimal"))
    parser.add_argument(
        "--window-features",
        default=str(Path("results") / "ds006623_fmri_gcc_window_features.csv"),
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--output-prefix", default="ds006623_leakage_free")
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=20260426)
    parser.add_argument(
        "--permutation-tasks",
        default="prelor_vs_lor_task2,lor_vs_ror_task3,responsive_vs_lor_all_phases",
        help="Comma-separated task names for subject-wise label-permutation controls.",
    )
    parser.add_argument(
        "--permutation-models",
        default=",".join(FEATURE_BLOCKS.keys()),
        help="Comma-separated model names for permutation controls.",
    )
    return parser.parse_args()


def read_1d(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=float)
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
    return np.asarray(values, dtype=float)


def window_drug_summary(values: np.ndarray, start: int, stop: int) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    start = max(0, min(start, values.size))
    stop = max(start, min(stop, values.size))
    if stop <= start:
        return math.nan, math.nan
    segment = values[start:stop]
    return float(np.nanmean(segment)), float(np.nanmax(segment))


def add_propofol_features(window_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    esc_cache: dict[tuple[str, int, str], np.ndarray] = {}
    rows: list[dict[str, float]] = []
    for _, row in window_df.iterrows():
        subject = str(row["subject"])
        run = int(row["run"])
        start = int(row["window_start_tr"])
        stop = int(row["window_stop_tr"])
        summary: dict[str, float] = {}
        for kind, prefix in [("ESC", "esc"), ("PLAS", "plas")]:
            key = (subject, run, kind)
            if key not in esc_cache:
                path = data_root / "derivatives" / "Propofol_Infusion" / subject / f"{subject}_task{run}_{kind}.1D"
                esc_cache[key] = read_1d(path)
            mean, max_value = window_drug_summary(esc_cache[key], start, stop)
            summary[f"{prefix}_mean"] = mean
            summary[f"{prefix}_max"] = max_value
        rows.append(summary)
    drug_df = pd.DataFrame(rows, index=window_df.index)
    return pd.concat([window_df.reset_index(drop=True), drug_df.reset_index(drop=True)], axis=1)


def aggregate_phase_features(window_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject", "run", "phase"]
    agg = {
        "inside_baseline_regime": "mean",
        "R_phase": "mean",
        "D_eff": "mean",
        "M_tau": "mean",
        "mean_fc": "mean",
        "mean_abs_fc": "mean",
        "fd_mean": "mean",
        "fd_max": "max",
        "esc_mean": "mean",
        "esc_max": "max",
        "plas_mean": "mean",
        "plas_max": "max",
        "window_start_tr": "count",
    }
    phase = window_df.groupby(group_cols, as_index=False).agg(agg)
    phase = phase.rename(
        columns={
            "inside_baseline_regime": "Pi_fMRI",
            "R_phase": "R_phase_mean",
            "D_eff": "D_eff_mean",
            "M_tau": "M_tau_mean",
            "window_start_tr": "n_windows",
        }
    )
    return phase


def build_task_frame(phase_df: pd.DataFrame, task_name: str, spec: dict[str, object]) -> pd.DataFrame:
    phases = list(spec["phases"])
    positive = set(spec["positive"])
    data = phase_df[phase_df["phase"].isin(phases)].copy()
    if bool(spec.get("paired_required", False)):
        counts = data.groupby("subject")["phase"].nunique()
        complete_subjects = counts[counts == len(phases)].index
        data = data[data["subject"].isin(complete_subjects)].copy()
    data["target"] = data["phase"].isin(positive).astype(int)
    data["task_name"] = task_name
    return data.sort_values(["subject", "phase"]).reset_index(drop=True)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=500,
                    random_state=13,
                ),
            ),
        ]
    )


def loso_predict(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    subjects = sorted(data["subject"].unique())
    for subject in subjects:
        train = data[data["subject"] != subject]
        test = data[data["subject"] == subject]
        if train["target"].nunique() < 2 or test.empty:
            continue
        model = make_model()
        model.fit(train[features], train["target"])
        proba = model.predict_proba(test[features])[:, 1]
        fold = test[["subject", "run", "phase", "target", "task_name"]].copy()
        fold["probability"] = proba
        fold["prediction"] = (proba >= 0.5).astype(int)
        predictions.append(fold)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return math.nan
    return float(average_precision_score(y, p))


def compute_metrics(pred: pd.DataFrame, eps: float) -> dict[str, float]:
    y = pred["target"].to_numpy(dtype=int)
    p = np.clip(pred["probability"].to_numpy(dtype=float), eps, 1 - eps)
    yhat = pred["prediction"].to_numpy(dtype=int)
    return {
        "n_rows": int(len(pred)),
        "n_subjects": int(pred["subject"].nunique()),
        "positive_rate": float(np.mean(y)),
        "roc_auc": safe_auc(y, p),
        "average_precision": safe_ap(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "accuracy": float(accuracy_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def subject_losses(pred: pd.DataFrame, eps: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, group in pred.groupby("subject"):
        y = group["target"].to_numpy(dtype=int)
        p = np.clip(group["probability"].to_numpy(dtype=float), eps, 1 - eps)
        rows.append(
            {
                "subject": subject,
                "subject_brier": float(np.mean((p - y) ** 2)),
                "subject_log_loss": float(log_loss(y, p, labels=[0, 1])),
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def paired_model_tests(loss_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = sorted(loss_df["model"].unique())
    baselines = ["confounds_motion_propofol", "standard_fc_plus_confounds"]
    for task in sorted(loss_df["task_name"].unique()):
        task_df = loss_df[loss_df["task_name"] == task]
        for baseline in baselines:
            if baseline not in model_names:
                continue
            for model in model_names:
                if model == baseline:
                    continue
                wide = task_df[task_df["model"].isin([baseline, model])].pivot_table(
                    index="subject", columns="model", values="subject_log_loss"
                )
                wide = wide.dropna()
                if len(wide) < 4:
                    continue
                delta = wide[model] - wide[baseline]
                try:
                    stat, p_value = wilcoxon(delta, zero_method="wilcox")
                except ValueError:
                    stat, p_value = math.nan, math.nan
                rows.append(
                    {
                        "task_name": task,
                        "baseline_model": baseline,
                        "comparison_model": model,
                        "n_subjects": int(len(delta)),
                        "mean_log_loss_delta": float(delta.mean()),
                        "median_log_loss_delta": float(delta.median()),
                        "wilcoxon_p": float(p_value) if not math.isnan(p_value) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def make_auc_plot(metrics_df: pd.DataFrame, figure_dir: Path, output_prefix: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    primary_tasks = ["prelor_vs_lor_task2", "lor_vs_ror_task3", "responsive_vs_lor_all_phases"]
    plot_df = metrics_df[metrics_df["task_name"].isin(primary_tasks)].copy()
    if plot_df.empty:
        return None
    model_order = list(FEATURE_BLOCKS)
    task_order = [task for task in primary_tasks if task in set(plot_df["task_name"])]

    x = np.arange(len(task_order))
    width = 0.14
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, model in enumerate(model_order):
        vals = []
        for task in task_order:
            row = plot_df[(plot_df["task_name"] == task) & (plot_df["model"] == model)]
            vals.append(float(row["roc_auc"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 2) * width, vals, width=width, label=model.replace("_", " "))
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Leave-one-subject-out ROC AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace("_", "\n") for task in task_order], fontsize=9)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title("ds006623 leakage-free model comparison")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{output_prefix}_model_comparison_auc.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def permute_targets_within_subject(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    permuted = data.copy()
    for _, group in data.groupby("subject"):
        idx = group.index.to_numpy()
        labels = group["target"].to_numpy(dtype=int).copy()
        rng.shuffle(labels)
        permuted.loc[idx, "target"] = labels
    return permuted


def permutation_p_values(metrics_df: pd.DataFrame, permutation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if metrics_df.empty or permutation_df.empty:
        return pd.DataFrame()
    for _, obs in metrics_df.iterrows():
        subset = permutation_df[
            (permutation_df["task_name"] == obs["task_name"]) & (permutation_df["model"] == obs["model"])
        ]
        if subset.empty:
            continue
        n = len(subset)
        rows.append(
            {
                "task_name": obs["task_name"],
                "model": obs["model"],
                "n_permutations": int(n),
                "observed_roc_auc": float(obs["roc_auc"]),
                "auc_permutation_p": float((np.sum(subset["roc_auc"] >= obs["roc_auc"]) + 1) / (n + 1)),
                "observed_balanced_accuracy": float(obs["balanced_accuracy"]),
                "balanced_accuracy_permutation_p": float(
                    (np.sum(subset["balanced_accuracy"] >= obs["balanced_accuracy"]) + 1) / (n + 1)
                ),
                "observed_log_loss": float(obs["log_loss"]),
                "log_loss_permutation_p": float((np.sum(subset["log_loss"] <= obs["log_loss"]) + 1) / (n + 1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    window_df = pd.read_csv(args.window_features)
    augmented_window = add_propofol_features(window_df, data_root)
    augmented_phase = aggregate_phase_features(augmented_window)

    window_aug_path = output_dir / f"{args.output_prefix}_window_features_with_propofol.csv"
    phase_aug_path = output_dir / f"{args.output_prefix}_phase_features_with_propofol.csv"
    augmented_window.to_csv(window_aug_path, index=False)
    augmented_phase.to_csv(phase_aug_path, index=False)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    loss_rows: list[pd.DataFrame] = []
    permutation_rows: list[dict[str, object]] = []
    task_sizes: dict[str, dict[str, object]] = {}
    permutation_tasks = {item.strip() for item in args.permutation_tasks.split(",") if item.strip()}
    permutation_models = {item.strip() for item in args.permutation_models.split(",") if item.strip()}
    rng = np.random.default_rng(args.permutation_seed)

    for task_name, spec in TASKS.items():
        task_df = build_task_frame(augmented_phase, task_name, spec)
        task_sizes[task_name] = {
            "rows": int(len(task_df)),
            "subjects": int(task_df["subject"].nunique()) if not task_df.empty else 0,
            "positive_rate": float(task_df["target"].mean()) if not task_df.empty else math.nan,
            "note": spec["note"],
        }
        if task_df.empty or task_df["target"].nunique() < 2:
            continue
        for model_name, features in FEATURE_BLOCKS.items():
            pred = loso_predict(task_df, features)
            if pred.empty:
                continue
            pred["model"] = model_name
            prediction_rows.append(pred)
            metrics = compute_metrics(pred, args.clip_eps)
            metrics.update({"task_name": task_name, "model": model_name, "features": ",".join(features)})
            metrics_rows.append(metrics)
            losses = subject_losses(pred, args.clip_eps)
            losses["task_name"] = task_name
            losses["model"] = model_name
            loss_rows.append(losses)

            if args.permutations > 0 and task_name in permutation_tasks and model_name in permutation_models:
                for perm_id in range(1, args.permutations + 1):
                    perm_task_df = permute_targets_within_subject(task_df, rng)
                    perm_pred = loso_predict(perm_task_df, features)
                    if perm_pred.empty:
                        continue
                    perm_metrics = compute_metrics(perm_pred, args.clip_eps)
                    permutation_rows.append(
                        {
                            "task_name": task_name,
                            "model": model_name,
                            "permutation": perm_id,
                            **perm_metrics,
                        }
                    )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    loss_df = pd.concat(loss_rows, ignore_index=True) if loss_rows else pd.DataFrame()
    permutation_df = pd.DataFrame(permutation_rows)
    tests_df = paired_model_tests(loss_df) if not loss_df.empty else pd.DataFrame()
    permutation_p_df = permutation_p_values(metrics_df, permutation_df)

    metrics_path = output_dir / f"{args.output_prefix}_model_metrics.csv"
    predictions_path = output_dir / f"{args.output_prefix}_model_predictions.csv"
    loss_path = output_dir / f"{args.output_prefix}_subject_losses.csv"
    tests_path = output_dir / f"{args.output_prefix}_model_delta_tests.csv"
    permutation_path = output_dir / f"{args.output_prefix}_permutation_metrics.csv"
    permutation_p_path = output_dir / f"{args.output_prefix}_permutation_pvalues.csv"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    loss_df.to_csv(loss_path, index=False)
    tests_df.to_csv(tests_path, index=False)
    permutation_df.to_csv(permutation_path, index=False)
    permutation_p_df.to_csv(permutation_p_path, index=False)
    figure_path = make_auc_plot(metrics_df, figure_dir, args.output_prefix)

    summary = {
        "analysis": "ds006623 leakage-free leave-one-subject-out phase-level model comparison",
        "data_root": str(data_root.resolve()),
        "window_augmented": str(window_aug_path.resolve()),
        "phase_augmented": str(phase_aug_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "subject_losses": str(loss_path.resolve()),
        "model_delta_tests": str(tests_path.resolve()),
        "permutation_metrics": str(permutation_path.resolve()),
        "permutation_pvalues": str(permutation_p_path.resolve()),
        "figure": str(figure_path.resolve()) if figure_path else None,
        "permutations": args.permutations,
        "permutation_tasks": sorted(permutation_tasks),
        "permutation_models": sorted(permutation_models),
        "feature_blocks": FEATURE_BLOCKS,
        "task_sizes": task_sizes,
        "topline": metrics_df.sort_values(["task_name", "roc_auc"], ascending=[True, False])[
            ["task_name", "model", "n_rows", "n_subjects", "roc_auc", "balanced_accuracy", "log_loss", "brier"]
        ].to_dict(orient="records")
        if not metrics_df.empty
        else [],
    }
    summary_path = output_dir / "ds006623_leakage_free_model_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
