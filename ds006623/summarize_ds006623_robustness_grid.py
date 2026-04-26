"""Summarize ds006623 robustness analyses across GSR and atlas choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COMBOS = ["withoutGSR_4S156", "withoutGSR_4S256", "withGSR_4S156", "withGSR_4S256"]
PRIMARY_TASKS = ["prelor_vs_lor_task2", "lor_vs_ror_task3", "responsive_vs_lor_all_phases"]
MODELS = [
    "confounds_motion_propofol",
    "standard_fc_plus_confounds",
    "gcc_core_plus_confounds",
    "gcc_regime_plus_confounds",
    "full_standard_plus_gcc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--output-prefix", default="ds006623_robustness_grid")
    return parser.parse_args()


def read_combo(results_dir: Path, combo: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = f"ds006623_{combo}_leakage_free"
    metrics = pd.read_csv(results_dir / f"{prefix}_model_metrics.csv")
    pvals = pd.read_csv(results_dir / f"{prefix}_permutation_pvalues.csv")
    deltas = pd.read_csv(results_dir / f"{prefix}_model_delta_tests.csv")
    for frame in (metrics, pvals, deltas):
        frame["combo"] = combo
        frame["gsr"] = "with_GSR" if combo.startswith("withGSR") else "without_GSR"
        frame["atlas"] = combo.split("_")[-1]
    return metrics, pvals, deltas


def make_plots(metrics: pd.DataFrame, output_prefix: str, figures_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    plot_df = metrics[
        metrics["task_name"].isin(PRIMARY_TASKS)
        & metrics["model"].isin(["confounds_motion_propofol", "standard_fc_plus_confounds", "gcc_core_plus_confounds"])
    ].copy()
    if not plot_df.empty:
        task_labels = {
            "prelor_vs_lor_task2": "PreLOR vs LOR",
            "lor_vs_ror_task3": "LOR vs ROR",
            "responsive_vs_lor_all_phases": "All-phase responsive vs LOR",
        }
        model_labels = {
            "confounds_motion_propofol": "Motion+propofol",
            "standard_fc_plus_confounds": "+ standard FC",
            "gcc_core_plus_confounds": "+ GCC core",
        }
        for task in PRIMARY_TASKS:
            sub = plot_df[plot_df["task_name"] == task]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.2))
            x_labels = COMBOS
            x = range(len(x_labels))
            width = 0.24
            for i, model in enumerate(model_labels):
                values = []
                for combo in x_labels:
                    row = sub[(sub["combo"] == combo) & (sub["model"] == model)]
                    values.append(float(row["roc_auc"].iloc[0]) if not row.empty else float("nan"))
                ax.bar([v + (i - 1) * width for v in x], values, width=width, label=model_labels[model])
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(0.9, 1.01)
            ax.set_ylabel("LOSO ROC AUC")
            ax.set_xticks(list(x))
            ax.set_xticklabels([label.replace("_", "\n") for label in x_labels], fontsize=8)
            ax.set_title(task_labels.get(task, task))
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            path = figures_dir / f"{output_prefix}_{task}_auc.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(str(path.resolve()))
    return paths


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    metrics_frames = []
    pval_frames = []
    delta_frames = []
    for combo in COMBOS:
        metrics, pvals, deltas = read_combo(results_dir, combo)
        metrics_frames.append(metrics)
        pval_frames.append(pvals)
        delta_frames.append(deltas)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    pvals = pd.concat(pval_frames, ignore_index=True)
    deltas = pd.concat(delta_frames, ignore_index=True)
    merged = metrics.merge(
        pvals[
            [
                "combo",
                "task_name",
                "model",
                "n_permutations",
                "auc_permutation_p",
                "balanced_accuracy_permutation_p",
                "log_loss_permutation_p",
            ]
        ],
        on=["combo", "task_name", "model"],
        how="left",
    )

    primary = merged[merged["task_name"].isin(PRIMARY_TASKS) & merged["model"].isin(MODELS)].copy()
    primary_path = results_dir / f"{args.output_prefix}_primary_metrics.csv"
    delta_path = results_dir / f"{args.output_prefix}_delta_tests.csv"
    pval_path = results_dir / f"{args.output_prefix}_permutation_pvalues.csv"
    primary.to_csv(primary_path, index=False)
    deltas.to_csv(delta_path, index=False)
    pvals.to_csv(pval_path, index=False)

    best = (
        primary.sort_values(["combo", "task_name", "roc_auc"], ascending=[True, True, False])
        .groupby(["combo", "task_name"], as_index=False)
        .head(1)
    )
    best_path = results_dir / f"{args.output_prefix}_best_models.csv"
    best.to_csv(best_path, index=False)

    figures = make_plots(primary, args.output_prefix, figures_dir)

    improvement_rows = []
    for combo in COMBOS:
        for task in PRIMARY_TASKS:
            sub = primary[(primary["combo"] == combo) & (primary["task_name"] == task)]
            conf = sub[sub["model"] == "confounds_motion_propofol"]
            std = sub[sub["model"] == "standard_fc_plus_confounds"]
            gcc = sub[sub["model"] == "gcc_core_plus_confounds"]
            if conf.empty or std.empty or gcc.empty:
                continue
            improvement_rows.append(
                {
                    "combo": combo,
                    "task_name": task,
                    "gcc_core_auc_minus_confounds": float(gcc["roc_auc"].iloc[0] - conf["roc_auc"].iloc[0]),
                    "gcc_core_auc_minus_standard_fc": float(gcc["roc_auc"].iloc[0] - std["roc_auc"].iloc[0]),
                    "gcc_core_log_loss_minus_confounds": float(gcc["log_loss"].iloc[0] - conf["log_loss"].iloc[0]),
                    "gcc_core_log_loss_minus_standard_fc": float(gcc["log_loss"].iloc[0] - std["log_loss"].iloc[0]),
                }
            )
    improvements = pd.DataFrame(improvement_rows)
    improvements_path = results_dir / f"{args.output_prefix}_gcc_core_improvements.csv"
    improvements.to_csv(improvements_path, index=False)

    improvement_summary_raw = improvements.groupby("task_name").agg(
        {
            "gcc_core_auc_minus_confounds": ["min", "mean", "max"],
            "gcc_core_auc_minus_standard_fc": ["min", "mean", "max"],
            "gcc_core_log_loss_minus_confounds": ["min", "mean", "max"],
            "gcc_core_log_loss_minus_standard_fc": ["min", "mean", "max"],
        }
    ).round(6)
    improvement_summary: dict[str, dict[str, float]] = {}
    for task, row in improvement_summary_raw.iterrows():
        improvement_summary[str(task)] = {
            f"{metric}_{stat}": float(value)
            for (metric, stat), value in row.items()
        }

    summary = {
        "combos": COMBOS,
        "primary_tasks": PRIMARY_TASKS,
        "models": MODELS,
        "primary_metrics": str(primary_path.resolve()),
        "best_models": str(best_path.resolve()),
        "delta_tests": str(delta_path.resolve()),
        "permutation_pvalues": str(pval_path.resolve()),
        "gcc_core_improvements": str(improvements_path.resolve()),
        "figures": figures,
        "best_models_records": best[
            ["combo", "task_name", "model", "roc_auc", "balanced_accuracy", "log_loss", "auc_permutation_p"]
        ].to_dict(orient="records"),
        "gcc_core_improvement_summary": improvement_summary,
    }
    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
