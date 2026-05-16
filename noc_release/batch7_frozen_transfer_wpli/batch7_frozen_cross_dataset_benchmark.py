#!/usr/bin/env python
"""NoC-oriented frozen cross-dataset benchmark for GCC.

This script assembles standard GCC, spectral baselines, and wPLI-GCC into a
single binary state-discrimination benchmark. It reports:

1. parameter-free sign-rule performance (`Pi` declines from baseline);
2. fixed logistic train-on-one-dataset / test-on-the-other transfer;
3. baseline comparisons against R-only, D-only, M-only, spectral features,
   and wPLI variants.

The purpose is not to maximize classification performance. The purpose is to
make the evidential status transparent under frozen, reviewer-auditable rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STANDARD_FEATURE_SETS = {
    "R_only": ["dR"],
    "D_only": ["dD"],
    "M_only": ["dlogM"],
    "GCC_Pi_delta": ["dPi"],
    "GCC_triad_delta": ["dR", "dD", "dlogM"],
    "GCC_triad_plus_Pi": ["dR", "dD", "dlogM", "dPi"],
    "spectral_bandpowers": ["d_theta_power", "d_alpha_power", "d_beta_power", "d_gamma_power"],
    "spectral_all": [
        "d_theta_power",
        "d_alpha_power",
        "d_beta_power",
        "d_gamma_power",
        "d_alpha_gamma_ratio",
        "d_spectral_entropy",
    ],
    "spectral_plus_GCC_Pi": [
        "d_theta_power",
        "d_alpha_power",
        "d_beta_power",
        "d_gamma_power",
        "d_alpha_gamma_ratio",
        "d_spectral_entropy",
        "dPi",
    ],
    "spectral_plus_GCC_triad": [
        "d_theta_power",
        "d_alpha_power",
        "d_beta_power",
        "d_gamma_power",
        "d_alpha_gamma_ratio",
        "d_spectral_entropy",
        "dR",
        "dD",
        "dlogM",
        "dPi",
    ],
}

WPLI_FEATURE_SETS = {
    "wPLI_R_only": ["dR_wpli"],
    "wPLI_D_only": ["dD_wpli"],
    "wPLI_M_only": ["dlogM_wpli"],
    "wPLI_Pi_delta": ["dPi_wpli"],
    "wPLI_triad_delta": ["dR_wpli", "dD_wpli", "dlogM_wpli"],
    "wPLI_triad_plus_Pi": ["dR_wpli", "dD_wpli", "dlogM_wpli", "dPi_wpli"],
}


def auc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def sign_rule(df: pd.DataFrame, score_col: str, positive_when_decreases: bool = True) -> dict[str, float]:
    y = df["y"].to_numpy()
    # Baselines have zero delta. Sedation is predicted if Pi has dropped.
    score = -df[score_col].to_numpy() if positive_when_decreases else df[score_col].to_numpy()
    pred = (score > 0).astype(int)
    return {
        "auc": auc_safe(y, score),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(df)),
    }


def logistic_transfer(df: pd.DataFrame, feature_cols: list[str], train_dataset: str, test_dataset: str) -> dict[str, float]:
    train = df[df["dataset"] == train_dataset].copy()
    test = df[df["dataset"] == test_dataset].copy()
    train = train.dropna(subset=feature_cols + ["y"])
    test = test.dropna(subset=feature_cols + ["y"])
    if len(train) == 0 or len(test) == 0:
        return {"auc": float("nan"), "balanced_accuracy": float("nan"), "n_train": 0, "n_test": 0}
    x_train = train[feature_cols].to_numpy()
    y_train = train["y"].astype(int).to_numpy()
    x_test = test[feature_cols].to_numpy()
    y_test = test["y"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc": float("nan"), "balanced_accuracy": float("nan"), "n_train": int(len(train)), "n_test": int(len(test))}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260514),
    )
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def paired_delta_features(df: pd.DataFrame, dataset: str, band: str, subject_col: str, condition_col: str) -> pd.DataFrame:
    rows = []
    id_cols = [subject_col]
    for subject, sub in df.groupby(subject_col):
        sub = sub.copy()
        baseline_name = "baseline" if dataset == "Chennu" else "awake"
        base = sub[sub[condition_col] == baseline_name]
        if base.empty:
            continue
        base = base.iloc[0]
        target_conditions = ["moderate"] if dataset == "Chennu" else ["sed", "sed2"]
        rows.append(
            {
                "dataset": dataset,
                "band": band,
                "subject": str(subject),
                "condition": baseline_name,
                "y": 0,
                "dR_wpli": 0.0,
                "dD_wpli": 0.0,
                "dlogM_wpli": 0.0,
                "dPi_wpli": 0.0,
            }
        )
        for cond in target_conditions:
            target = sub[sub[condition_col] == cond]
            if target.empty:
                continue
            target = target.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "band": band,
                    "subject": str(subject),
                    "condition": cond,
                    "y": 1,
                    "dR_wpli": float(target["R_wpli_mean"] - base["R_wpli_mean"]),
                    "dD_wpli": float(target["D_eff_mean"] - base["D_eff_mean"]),
                    "dlogM_wpli": float(np.log10(target["M_tau_mean"] + 1e-12) - np.log10(base["M_tau_mean"] + 1e-12)),
                    "dPi_wpli": float(target["Pi"] - base["Pi"]),
                }
            )
    return pd.DataFrame(rows)


def load_wpli_features(chennu_dir: Path, ds_dir: Path) -> pd.DataFrame:
    frames = []
    for band in ["alpha", "gamma"]:
        ch = pd.read_csv(chennu_dir / f"chennu_{band}_wpli_gcc.csv")
        ds = pd.read_csv(ds_dir / f"ds005620_{band}_wpli_gcc.csv")
        frames.append(paired_delta_features(ch, "Chennu", band, "subject", "condition"))
        frames.append(paired_delta_features(ds, "DS005620", band, "subject", "condition"))
    return pd.concat(frames, ignore_index=True)


def benchmark_family(df: pd.DataFrame, feature_sets: dict[str, list[str]], prefix: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    nested = {}
    for band in ["alpha", "gamma"]:
        band_df = df[df["band"] == band].copy()
        nested[band] = {}

        if prefix == "standard":
            for col, name in [("dPi", "GCC_Pi_sign_rule")]:
                res_all = sign_rule(band_df, col)
                nested[band][name] = {"all": res_all}
                rows.append({"family": prefix, "band": band, "model": name, "direction": "pooled_sign_rule", **res_all})
        if prefix == "wpli":
            res_all = sign_rule(band_df, "dPi_wpli")
            nested[band]["wPLI_Pi_sign_rule"] = {"all": res_all}
            rows.append({"family": prefix, "band": band, "model": "wPLI_Pi_sign_rule", "direction": "pooled_sign_rule", **res_all})

        for name, cols in feature_sets.items():
            nested[band][name] = {}
            for train_dataset, test_dataset, direction in [
                ("Chennu", "DS005620", "train_chennu_test_ds"),
                ("DS005620", "Chennu", "train_ds_test_chennu"),
            ]:
                res = logistic_transfer(band_df, cols, train_dataset, test_dataset)
                nested[band][name][direction] = res
                rows.append({"family": prefix, "band": band, "model": name, "direction": direction, **res})
    return pd.DataFrame(rows), nested


def plot_benchmark(summary_df: pd.DataFrame, outdir: Path) -> None:
    plot_df = summary_df[
        (summary_df["direction"].isin(["train_chennu_test_ds", "train_ds_test_chennu"]))
        & (
            summary_df["model"].isin(
                [
                    "GCC_Pi_delta",
                    "GCC_triad_plus_Pi",
                    "spectral_bandpowers",
                    "spectral_all",
                    "spectral_plus_GCC_Pi",
                    "wPLI_Pi_delta",
                    "wPLI_triad_plus_Pi",
                ]
            )
        )
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180, sharey=True)
    for ax, band in zip(axes, ["alpha", "gamma"]):
        sub = plot_df[plot_df["band"] == band].copy()
        labels = []
        vals = []
        colors = []
        for _, row in sub.iterrows():
            labels.append(f"{row['model']}\n{row['direction'].replace('train_', '').replace('_test_', '→')}")
            vals.append(row["auc"])
            colors.append("#8A4F2A" if row["family"] == "standard" else "#1F6F78")
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.axhline(0.5, color="black", lw=1, ls="--")
        ax.set_xticks(x, labels, rotation=80, ha="right", fontsize=7)
        ax.set_title(f"{band} frozen transfer AUC")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(outdir / "noc_frozen_transfer_benchmark_auc.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-features", type=Path, required=True)
    parser.add_argument("--chennu-wpli-dir", type=Path, required=True)
    parser.add_argument("--ds-wpli-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    standard = pd.read_csv(args.standard_features)
    wpli = load_wpli_features(args.chennu_wpli_dir, args.ds_wpli_dir)
    standard.to_csv(args.outdir / "noc_standard_delta_features.csv", index=False)
    wpli.to_csv(args.outdir / "noc_wpli_delta_features.csv", index=False)

    standard_rows, standard_nested = benchmark_family(standard, STANDARD_FEATURE_SETS, "standard")
    wpli_rows, wpli_nested = benchmark_family(wpli, WPLI_FEATURE_SETS, "wpli")
    summary_df = pd.concat([standard_rows, wpli_rows], ignore_index=True)
    summary_df.to_csv(args.outdir / "noc_frozen_benchmark_results.csv", index=False)
    plot_benchmark(summary_df, args.outdir)

    summary = {
        "design": {
            "standard_features": str(args.standard_features),
            "wpli_chennu": str(args.chennu_wpli_dir),
            "wpli_ds005620": str(args.ds_wpli_dir),
            "rule": "All thresholds, scaling, and logistic weights are fitted only on the training dataset; sign-rule models use no fitted parameters.",
        },
        "standard": standard_nested,
        "wpli": wpli_nested,
    }
    with open(args.outdir / "noc_frozen_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
