#!/usr/bin/env python
"""Residualization of GCC Pi against spectral bandpower.

Tests whether subject-normalized GCC regime occupancy (dPi) carries condition
information after relative theta/alpha/beta/gamma bandpower shifts are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDPOWER = ["d_theta_power", "d_alpha_power", "d_beta_power", "d_gamma_power"]
ALL_SPECTRAL = BANDPOWER + ["d_alpha_gamma_ratio", "d_spectral_entropy"]


def _logit() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260513),
    )


def residualize(train: pd.DataFrame, test: pd.DataFrame, spectral_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    reg = make_pipeline(StandardScaler(), LinearRegression())
    reg.fit(train[spectral_cols], train["dPi"])
    train_resid = train["dPi"].to_numpy() - reg.predict(train[spectral_cols])
    test_resid = test["dPi"].to_numpy() - reg.predict(test[spectral_cols])
    return train_resid.reshape(-1, 1), test_resid.reshape(-1, 1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, mode: str, spectral_cols: list[str]) -> np.ndarray:
    if mode == "spectral":
        x_train = train[spectral_cols]
        x_test = test[spectral_cols]
    elif mode == "spectral_plus_pi":
        x_train = train[spectral_cols + ["dPi"]]
        x_test = test[spectral_cols + ["dPi"]]
    elif mode == "pi_only":
        x_train = train[["dPi"]]
        x_test = test[["dPi"]]
    elif mode == "residual_pi":
        x_train, x_test = residualize(train, test, spectral_cols)
    else:
        raise ValueError(mode)
    clf = _logit()
    clf.fit(x_train, train["y"])
    return clf.predict_proba(x_test)[:, 1]


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def within_dataset_loso(df: pd.DataFrame, spectral_cols: list[str]) -> dict:
    modes = ["spectral", "pi_only", "residual_pi", "spectral_plus_pi"]
    scores = {mode: np.full(len(df), np.nan) for mode in modes}
    logo = LeaveOneGroupOut()
    groups = df["subject"].astype(str).to_numpy()
    y = df["y"].to_numpy()
    for train_idx, test_idx in logo.split(df, y, groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if train["y"].nunique() < 2 or test["y"].nunique() < 2:
            continue
        for mode in modes:
            scores[mode][test_idx] = fit_predict(train, test, mode, spectral_cols)
    out = {}
    mask_all = np.isfinite(next(iter(scores.values())))
    for mode, score in scores.items():
        mask = np.isfinite(score)
        out[mode] = metrics(y[mask], score[mask])
        out[mode]["n"] = int(mask.sum())
    out["auc_delta_spectral_plus_pi_minus_spectral"] = float(out["spectral_plus_pi"]["auc"] - out["spectral"]["auc"])
    out["auc_delta_residual_pi_minus_spectral"] = float(out["residual_pi"]["auc"] - out["spectral"]["auc"])
    return out


def cross_dataset(train: pd.DataFrame, test: pd.DataFrame, spectral_cols: list[str]) -> dict:
    out = {}
    y = test["y"].to_numpy()
    for mode in ["spectral", "pi_only", "residual_pi", "spectral_plus_pi"]:
        score = fit_predict(train, test, mode, spectral_cols)
        out[mode] = metrics(y, score)
        out[mode]["n_train"] = int(len(train))
        out[mode]["n_test"] = int(len(test))
    out["auc_delta_spectral_plus_pi_minus_spectral"] = float(out["spectral_plus_pi"]["auc"] - out["spectral"]["auc"])
    out["auc_delta_residual_pi_minus_spectral"] = float(out["residual_pi"]["auc"] - out["spectral"]["auc"])
    return out


def plot_summary(summary: dict, outdir: Path, tag: str) -> None:
    rows = []
    for scope, scope_res in summary.items():
        for model, vals in scope_res.items():
            if not isinstance(vals, dict) or "auc" not in vals:
                continue
            rows.append({"scope": scope, "model": model, "auc": vals["auc"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for scope, sub in df.groupby("scope"):
        fig, ax = plt.subplots(figsize=(7, 4))
        order = ["spectral", "pi_only", "residual_pi", "spectral_plus_pi"]
        vals = [sub[sub["model"] == m]["auc"].iloc[0] for m in order if m in sub["model"].values]
        labels = [m for m in order if m in sub["model"].values]
        ax.bar(labels, vals, color=["#6B7280", "#2563EB", "#0891B2", "#DC7F37"])
        ax.axhline(0.5, color="black", linewidth=1, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title(f"{tag}: {scope}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        safe_scope = scope.replace(" ", "_").replace("→", "to").replace("/", "_")
        fig.savefig(outdir / f"bandpower_residualization_{tag}_{safe_scope}.png", dpi=180)
        plt.close(fig)


def run(df: pd.DataFrame, spectral_cols: list[str], tag: str, outdir: Path) -> dict:
    summary = {}
    for band in sorted(df["band"].unique()):
        band_df = df[df["band"] == band].copy()
        summary[band] = {}
        for dataset in ["Chennu", "DS005620"]:
            sub = band_df[band_df["dataset"] == dataset].copy()
            summary[band][f"within_{dataset}_LOSO"] = within_dataset_loso(sub, spectral_cols)
        ch = band_df[band_df["dataset"] == "Chennu"].copy()
        ds = band_df[band_df["dataset"] == "DS005620"].copy()
        summary[band]["train_Chennu_test_DS005620"] = cross_dataset(ch, ds, spectral_cols)
        summary[band]["train_DS005620_test_Chennu"] = cross_dataset(ds, ch, spectral_cols)
        plot_summary(summary[band], outdir, f"{tag}_{band}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    summaries = {
        "bandpower_only_residualization": run(df, BANDPOWER, "bandpower", args.outdir),
        "all_spectral_residualization": run(df, ALL_SPECTRAL, "all_spectral", args.outdir),
    }
    with open(args.outdir / "bandpower_residualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2)[:12000])


if __name__ == "__main__":
    main()
