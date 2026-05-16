from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPECTRAL = ["theta_power", "alpha_power", "beta_power", "gamma_power", "alpha_gamma_ratio", "spectral_entropy"]
GCCO = ["R", "D_eff", "log_M", "Pi"]


def residualize(train: pd.DataFrame, test: pd.DataFrame, target_cols: list[str], covariate_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train_cov = train[covariate_cols].to_numpy(float)
    x_test_cov = test[covariate_cols].to_numpy(float)
    train_resids = []
    test_resids = []
    for col in target_cols:
        reg = make_pipeline(StandardScaler(), LinearRegression())
        reg.fit(x_train_cov, train[col].to_numpy(float))
        train_resids.append(train[col].to_numpy(float) - reg.predict(x_train_cov))
        test_resids.append(test[col].to_numpy(float) - reg.predict(x_test_cov))
    return np.column_stack(train_resids), np.column_stack(test_resids)


def logo_auc(df: pd.DataFrame, positive: str, negative: str, model: str) -> dict[str, float]:
    sub = df[df["state"].isin([positive, negative])].replace([np.inf, -np.inf], np.nan).dropna(subset=SPECTRAL + GCCO)
    y = (sub["state"] == positive).astype(int).to_numpy()
    groups = sub["subject"].to_numpy()
    scores = np.full(len(sub), np.nan)
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(sub, y, groups):
        train = sub.iloc[train_idx]
        test = sub.iloc[test_idx]
        if len(np.unique(y[train_idx])) < 2:
            continue
        if model == "spectral_all":
            x_train = train[SPECTRAL].to_numpy(float)
            x_test = test[SPECTRAL].to_numpy(float)
        elif model == "gcco_triad_plus_pi":
            x_train = train[GCCO].to_numpy(float)
            x_test = test[GCCO].to_numpy(float)
        elif model == "spectral_all_plus_gcco":
            x_train = train[SPECTRAL + GCCO].to_numpy(float)
            x_test = test[SPECTRAL + GCCO].to_numpy(float)
        elif model == "residual_gcco_after_spectral_all":
            x_train, x_test = residualize(train, test, GCCO, SPECTRAL)
        else:
            raise ValueError(model)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260514))
        clf.fit(x_train, y[train_idx])
        scores[test_idx] = clf.predict_proba(x_test)[:, 1]
    mask = np.isfinite(scores)
    pred = (scores[mask] >= 0.5).astype(int)
    return {
        "n": int(mask.sum()),
        "auc": float(roc_auc_score(y[mask], scores[mask])) if mask.sum() and len(np.unique(y[mask])) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)) if mask.sum() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    df["log_M"] = np.log10(df["M_tau"] + 1e-12)
    rows = []
    for band, bdf in df.groupby("band"):
        for contrast in [("Wake", "NREM"), ("REM", "NREM")]:
            for model in ["spectral_all", "gcco_triad_plus_pi", "spectral_all_plus_gcco", "residual_gcco_after_spectral_all"]:
                rows.append({"band": band, "contrast": f"{contrast[0]}_vs_{contrast[1]}", "model": model, **logo_auc(bdf, contrast[0], contrast[1], model)})
    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "gcco_sleep_residual_cv_metrics.csv", index=False)
    print(json.dumps({"out": str(args.outdir / "gcco_sleep_residual_cv_metrics.csv"), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
