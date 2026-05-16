from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RNG = np.random.default_rng(20260514)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def load_wide(alpha_csv: Path, gamma_csv: Path) -> pd.DataFrame:
    frames = []
    for band, path in [("alpha", alpha_csv), ("gamma", gamma_csv)]:
        df = pd.read_csv(path)
        keep = ["subject", "Group", "GroupName", "MMSE", "Pi", "R_mean", "D_mean", "M_mean"]
        df = df[keep].copy()
        df = df.rename(
            columns={
                "Pi": f"{band}_Pi",
                "R_mean": f"{band}_R",
                "D_mean": f"{band}_D",
                "M_mean": f"{band}_M",
            }
        )
        frames.append(df)

    wide = frames[0].merge(
        frames[1],
        on=["subject", "Group", "GroupName", "MMSE"],
        how="inner",
        validate="one_to_one",
    )
    wide["is_dementia_proxy"] = (wide["GroupName"] != "control").astype(int)
    return wide


def loo_eval(df: pd.DataFrame, features: list[str], label_col: str, positive_label: int = 1) -> dict:
    sub = df.dropna(subset=[label_col]).copy()
    y = sub[label_col].astype(int).to_numpy()
    X = sub[features].to_numpy(dtype=float)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"n": len(y), "auc": np.nan, "balanced_accuracy": np.nan, "accuracy": np.nan}

    proba = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(y), dtype=int)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="liblinear", random_state=20260514),
        )
        model.fit(X[train_idx], y[train_idx])
        classes = model.named_steps["logisticregression"].classes_
        pos_index = int(np.where(classes == positive_label)[0][0])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, pos_index]
        pred[test_idx] = model.predict(X[test_idx])

    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def permutation_auc(df: pd.DataFrame, features: list[str], label_col: str, observed_auc: float, n_perm: int) -> float:
    if not np.isfinite(observed_auc):
        return np.nan
    y_true = df[label_col].astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return np.nan
    aucs = []
    tmp = df.copy()
    for _ in range(n_perm):
        tmp[label_col] = RNG.permutation(y_true)
        aucs.append(loo_eval(tmp, features, label_col)["auc"])
    aucs = np.asarray(aucs, dtype=float)
    return float((np.sum(aucs >= observed_auc) + 1) / (np.sum(np.isfinite(aucs)) + 1))


def classify_sets(wide: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    feature_sets = {
        "alpha_pi": ["alpha_Pi"],
        "gamma_pi": ["gamma_Pi"],
        "alpha_triad": ["alpha_R", "alpha_D", "alpha_M"],
        "gamma_triad": ["gamma_R", "gamma_D", "gamma_M"],
        "combined_pi": ["alpha_Pi", "gamma_Pi"],
        "combined_triad": ["alpha_R", "alpha_D", "alpha_M", "gamma_R", "gamma_D", "gamma_M"],
        "combined_all": ["alpha_Pi", "alpha_R", "alpha_D", "alpha_M", "gamma_Pi", "gamma_R", "gamma_D", "gamma_M"],
    }
    tasks = []
    tasks.append(("dementia_proxy_vs_control", wide.assign(label=wide["is_dementia_proxy"])))
    tasks.append(("alzheimer_vs_control", wide[wide["GroupName"].isin(["alzheimer", "control"])].assign(label=lambda d: (d["GroupName"] == "alzheimer").astype(int))))
    tasks.append(("frontotemporal_vs_control", wide[wide["GroupName"].isin(["frontotemporal", "control"])].assign(label=lambda d: (d["GroupName"] == "frontotemporal").astype(int))))
    tasks.append(("alzheimer_vs_frontotemporal", wide[wide["GroupName"].isin(["alzheimer", "frontotemporal"])].assign(label=lambda d: (d["GroupName"] == "alzheimer").astype(int))))

    rows = []
    for task, df in tasks:
        for set_name, features in feature_sets.items():
            metrics = loo_eval(df, features, "label")
            p_perm = permutation_auc(df, features, "label", metrics["auc"], n_perm=n_perm)
            rows.append({"task": task, "feature_set": set_name, **metrics, "p_perm_auc_ge_observed": p_perm})
    return pd.DataFrame(rows)


def group_effects(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["alpha_Pi", "alpha_R", "alpha_D", "alpha_M", "gamma_Pi", "gamma_R", "gamma_D", "gamma_M"]
    controls = wide[wide["GroupName"] == "control"]
    for group in ["alzheimer", "frontotemporal"]:
        gdf = wide[wide["GroupName"] == group]
        for feature in features:
            x = gdf[feature].to_numpy(dtype=float)
            y = controls[feature].to_numpy(dtype=float)
            try:
                u_p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
            except ValueError:
                u_p = np.nan
            rows.append(
                {
                    "contrast": f"{group}_minus_control",
                    "feature": feature,
                    "group_mean": float(np.nanmean(x)),
                    "control_mean": float(np.nanmean(y)),
                    "mean_difference": float(np.nanmean(x) - np.nanmean(y)),
                    "cohen_d_group_minus_control": cohen_d(x, y),
                    "mannwhitney_p": float(u_p) if np.isfinite(u_p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mmse_correlations(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["alpha_Pi", "alpha_R", "alpha_D", "alpha_M", "gamma_Pi", "gamma_R", "gamma_D", "gamma_M"]
    subsets = {
        "all_subjects": wide,
        "dementia_proxy_only": wide[wide["GroupName"] != "control"],
    }
    for subset_name, df in subsets.items():
        for feature in features:
            tmp = df[["MMSE", feature]].dropna()
            if len(tmp) < 6:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(tmp["MMSE"], tmp[feature])
            rows.append({"subset": subset_name, "feature": feature, "n": int(len(tmp)), "spearman_rho": rho, "p": p})
    return pd.DataFrame(rows)


def write_report(outdir: Path, wide: pd.DataFrame, effects: pd.DataFrame, corr: pd.DataFrame, cls: pd.DataFrame) -> None:
    group_summary = (
        wide.groupby("GroupName")
        .agg(
            n=("subject", "count"),
            MMSE_mean=("MMSE", "mean"),
            alpha_Pi_mean=("alpha_Pi", "mean"),
            gamma_Pi_mean=("gamma_Pi", "mean"),
            alpha_R_mean=("alpha_R", "mean"),
            gamma_R_mean=("gamma_R", "mean"),
        )
        .reset_index()
    )
    top_cls = cls.sort_values(["task", "auc"], ascending=[True, False]).groupby("task").head(3)
    top_corr = corr.reindex(corr["spearman_rho"].abs().sort_values(ascending=False).index).head(10)
    top_effects = effects.reindex(effects["cohen_d_group_minus_control"].abs().sort_values(ascending=False).index).head(12)

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

    report = []
    report.append("# Batch 12 ds004504 Degradation-Proxy GCC Analysis\n")
    report.append("Date: 2026-05-14\n")
    report.append("## Scope\n")
    report.append(
        "This analysis treats ds004504 as a neurodegeneration/degradation proxy, not as a direct level-of-consciousness or terminal-lucidity dataset. "
        "It tests whether GCC observables carry clinically relevant structure across Alzheimer, frontotemporal dementia, and healthy control groups.\n"
    )
    report.append("## Group Summary\n")
    report.append(md_table(group_summary))
    report.append("\n\n## Strongest Group Effects\n")
    report.append(md_table(top_effects))
    report.append("\n\n## Strongest MMSE Associations\n")
    report.append(md_table(top_corr))
    report.append("\n\n## Best Leave-One-Subject-Out Classifiers\n")
    report.append(md_table(top_cls))
    report.append("\n\n## Interpretation\n")
    report.append(
        "The appropriate claim is limited: GCC features show degradation-proxy sensitivity and can be benchmarked against clinical group labels, "
        "but this does not validate the re-entry mechanism or conscious-access criterion. It strengthens empirical breadth by adding a non-anesthesia, "
        "clinical degradation axis.\n"
    )
    (outdir / "BATCH12_DEGRADATION_PROXY_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-csv", type=Path, required=True)
    parser.add_argument("--gamma-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=300)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    wide = load_wide(args.alpha_csv, args.gamma_csv)
    effects = group_effects(wide)
    corr = mmse_correlations(wide)
    cls = classify_sets(wide, n_perm=args.n_perm)

    wide.to_csv(args.outdir / "ds004504_wide_gcc_features.csv", index=False)
    effects.to_csv(args.outdir / "ds004504_group_effects.csv", index=False)
    corr.to_csv(args.outdir / "ds004504_mmse_correlations.csv", index=False)
    cls.to_csv(args.outdir / "ds004504_loso_classification.csv", index=False)
    write_report(args.outdir, wide, effects, corr, cls)

    print(
        json.dumps(
            {
                "n_subjects": int(len(wide)),
                "groups": wide["GroupName"].value_counts().to_dict(),
                "outputs": [
                    "ds004504_wide_gcc_features.csv",
                    "ds004504_group_effects.csv",
                    "ds004504_mmse_correlations.csv",
                    "ds004504_loso_classification.csv",
                    "BATCH12_DEGRADATION_PROXY_REPORT.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
