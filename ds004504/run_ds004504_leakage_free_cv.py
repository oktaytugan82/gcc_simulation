from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = Path(os.environ.get("DS004504_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = REPO_ROOT / "figures"
FEATURES_CSV = RESULTS_DIR / "ds004504_gcc_features_by_band.csv"
WINDOWS_CSV = RESULTS_DIR / "ds004504_gcc_window_features.csv"
WPLI_NPZ = RESULTS_DIR / "ds004504_wpli_matrices.npz"

OUT_PRED = RESULTS_DIR / "ds004504_cv_predictions.csv"
OUT_FOLD = RESULTS_DIR / "ds004504_cv_fold_metrics.csv"
OUT_SUBJECT = RESULTS_DIR / "ds004504_cv_subject_mean_predictions.csv"
OUT_SUMMARY = RESULTS_DIR / "ds004504_cv_model_comparison_summary.json"
OUT_PLOT = FIGURES_DIR / "ds004504_cv_model_comparison.png"

BANDS = ["theta", "alpha", "beta", "low_gamma"]
GCC_WINDOW_FEATURES = ["R", "D_eff", "M_tau"]
BACKBONE_TOP_FRACTION = 0.20
N_SPLITS = 5
N_REPEATS = 100
RANDOM_STATE = 20260425
ALPHAS = np.logspace(-4, 4, 41)
EPS = 1e-12


def upper_triangular_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def top_backbone_mask(reference: np.ndarray, fraction: float) -> np.ndarray:
    n = reference.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    tri = np.triu_indices(n, k=1)
    values = reference[tri]
    n_edges = max(1, int(round(values.size * fraction)))
    chosen = np.argsort(values)[-n_edges:]
    mask[tri[0][chosen], tri[1][chosen]] = True
    mask = mask | mask.T
    return mask


def random_backbone_mask(reference: np.ndarray, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    n = reference.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    tri = np.triu_indices(n, k=1)
    chosen = rng.choice(np.arange(len(tri[0])), size=n_edges, replace=False)
    mask[tri[0][chosen], tri[1][chosen]] = True
    mask = mask | mask.T
    return mask


def masked_mean(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = matrix[mask]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def make_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "n_test": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(np.unique(y_true)) > 1:
        out["r2"] = float(r2_score(y_true, y_pred))
        out["spearman_rho"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred).statistic)
    else:
        out["r2"] = float("nan")
        out["spearman_rho"] = float("nan")
        out["pearson_r"] = float("nan")
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = float(np.mean(sample))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
    }


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, n_perm: int = 100000) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)), replace=True)
    null = np.mean(signs * values[None, :], axis=1)
    # One-sided test: improvement > 0.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def build_fold_features(
    participants: pd.DataFrame,
    subject_band: pd.DataFrame,
    windows: pd.DataFrame,
    wpli: np.lib.npyio.NpzFile,
    train_subjects: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    train_controls = set(
        participants.loc[
            (participants["participant_id"].isin(train_subjects)) & (participants["group"] == "C"),
            "participant_id",
        ]
    )
    if not train_controls:
        raise RuntimeError("No training controls available for fold calibration.")

    rows: list[dict] = []
    base = participants.set_index("participant_id")

    for subject_id, meta in base.iterrows():
        row = {
            "participant_id": subject_id,
            "group": meta["group"],
            "gender": meta["gender"],
            "age": float(meta["age"]),
            "mmse": float(meta["mmse"]),
            "diagnosis": "F" if meta["group"] == "F" else "A_or_C",
        }
        if meta["group"] in {"A", "F"}:
            row["diagnosis"] = meta["group"]

        for band in BANDS:
            sb = subject_band[
                (subject_band["participant_id"] == subject_id) & (subject_band["band"] == band)
            ].iloc[0]
            row[f"{band}__log_power"] = float(np.log10(float(sb["band_power_mean"]) + EPS))
            row[f"{band}__mean_wpli"] = float(sb["mean_wpli"])
            row[f"{band}__R_mean"] = float(sb["R_mean"])
            row[f"{band}__D_eff_mean"] = float(sb["D_eff_mean"])
            row[f"{band}__M_tau_mean"] = float(sb["M_tau_mean"])

            train_control_windows = windows[
                (windows["band"] == band) & (windows["participant_id"].isin(train_controls))
            ]
            bounds = {}
            for feature in GCC_WINDOW_FEATURES:
                bounds[feature] = (
                    float(train_control_windows[feature].quantile(0.10)),
                    float(train_control_windows[feature].quantile(0.90)),
                )

            subject_windows = windows[
                (windows["band"] == band) & (windows["participant_id"] == subject_id)
            ]
            inside = np.ones(len(subject_windows), dtype=bool)
            for feature, (lo, hi) in bounds.items():
                values = subject_windows[feature].to_numpy(dtype=float)
                inside &= (values >= lo) & (values <= hi)
            row[f"{band}__Pi_control_window"] = float(np.mean(inside)) if len(inside) else float("nan")

            control_mats = [
                wpli[f"{control_id}__{band}"].astype(float)
                for control_id in train_controls
                if f"{control_id}__{band}" in wpli.files
            ]
            reference = np.mean(control_mats, axis=0)
            real_mask = top_backbone_mask(reference, BACKBONE_TOP_FRACTION)
            n_edges = int(np.sum(np.triu(real_mask, k=1)))
            random_mask = random_backbone_mask(reference, n_edges, rng)
            matrix = wpli[f"{subject_id}__{band}"].astype(float)

            for label, mask in [("backbone", real_mask), ("random_backbone", random_mask)]:
                non_mask = (~mask) & (~np.eye(mask.shape[0], dtype=bool))
                backbone_value = masked_mean(matrix, mask)
                non_value = masked_mean(matrix, non_mask)
                row[f"{band}__{label}_wpli"] = backbone_value
                row[f"{band}__{label}_non_wpli"] = non_value
                row[f"{band}__{label}_selective_ratio"] = backbone_value / (non_value + EPS)

        if "theta__log_power" in row and "alpha__log_power" in row:
            row["theta_alpha_log_power_ratio"] = row["theta__log_power"] - row["alpha__log_power"]
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    participants = pd.read_csv(ROOT / "data" / "ds004504-main" / "participants.tsv", sep="\t")
    participants = participants.rename(columns={"Gender": "gender", "Age": "age", "Group": "group", "MMSE": "mmse"})
    participants["group"] = participants["group"].astype(str)
    participants["gender"] = participants["gender"].astype(str)
    participants["age"] = participants["age"].astype(float)
    participants["mmse"] = participants["mmse"].astype(float)

    subject_band = pd.read_csv(FEATURES_CSV)
    windows = pd.read_csv(WINDOWS_CSV)
    wpli = np.load(WPLI_NPZ)

    subjects = participants["participant_id"].to_numpy()
    strata = participants["group"].to_numpy()
    patients = set(participants.loc[participants["group"].isin(["A", "F"]), "participant_id"])

    clinical_numeric = ["age"]
    clinical_categorical = ["gender", "diagnosis"]

    standard_numeric = clinical_numeric + ["theta_alpha_log_power_ratio"]
    for band in BANDS:
        standard_numeric.extend([f"{band}__log_power", f"{band}__mean_wpli"])

    gcc_numeric = list(standard_numeric)
    for band in BANDS:
        gcc_numeric.extend(
            [
                f"{band}__R_mean",
                f"{band}__D_eff_mean",
                f"{band}__M_tau_mean",
                f"{band}__Pi_control_window",
                f"{band}__backbone_wpli",
                f"{band}__backbone_non_wpli",
                f"{band}__backbone_selective_ratio",
            ]
        )

    random_backbone_numeric = list(standard_numeric)
    for band in BANDS:
        random_backbone_numeric.extend(
            [
                f"{band}__R_mean",
                f"{band}__D_eff_mean",
                f"{band}__M_tau_mean",
                f"{band}__Pi_control_window",
                f"{band}__random_backbone_wpli",
                f"{band}__random_backbone_non_wpli",
                f"{band}__random_backbone_selective_ratio",
            ]
        )

    models = {
        "A_clinical": (clinical_categorical, clinical_numeric),
        "B_standard_eeg": (clinical_categorical, standard_numeric),
        "C_gcc_real_backbone": (clinical_categorical, gcc_numeric),
        "D_gcc_random_backbone": (clinical_categorical, random_backbone_numeric),
    }

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    predictions: list[dict] = []
    fold_metrics_rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(subjects, strata), start=1):
        train_subjects = set(subjects[train_idx])
        test_subjects = set(subjects[test_idx])
        feature_table = build_fold_features(
            participants=participants,
            subject_band=subject_band,
            windows=windows,
            wpli=wpli,
            train_subjects=train_subjects,
            rng=rng,
        )

        train_df = feature_table[
            feature_table["participant_id"].isin(train_subjects)
            & feature_table["participant_id"].isin(patients)
        ].copy()
        test_df = feature_table[
            feature_table["participant_id"].isin(test_subjects)
            & feature_table["participant_id"].isin(patients)
        ].copy()
        y_train = train_df["mmse"].to_numpy(dtype=float)
        y_test = test_df["mmse"].to_numpy(dtype=float)

        fold_preds: dict[str, np.ndarray] = {}
        for model_name, (categorical, numeric) in models.items():
            model = make_model(categorical, numeric)
            model.fit(train_df[categorical + numeric], y_train)
            pred = model.predict(test_df[categorical + numeric])
            fold_preds[model_name] = pred

            metrics = fold_metrics(y_test, pred)
            metrics.update(
                {
                    "fold_id": fold_id,
                    "repeat": (fold_id - 1) // N_SPLITS,
                    "split": (fold_id - 1) % N_SPLITS,
                    "model": model_name,
                    "n_train_patients": int(len(train_df)),
                    "n_test_patients": int(len(test_df)),
                    "n_train_controls_for_calibration": int(
                        participants[
                            participants["participant_id"].isin(train_subjects)
                            & (participants["group"] == "C")
                        ].shape[0]
                    ),
                }
            )
            fold_metrics_rows.append(metrics)

            for subject_id, group, observed, predicted in zip(
                test_df["participant_id"], test_df["group"], y_test, pred
            ):
                predictions.append(
                    {
                        "fold_id": fold_id,
                        "repeat": (fold_id - 1) // N_SPLITS,
                        "split": (fold_id - 1) % N_SPLITS,
                        "participant_id": subject_id,
                        "group": group,
                        "model": model_name,
                        "observed_mmse": float(observed),
                        "predicted_mmse": float(predicted),
                    }
                )

        if fold_id % 25 == 0:
            b_mae = mean_absolute_error(y_test, fold_preds["B_standard_eeg"])
            c_mae = mean_absolute_error(y_test, fold_preds["C_gcc_real_backbone"])
            print(f"fold {fold_id:03d}: B_MAE={b_mae:.3f} C_MAE={c_mae:.3f}")

    pred_df = pd.DataFrame(predictions)
    fold_df = pd.DataFrame(fold_metrics_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_PRED, index=False)
    fold_df.to_csv(OUT_FOLD, index=False)

    subject_mean = (
        pred_df.groupby(["participant_id", "group", "model"], as_index=False)
        .agg(observed_mmse=("observed_mmse", "first"), predicted_mmse=("predicted_mmse", "mean"))
    )
    subject_mean.to_csv(OUT_SUBJECT, index=False)

    summary_models = {}
    for model_name in models:
        sm = subject_mean[subject_mean["model"] == model_name]
        y = sm["observed_mmse"].to_numpy(dtype=float)
        pred = sm["predicted_mmse"].to_numpy(dtype=float)
        summary_models[model_name] = fold_metrics(y, pred)

    pivot = subject_mean.pivot(
        index=["participant_id", "group", "observed_mmse"],
        columns="model",
        values="predicted_mmse",
    ).reset_index()
    pivot["abs_error_B"] = np.abs(pivot["observed_mmse"] - pivot["B_standard_eeg"])
    pivot["abs_error_C"] = np.abs(pivot["observed_mmse"] - pivot["C_gcc_real_backbone"])
    pivot["abs_error_D"] = np.abs(pivot["observed_mmse"] - pivot["D_gcc_random_backbone"])
    improvement_bc = pivot["abs_error_B"].to_numpy() - pivot["abs_error_C"].to_numpy()
    improvement_dc = pivot["abs_error_D"].to_numpy() - pivot["abs_error_C"].to_numpy()

    fold_pivot = fold_df.pivot(index="fold_id", columns="model", values="mae")
    fold_pivot["B_minus_C_mae"] = fold_pivot["B_standard_eeg"] - fold_pivot["C_gcc_real_backbone"]
    fold_pivot["D_minus_C_mae"] = fold_pivot["D_gcc_random_backbone"] - fold_pivot["C_gcc_real_backbone"]

    summary = {
        "analysis": "Leakage-free repeated stratified 5-fold CV; controls in each training fold define GCC access windows and wPLI backbone.",
        "primary_population": "AD + FTD patients only; controls used only for fold-local calibration.",
        "n_subjects_total": int(len(participants)),
        "n_patients": int(len(patients)),
        "n_controls": int((participants["group"] == "C").sum()),
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_folds": int(N_SPLITS * N_REPEATS),
        "backbone_top_fraction": BACKBONE_TOP_FRACTION,
        "models_subject_mean_out_of_sample": summary_models,
        "primary_subject_level_mae_improvement_B_minus_C": bootstrap_ci(improvement_bc, rng),
        "primary_subject_level_mae_improvement_B_minus_C_signflip_p_one_sided": sign_flip_p(improvement_bc, rng),
        "random_control_subject_level_mae_improvement_D_minus_C": bootstrap_ci(improvement_dc, rng),
        "fold_level_mean_mae": fold_df.groupby("model")["mae"].mean().to_dict(),
        "fold_level_sd_mae": fold_df.groupby("model")["mae"].std().to_dict(),
        "fold_level_B_minus_C_mae_mean": float(fold_pivot["B_minus_C_mae"].mean()),
        "fold_level_B_minus_C_mae_wilcoxon_p_two_sided": float(
            stats.wilcoxon(fold_pivot["B_standard_eeg"], fold_pivot["C_gcc_real_backbone"]).pvalue
        ),
        "outputs": {
            "predictions": str(OUT_PRED),
            "fold_metrics": str(OUT_FOLD),
            "subject_mean_predictions": str(OUT_SUBJECT),
            "plot": str(OUT_PLOT),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_df = pd.DataFrame(
        [
            {"model": name, "MAE": values["mae"], "RMSE": values["rmse"], "R2": values["r2"]}
            for name, values in summary_models.items()
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(plot_df["model"], plot_df["MAE"], color=["#8a8f98", "#4b77be", "#2b9a66", "#c77c2b"])
    axes[0].set_ylabel("Subject-mean out-of-sample MAE (MMSE)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Model comparison")
    c = subject_mean[subject_mean["model"] == "C_gcc_real_backbone"]
    axes[1].scatter(c["observed_mmse"], c["predicted_mmse"], c="#2b9a66", alpha=0.85)
    lo = min(c["observed_mmse"].min(), c["predicted_mmse"].min())
    hi = max(c["observed_mmse"].max(), c["predicted_mmse"].max())
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[1].set_xlabel("Observed MMSE")
    axes[1].set_ylabel("Predicted MMSE")
    axes[1].set_title("GCC + real backbone")
    fig.savefig(OUT_PLOT, dpi=200)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PRED}")
    print(f"Wrote {OUT_FOLD}")
    print(f"Wrote {OUT_SUBJECT}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
