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
DATASET_ROOT = Path(os.environ.get("DS004504_ROOT", REPO_ROOT / "data" / "ds004504-main"))
RESULTS_DIR = Path(os.environ.get("DS004504_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = REPO_ROOT / "figures"
FEATURES_CSV = RESULTS_DIR / "ds004504_gcc_features_by_band.csv"
WINDOWS_CSV = RESULTS_DIR / "ds004504_gcc_window_features.csv"
WPLI_NPZ = RESULTS_DIR / "ds004504_wpli_matrices.npz"

OUT_PRED = RESULTS_DIR / "ds004504_compact_cv_predictions.csv"
OUT_FOLD = RESULTS_DIR / "ds004504_compact_cv_fold_metrics.csv"
OUT_SUBJECT = RESULTS_DIR / "ds004504_compact_cv_subject_mean_predictions.csv"
OUT_SUMMARY = RESULTS_DIR / "ds004504_compact_cv_summary.json"
OUT_PLOT = FIGURES_DIR / "ds004504_compact_cv_model_comparison.png"

THEORY_BANDS = ["alpha", "low_gamma"]
GCC_WINDOW_FEATURES = ["R", "D_eff", "M_tau"]
BACKBONE_TOP_FRACTION = 0.20
N_RANDOM_BACKBONES = 50
N_SPLITS = 5
N_REPEATS = 100
RANDOM_STATE = 20260426
ALPHAS = np.logspace(-4, 4, 41)
EPS = 1e-12


def top_backbone_mask(reference: np.ndarray, fraction: float) -> np.ndarray:
    n = reference.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    tri = np.triu_indices(n, k=1)
    values = reference[tri]
    n_edges = max(1, int(round(values.size * fraction)))
    chosen = np.argsort(values)[-n_edges:]
    mask[tri[0][chosen], tri[1][chosen]] = True
    return mask | mask.T


def random_backbone_masks(n: int, n_edges: int, n_masks: int, rng: np.random.Generator) -> list[np.ndarray]:
    tri = np.triu_indices(n, k=1)
    masks: list[np.ndarray] = []
    for _ in range(n_masks):
        mask = np.zeros((n, n), dtype=bool)
        chosen = rng.choice(np.arange(len(tri[0])), size=n_edges, replace=False)
        mask[tri[0][chosen], tri[1][chosen]] = True
        masks.append(mask | mask.T)
    return masks


def masked_mean(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = matrix[mask]
    return float(np.mean(values)) if values.size else float("nan")


def backbone_features(matrix: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    non_mask = (~mask) & (~np.eye(mask.shape[0], dtype=bool))
    on = masked_mean(matrix, mask)
    off = masked_mean(matrix, non_mask)
    return {
        f"{prefix}_wpli": on,
        f"{prefix}_non_wpli": off,
        f"{prefix}_selective_ratio": on / (off + EPS),
    }


def random_backbone_features(matrix: np.ndarray, masks: list[np.ndarray], prefix: str) -> dict[str, float]:
    on_values = []
    off_values = []
    ratios = []
    for mask in masks:
        values = backbone_features(matrix, mask, prefix)
        on_values.append(values[f"{prefix}_wpli"])
        off_values.append(values[f"{prefix}_non_wpli"])
        ratios.append(values[f"{prefix}_selective_ratio"])
    return {
        f"{prefix}_wpli": float(np.mean(on_values)),
        f"{prefix}_non_wpli": float(np.mean(off_values)),
        f"{prefix}_selective_ratio": float(np.mean(ratios)),
    }


def make_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", pre), ("ridge", RidgeCV(alphas=ALPHAS))])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 20000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, n_perm: int = 100000) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)), replace=True)
    null = np.mean(signs * values[None, :], axis=1)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    participants = pd.read_csv(DATASET_ROOT / "participants.tsv", sep="\t")
    participants = participants.rename(columns={"Gender": "gender", "Age": "age", "Group": "group", "MMSE": "mmse"})
    participants["age"] = participants["age"].astype(float)
    participants["mmse"] = participants["mmse"].astype(float)
    participants["gender"] = participants["gender"].astype(str)
    participants["group"] = participants["group"].astype(str)
    participants["diagnosis"] = participants["group"].where(participants["group"].isin(["A", "F"]), "C")

    subject_band = pd.read_csv(FEATURES_CSV)
    subject_band = subject_band[subject_band["band"].isin(THEORY_BANDS)].copy()

    windows = pd.read_csv(WINDOWS_CSV)
    windows = windows[windows["band"].isin(THEORY_BANDS)].copy()
    window_map: dict[str, dict[str, np.ndarray]] = {band: {} for band in THEORY_BANDS}
    for (band, subject_id), group in windows.groupby(["band", "participant_id"]):
        window_map[band][subject_id] = group[GCC_WINDOW_FEATURES].to_numpy(dtype=float)

    wpli_npz = np.load(WPLI_NPZ)
    wpli_map: dict[str, dict[str, np.ndarray]] = {band: {} for band in THEORY_BANDS}
    for key in wpli_npz.files:
        subject_id, band = key.split("__")
        if band in THEORY_BANDS:
            wpli_map[band][subject_id] = wpli_npz[key].astype(float)

    return participants, subject_band, window_map, wpli_map


def build_features_for_fold(
    participants: pd.DataFrame,
    subject_band: pd.DataFrame,
    window_map: dict[str, dict[str, np.ndarray]],
    wpli_map: dict[str, dict[str, np.ndarray]],
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
        raise RuntimeError("No training controls found for fold calibration.")

    sb_lookup = subject_band.set_index(["participant_id", "band"])
    rows: list[dict] = []

    fold_calibration: dict[str, dict] = {}
    for band in THEORY_BANDS:
        train_arrays = [window_map[band][sid] for sid in train_controls]
        train_windows = np.vstack(train_arrays)
        bounds = {}
        for i, feature in enumerate(GCC_WINDOW_FEATURES):
            bounds[feature] = (
                float(np.quantile(train_windows[:, i], 0.10)),
                float(np.quantile(train_windows[:, i], 0.90)),
            )

        control_mats = [wpli_map[band][sid] for sid in train_controls]
        reference = np.mean(control_mats, axis=0)
        real_mask = top_backbone_mask(reference, BACKBONE_TOP_FRACTION)
        n_edges = int(np.sum(np.triu(real_mask, k=1)))
        random_masks = random_backbone_masks(reference.shape[0], n_edges, N_RANDOM_BACKBONES, rng)
        fold_calibration[band] = {
            "bounds": bounds,
            "real_mask": real_mask,
            "random_masks": random_masks,
        }

    for _, meta in participants.iterrows():
        subject_id = meta["participant_id"]
        row = {
            "participant_id": subject_id,
            "group": meta["group"],
            "diagnosis": meta["diagnosis"],
            "gender": meta["gender"],
            "age": float(meta["age"]),
            "mmse": float(meta["mmse"]),
        }

        for band in THEORY_BANDS:
            sb = sb_lookup.loc[(subject_id, band)]
            row[f"{band}__log_power"] = float(np.log10(float(sb["band_power_mean"]) + EPS))
            row[f"{band}__mean_wpli"] = float(sb["mean_wpli"])
            row[f"{band}__D_eff_mean"] = float(sb["D_eff_mean"])
            row[f"{band}__M_tau_mean"] = float(sb["M_tau_mean"])

            arr = window_map[band][subject_id]
            inside = np.ones(arr.shape[0], dtype=bool)
            for i, feature in enumerate(GCC_WINDOW_FEATURES):
                lo, hi = fold_calibration[band]["bounds"][feature]
                inside &= (arr[:, i] >= lo) & (arr[:, i] <= hi)
            row[f"{band}__Pi_control_window"] = float(np.mean(inside))

            matrix = wpli_map[band][subject_id]
            row.update(backbone_features(matrix, fold_calibration[band]["real_mask"], f"{band}__real_backbone"))
            row.update(
                random_backbone_features(
                    matrix,
                    fold_calibration[band]["random_masks"],
                    f"{band}__random_backbone",
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    participants, subject_band, window_map, wpli_map = load_inputs()

    subjects = participants["participant_id"].to_numpy()
    strata = participants["group"].to_numpy()
    patient_ids = set(participants.loc[participants["group"].isin(["A", "F"]), "participant_id"])

    categorical = ["gender", "diagnosis"]
    clinical_numeric = ["age"]

    standard_numeric = ["age"]
    for band in THEORY_BANDS:
        standard_numeric.extend([f"{band}__log_power", f"{band}__mean_wpli"])

    gcc_dynamics_numeric = list(standard_numeric)
    for band in THEORY_BANDS:
        gcc_dynamics_numeric.extend(
            [
                f"{band}__Pi_control_window",
                f"{band}__D_eff_mean",
                f"{band}__M_tau_mean",
            ]
        )

    real_numeric = list(gcc_dynamics_numeric)
    random_numeric = list(gcc_dynamics_numeric)
    backbone_only_numeric = list(standard_numeric)
    for band in THEORY_BANDS:
        real_numeric.extend([f"{band}__real_backbone_wpli", f"{band}__real_backbone_selective_ratio"])
        random_numeric.extend([f"{band}__random_backbone_wpli", f"{band}__random_backbone_selective_ratio"])
        backbone_only_numeric.extend([f"{band}__real_backbone_wpli", f"{band}__real_backbone_selective_ratio"])

    models = {
        "A_clinical": (categorical, clinical_numeric),
        "B_standard_alpha_lowgamma": (categorical, standard_numeric),
        "C_gcc_dynamics_only": (categorical, gcc_dynamics_numeric),
        "D_gcc_real_backbone_compact": (categorical, real_numeric),
        "E_gcc_random_backbone_compact": (categorical, random_numeric),
        "F_real_backbone_only": (categorical, backbone_only_numeric),
    }

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    pred_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(subjects, strata), start=1):
        train_subjects = set(subjects[train_idx])
        test_subjects = set(subjects[test_idx])
        features = build_features_for_fold(participants, subject_band, window_map, wpli_map, train_subjects, rng)

        train = features[
            features["participant_id"].isin(train_subjects) & features["participant_id"].isin(patient_ids)
        ].copy()
        test = features[
            features["participant_id"].isin(test_subjects) & features["participant_id"].isin(patient_ids)
        ].copy()
        y_train = train["mmse"].to_numpy(dtype=float)
        y_test = test["mmse"].to_numpy(dtype=float)

        for model_name, (cat_cols, num_cols) in models.items():
            model = make_model(cat_cols, num_cols)
            model.fit(train[cat_cols + num_cols], y_train)
            y_pred = model.predict(test[cat_cols + num_cols])

            fold_metric = metrics(y_test, y_pred)
            fold_metric.update(
                {
                    "fold_id": fold_id,
                    "repeat": (fold_id - 1) // N_SPLITS,
                    "split": (fold_id - 1) % N_SPLITS,
                    "model": model_name,
                    "n_train_patients": int(len(train)),
                    "n_test_patients": int(len(test)),
                    "n_train_controls_for_calibration": int(
                        participants[
                            participants["participant_id"].isin(train_subjects)
                            & (participants["group"] == "C")
                        ].shape[0]
                    ),
                }
            )
            fold_rows.append(fold_metric)

            for subject_id, group, observed, predicted in zip(
                test["participant_id"], test["group"], y_test, y_pred
            ):
                pred_rows.append(
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

        if fold_id % 50 == 0:
            recent = pd.DataFrame(fold_rows)
            means = recent[recent["fold_id"] > fold_id - 50].groupby("model")["mae"].mean().to_dict()
            print(f"fold {fold_id:03d}: " + ", ".join(f"{k}={v:.3f}" for k, v in means.items()))

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
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
        summary_models[model_name] = metrics(
            sm["observed_mmse"].to_numpy(dtype=float),
            sm["predicted_mmse"].to_numpy(dtype=float),
        )

    pivot = subject_mean.pivot(
        index=["participant_id", "group", "observed_mmse"], columns="model", values="predicted_mmse"
    ).reset_index()
    for model_name in models:
        pivot[f"abs_error__{model_name}"] = np.abs(pivot["observed_mmse"] - pivot[model_name])

    improvements = {}
    comparisons = [
        ("B_minus_D_real", "B_standard_alpha_lowgamma", "D_gcc_real_backbone_compact"),
        ("C_minus_D_real", "C_gcc_dynamics_only", "D_gcc_real_backbone_compact"),
        ("E_random_minus_D_real", "E_gcc_random_backbone_compact", "D_gcc_real_backbone_compact"),
        ("B_minus_C_dynamics", "B_standard_alpha_lowgamma", "C_gcc_dynamics_only"),
        ("B_minus_F_backbone_only", "B_standard_alpha_lowgamma", "F_real_backbone_only"),
    ]
    for label, base, candidate in comparisons:
        values = pivot[f"abs_error__{base}"].to_numpy() - pivot[f"abs_error__{candidate}"].to_numpy()
        improvements[label] = {
            **bootstrap_ci(values, rng),
            "signflip_p_one_sided": sign_flip_p(values, rng),
        }

    fold_pivot = fold_df.pivot(index="fold_id", columns="model", values="mae")
    fold_comparisons = {}
    for label, base, candidate in comparisons:
        diff = fold_pivot[base] - fold_pivot[candidate]
        fold_comparisons[label] = {
            "mean": float(diff.mean()),
            "sd": float(diff.std()),
            "wilcoxon_p_two_sided": float(stats.wilcoxon(fold_pivot[base], fold_pivot[candidate]).pvalue),
        }

    summary = {
        "analysis": "Compact theory-driven leakage-free CV: only alpha and low-gamma Pi, D_eff, M_tau, and fold-local functional backbone features.",
        "primary_population": "AD + FTD patients only; controls used only for fold-local access-window and backbone calibration.",
        "n_subjects_total": int(len(participants)),
        "n_patients": int(len(patient_ids)),
        "n_controls": int((participants["group"] == "C").sum()),
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_folds": int(N_SPLITS * N_REPEATS),
        "theory_bands": THEORY_BANDS,
        "backbone_top_fraction": BACKBONE_TOP_FRACTION,
        "n_random_backbones_per_fold_band": N_RANDOM_BACKBONES,
        "models_subject_mean_out_of_sample": summary_models,
        "subject_level_mae_improvements_positive_means_candidate_better": improvements,
        "fold_level_mean_mae": fold_df.groupby("model")["mae"].mean().to_dict(),
        "fold_level_sd_mae": fold_df.groupby("model")["mae"].std().to_dict(),
        "fold_level_mae_improvements_positive_means_candidate_better": fold_comparisons,
        "outputs": {
            "predictions": str(OUT_PRED),
            "fold_metrics": str(OUT_FOLD),
            "subject_mean_predictions": str(OUT_SUBJECT),
            "plot": str(OUT_PLOT),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_df = pd.DataFrame(
        [{"model": name, "MAE": values["mae"], "R2": values["r2"]} for name, values in summary_models.items()]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = ["#8a8f98", "#4b77be", "#7a6bb8", "#2b9a66", "#c77c2b", "#5a9aa6"]
    axes[0].bar(plot_df["model"], plot_df["MAE"], color=colors)
    axes[0].set_ylabel("Subject-mean out-of-sample MAE (MMSE)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Compact theory-driven CV")

    c = subject_mean[subject_mean["model"] == "D_gcc_real_backbone_compact"]
    axes[1].scatter(c["observed_mmse"], c["predicted_mmse"], c="#2b9a66", alpha=0.85)
    lo = min(c["observed_mmse"].min(), c["predicted_mmse"].min())
    hi = max(c["observed_mmse"].max(), c["predicted_mmse"].max())
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[1].set_xlabel("Observed MMSE")
    axes[1].set_ylabel("Predicted MMSE")
    axes[1].set_title("GCC dynamics + real backbone")
    fig.savefig(OUT_PLOT, dpi=200)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PRED}")
    print(f"Wrote {OUT_FOLD}")
    print(f"Wrote {OUT_SUBJECT}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
