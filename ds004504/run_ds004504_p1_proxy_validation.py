from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"C:\Users\oktay\OneDrive\Dokumente\New project")
FEATURES_CSV = ROOT / "ds004504_gcc_features_by_band.csv"
WINDOWS_CSV = ROOT / "ds004504_gcc_window_features.csv"
WPLI_NPZ = ROOT / "ds004504_wpli_matrices.npz"

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_FEATURES = RESULTS / "ds004504_p1_proxy_subject_features.csv"
OUT_PRED = RESULTS / "ds004504_p1_proxy_cv_predictions.csv"
OUT_SUBJECT = RESULTS / "ds004504_p1_proxy_subject_mean_predictions.csv"
OUT_FOLD = RESULTS / "ds004504_p1_proxy_fold_metrics.csv"
OUT_SUMMARY = RESULTS / "ds004504_p1_proxy_validation_summary.json"
OUT_REPORT = RESULTS / "ds004504_p1_proxy_report.md"
OUT_BAR = FIGURES / "ds004504_p1_proxy_model_comparison.png"
OUT_SCATTER = FIGURES / "ds004504_p1_proxy_scatter.png"

BANDS = ["theta", "alpha", "beta", "low_gamma"]
P1_BANDS = ["alpha", "low_gamma"]
GCC_WINDOW_FEATURES = ["R", "D_eff", "M_tau"]
BACKBONE_TOP_FRACTION = 0.20
N_RANDOM_BACKBONES = 200
N_SPLITS = 5
N_REPEATS = 100
RANDOM_STATE = 20260426
ALPHAS = np.logspace(-4, 4, 41)
EPS = 1e-12


@dataclass(frozen=True)
class FeatureSet:
    numeric: list[str]
    categorical: list[str]


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def upper_mean(matrix: np.ndarray) -> float:
    return float(np.mean(upper_values(matrix)))


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
    masks = []
    edge_ids = np.arange(len(tri[0]))
    for _ in range(n_masks):
        chosen = rng.choice(edge_ids, size=n_edges, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        mask[tri[0][chosen], tri[1][chosen]] = True
        masks.append(mask | mask.T)
    return masks


def mask_stats(matrix: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    eye = np.eye(mask.shape[0], dtype=bool)
    off_mask = (~mask) & (~eye)
    on = float(np.mean(matrix[mask]))
    off = float(np.mean(matrix[off_mask]))
    ratio = on / (off + EPS)
    diff = on - off
    return on, off, ratio, diff


def load_subject_band_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["mmse"] = pd.to_numeric(df["mmse"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["group"] = df["group"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip()
    df["log_band_power_mean"] = np.log10(np.clip(df["band_power_mean"].astype(float), EPS, None))
    return df


def make_standard_wide(features: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["participant_id", "group", "gender", "age", "mmse"]
    base = features[id_cols].drop_duplicates("participant_id").copy()
    value_cols = [
        "R_mean",
        "D_eff_mean",
        "M_tau_mean",
        "mean_wpli",
        "log_band_power_mean",
        "band_power_mean",
    ]
    wide = features.pivot(index="participant_id", columns="band", values=value_cols)
    wide.columns = [f"{band}_{feature}" for feature, band in wide.columns]
    wide = wide.reset_index()
    out = base.merge(wide, on="participant_id", how="left")
    out["theta_alpha_log_power_ratio"] = out["theta_log_band_power_mean"] - out["alpha_log_band_power_mean"]
    out["lowgamma_alpha_log_power_ratio"] = out["low_gamma_log_band_power_mean"] - out["alpha_log_band_power_mean"]
    return out


def add_control_calibrated_pi(subjects: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    controls = windows[windows["group"].astype(str).str.strip() == "C"].copy()
    result = subjects.copy()
    for band in P1_BANDS:
        ctrl_band = controls[controls["band"] == band]
        bounds = {
            feature: (
                float(ctrl_band[feature].quantile(0.10)),
                float(ctrl_band[feature].quantile(0.90)),
            )
            for feature in GCC_WINDOW_FEATURES
        }
        rows = windows[windows["band"] == band].copy()
        inside = np.ones(len(rows), dtype=bool)
        for feature, (lo, hi) in bounds.items():
            inside &= rows[feature].between(lo, hi, inclusive="both").to_numpy()
        pi = (
            pd.DataFrame({"participant_id": rows["participant_id"].to_numpy(), f"{band}_control_pi": inside.astype(float)})
            .groupby("participant_id", as_index=False)
            .mean()
        )
        result = result.merge(pi, on="participant_id", how="left")
    return result


def add_control_backbone_features(subjects: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    matrices = np.load(WPLI_NPZ)
    controls = subjects[subjects["group"] == "C"]["participant_id"].tolist()
    result = subjects.copy()

    for band in P1_BANDS:
        control_stack = np.stack([matrices[f"{sid}__{band}"] for sid in controls], axis=0)
        control_ref = np.mean(control_stack, axis=0)
        true_mask = top_backbone_mask(control_ref, BACKBONE_TOP_FRACTION)
        n_edges = int(np.sum(true_mask) // 2)
        random_masks = random_backbone_masks(control_ref.shape[0], n_edges, N_RANDOM_BACKBONES, rng)

        ctrl_global = upper_mean(control_ref)
        ctrl_on, ctrl_off, ctrl_ratio, _ = mask_stats(control_ref, true_mask)

        band_rows = []
        for sid in subjects["participant_id"]:
            matrix = matrices[f"{sid}__{band}"]
            global_wpli = upper_mean(matrix)
            on, off, ratio, diff = mask_stats(matrix, true_mask)
            random_ratios = []
            random_diffs = []
            random_ons = []
            for random_mask in random_masks:
                random_on, random_off, random_ratio, random_diff = mask_stats(matrix, random_mask)
                random_ons.append(random_on)
                random_ratios.append(random_ratio)
                random_diffs.append(random_diff)
            random_ratio_mean = float(np.mean(random_ratios))
            random_diff_mean = float(np.mean(random_diffs))
            random_on_mean = float(np.mean(random_ons))

            band_rows.append(
                {
                    "participant_id": sid,
                    f"{band}_norm_backbone_on": on,
                    f"{band}_norm_backbone_off": off,
                    f"{band}_norm_backbone_ratio": ratio,
                    f"{band}_norm_backbone_diff": diff,
                    f"{band}_norm_backbone_on_rel_control": on / (ctrl_on + EPS),
                    f"{band}_norm_backbone_off_rel_control": off / (ctrl_off + EPS),
                    f"{band}_norm_global_rel_control": global_wpli / (ctrl_global + EPS),
                    f"{band}_selective_preservation": (on / (ctrl_on + EPS)) - (off / (ctrl_off + EPS)),
                    f"{band}_backbone_excess_over_random_ratio": ratio - random_ratio_mean,
                    f"{band}_backbone_excess_over_random_diff": diff - random_diff_mean,
                    f"{band}_random_backbone_on_mean": random_on_mean,
                    f"{band}_random_backbone_ratio_mean": random_ratio_mean,
                    f"{band}_random_backbone_diff_mean": random_diff_mean,
                }
            )
        result = result.merge(pd.DataFrame(band_rows), on="participant_id", how="left")
    return result


def build_p1_composite(subjects: pd.DataFrame) -> pd.DataFrame:
    result = subjects.copy()
    p1_cols = [
        "alpha_control_pi",
        "low_gamma_control_pi",
        "alpha_selective_preservation",
        "low_gamma_selective_preservation",
        "alpha_backbone_excess_over_random_ratio",
        "low_gamma_backbone_excess_over_random_ratio",
    ]
    patients = result["group"].isin(["A", "F"])
    z = result[p1_cols].copy()
    means = z.loc[patients].mean(axis=0)
    stds = z.loc[patients].std(axis=0, ddof=1).replace(0, np.nan)
    result["p1_proxy_composite"] = ((z - means) / stds).mean(axis=1)
    return result


def make_preprocessor(feature_set: FeatureSet) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_set.numeric),
            ("cat", categorical_pipe, feature_set.categorical),
        ],
        remainder="drop",
    )


def model_pipeline(feature_set: FeatureSet) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(feature_set)),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    rho = stats.spearmanr(y_true, y_pred).statistic
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return {
        "n_test": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(rho),
        "pearson_r": float(pearson),
    }


def stratification_labels(patients: pd.DataFrame) -> np.ndarray:
    # The patient sample is modest (36 AD, 23 FTD). Stratifying by diagnosis is
    # stable across 5 folds; adding MMSE bins creates sparse strata.
    return patients["group"].astype(str).to_numpy()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = RANDOM_STATE) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
    }


def signflip_p_one_sided(improvements: np.ndarray, seed: int = RANDOM_STATE, n_perm: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    improvements = np.asarray(improvements, dtype=float)
    observed = improvements.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(improvements)), replace=True)
    null_means = (signs * improvements).mean(axis=1)
    return float((np.sum(null_means >= observed) + 1) / (n_perm + 1))


def partial_spearman(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> dict[str, float]:
    ranked = df[[x, y] + covariates].rank()
    cov = ranked[covariates].to_numpy(dtype=float)
    cov = np.column_stack([np.ones(len(cov)), cov])
    x_resid = ranked[x].to_numpy(dtype=float) - cov @ np.linalg.lstsq(cov, ranked[x].to_numpy(dtype=float), rcond=None)[0]
    y_resid = ranked[y].to_numpy(dtype=float) - cov @ np.linalg.lstsq(cov, ranked[y].to_numpy(dtype=float), rcond=None)[0]
    rho, p_value = stats.pearsonr(x_resid, y_resid)
    return {"rho": float(rho), "p_value": float(p_value), "n": int(len(df))}


def run_cv(subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    patients = subjects[subjects["group"].isin(["A", "F"])].copy().reset_index(drop=True)
    y = patients["mmse"].to_numpy(dtype=float)
    strat = stratification_labels(patients)

    clinical = FeatureSet(numeric=["age"], categorical=["gender", "group"])
    global_activity = FeatureSet(
        numeric=[
            "age",
            "theta_log_band_power_mean",
            "alpha_log_band_power_mean",
            "beta_log_band_power_mean",
            "low_gamma_log_band_power_mean",
            "theta_alpha_log_power_ratio",
            "lowgamma_alpha_log_power_ratio",
        ],
        categorical=["gender", "group"],
    )
    standard_eeg = FeatureSet(
        numeric=global_activity.numeric
        + [
            f"{band}_{feature}"
            for band in BANDS
            for feature in ["R_mean", "D_eff_mean", "M_tau_mean", "mean_wpli"]
        ],
        categorical=["gender", "group"],
    )
    p1_core = FeatureSet(
        numeric=global_activity.numeric
        + [
            f"{band}_{feature}"
            for band in P1_BANDS
            for feature in [
                "control_pi",
                "norm_backbone_on_rel_control",
                "norm_backbone_off_rel_control",
                "selective_preservation",
                "backbone_excess_over_random_ratio",
            ]
        ],
        categorical=["gender", "group"],
    )
    standard_plus_p1 = FeatureSet(
        numeric=standard_eeg.numeric
        + [
            f"{band}_{feature}"
            for band in P1_BANDS
            for feature in [
                "control_pi",
                "norm_backbone_on_rel_control",
                "norm_backbone_off_rel_control",
                "selective_preservation",
                "backbone_excess_over_random_ratio",
            ]
        ],
        categorical=["gender", "group"],
    )
    random_backbone_control = FeatureSet(
        numeric=global_activity.numeric
        + [
            f"{band}_{feature}"
            for band in P1_BANDS
            for feature in ["control_pi", "random_backbone_ratio_mean", "random_backbone_diff_mean"]
        ],
        categorical=["gender", "group"],
    )

    model_specs = {
        "A_clinical": clinical,
        "B_global_activity": global_activity,
        "C_standard_eeg_fc": standard_eeg,
        "D_p1_core_true_backbone": p1_core,
        "E_standard_plus_p1": standard_plus_p1,
        "F_random_backbone_control": random_backbone_control,
    }

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    pred_rows = []
    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(patients, strat)):
        train = patients.iloc[train_idx].copy()
        test = patients.iloc[test_idx].copy()
        y_train = train["mmse"].to_numpy(dtype=float)
        y_test = test["mmse"].to_numpy(dtype=float)

        for model_name, feature_set in model_specs.items():
            pipe = model_pipeline(feature_set)
            pipe.fit(train, y_train)
            pred = pipe.predict(test)
            m = metrics(y_test, pred)
            fold_rows.append({"fold_id": fold_id, "model": model_name, **m})
            for sid, true, pred_value in zip(test["participant_id"], y_test, pred):
                pred_rows.append(
                    {
                        "fold_id": fold_id,
                        "participant_id": sid,
                        "model": model_name,
                        "true_mmse": float(true),
                        "pred_mmse": float(pred_value),
                        "abs_error": float(abs(true - pred_value)),
                    }
                )

    predictions = pd.DataFrame(pred_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    subject_mean = (
        predictions.groupby(["participant_id", "model"], as_index=False)
        .agg(true_mmse=("true_mmse", "mean"), pred_mmse=("pred_mmse", "mean"), abs_error=("abs_error", "mean"))
        .merge(patients[["participant_id", "group", "gender", "age"]], on="participant_id", how="left")
    )

    model_metrics = {}
    for model_name in model_specs:
        rows = subject_mean[subject_mean["model"] == model_name]
        model_metrics[model_name] = metrics(rows["true_mmse"].to_numpy(), rows["pred_mmse"].to_numpy())

    comparisons = {}
    wide_err = subject_mean.pivot(index="participant_id", columns="model", values="abs_error")
    for label, baseline, candidate in [
        ("standard_minus_standard_plus_p1", "C_standard_eeg_fc", "E_standard_plus_p1"),
        ("global_minus_p1_core", "B_global_activity", "D_p1_core_true_backbone"),
        ("random_minus_p1_core", "F_random_backbone_control", "D_p1_core_true_backbone"),
        ("clinical_minus_p1_core", "A_clinical", "D_p1_core_true_backbone"),
    ]:
        improvements = wide_err[baseline].to_numpy() - wide_err[candidate].to_numpy()
        comparisons[label] = {
            **bootstrap_mean_ci(improvements),
            "signflip_p_one_sided": signflip_p_one_sided(improvements),
            "positive_mean_means_candidate_better": True,
            "baseline": baseline,
            "candidate": candidate,
        }

    fold_comparisons = {}
    wide_fold = fold_metrics.pivot(index="fold_id", columns="model", values="mae")
    for label, baseline, candidate in [
        ("standard_minus_standard_plus_p1", "C_standard_eeg_fc", "E_standard_plus_p1"),
        ("global_minus_p1_core", "B_global_activity", "D_p1_core_true_backbone"),
        ("random_minus_p1_core", "F_random_backbone_control", "D_p1_core_true_backbone"),
    ]:
        improvements = wide_fold[baseline].to_numpy() - wide_fold[candidate].to_numpy()
        fold_comparisons[label] = {
            "mean": float(improvements.mean()),
            "sd": float(improvements.std(ddof=1)),
            "wilcoxon_p_two_sided": float(stats.wilcoxon(improvements).pvalue),
        }

    summary = {
        "analysis": "ds004504 P1-lite clinical proxy validation: MMSE prediction in AD+FTD patients from control-calibrated access-regime occupancy and normative residual-backbone preservation.",
        "interpretation_rule": "P1-compatible support would require true-backbone P1 features to predict MMSE beyond age/sex/diagnosis, global activity, standard EEG/FC features, and random-backbone controls.",
        "n_subjects_total": int(len(subjects)),
        "n_controls": int((subjects["group"] == "C").sum()),
        "n_patients": int((subjects["group"].isin(["A", "F"])).sum()),
        "n_ad": int((subjects["group"] == "A").sum()),
        "n_ftd": int((subjects["group"] == "F").sum()),
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "n_folds": int(N_SPLITS * N_REPEATS),
            "ridge_alphas": [float(ALPHAS.min()), float(ALPHAS.max()), int(len(ALPHAS))],
        },
        "calibration": {
            "controls_only_for_pi_and_normative_backbone": True,
            "p1_bands": P1_BANDS,
            "backbone_top_fraction": BACKBONE_TOP_FRACTION,
            "n_random_backbones_per_band": N_RANDOM_BACKBONES,
        },
        "models_subject_mean_out_of_sample": model_metrics,
        "subject_level_mae_improvements_positive_means_candidate_better": comparisons,
        "fold_level_mean_mae": fold_metrics.groupby("model")["mae"].mean().to_dict(),
        "fold_level_sd_mae": fold_metrics.groupby("model")["mae"].std().to_dict(),
        "fold_level_mae_improvements_positive_means_candidate_better": fold_comparisons,
    }

    return predictions, subject_mean, fold_metrics, summary


def add_partial_correlations(summary: dict, subjects: pd.DataFrame) -> None:
    patients = subjects[subjects["group"].isin(["A", "F"])].copy()
    patients["group_code"] = (patients["group"] == "F").astype(float)
    patients["gender_code"] = (patients["gender"] == "M").astype(float)
    cov_base = ["age", "group_code", "gender_code"]
    cov_global = cov_base + [
        "theta_log_band_power_mean",
        "alpha_log_band_power_mean",
        "beta_log_band_power_mean",
        "low_gamma_log_band_power_mean",
    ]
    summary["partial_spearman"] = {
        "p1_composite_adjusted_for_clinical": partial_spearman(patients, "p1_proxy_composite", "mmse", cov_base),
        "p1_composite_adjusted_for_clinical_and_power": partial_spearman(
            patients, "p1_proxy_composite", "mmse", cov_global
        ),
        "low_gamma_mean_wpli_unadjusted": {
            "rho": float(stats.spearmanr(patients["low_gamma_mean_wpli"], patients["mmse"]).statistic),
            "p_value": float(stats.spearmanr(patients["low_gamma_mean_wpli"], patients["mmse"]).pvalue),
            "n": int(len(patients)),
        },
    }
    key_features = [
        "low_gamma_mean_wpli",
        "low_gamma_norm_backbone_on",
        "low_gamma_norm_backbone_ratio",
        "low_gamma_control_pi",
        "low_gamma_selective_preservation",
        "alpha_control_pi",
        "alpha_selective_preservation",
    ]
    summary["key_feature_partial_spearman"] = {}
    for feature in key_features:
        unadjusted = stats.spearmanr(patients[feature], patients["mmse"])
        summary["key_feature_partial_spearman"][feature] = {
            "unadjusted": {"rho": float(unadjusted.statistic), "p_value": float(unadjusted.pvalue), "n": int(len(patients))},
            "adjusted_for_clinical": partial_spearman(patients, feature, "mmse", cov_base),
            "adjusted_for_clinical_and_power": partial_spearman(patients, feature, "mmse", cov_global),
        }


def write_report(summary: dict) -> None:
    mm = summary["models_subject_mean_out_of_sample"]
    cmp = summary["subject_level_mae_improvements_positive_means_candidate_better"]
    partials = summary["key_feature_partial_spearman"]
    lines = [
        "# ds004504 P1-Lite Proxy Validation",
        "",
        "## Design",
        "",
        f"- Population: AD+FTD patients only for MMSE prediction (n={summary['n_patients']}); controls (n={summary['n_controls']}) used only for control-calibrated access windows and normative backbone definition.",
        f"- Cross-validation: repeated stratified {summary['cv']['n_splits']}-fold CV, {summary['cv']['n_repeats']} repeats ({summary['cv']['n_folds']} folds).",
        f"- P1 features: alpha/low-gamma control-regime occupancy, normative backbone preservation, selective preservation, and random-backbone controls.",
        "",
        "## Main CV Result",
        "",
        f"- Clinical baseline MAE: {mm['A_clinical']['mae']:.3f}, Spearman rho: {mm['A_clinical']['spearman_rho']:.3f}.",
        f"- Global-activity model MAE: {mm['B_global_activity']['mae']:.3f}, Spearman rho: {mm['B_global_activity']['spearman_rho']:.3f}.",
        f"- Standard EEG+FC model MAE: {mm['C_standard_eeg_fc']['mae']:.3f}, Spearman rho: {mm['C_standard_eeg_fc']['spearman_rho']:.3f}.",
        f"- P1 true-backbone core model MAE: {mm['D_p1_core_true_backbone']['mae']:.3f}, Spearman rho: {mm['D_p1_core_true_backbone']['spearman_rho']:.3f}.",
        f"- Standard EEG+FC plus P1 model MAE: {mm['E_standard_plus_p1']['mae']:.3f}, Spearman rho: {mm['E_standard_plus_p1']['spearman_rho']:.3f}.",
        "",
        "## Incremental Tests",
        "",
        f"- Standard EEG+FC minus Standard+P1 MAE improvement: {cmp['standard_minus_standard_plus_p1']['mean']:.3f} (positive would favor P1; one-sided sign-flip p={cmp['standard_minus_standard_plus_p1']['signflip_p_one_sided']:.3f}).",
        f"- Global activity minus P1-core MAE improvement: {cmp['global_minus_p1_core']['mean']:.3f} (positive would favor P1; one-sided sign-flip p={cmp['global_minus_p1_core']['signflip_p_one_sided']:.3f}).",
        f"- Random-backbone minus P1-core MAE improvement: {cmp['random_minus_p1_core']['mean']:.3f} (positive would favor true backbone; one-sided sign-flip p={cmp['random_minus_p1_core']['signflip_p_one_sided']:.3f}).",
        "",
        "## Key Single-Feature Checks",
        "",
        f"- Low-gamma mean wPLI vs MMSE survives adjustment for clinical covariates and power: rho={partials['low_gamma_mean_wpli']['adjusted_for_clinical_and_power']['rho']:.3f}, p={partials['low_gamma_mean_wpli']['adjusted_for_clinical_and_power']['p_value']:.4f}.",
        f"- Low-gamma normative-backbone on-strength also survives adjustment: rho={partials['low_gamma_norm_backbone_on']['adjusted_for_clinical_and_power']['rho']:.3f}, p={partials['low_gamma_norm_backbone_on']['adjusted_for_clinical_and_power']['p_value']:.4f}.",
        f"- Low-gamma backbone ratio does not survive as a P1-selectivity marker: rho={partials['low_gamma_norm_backbone_ratio']['adjusted_for_clinical_and_power']['rho']:.3f}, p={partials['low_gamma_norm_backbone_ratio']['adjusted_for_clinical_and_power']['p_value']:.4f}.",
        f"- P1 composite adjusted for clinical covariates and power: rho={summary['partial_spearman']['p1_composite_adjusted_for_clinical_and_power']['rho']:.3f}, p={summary['partial_spearman']['p1_composite_adjusted_for_clinical_and_power']['p_value']:.4f}.",
        "",
        "## Interpretation",
        "",
        "This analysis provides a useful negative boundary. ds004504 supports a weaker claim that preserved low-gamma functional connectivity is associated with cognitive preservation in AD/FTD, but it does not validate the stronger P1 selective-preservation mechanism. In the manuscript, this should be framed as a clinical proxy stress test that constrains P1 rather than confirms it.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(subjects: pd.DataFrame, subject_mean: pd.DataFrame) -> None:
    order = [
        "A_clinical",
        "B_global_activity",
        "C_standard_eeg_fc",
        "D_p1_core_true_backbone",
        "E_standard_plus_p1",
        "F_random_backbone_control",
    ]
    labels = [
        "Clinical",
        "Global\nactivity",
        "Standard\nEEG+FC",
        "P1 core\ntrue backbone",
        "Standard\n+ P1",
        "Random\nbackbone",
    ]
    mae = []
    for model in order:
        rows = subject_mean[subject_mean["model"] == model]
        mae.append(mean_absolute_error(rows["true_mmse"], rows["pred_mmse"]))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#6b7280", "#4b6f8f", "#2563eb", "#b45309", "#059669", "#9ca3af"]
    ax.bar(labels, mae, color=colors)
    ax.set_ylabel("Subject-level out-of-sample MAE (MMSE points)")
    ax.set_title("ds004504 P1-lite proxy validation: model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_BAR, dpi=200)
    plt.close(fig)

    patients = subjects[subjects["group"].isin(["A", "F"])].copy()
    rho, p_value = stats.spearmanr(patients["p1_proxy_composite"], patients["mmse"])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for group, color, label in [("A", "#1d4ed8", "AD"), ("F", "#b45309", "FTD")]:
        rows = patients[patients["group"] == group]
        ax.scatter(rows["p1_proxy_composite"], rows["mmse"], color=color, label=label, alpha=0.85)
    x = patients["p1_proxy_composite"].to_numpy()
    y = patients["mmse"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(xx, slope * xx + intercept, color="#111827", linewidth=1.5)
    ax.set_xlabel("P1 proxy composite (control-calibrated)")
    ax.set_ylabel("MMSE")
    ax.set_title(f"P1 proxy vs. MMSE in AD+FTD (rho={rho:.2f}, p={p_value:.3f})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    features = load_subject_band_features()
    windows = pd.read_csv(WINDOWS_CSV)
    windows["group"] = windows["group"].astype(str).str.strip()
    windows["mmse"] = pd.to_numeric(windows["mmse"], errors="coerce")
    windows["age"] = pd.to_numeric(windows["age"], errors="coerce")

    subjects = make_standard_wide(features)
    subjects = add_control_calibrated_pi(subjects, windows)
    subjects = add_control_backbone_features(subjects)
    subjects = build_p1_composite(subjects)
    subjects.to_csv(OUT_FEATURES, index=False)

    predictions, subject_mean, fold_metrics, summary = run_cv(subjects)
    add_partial_correlations(summary, subjects)
    plot_outputs(subjects, subject_mean)

    predictions.to_csv(OUT_PRED, index=False)
    subject_mean.to_csv(OUT_SUBJECT, index=False)
    fold_metrics.to_csv(OUT_FOLD, index=False)

    summary["outputs"] = {
        "subject_features": str(OUT_FEATURES),
        "predictions": str(OUT_PRED),
        "subject_mean_predictions": str(OUT_SUBJECT),
        "fold_metrics": str(OUT_FOLD),
        "report": str(OUT_REPORT),
        "model_comparison_plot": str(OUT_BAR),
        "p1_proxy_scatter": str(OUT_SCATTER),
    }
    write_report(summary)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
