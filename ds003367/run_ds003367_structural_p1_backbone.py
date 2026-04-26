#!/usr/bin/env python
"""
Structural P1-backbone proxy analysis for OpenNeuro ds003367.

This script uses the DSI Studio connectometry database distributed with the
OpenNeuro/GitHub mirror of ds003367 (`sub-all.dz`). It does not claim to test
terminal lucidity or dynamic re-entry. It tests a narrower structural premise:
whether a preregistered ascending-arousal / thalamo-cortical residual backbone
is more preserved in recovered/healthy scans than in chronic DoC scans, and
whether that structural effect is stronger than random backbone controls.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut


RECOVERY_LONGITUDINAL = {
    "TCRp001",
    "TCRp003",
    "TCRp005",
    "TCRp006",
    "TCRp008",
    "TCRp013",
    "TCRp017",
    "TCRp018",
    "TCRp023",
}

CHRONIC_DOC = {
    "TCRp002",
    "TCRp019",
    "TCRp020",
    "TCRp021",
    "TCRp024",
    "TCRp025",
}

# A priori structural P1-backbone proxy.
# Coordinates are MNI-ish centers in the QSDR 2 mm template used by DSI Studio.
# The set deliberately emphasizes ascending arousal and thalamo-cortical relay
# white-matter/subcortical regions, not broad whole-brain integrity.
P1_ROIS = [
    ("mesopontine_tegmentum", (0.0, -28.0, -14.0), "aan_core"),
    ("midbrain_tegmentum", (0.0, -22.0, -8.0), "aan_core"),
    ("left_intralaminar_thalamus", (-8.0, -18.0, 8.0), "aan_core"),
    ("right_intralaminar_thalamus", (8.0, -18.0, 8.0), "aan_core"),
    ("left_hypothalamic_basal_forebrain", (-8.0, -4.0, -8.0), "aan_core"),
    ("right_hypothalamic_basal_forebrain", (8.0, -4.0, -8.0), "aan_core"),
    ("left_internal_capsule", (-22.0, -12.0, 10.0), "relay_tract"),
    ("right_internal_capsule", (22.0, -12.0, 10.0), "relay_tract"),
    ("left_thalamocortical_corona", (-24.0, -24.0, 28.0), "relay_tract"),
    ("right_thalamocortical_corona", (24.0, -24.0, 28.0), "relay_tract"),
    ("left_temporal_stem", (-34.0, -12.0, -10.0), "relay_tract"),
    ("right_temporal_stem", (34.0, -12.0, -10.0), "relay_tract"),
]

METRICS = ("qa", "dti_fa")


@dataclass(frozen=True)
class DzData:
    names: list[str]
    metrics: dict[str, np.ndarray]
    mask: np.ndarray
    rank_lookup: np.ndarray
    dim: np.ndarray
    voxel_to_mni: np.ndarray
    control_indices: np.ndarray


def decode_uint8_text(arr: np.ndarray) -> str:
    return bytes(np.asarray(arr, dtype=np.uint8)).decode("utf-8", errors="ignore").rstrip("\x00")


def parse_scan_name(name: str, occurrence: int = 1) -> dict[str, str]:
    m = re.match(r"sub-(TCR[cp]\d+)_ses-([^_]+)_dwi$", name)
    if not m:
        raise ValueError(f"Cannot parse subject/session from {name!r}")
    subject, session = m.group(1), m.group(2)
    if subject.startswith("TCRc"):
        group = "control"
    elif subject in CHRONIC_DOC and session == "late":
        group = "chronic_doc"
    elif subject in RECOVERY_LONGITUDINAL and session == "early":
        group = "recovery_early"
    elif subject in RECOVERY_LONGITUDINAL and session == "late":
        group = "recovery_late"
    elif session == "early":
        group = "acute_only"
    else:
        group = "other"
    scan_id = f"{name}__dup{occurrence}" if occurrence > 1 else name
    return {"scan_id": scan_id, "scan_name": name, "subject": subject, "session": session, "group": group}


def load_dz(path: Path) -> tuple[DzData, pd.DataFrame]:
    mat = loadmat(io.BytesIO(gzip.decompress(path.read_bytes())), squeeze_me=True, struct_as_record=False)
    names = [n.strip() for n in decode_uint8_text(mat["subject_names"]).splitlines() if n.strip()]
    dim = np.asarray(mat["dimension"], dtype=int)
    trans = np.asarray(mat["trans"], dtype=float)
    voxel_to_mni = trans.T if trans.shape == (4, 4) else trans.reshape(4, 4)
    mask = np.asarray(mat["mask"], dtype=bool)

    flat_mask = mask.ravel(order="F")
    rank_lookup = np.full(flat_mask.size, -1, dtype=int)
    rank_lookup[flat_mask] = np.arange(int(flat_mask.sum()))

    metrics: dict[str, np.ndarray] = {}
    for metric in METRICS:
        raw = np.asarray(mat[metric], dtype=np.float32)
        slope = float(np.asarray(mat[f"{metric}.slope"]).squeeze())
        inter = float(np.asarray(mat[f"{metric}.inter"]).squeeze())
        metrics[metric] = raw * slope + inter

    occurrences: dict[str, int] = {}
    rows = []
    for idx, name in enumerate(names):
        occurrences[name] = occurrences.get(name, 0) + 1
        meta = parse_scan_name(name, occurrences[name])
        meta["row_index"] = idx
        rows.append(meta)
    scans = pd.DataFrame(rows)
    control_indices = scans.index[scans["group"] == "control"].to_numpy()

    data = DzData(
        names=names,
        metrics=metrics,
        mask=mask,
        rank_lookup=rank_lookup,
        dim=dim,
        voxel_to_mni=voxel_to_mni,
        control_indices=control_indices,
    )
    return data, scans


def sphere_indices(data: DzData, center: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    inv = np.linalg.inv(data.voxel_to_mni)
    center_h = np.array([center[0], center[1], center[2], 1.0], dtype=float)
    center_ijk = inv @ center_h
    radius_vox = int(math.ceil(radius_mm / float(np.min(np.abs(np.diag(data.voxel_to_mni[:3, :3])))))) + 2

    i0, j0, k0 = np.round(center_ijk[:3]).astype(int)
    imin, imax = max(0, i0 - radius_vox), min(int(data.dim[0]) - 1, i0 + radius_vox)
    jmin, jmax = max(0, j0 - radius_vox), min(int(data.dim[1]) - 1, j0 + radius_vox)
    kmin, kmax = max(0, k0 - radius_vox), min(int(data.dim[2]) - 1, k0 + radius_vox)

    ranks: list[int] = []
    nrows = int(data.dim[0] * data.dim[1])
    shape = (nrows, int(data.dim[2]))
    center_xyz = np.asarray(center, dtype=float)
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            row = i + j * int(data.dim[0])
            for k in range(kmin, kmax + 1):
                if not data.mask[row, k]:
                    continue
                xyz = (data.voxel_to_mni @ np.array([i, j, k, 1.0], dtype=float))[:3]
                if float(np.linalg.norm(xyz - center_xyz)) <= radius_mm:
                    flat = np.ravel_multi_index((row, k), shape, order="F")
                    rank = int(data.rank_lookup[flat])
                    if rank >= 0:
                        ranks.append(rank)
    return np.asarray(sorted(set(ranks)), dtype=int)


def compute_roi_features(
    data: DzData,
    scans: pd.DataFrame,
    rois: list[tuple[str, tuple[float, float, float], str]],
    radius_mm: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    roi_indices: dict[str, np.ndarray] = {}
    rows = []
    for roi_name, center, component in rois:
        idx = sphere_indices(data, center, radius_mm)
        roi_indices[roi_name] = idx
        if len(idx) == 0:
            raise RuntimeError(f"ROI {roi_name} has no voxels in the QSDR mask")
        for metric in METRICS:
            values = data.metrics[metric][:, idx].mean(axis=1)
            ctrl = values[data.control_indices]
            ctrl_mean = float(np.mean(ctrl))
            ctrl_sd = float(np.std(ctrl, ddof=1))
            if not np.isfinite(ctrl_sd) or ctrl_sd < 1e-12:
                z = np.full_like(values, np.nan, dtype=float)
            else:
                z = (values - ctrl_mean) / ctrl_sd
            for scan_idx, value, z_value in zip(range(len(scans)), values, z):
                rows.append(
                    {
                        "scan_id": scans.loc[scan_idx, "scan_id"],
                        "scan_name": scans.loc[scan_idx, "scan_name"],
                        "subject": scans.loc[scan_idx, "subject"],
                        "session": scans.loc[scan_idx, "session"],
                        "group": scans.loc[scan_idx, "group"],
                        "roi": roi_name,
                        "component": component,
                        "metric": metric,
                        "n_voxels": int(len(idx)),
                        "value": float(value),
                        "control_z": float(z_value),
                    }
                )
    return pd.DataFrame(rows), roi_indices


def compute_scan_scores(data: DzData, scans: pd.DataFrame, roi_features: pd.DataFrame) -> pd.DataFrame:
    score = (
        roi_features.groupby(["scan_id", "component"])["control_z"]
        .mean()
        .unstack("component")
        .reset_index()
        .rename(columns={"aan_core": "p1_aan_core_z", "relay_tract": "p1_relay_tract_z"})
    )
    p1_all = roi_features.groupby("scan_id")["control_z"].mean().rename("p1_backbone_z").reset_index()
    score = score.merge(p1_all, on="scan_id", how="left")
    out = scans.merge(score, on="scan_id", how="left")

    for metric in METRICS:
        values = data.metrics[metric].mean(axis=1)
        ctrl = values[data.control_indices]
        out[f"global_{metric}"] = values
        out[f"global_{metric}_z"] = (values - np.mean(ctrl)) / np.std(ctrl, ddof=1)
    out["global_integrity_z"] = out[[f"global_{m}_z" for m in METRICS]].mean(axis=1)
    out["p1_specificity_residual"] = residualize(out["p1_backbone_z"].to_numpy(), out["global_integrity_z"].to_numpy())
    return out


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    res = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < 3:
        return res
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    res[ok] = y[ok] - X @ beta
    return res


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def group_test(scores: pd.DataFrame, feature: str, group_a: str, group_b: str) -> dict[str, float]:
    a = scores.loc[scores["group"] == group_a, feature].to_numpy(dtype=float)
    b = scores.loc[scores["group"] == group_b, feature].to_numpy(dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else float("nan")
    t_p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue if len(a) > 1 and len(b) > 1 else float("nan")
    return {
        "feature": feature,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)) if len(a) else float("nan"),
        "mean_b": float(np.mean(b)) if len(b) else float("nan"),
        "median_a": float(np.median(a)) if len(a) else float("nan"),
        "median_b": float(np.median(b)) if len(b) else float("nan"),
        "cohens_d_a_minus_b": cohens_d(a, b),
        "mannwhitney_p_two_sided": float(u_p),
        "welch_p_two_sided": float(t_p),
    }


def paired_longitudinal_test(scores: pd.DataFrame, feature: str) -> dict[str, float]:
    wide = scores[scores["subject"].isin(RECOVERY_LONGITUDINAL)].pivot_table(
        index="subject", columns="session", values=feature, aggfunc="mean"
    )
    wide = wide.dropna(subset=["early", "late"])
    delta = wide["late"] - wide["early"]
    if len(delta) > 1:
        try:
            wilcoxon_p = float(stats.wilcoxon(delta, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        t_p = float(stats.ttest_rel(wide["late"], wide["early"]).pvalue)
    else:
        wilcoxon_p = float("nan")
        t_p = float("nan")
    return {
        "feature": feature,
        "n_pairs": int(len(delta)),
        "mean_delta_late_minus_early": float(delta.mean()) if len(delta) else float("nan"),
        "median_delta_late_minus_early": float(delta.median()) if len(delta) else float("nan"),
        "wilcoxon_p_two_sided": wilcoxon_p,
        "paired_t_p_two_sided": t_p,
    }


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (x_train - mean) / sd, (x_test - mean) / sd


def loocv_logistic_auc(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    def score_for_labels(labels: np.ndarray) -> tuple[float, float]:
        probs = np.zeros(len(labels), dtype=float)
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(x):
            x_train, x_test = standardize_train_test(x[train_idx], x[test_idx])
            clf = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed)
            clf.fit(x_train, labels[train_idx])
            probs[test_idx[0]] = clf.predict_proba(x_test)[0, 1]
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, probs >= 0.5)
        return float(auc), float(acc)

    observed_auc, observed_acc = score_for_labels(y)
    perm_aucs = []
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        if len(np.unique(perm_y)) < 2:
            continue
        perm_auc, _ = score_for_labels(perm_y)
        perm_aucs.append(perm_auc)
    perm_aucs_arr = np.asarray(perm_aucs, dtype=float)
    p_upper = float((1 + np.sum(perm_aucs_arr >= observed_auc)) / (len(perm_aucs_arr) + 1))
    return {
        "auc": observed_auc,
        "accuracy": observed_acc,
        "permutation_p_upper": p_upper,
        "n_permutations": int(len(perm_aucs_arr)),
    }


def compute_random_backbone_controls(
    data: DzData,
    scores: pd.DataFrame,
    radius_mm: float,
    n_random: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    control_mean_fa = data.metrics["dti_fa"][data.control_indices].mean(axis=0)

    flat = np.where(data.mask.ravel(order="F"))[0]
    rows, zs = np.unravel_index(flat, data.mask.shape, order="F")
    is_ = rows % int(data.dim[0])
    js = rows // int(data.dim[0])
    ijk = np.column_stack([is_, js, zs])
    mni = (data.voxel_to_mni @ np.column_stack([ijk, np.ones(len(ijk))]).T).T[:, :3]

    candidate_rank = np.where((control_mean_fa > 0.18) & (mni[:, 2] > -35.0) & (mni[:, 2] < 60.0))[0]
    if len(candidate_rank) < len(P1_ROIS):
        raise RuntimeError("Not enough candidate voxels for random backbone controls")

    # Observed effects for reference.
    obs_cross = effect_recovery_late_minus_chronic(scores["p1_backbone_z"].to_numpy(), scores)
    obs_long = longitudinal_mean_delta(scores, "p1_backbone_z")

    rows_out = []
    n_rois = len(P1_ROIS)
    for rep in range(n_random):
        center_ranks = rng.choice(candidate_rank, size=n_rois, replace=False)
        roi_defs = []
        for ii, rank in enumerate(center_ranks):
            center = tuple(float(v) for v in mni[rank])
            roi_defs.append((f"random_{ii:02d}", center, "random"))
        rf, _ = compute_roi_features(data, scores, roi_defs, radius_mm)
        random_scan_score = rf.groupby("scan_id")["control_z"].mean().rename("random_backbone_z")
        merged = scores[["scan_id", "subject", "session", "group"]].merge(random_scan_score, on="scan_id", how="left")
        rows_out.append(
            {
                "replicate": rep,
                "cross_d_recovery_late_minus_chronic": effect_recovery_late_minus_chronic(
                    merged["random_backbone_z"].to_numpy(), merged
                ),
                "longitudinal_mean_delta": longitudinal_mean_delta(merged, "random_backbone_z"),
            }
        )
    null = pd.DataFrame(rows_out)
    summary = {
        "n_random_backbones": int(n_random),
        "candidate_voxels": int(len(candidate_rank)),
        "observed_cross_d_recovery_late_minus_chronic": float(obs_cross),
        "observed_longitudinal_mean_delta": float(obs_long),
        "random_cross_p_upper": float((1 + np.sum(null["cross_d_recovery_late_minus_chronic"] >= obs_cross)) / (n_random + 1)),
        "random_longitudinal_p_upper": float((1 + np.sum(null["longitudinal_mean_delta"] >= obs_long)) / (n_random + 1)),
        "random_cross_d_mean": float(null["cross_d_recovery_late_minus_chronic"].mean()),
        "random_cross_d_sd": float(null["cross_d_recovery_late_minus_chronic"].std(ddof=1)),
        "random_longitudinal_delta_mean": float(null["longitudinal_mean_delta"].mean()),
        "random_longitudinal_delta_sd": float(null["longitudinal_mean_delta"].std(ddof=1)),
    }
    return null, summary


def effect_recovery_late_minus_chronic(values: np.ndarray, meta: pd.DataFrame) -> float:
    tmp = meta.copy()
    tmp["_v"] = values
    a = tmp.loc[tmp["group"] == "recovery_late", "_v"].to_numpy(dtype=float)
    b = tmp.loc[tmp["group"] == "chronic_doc", "_v"].to_numpy(dtype=float)
    return cohens_d(a, b)


def longitudinal_mean_delta(scores: pd.DataFrame, feature: str) -> float:
    wide = scores[scores["subject"].isin(RECOVERY_LONGITUDINAL)].pivot_table(
        index="subject", columns="session", values=feature, aggfunc="mean"
    )
    wide = wide.dropna(subset=["early", "late"])
    if wide.empty:
        return float("nan")
    return float((wide["late"] - wide["early"]).mean())


def run_model_comparison(scores: pd.DataFrame, n_perm: int, seed: int) -> dict[str, dict[str, float]]:
    subset = scores[scores["group"].isin(["recovery_late", "chronic_doc"])].copy()
    subset = subset.dropna(subset=["p1_backbone_z", "global_integrity_z", "p1_aan_core_z", "p1_relay_tract_z"])
    y = (subset["group"] == "recovery_late").astype(int).to_numpy()
    models = {
        "global_integrity": subset[["global_integrity_z"]].to_numpy(dtype=float),
        "p1_backbone": subset[["p1_backbone_z"]].to_numpy(dtype=float),
        "global_plus_p1": subset[["global_integrity_z", "p1_backbone_z"]].to_numpy(dtype=float),
        "p1_components": subset[["p1_aan_core_z", "p1_relay_tract_z"]].to_numpy(dtype=float),
    }
    return {name: loocv_logistic_auc(x, y, n_perm=n_perm, seed=seed) | {"n": int(len(y))} for name, x in models.items()}


def make_score_plot(scores: pd.DataFrame, out_path: Path) -> None:
    group_order = ["control", "recovery_early", "recovery_late", "chronic_doc", "acute_only"]
    labels = ["Control", "Recovery early", "Recovery late", "Chronic DoC", "Acute only"]
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    data = [scores.loc[scores["group"] == g, "p1_backbone_z"].dropna().to_numpy() for g in group_order]
    ax.boxplot(data, positions=np.arange(len(group_order)), widths=0.55, showfliers=False, patch_artist=True)
    colors = ["#5B8DEF", "#A26DDC", "#36A269", "#D14F45", "#8A8A8A"]
    for i, vals in enumerate(data):
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=38, color=colors[i], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("P1 structural backbone preservation (control-z)")
    ax.set_title("ds003367 structural P1-backbone proxy")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_random_plot(null: pd.DataFrame, random_summary: dict[str, float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    axes[0].hist(null["cross_d_recovery_late_minus_chronic"], bins=28, color="#B8C7D9", edgecolor="white")
    axes[0].axvline(random_summary["observed_cross_d_recovery_late_minus_chronic"], color="#D14F45", linewidth=2)
    axes[0].set_xlabel("Random-backbone Cohen d\nRecovery late minus Chronic DoC")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cross-sectional specificity")
    axes[1].hist(null["longitudinal_mean_delta"], bins=28, color="#B8C7D9", edgecolor="white")
    axes[1].axvline(random_summary["observed_longitudinal_mean_delta"], color="#D14F45", linewidth=2)
    axes[1].set_xlabel("Random-backbone mean delta\nlate minus early")
    axes[1].set_title("Longitudinal specificity")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(
    out_path: Path,
    data_path: Path,
    scores: pd.DataFrame,
    group_tests: list[dict[str, float]],
    longitudinal: list[dict[str, float]],
    model_comparison: dict[str, dict[str, float]],
    random_summary: dict[str, float],
) -> None:
    counts = scores["group"].value_counts().to_dict()
    gt = pd.DataFrame(group_tests)
    lt = pd.DataFrame(longitudinal)
    model = pd.DataFrame(model_comparison).T.reset_index().rename(columns={"index": "model"})

    lines = [
        "# ds003367 Structural P1-Backbone Proxy Analysis",
        "",
        f"Input: `{data_path}`",
        "",
        "## What This Tests",
        "",
        "This analysis tests a structural prerequisite of P1: whether a predefined ascending-arousal / thalamo-cortical backbone is more preserved in recovered/healthy scans than in chronic post-traumatic DoC scans.",
        "",
        "It does not test terminal lucidity, conscious report, or dynamic re-entry. It is a structural constraint layer.",
        "",
        "## Cohorts",
        "",
        json.dumps(counts, indent=2),
        "",
        "Primary cross-sectional contrast: `recovery_late` vs `chronic_doc`.",
        "Primary longitudinal contrast: paired `recovery_early` to `recovery_late` in the nine recovery subjects.",
        "",
        "## Group Tests",
        "",
        markdown_table(gt),
        "",
        "## Longitudinal Tests",
        "",
        markdown_table(lt),
        "",
        "## Leakage-Free Compact Model Comparison",
        "",
        "Outcome: recovery_late vs chronic_doc. Model assessment uses leave-one-out CV with scaling fit inside each training fold and label permutations.",
        "",
        markdown_table(model),
        "",
        "## Random-Backbone Specificity Controls",
        "",
        json.dumps(random_summary, indent=2),
        "",
        "## Manuscript-Safe Interpretation",
        "",
        "- Positive result: supports the structural plausibility of a residual backbone constraint in traumatic DoC/recovery data.",
        "- Negative or mixed result: constrains P1 by showing that this public HARDI-only dataset does not isolate the predicted structural substrate.",
        "- In either case, this should be framed as an independent structural proxy/constraint, not as empirical validation of terminal lucidity or re-entry dynamics.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    """Small local markdown-table formatter to avoid optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "nan" if pd.isna(x) else f"{float(x):.{digits}g}")
        else:
            out[col] = out[col].astype(str)
    headers = list(out.columns)
    rows = out.values.tolist()
    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(str(header)), *(len(str(row[idx])) for row in rows)))
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/ds003367"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--radius-mm", type=float, default=8.0)
    parser.add_argument("--n-random", type=int, default=500)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "sub-all.dz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    data, scans = load_dz(data_path)
    roi_features, roi_indices = compute_roi_features(data, scans, P1_ROIS, args.radius_mm)
    scores = compute_scan_scores(data, scans, roi_features)

    group_tests = []
    for feature in ["p1_backbone_z", "p1_aan_core_z", "p1_relay_tract_z", "global_integrity_z", "p1_specificity_residual"]:
        group_tests.append(group_test(scores, feature, "recovery_late", "chronic_doc"))
        group_tests.append(group_test(scores, feature, "control", "chronic_doc"))

    longitudinal = []
    for feature in ["p1_backbone_z", "p1_aan_core_z", "p1_relay_tract_z", "global_integrity_z", "p1_specificity_residual"]:
        longitudinal.append(paired_longitudinal_test(scores, feature))

    model_comparison = run_model_comparison(scores, n_perm=args.n_permutations, seed=args.seed)
    null, random_summary = compute_random_backbone_controls(
        data, scores, radius_mm=args.radius_mm, n_random=args.n_random, seed=args.seed
    )

    roi_features.to_csv(args.results_dir / "ds003367_structural_p1_roi_features.csv", index=False)
    scores.to_csv(args.results_dir / "ds003367_structural_p1_scan_scores.csv", index=False)
    pd.DataFrame(group_tests).to_csv(args.results_dir / "ds003367_structural_p1_group_tests.csv", index=False)
    pd.DataFrame(longitudinal).to_csv(args.results_dir / "ds003367_structural_p1_longitudinal_tests.csv", index=False)
    pd.DataFrame(model_comparison).T.to_csv(args.results_dir / "ds003367_structural_p1_model_comparison.csv")
    null.to_csv(args.results_dir / "ds003367_structural_p1_random_backbones.csv", index=False)

    summary = {
        "dataset": "OpenNeuro ds003367 / data-openneuro disease release ds003367",
        "input": str(data_path),
        "n_scans": int(len(scans)),
        "group_counts": {k: int(v) for k, v in scores["group"].value_counts().to_dict().items()},
        "radius_mm": float(args.radius_mm),
        "roi_voxel_counts": {name: int(len(idx)) for name, idx in roi_indices.items()},
        "group_tests": group_tests,
        "longitudinal_tests": longitudinal,
        "model_comparison": model_comparison,
        "random_backbone_controls": random_summary,
        "non_claim": "Structural HARDI/QSDR proxy only; does not empirically test terminal lucidity or dynamic P1 re-entry.",
    }
    (args.results_dir / "ds003367_structural_p1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_score_plot(scores, args.figures_dir / "ds003367_structural_p1_backbone_scores.png")
    make_random_plot(null, random_summary, args.figures_dir / "ds003367_structural_p1_random_controls.png")
    write_report(
        args.results_dir / "ds003367_structural_p1_report.md",
        data_path,
        scores,
        group_tests,
        longitudinal,
        model_comparison,
        random_summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
