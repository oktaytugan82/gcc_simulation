"""Run GCC re-entry simulations on HCP-derived connectome topologies.

This analysis uses the Budapest Reference Connectome matrices prepared by
``prepare_budapest_connectome.py``. It tests whether the model's selective
backbone re-entry signature is stronger on the empirical HCP-derived topology
than on degree-preserving and degree+strength-preserving random controls.

The script is deliberately conservative:
  * the backbone is pre-defined as the top strength nodes of the real topology;
  * the same backbone node set is used in the null graphs;
  * binary degree is preserved exactly by double-edge swaps;
  * the stricter null additionally matches node strengths by iterative scaling;
  * isolated nodes outside the largest connected component are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = Path(
    os.environ.get("BUDAPEST_CONNECTOME_DIR", SCRIPT_DIR / "data" / "budapest_connectome")
)
MATRIX_NPZ = DATA_DIR / "budapest_connectome_matrices.npz"
if not MATRIX_NPZ.exists():
    bundled_matrix = REPO_ROOT / "data_manifests" / "budapest_connectome_matrices.npz"
    if bundled_matrix.exists():
        MATRIX_NPZ = bundled_matrix

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
OUT_RESULTS = RESULTS_DIR / "hcp_gcc_connectome_simulation_results.csv"
OUT_NULL_STATS = RESULTS_DIR / "hcp_gcc_connectome_null_model_stats.csv"
OUT_COMPARISON = RESULTS_DIR / "hcp_gcc_connectome_group_comparison.csv"
OUT_SUMMARY = RESULTS_DIR / "hcp_gcc_connectome_summary.json"
OUT_REENTRY_PLOT = FIGURES_DIR / "hcp_gcc_connectome_reentry_comparison.png"
OUT_ALPHA_PLOT = FIGURES_DIR / "hcp_gcc_connectome_structural_alpha.png"

EPS = 1e-12


VARIANT_LABELS = {
    "brc_v3_20k_fibercount_conf50_default": "20k FC 50%",
    "brc_v3_20k_fibercount_conf25": "20k FC 25%",
    "brc_v3_20k_fibercount_conf10": "20k FC 10%",
    "brc_v3_200k_fibercount_conf50": "200k FC 50%",
    "brc_v3_1m_fibercount_conf50": "1m FC 50%",
    "brc_v3_20k_electrical_conf50": "20k EC 50%",
}


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.04
    total_time: float = 12.0
    event_start: float = 5.0
    event_end: float = 8.0
    baseline_start: float = 2.0
    baseline_end: float = 4.5
    base_freq_hz: float = 1.0
    freq_std_hz: float = 0.10
    sigma0: float = 0.22
    s_backbone: float = 0.65
    s_other: float = 0.25
    phi_base: float = 0.55
    phi_backbone_event: float = 0.95
    phi_other_event: float = 0.30
    backbone_global_gain: float = 0.10
    backbone_fraction: float = 0.10
    min_backbone_nodes: int = 20
    max_backbone_nodes: int = 120
    n_controls: int = 20
    degree_swap_factor: int = 12
    strength_match_iterations: int = 600
    seed: int = 20260426


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def np_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): np_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [np_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return safe_float(obj)
    return obj


def variant_ids_from_npz(keys: Iterable[str]) -> list[str]:
    ids = []
    for key in keys:
        if key.endswith("_weight"):
            ids.append(key.removesuffix("_weight"))
    return sorted(ids)


def largest_connected_component(A: np.ndarray) -> np.ndarray:
    binary = A > 0
    n = binary.shape[0]
    seen = np.zeros(n, dtype=bool)
    best: list[int] = []
    neighbors = [np.flatnonzero(binary[i]) for i in range(n)]
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in neighbors[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(int(nbr))
        if len(comp) > len(best):
            best = comp
    return np.asarray(sorted(best), dtype=int)


def top_strength_backbone(A: np.ndarray, cfg: SimConfig) -> np.ndarray:
    n = A.shape[0]
    k = int(round(cfg.backbone_fraction * n))
    k = max(cfg.min_backbone_nodes, min(cfg.max_backbone_nodes, k, n - 1))
    strengths = A.sum(axis=1)
    order = np.argsort(strengths)[::-1]
    return np.asarray(sorted(order[:k]), dtype=int)


def edge_arrays_from_matrix(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices_from(A, k=1)
    mask = A[rows, cols] > 0
    return rows[mask].astype(np.int32), cols[mask].astype(np.int32), A[rows[mask], cols[mask]].astype(float)


def row_normalized_directed_edges(
    i: np.ndarray, j: np.ndarray, weights: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strengths = np.zeros(n, dtype=float)
    np.add.at(strengths, i, weights)
    np.add.at(strengths, j, weights)

    src = np.concatenate([i, j]).astype(np.int32)
    dst = np.concatenate([j, i]).astype(np.int32)
    raw_w = np.concatenate([weights, weights]).astype(float)
    norm_w = raw_w / np.maximum(strengths[src], EPS)
    return src, dst, norm_w


def edge_set_from_arrays(i: np.ndarray, j: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(i, j)}


def double_edge_swap(
    i: np.ndarray,
    j: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
    swap_factor: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    edges = [tuple(sorted((int(a), int(b)))) for a, b in zip(i, j)]
    edge_set = set(edges)
    n_edges = len(edges)
    target_swaps = max(n_edges * swap_factor, n_edges)
    max_attempts = target_swaps * 30
    swaps = 0
    attempts = 0

    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        e1_idx, e2_idx = rng.choice(n_edges, size=2, replace=False)
        a, b = edges[int(e1_idx)]
        c, d = edges[int(e2_idx)]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new1 = tuple(sorted((a, d)))
            new2 = tuple(sorted((c, b)))
        else:
            new1 = tuple(sorted((a, c)))
            new2 = tuple(sorted((b, d)))
        if new1[0] == new1[1] or new2[0] == new2[1] or new1 == new2:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue

        edge_set.remove(edges[int(e1_idx)])
        edge_set.remove(edges[int(e2_idx)])
        edge_set.add(new1)
        edge_set.add(new2)
        edges[int(e1_idx)] = new1
        edges[int(e2_idx)] = new2
        swaps += 1

    out = np.asarray(edges, dtype=np.int32)
    return out[:, 0], out[:, 1], {"requested_swaps": target_swaps, "completed_swaps": swaps, "attempts": attempts}


def match_node_strengths(
    i: np.ndarray,
    j: np.ndarray,
    initial_weights: np.ndarray,
    target_strengths: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.asarray(initial_weights, dtype=float).copy()
    weights = np.maximum(weights, EPS)
    n = target_strengths.size
    positive_target = target_strengths > EPS

    for _ in range(iterations):
        current = np.zeros(n, dtype=float)
        np.add.at(current, i, weights)
        np.add.at(current, j, weights)
        ratio = np.ones(n, dtype=float)
        ratio[positive_target] = target_strengths[positive_target] / np.maximum(current[positive_target], EPS)
        ratio[~positive_target] = 1.0
        weights *= np.sqrt(ratio[i] * ratio[j])

    final = np.zeros(n, dtype=float)
    np.add.at(final, i, weights)
    np.add.at(final, j, weights)
    rel_err = np.abs(final[positive_target] - target_strengths[positive_target]) / np.maximum(
        target_strengths[positive_target], EPS
    )
    return weights, {
        "strength_rel_error_mean": float(np.mean(rel_err)) if rel_err.size else 0.0,
        "strength_rel_error_median": float(np.median(rel_err)) if rel_err.size else 0.0,
        "strength_rel_error_max": float(np.max(rel_err)) if rel_err.size else 0.0,
    }


def build_null_graph(
    A: np.ndarray,
    rng: np.random.Generator,
    graph_type: str,
    cfg: SimConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    n = A.shape[0]
    i0, j0, weights0 = edge_arrays_from_matrix(A)
    strengths0 = A.sum(axis=1)
    i, j, swap_stats = double_edge_swap(i0, j0, n, rng, cfg.degree_swap_factor)
    permuted = rng.permutation(weights0)

    stats_out: dict[str, float | int | str] = {
        "graph_type": graph_type,
        "n_nodes": int(n),
        "n_edges": int(len(i)),
        **swap_stats,
    }
    if graph_type == "degree_strength_preserving":
        weights, strength_stats = match_node_strengths(
            i, j, permuted, strengths0, iterations=cfg.strength_match_iterations
        )
        stats_out.update(strength_stats)
    elif graph_type == "degree_preserving":
        weights = permuted
        current = np.zeros(n, dtype=float)
        np.add.at(current, i, weights)
        np.add.at(current, j, weights)
        positive = strengths0 > EPS
        rel_err = np.abs(current[positive] - strengths0[positive]) / np.maximum(strengths0[positive], EPS)
        stats_out.update(
            {
                "strength_rel_error_mean": float(np.mean(rel_err)),
                "strength_rel_error_median": float(np.median(rel_err)),
                "strength_rel_error_max": float(np.max(rel_err)),
            }
        )
    else:
        raise ValueError(f"Unknown null graph type: {graph_type}")

    A_null = np.zeros_like(A, dtype=float)
    A_null[i, j] = weights
    A_null[j, i] = weights
    return A_null, stats_out


def structural_backbone_metrics(A: np.ndarray, backbone: np.ndarray) -> dict[str, float | int]:
    n = A.shape[0]
    i, j, weights = edge_arrays_from_matrix(A)
    src, dst, w_norm = row_normalized_directed_edges(i, j, weights, n)
    is_b = np.zeros(n, dtype=bool)
    is_b[backbone] = True
    b_src = is_b[src]
    internal_directed = b_src & is_b[dst]
    external_directed = b_src & (~is_b[dst])

    outgoing_internal = np.zeros(n, dtype=float)
    outgoing_external = np.zeros(n, dtype=float)
    np.add.at(outgoing_internal, src[internal_directed], w_norm[internal_directed])
    np.add.at(outgoing_external, src[external_directed], w_norm[external_directed])

    raw_total = A.sum() / 2.0
    raw_internal = A[np.ix_(backbone, backbone)].sum() / 2.0
    incident = A[backbone, :].sum()
    raw_external = incident - 2.0 * raw_internal

    return {
        "n_nodes": int(n),
        "n_edges": int(len(weights)),
        "backbone_nodes": int(len(backbone)),
        "backbone_fraction": float(len(backbone) / n),
        "alpha_S_max": float(np.max(outgoing_external[backbone])) if len(backbone) else float("nan"),
        "external_mass_mean": float(np.mean(outgoing_external[backbone])) if len(backbone) else float("nan"),
        "internal_mass_mean": float(np.mean(outgoing_internal[backbone])) if len(backbone) else float("nan"),
        "internal_minus_external_mean": float(
            np.mean(outgoing_internal[backbone] - outgoing_external[backbone])
        )
        if len(backbone)
        else float("nan"),
        "raw_backbone_internal_weight": float(raw_internal),
        "raw_backbone_external_weight": float(raw_external),
        "raw_backbone_internal_fraction_of_incident": float((2.0 * raw_internal) / max(incident, EPS)),
        "raw_backbone_internal_fraction_of_total": float(raw_internal / max(raw_total, EPS)),
    }


def effective_dimensionality_window(theta_window: np.ndarray) -> tuple[float, float]:
    if theta_window.shape[0] < 4:
        return float("nan"), float("nan")
    x = np.cos(theta_window)
    x = x - x.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (svals**2) / max(theta_window.shape[0], 1)
    total = float(np.sum(eig))
    denom = float(np.sum(eig**2))
    d_eff = (total * total) / max(denom, EPS)
    return float(d_eff), float(d_eff / theta_window.shape[1])


def window_metrics(theta: np.ndarray, t: np.ndarray, mask: np.ndarray, backbone: np.ndarray) -> dict[str, float]:
    n = theta.shape[1]
    comp_mask = np.ones(n, dtype=bool)
    comp_mask[backbone] = False
    comp = np.flatnonzero(comp_mask)
    z = np.exp(1j * theta[mask])
    r_global = np.abs(np.mean(z, axis=1))
    r_backbone = np.abs(np.mean(z[:, backbone], axis=1))
    r_complement = np.abs(np.mean(z[:, comp], axis=1))
    d_eff, d_eff_norm = effective_dimensionality_window(theta[mask])
    return {
        "R_global": float(np.mean(r_global)),
        "R_backbone": float(np.mean(r_backbone)),
        "R_complement": float(np.mean(r_complement)),
        "A_R": float(np.mean(r_backbone - r_complement)),
        "M_tau": float(np.var(r_global, ddof=1)) if r_global.size > 1 else float("nan"),
        "D_eff": d_eff,
        "D_eff_norm": d_eff_norm,
        "window_start_s": float(np.min(t[mask])) if np.any(mask) else float("nan"),
        "window_stop_s": float(np.max(t[mask])) if np.any(mask) else float("nan"),
    }


def simulate_gcc_reentry(A: np.ndarray, backbone: np.ndarray, K: float, cfg: SimConfig, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    i, j, weights = edge_arrays_from_matrix(A)
    src, dst, w_norm = row_normalized_directed_edges(i, j, weights, n)

    t = np.arange(0.0, cfg.total_time + cfg.dt / 2.0, cfg.dt)
    theta = np.zeros((t.size, n), dtype=np.float32)
    theta[0] = rng.uniform(-np.pi, np.pi, size=n)
    omega = 2.0 * np.pi * (cfg.base_freq_hz + cfg.freq_std_hz * rng.normal(size=n))

    is_backbone = np.zeros(n, dtype=bool)
    is_backbone[backbone] = True
    non_backbone = ~is_backbone
    s = np.full(n, cfg.s_other, dtype=float)
    s[is_backbone] = cfg.s_backbone
    sigma = np.full(n, cfg.sigma0, dtype=float)
    sqrt_dt = math.sqrt(cfg.dt)

    lambda_baseline = K * cfg.phi_base * s
    phi_event = np.full(n, cfg.phi_other_event, dtype=float)
    phi_event[is_backbone] = cfg.phi_backbone_event
    lambda_event = K * (1.0 + cfg.backbone_global_gain) * phi_event * s

    for step in range(t.size - 1):
        current_t = t[step]
        if cfg.event_start <= current_t <= cfg.event_end:
            lambda_t = lambda_event
        else:
            lambda_t = lambda_baseline

        phase = theta[step].astype(float)
        coupling = np.zeros(n, dtype=float)
        np.add.at(coupling, src, w_norm * np.sin(phase[dst] - phase[src]))
        drift = omega + lambda_t * coupling
        theta[step + 1] = phase + drift * cfg.dt + sigma * sqrt_dt * rng.normal(size=n)

    baseline_mask = (t >= cfg.baseline_start) & (t <= cfg.baseline_end)
    event_mask = (t >= cfg.event_start) & (t <= cfg.event_end)
    baseline = window_metrics(theta, t, baseline_mask, backbone)
    event = window_metrics(theta, t, event_mask, backbone)

    out: dict[str, float] = {}
    for prefix, metrics in (("baseline", baseline), ("event", event)):
        for key, value in metrics.items():
            out[f"{prefix}_{key}"] = value
    for key in ["R_global", "R_backbone", "R_complement", "A_R", "M_tau", "D_eff", "D_eff_norm"]:
        out[f"delta_{key}"] = event[key] - baseline[key]

    out["Xi_baseline"] = float(
        np.mean(lambda_baseline[is_backbone]) / max(float(np.mean(lambda_baseline[non_backbone])), EPS)
    )
    out["Xi_event"] = float(
        np.mean(lambda_event[is_backbone]) / max(float(np.mean(lambda_event[non_backbone])), EPS)
    )
    out["delta_Xi"] = out["Xi_event"] - out["Xi_baseline"]
    return out


def summarize_comparisons(results: pd.DataFrame, null_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "event_A_R",
        "delta_A_R",
        "event_R_backbone",
        "delta_R_backbone",
        "event_R_global",
        "delta_R_global",
        "event_D_eff_norm",
        "delta_D_eff_norm",
        "event_M_tau",
        "delta_M_tau",
    ]
    for (variant_id, K), group in results.groupby(["variant_id", "K"]):
        real = group[group["graph_type"] == "real"]
        if real.empty:
            continue
        for null_type in ["degree_preserving", "degree_strength_preserving"]:
            null = group[group["graph_type"] == null_type]
            if null.empty:
                continue
            row: dict[str, object] = {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(str(variant_id), str(variant_id)),
                "K": float(K),
                "null_type": null_type,
                "n_real": int(len(real)),
                "n_null": int(len(null)),
            }
            for metric in metrics:
                real_values = real[metric].to_numpy(dtype=float)
                null_values = null[metric].to_numpy(dtype=float)
                real_mean = float(np.nanmean(real_values))
                null_mean = float(np.nanmean(null_values))
                row[f"{metric}_real_mean"] = real_mean
                row[f"{metric}_null_mean"] = null_mean
                row[f"{metric}_real_minus_null"] = real_mean - null_mean
                if stats is not None and np.isfinite(real_values).sum() >= 3 and np.isfinite(null_values).sum() >= 3:
                    try:
                        test = stats.mannwhitneyu(real_values, null_values, alternative="greater")
                        row[f"{metric}_mw_p_greater"] = float(test.pvalue)
                    except Exception:
                        row[f"{metric}_mw_p_greater"] = float("nan")
                else:
                    row[f"{metric}_mw_p_greater"] = float("nan")

            struct_group = null_stats[
                (null_stats["variant_id"] == variant_id) & (null_stats["graph_type"].isin(["real", null_type]))
            ]
            real_struct = struct_group[struct_group["graph_type"] == "real"]
            null_struct = struct_group[struct_group["graph_type"] == null_type]
            if not real_struct.empty and not null_struct.empty:
                real_alpha = float(real_struct["alpha_S_max"].iloc[0])
                null_alpha = null_struct["alpha_S_max"].to_numpy(dtype=float)
                row["alpha_S_max_real"] = real_alpha
                row["alpha_S_max_null_mean"] = float(np.nanmean(null_alpha))
                row["alpha_S_max_real_minus_null"] = real_alpha - float(np.nanmean(null_alpha))
                if stats is not None and np.isfinite(null_alpha).sum() >= 3:
                    row["alpha_S_max_p_real_lower_than_null"] = float((1.0 + np.sum(null_alpha <= real_alpha)) / (len(null_alpha) + 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(results: pd.DataFrame, null_stats: pd.DataFrame) -> None:
    if plt is None:
        return
    labels = [VARIANT_LABELS.get(v, v) for v in sorted(results["variant_id"].unique())]
    variants = sorted(results["variant_id"].unique())

    fig, axes = plt.subplots(len(variants), 1, figsize=(10, max(3, 2.2 * len(variants))), sharex=True)
    if len(variants) == 1:
        axes = [axes]
    colors = {
        "real": "#111111",
        "degree_preserving": "#5470c6",
        "degree_strength_preserving": "#91cc75",
    }
    for ax, variant_id, label in zip(axes, variants, labels):
        sub = results[results["variant_id"] == variant_id]
        for graph_type, g in sub.groupby("graph_type"):
            summary = g.groupby("K")["delta_A_R"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(
                summary["K"],
                summary["mean"],
                yerr=summary["sem"].fillna(0.0),
                marker="o",
                linewidth=1.6,
                color=colors.get(graph_type, None),
                label=graph_type,
            )
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Global coupling K")
    fig.supylabel("Delta backbone order advantage (event - baseline)")
    fig.tight_layout()
    fig.savefig(OUT_REENTRY_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants), dtype=float)
    width = 0.25
    for offset, graph_type in [(-width, "real"), (0.0, "degree_preserving"), (width, "degree_strength_preserving")]:
        means = []
        sems = []
        for variant_id in variants:
            values = null_stats[
                (null_stats["variant_id"] == variant_id) & (null_stats["graph_type"] == graph_type)
            ]["alpha_S_max"].to_numpy(dtype=float)
            means.append(float(np.nanmean(values)) if values.size else float("nan"))
            sems.append(float(np.nanstd(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
        ax.bar(x + offset, means, width=width, yerr=sems, label=graph_type, color=colors.get(graph_type, None), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=25, ha="right")
    ax.set_ylabel("alpha_S max: max outgoing mass from backbone to complement")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_ALPHA_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_k_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-controls", type=int, default=20)
    parser.add_argument("--k-values", type=str, default="1.6,2.4,3.2,4.0")
    parser.add_argument("--variant-regex", type=str, default=".*")
    parser.add_argument("--seed", type=int, default=20260426)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(n_controls=args.n_controls, seed=args.seed)
    k_values = parse_k_values(args.k_values)
    variant_pattern = re.compile(args.variant_regex)

    matrices = np.load(MATRIX_NPZ)
    variant_ids = [vid for vid in variant_ids_from_npz(matrices.files) if variant_pattern.search(vid)]
    if not variant_ids:
        raise RuntimeError("No connectome variants matched the requested pattern.")

    result_rows: list[dict[str, object]] = []
    null_stat_rows: list[dict[str, object]] = []

    for variant_idx, variant_id in enumerate(variant_ids):
        A_full = np.asarray(matrices[f"{variant_id}_weight"], dtype=float)
        np.fill_diagonal(A_full, 0.0)
        lcc = largest_connected_component(A_full)
        A = A_full[np.ix_(lcc, lcc)]
        backbone = top_strength_backbone(A, cfg)
        base_seed = cfg.seed + 10000 * variant_idx
        real_struct = structural_backbone_metrics(A, backbone)
        null_stat_rows.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                "graph_type": "real",
                "control_id": -1,
                "strength_rel_error_mean": 0.0,
                "strength_rel_error_median": 0.0,
                "strength_rel_error_max": 0.0,
                **real_struct,
            }
        )
        print(
            f"{variant_id}: LCC n={A.shape[0]}, edges={real_struct['n_edges']}, "
            f"backbone={len(backbone)}, alpha={real_struct['alpha_S_max']:.3f}"
        )

        graph_bank: list[tuple[str, int, np.ndarray]] = [("real", -1, A)]
        for graph_type in ["degree_preserving", "degree_strength_preserving"]:
            for control_id in range(cfg.n_controls):
                rng = np.random.default_rng(base_seed + (1 if graph_type == "degree_preserving" else 2) * 1000 + control_id)
                A_null, null_stats = build_null_graph(A, rng, graph_type, cfg)
                struct = structural_backbone_metrics(A_null, backbone)
                null_stat_rows.append(
                    {
                        "variant_id": variant_id,
                        "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                        "graph_type": graph_type,
                        "control_id": control_id,
                        **null_stats,
                        **struct,
                    }
                )
                graph_bank.append((graph_type, control_id, A_null))

        for K in k_values:
            for graph_type, control_id, graph_A in graph_bank:
                reps = cfg.n_controls if graph_type == "real" else 1
                for rep in range(reps):
                    sim_seed = base_seed + int(round(K * 1000)) + rep + (control_id + 1) * 100000
                    metrics = simulate_gcc_reentry(graph_A, backbone, K, cfg, sim_seed)
                    result_rows.append(
                        {
                            "variant_id": variant_id,
                            "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                            "graph_type": graph_type,
                            "control_id": control_id,
                            "replicate": rep,
                            "K": float(K),
                            "n_nodes_lcc": int(A.shape[0]),
                            "n_edges": int(structural_backbone_metrics(graph_A, backbone)["n_edges"]),
                            "backbone_nodes": int(len(backbone)),
                            "seed": int(sim_seed),
                            **metrics,
                        }
                    )

    results = pd.DataFrame(result_rows)
    null_stats = pd.DataFrame(null_stat_rows)
    comparison = summarize_comparisons(results, null_stats)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_RESULTS, index=False)
    null_stats.to_csv(OUT_NULL_STATS, index=False)
    comparison.to_csv(OUT_COMPARISON, index=False)
    plot_outputs(results, null_stats)

    primary = comparison[comparison["null_type"] == "degree_strength_preserving"].copy()
    if not primary.empty:
        best = primary.sort_values("delta_A_R_real_minus_null", ascending=False).head(5)
        primary_summary = best[
            [
                "variant_id",
                "K",
                "delta_A_R_real_mean",
                "delta_A_R_null_mean",
                "delta_A_R_real_minus_null",
                "delta_A_R_mw_p_greater",
                "alpha_S_max_real",
                "alpha_S_max_null_mean",
                "alpha_S_max_real_minus_null",
            ]
        ].to_dict(orient="records")
    else:
        primary_summary = []

    summary = {
        "config": asdict(cfg),
        "k_values": k_values,
        "variant_ids": variant_ids,
        "outputs": {
            "simulation_results": str(OUT_RESULTS),
            "null_model_stats": str(OUT_NULL_STATS),
            "group_comparison": str(OUT_COMPARISON),
            "summary_json": str(OUT_SUMMARY),
            "reentry_plot": str(OUT_REENTRY_PLOT) if OUT_REENTRY_PLOT.exists() else None,
            "structural_alpha_plot": str(OUT_ALPHA_PLOT) if OUT_ALPHA_PLOT.exists() else None,
        },
        "primary_comparison": {
            "null_type": "degree_strength_preserving",
            "metric": "delta_A_R",
            "interpretation": (
                "Positive real_minus_null means the empirical HCP-derived topology showed a larger event-driven "
                "increase in backbone order advantage than degree+strength-preserving null graphs."
            ),
            "top_rows": primary_summary,
        },
        "notes": [
            "Backbone is the top-strength 10% of the real largest connected component, bounded by 20 and 120 nodes.",
            "The same backbone node indices are used in the random controls.",
            "Degree-preserving controls use double-edge swaps and shuffled empirical weights.",
            "Degree+strength controls add iterative symmetric edge-weight scaling to match node strengths approximately.",
            "Rows are excluded only if outside the largest connected component of the corresponding empirical variant.",
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(np_jsonable(summary), indent=2), encoding="utf-8")

    print(f"Wrote {len(results)} simulation rows to {OUT_RESULTS}")
    print(f"Wrote {len(null_stats)} null/structural rows to {OUT_NULL_STATS}")
    print(f"Wrote {len(comparison)} comparison rows to {OUT_COMPARISON}")
    print(json.dumps(np_jsonable(summary["primary_comparison"]), indent=2))


if __name__ == "__main__":
    main()
