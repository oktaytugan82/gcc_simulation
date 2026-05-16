"""
Matched synthetic stress tests for GCC.

This complements the broad synthetic benchmark with pairwise controlled tests.
Each pair holds K, noise, lesion fraction, global gain, network type, and seed
fixed, and changes only the mechanism under test. This prevents trivial
baselines such as global gain from winning because of label leakage.

Tests:
1. Backbone re-entry vs uniform degradation at matched global gain.
2. Lesion sweep at fixed gain/noise.
3. Noise sweep at fixed topology/gain.
4. Network-type robustness of the matched re-entry effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from batch3_synthetic_benchmark import (
    apply_edge_lesion,
    calibrate,
    euler_simulate,
    make_g_vector,
    make_network,
    observables,
    score_metrics,
)


def simulate_scores(
    *,
    network: str,
    seed: int,
    n_nodes: int,
    mean_degree: int,
    lesion: float,
    mode: str,
    backbone: float,
    k: float,
    sigma: float,
    global_gain: float,
    t_seconds: float,
    dt: float,
    tau_d_steps: int,
    tau_m_steps: int,
    bounds: dict[str, float],
    subset_mask: np.ndarray | None = None,
) -> dict[str, float]:
    a_raw, _ = make_network(network, n_nodes, mean_degree, seed)
    _, w, g_pres = apply_edge_lesion(a_raw, lesion, seed + 1)
    g_local = make_g_vector(g_pres, mode, backbone, seed + 2)
    theta = euler_simulate(
        w,
        g_local,
        k=k,
        global_gain=global_gain,
        sigma=sigma,
        t=t_seconds,
        dt=dt,
        seed=seed + 3,
    )
    if subset_mask is not None:
        theta = theta[:, subset_mask]
    r, d, m = observables(theta, tau_d_steps, tau_m_steps)
    return score_metrics(r, d, m, bounds, global_gain)


def backbone_mask(n_nodes: int, backbone_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_nodes, dtype=bool)
    size = max(2, int(round(backbone_fraction * n_nodes)))
    mask[rng.choice(n_nodes, size=size, replace=False)] = True
    return mask


def g_for_mode(g_pres: np.ndarray, mode: str, mask: np.ndarray | None) -> np.ndarray:
    if mode == "healthy":
        return np.ones_like(g_pres)
    if mode == "uniform":
        return np.clip(0.30 * g_pres, 0.0, 1.0)
    if mode == "backbone":
        if mask is None:
            raise ValueError("backbone mode requires mask")
        g = np.full_like(g_pres, 0.10, dtype=float)
        g[mask] = np.maximum(g_pres[mask], 0.95)
        return np.clip(g, 0.0, 1.0)
    raise ValueError(mode)


def simulate_scores_matched(
    *,
    network: str,
    seed: int,
    n_nodes: int,
    mean_degree: int,
    lesion: float,
    mode: str,
    mask: np.ndarray | None,
    scope: str,
    k: float,
    sigma: float,
    global_gain: float,
    t_seconds: float,
    dt: float,
    tau_d_steps: int,
    tau_m_steps: int,
    bounds: dict[str, float],
) -> dict[str, float]:
    a_raw, _ = make_network(network, n_nodes, mean_degree, seed)
    _, w, g_pres = apply_edge_lesion(a_raw, lesion, seed + 1)
    g_local = g_for_mode(g_pres, mode, mask)
    theta = euler_simulate(
        w,
        g_local,
        k=k,
        global_gain=global_gain,
        sigma=sigma,
        t=t_seconds,
        dt=dt,
        seed=seed + 3,
    )
    if scope == "backbone":
        if mask is None:
            raise ValueError("backbone scope requires mask")
        theta = theta[:, mask]
    r, d, m = observables(theta, tau_d_steps, tau_m_steps)
    return score_metrics(r, d, m, bounds, global_gain)


def make_bounds(args: argparse.Namespace, network_types: list[str], tau_d_steps: int, tau_m_steps: int) -> dict[str, float]:
    rng = np.random.default_rng(args.seed)
    reference = []
    for network in network_types:
        for i in range(args.calib_seeds):
            a_raw, w = make_network(network, args.n_nodes, args.mean_degree, args.seed + 100 + i)
            theta = euler_simulate(
                w,
                np.ones(args.n_nodes),
                k=float(rng.choice([1.5, 1.8, 2.1, 2.4])),
                global_gain=1.0,
                sigma=0.10,
                t=args.t_seconds,
                dt=args.dt,
                seed=args.seed + 200 + i,
            )
            reference.append(observables(theta, tau_d_steps, tau_m_steps))
    return calibrate(reference, args.alpha)


def matched_reentry(args: argparse.Namespace, network_types: list[str], bounds: dict[str, float], tau_d_steps: int, tau_m_steps: int) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 1000)
    rows = []
    for i in range(args.n_pairs):
        network = network_types[i % len(network_types)]
        seed = args.seed + 10_000 + i
        lesion = float(rng.choice([0.35, 0.50, 0.65]))
        backbone = float(rng.choice([0.20, 0.30, 0.40]))
        k = float(rng.choice([1.7, 2.1, 2.5]))
        sigma = float(rng.choice([0.06, 0.10, 0.14]))
        global_gain = float(rng.choice([0.70, 0.90, 1.10]))
        mask = backbone_mask(args.n_nodes, backbone, seed + 2)
        common = dict(
            network=network,
            seed=seed,
            n_nodes=args.n_nodes,
            mean_degree=args.mean_degree,
            lesion=lesion,
            k=k,
            sigma=sigma,
            global_gain=global_gain,
            t_seconds=args.t_seconds,
            dt=args.dt,
            tau_d_steps=tau_d_steps,
            tau_m_steps=tau_m_steps,
            bounds=bounds,
        )
        for scope in ["full", "backbone"]:
            uniform = simulate_scores_matched(mode="uniform", mask=mask, scope=scope, **common)
            reentry = simulate_scores_matched(mode="backbone", mask=mask, scope=scope, **common)
            for condition, scores in [("uniform_degraded", uniform), ("backbone_reentry", reentry)]:
                rows.append(
                    {
                        "pair": i,
                        "network": network,
                        "scope": scope,
                        "condition": condition,
                        "lesion_fraction": lesion,
                        "backbone_fraction": backbone,
                        "K": k,
                        "sigma": sigma,
                        "global_gain": global_gain,
                        **scores,
                    }
                )
    return pd.DataFrame(rows)


def sweeps(args: argparse.Namespace, network_types: list[str], bounds: dict[str, float], tau_d_steps: int, tau_m_steps: int) -> pd.DataFrame:
    rows = []
    lesion_values = [0.0, 0.20, 0.35, 0.50, 0.65]
    noise_values = [0.04, 0.08, 0.12, 0.20, 0.32, 0.45]
    for i in range(args.n_sweep_seeds):
        network = network_types[i % len(network_types)]
        seed = args.seed + 30_000 + i
        for lesion in lesion_values:
            scores = simulate_scores(
                network=network,
                seed=seed,
                n_nodes=args.n_nodes,
                mean_degree=args.mean_degree,
                lesion=lesion,
                mode="uniform" if lesion > 0 else "healthy",
                backbone=0.0,
                k=2.0,
                sigma=0.10,
                global_gain=1.0,
                t_seconds=args.t_seconds,
                dt=args.dt,
                tau_d_steps=tau_d_steps,
                tau_m_steps=tau_m_steps,
                bounds=bounds,
            )
            rows.append({"sweep": "lesion", "network": network, "seed": seed, "x": lesion, **scores})
        for sigma in noise_values:
            scores = simulate_scores(
                network=network,
                seed=seed + 500,
                n_nodes=args.n_nodes,
                mean_degree=args.mean_degree,
                lesion=0.0,
                mode="healthy",
                backbone=0.0,
                k=2.0,
                sigma=sigma,
                global_gain=1.0,
                t_seconds=args.t_seconds,
                dt=args.dt,
                tau_d_steps=tau_d_steps,
                tau_m_steps=tau_m_steps,
                bounds=bounds,
            )
            rows.append({"sweep": "noise", "network": network, "seed": seed, "x": sigma, **scores})
    return pd.DataFrame(rows)


def summarize_matched(df: pd.DataFrame) -> dict[str, object]:
    metrics = [
        "score_gcc_triad",
        "score_r_only",
        "score_d_only",
        "score_m_only",
        "score_criticality_only",
        "score_global_gain_only",
    ]
    summary: dict[str, object] = {}
    for scope, scope_df in df.groupby("scope"):
        summary[scope] = {}
        wide = scope_df.pivot(index="pair", columns="condition", values=metrics)
        for metric in metrics:
            delta = wide[(metric, "backbone_reentry")] - wide[(metric, "uniform_degraded")]
            try:
                stat, p = wilcoxon(delta)
            except ValueError:
                stat, p = np.nan, np.nan
            summary[scope][metric] = {
                "mean_delta_reentry_minus_uniform": float(delta.mean()),
                "median_delta": float(delta.median()),
                "sd_delta": float(delta.std(ddof=1)),
                "wilcoxon_W": float(stat) if np.isfinite(stat) else None,
                "p": float(p) if np.isfinite(p) else None,
                "positive_pairs": int((delta > 0).sum()),
                "n_pairs": int(len(delta)),
            }
        by_network = []
        for network, sub in scope_df.groupby("network"):
            w = sub.pivot(index="pair", columns="condition", values="score_gcc_triad")
            d = w["backbone_reentry"] - w["uniform_degraded"]
            by_network.append({"network": network, "mean_delta_gcc": float(d.mean()), "n_pairs": int(len(d))})
        summary[scope]["by_network"] = by_network
    return summary


def summarize_sweeps(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for sweep, sub in df.groupby("sweep"):
        out[sweep] = {}
        for metric in ["score_gcc_triad", "score_r_only", "score_d_only", "score_m_only", "score_criticality_only"]:
            rho, p = spearmanr(sub["x"], sub[metric])
            out[sweep][metric] = {"spearman_rho": float(rho), "p": float(p)}
    return out


def plot_matched(df: pd.DataFrame, matched_summary: dict[str, object], sweeps_df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        "score_gcc_triad",
        "score_r_only",
        "score_d_only",
        "score_m_only",
        "score_criticality_only",
        "score_global_gain_only",
    ]
    labels = ["GCC triad", "R only", "D only", "M only", "Criticality", "Global gain"]
    deltas = [matched_summary["backbone"][m]["mean_delta_reentry_minus_uniform"] for m in metrics]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(labels, deltas, color=["#0f766e", "#64748b", "#64748b", "#64748b", "#94a3b8", "#94a3b8"])
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Mean paired delta: re-entry minus uniform")
    ax.set_title("Matched synthetic re-entry test on residual backbone S")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(outdir / "synthetic_matched_reentry_delta.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, sweep in zip(axes, ["lesion", "noise"]):
        sub = sweeps_df[sweeps_df["sweep"] == sweep]
        means = sub.groupby("x")["score_gcc_triad"].agg(["mean", "std"]).reset_index()
        ax.errorbar(means["x"], means["mean"], yerr=means["std"], marker="o", capsize=4, color="#0f766e")
        ax.set_xlabel("Lesion fraction" if sweep == "lesion" else "Noise sigma")
        ax.set_ylabel("GCC triad Pi")
        ax.set_title(f"{sweep.title()} sweep")
        ax.grid(alpha=0.25)
    fig.savefig(outdir / "synthetic_lesion_noise_sweeps.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-pairs", type=int, default=120)
    ap.add_argument("--n-sweep-seeds", type=int, default=48)
    ap.add_argument("--n-nodes", type=int, default=72)
    ap.add_argument("--mean-degree", type=int, default=8)
    ap.add_argument("--t-seconds", type=float, default=7.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--tau-d", type=float, default=0.8)
    ap.add_argument("--tau-m", type=float, default=1.2)
    ap.add_argument("--calib-seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260513)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    network_types = ["small_world", "erdos", "barabasi", "modular"]
    tau_d_steps = max(10, int(round(args.tau_d / args.dt)))
    tau_m_steps = max(10, int(round(args.tau_m / args.dt)))
    bounds = make_bounds(args, network_types, tau_d_steps, tau_m_steps)
    matched_df = matched_reentry(args, network_types, bounds, tau_d_steps, tau_m_steps)
    sweeps_df = sweeps(args, network_types, bounds, tau_d_steps, tau_m_steps)
    matched_summary = summarize_matched(matched_df)
    sweep_summary = summarize_sweeps(sweeps_df)
    meta = {"bounds": bounds, "matched": matched_summary, "sweeps": sweep_summary, "parameters": vars(args)}

    matched_df.to_csv(outdir / "synthetic_matched_reentry_cases.csv", index=False)
    sweeps_df.to_csv(outdir / "synthetic_sweep_cases.csv", index=False)
    (outdir / "synthetic_matched_stress_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_matched(matched_df, matched_summary, sweeps_df, outdir)

    print("Bounds:")
    print(json.dumps(bounds, indent=2))
    print("\nMatched re-entry summary:")
    print(json.dumps(matched_summary, indent=2))
    print("\nSweep summary:")
    print(json.dumps(sweep_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
