"""
Extended synthetic GCC validation benchmark.

Purpose
-------
This script expands the original synthetic validation into a reviewer-facing
benchmark:

- network types: small-world, Erdős-Rényi, Barabási-Albert, modular
- parameter sweeps: K, noise, global gain, lesion fraction, backbone size
- baselines: R-only, D-only, M-only, criticality-only, global-gain-only

The benchmark is deliberately labelled by experimental condition rather than by
the GCC score itself. Positive cases are balanced healthy or selective-backbone
re-entry regimes. Negative cases are uniform degradation, undercoupling,
overcoupling, or high-noise collapse. This lets us ask whether the GCC triad
recovers the designed access-compatible regime better than single-observable
baselines.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def row_normalize(a: np.ndarray) -> np.ndarray:
    row_sums = a.sum(axis=1, keepdims=True)
    return np.divide(a, row_sums, out=np.zeros_like(a, dtype=float), where=row_sums > 0)


def make_network(kind: str, n: int, mean_degree: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "small_world":
        g = nx.watts_strogatz_graph(n, k=mean_degree, p=0.12, seed=seed)
    elif kind == "erdos":
        p = mean_degree / max(1, n - 1)
        g = nx.erdos_renyi_graph(n, p=p, seed=seed)
    elif kind == "barabasi":
        g = nx.barabasi_albert_graph(n, m=max(1, mean_degree // 2), seed=seed)
    elif kind == "modular":
        sizes = [n // 4, n // 4, n // 4, n - 3 * (n // 4)]
        probs = np.full((4, 4), 0.025)
        np.fill_diagonal(probs, 0.22)
        g = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)
    else:
        raise ValueError(f"Unknown network kind: {kind}")
    if not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for c1, c2 in zip(comps[:-1], comps[1:]):
            g.add_edge(rng.choice(c1), rng.choice(c2))
    a = nx.to_numpy_array(g, dtype=float)
    return a, row_normalize(a)


def apply_edge_lesion(a_raw: np.ndarray, lesion_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = a_raw.copy()
    edges = np.argwhere(a > 0)
    if lesion_fraction > 0 and len(edges) > 0:
        idx = rng.choice(len(edges), size=int(round(lesion_fraction * len(edges))), replace=False)
        a[edges[idx, 0], edges[idx, 1]] = 0.0
    m_ref = a_raw.sum(axis=1)
    m_now = a.sum(axis=1)
    g_pres = np.divide(m_now, m_ref, out=np.zeros_like(m_now), where=m_ref > 0)
    return a, row_normalize(a), np.clip(g_pres, 0.0, 1.0)


def euler_simulate(
    w: np.ndarray,
    g_local: np.ndarray,
    *,
    k: float,
    global_gain: float,
    sigma: float,
    t: float,
    dt: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = w.shape[0]
    omega = rng.normal(0.0, 0.45, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    burn = int(round(2.0 / dt))
    steps = int(round(t / dt))

    def step(th: np.ndarray) -> np.ndarray:
        diff = th[np.newaxis, :] - th[:, np.newaxis]
        coupling = (w * np.sin(diff)).sum(axis=1)
        drift = omega + k * global_gain * g_local * coupling
        return th + drift * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)

    for _ in range(burn):
        theta = step(theta)
    hist = np.empty((steps, n), dtype=float)
    for i in range(steps):
        theta = step(theta)
        hist[i] = theta
    return hist


def observables(theta: np.ndarray, tau_d: int, tau_m: int, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.exp(1j * theta)
    r = np.abs(z.mean(axis=1))
    x = np.cos(theta)
    n = x.shape[1]
    d = np.empty(len(x), dtype=float)
    half = tau_d // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half)
        xw = x[lo:hi] - x[lo:hi].mean(axis=0, keepdims=True)
        if len(xw) < 3:
            d[i] = 1.0
            continue
        c = (xw.T @ xw) / max(1, len(xw) - 1)
        eps = max(ridge * np.trace(c) / n, 1e-6)
        c = c + eps * np.eye(n)
        tr = np.trace(c)
        tr2 = np.trace(c @ c)
        d[i] = (tr * tr) / tr2 if tr2 > 0 else 1.0
    m = np.empty_like(r)
    half = tau_m // 2
    for i in range(len(r)):
        lo = max(0, i - half)
        hi = min(len(r), i + half)
        m[i] = np.var(r[lo:hi]) if hi - lo > 2 else 0.0
    return r, d, m


def calibrate(reference: list[tuple[np.ndarray, np.ndarray, np.ndarray]], alpha: float) -> dict[str, float]:
    r = np.concatenate([x[0] for x in reference])
    d = np.concatenate([x[1] for x in reference])
    m = np.concatenate([x[2] for x in reference])
    return {
        "R_min": float(np.quantile(r, alpha)),
        "D_min": float(np.quantile(d, alpha)),
        "D_max": float(np.quantile(d, 1 - alpha)),
        "M_min": float(np.quantile(m, alpha)),
        "M_max": float(np.quantile(m, 1 - alpha)),
    }


def score_metrics(r: np.ndarray, d: np.ndarray, m: np.ndarray, bounds: dict[str, float], global_gain: float) -> dict[str, float]:
    in_r = r >= bounds["R_min"]
    in_d = (d > bounds["D_min"]) & (d < bounds["D_max"])
    in_m = (m > bounds["M_min"]) & (m < bounds["M_max"])
    triad = in_r & in_d & in_m
    criticality = in_d & in_m
    return {
        "score_gcc_triad": float(np.mean(triad)),
        "score_r_only": float(np.mean(in_r)),
        "score_d_only": float(np.mean(in_d)),
        "score_m_only": float(np.mean(in_m)),
        "score_criticality_only": float(np.mean(criticality)),
        "score_global_gain_only": float(global_gain),
        "R_mean": float(np.mean(r)),
        "D_mean": float(np.mean(d)),
        "M_mean": float(np.mean(m)),
    }


def scenario_params(label: str, rng: np.random.Generator) -> dict[str, float]:
    if label == "balanced":
        return {
            "target": 1,
            "K": float(rng.choice([1.5, 1.8, 2.1, 2.4])),
            "sigma": float(rng.choice([0.06, 0.10, 0.14])),
            "G": float(rng.choice([0.85, 1.0, 1.15])),
            "lesion": 0.0,
            "backbone": 1.0,
            "mode": "healthy",
        }
    if label == "undercoupled":
        return {
            "target": 0,
            "K": float(rng.choice([0.4, 0.7, 1.0])),
            "sigma": float(rng.choice([0.10, 0.16, 0.22])),
            "G": float(rng.choice([0.45, 0.60, 0.75])),
            "lesion": float(rng.choice([0.20, 0.40])),
            "backbone": 0.0,
            "mode": "uniform",
        }
    if label == "overcoupled":
        return {
            "target": 0,
            "K": float(rng.choice([3.2, 4.0, 5.0])),
            "sigma": float(rng.choice([0.02, 0.05, 0.08])),
            "G": float(rng.choice([1.15, 1.35, 1.55])),
            "lesion": 0.0,
            "backbone": 1.0,
            "mode": "healthy",
        }
    if label == "high_noise":
        return {
            "target": 0,
            "K": float(rng.choice([1.5, 2.0, 2.5])),
            "sigma": float(rng.choice([0.28, 0.36, 0.45])),
            "G": float(rng.choice([0.8, 1.0, 1.2])),
            "lesion": float(rng.choice([0.0, 0.20])),
            "backbone": 1.0,
            "mode": "healthy",
        }
    if label == "uniform_degraded":
        return {
            "target": 0,
            "K": float(rng.choice([1.5, 2.0, 2.5])),
            "sigma": float(rng.choice([0.08, 0.12, 0.16])),
            "G": float(rng.choice([0.45, 0.60, 0.75])),
            "lesion": float(rng.choice([0.35, 0.50, 0.65])),
            "backbone": 0.0,
            "mode": "uniform",
        }
    if label == "backbone_reentry":
        return {
            "target": 1,
            "K": float(rng.choice([1.8, 2.2, 2.6])),
            "sigma": float(rng.choice([0.06, 0.10, 0.14])),
            "G": float(rng.choice([0.80, 1.0, 1.20])),
            "lesion": float(rng.choice([0.35, 0.50, 0.65])),
            "backbone": float(rng.choice([0.20, 0.30, 0.40])),
            "mode": "backbone",
        }
    raise ValueError(label)


def make_g_vector(g_pres: np.ndarray, mode: str, backbone: float, seed: int) -> np.ndarray:
    if mode == "healthy":
        return np.ones_like(g_pres)
    if mode == "uniform":
        return g_pres
    if mode == "backbone":
        rng = np.random.default_rng(seed)
        n = len(g_pres)
        s_size = max(2, int(round(backbone * n)))
        idx = rng.choice(n, size=s_size, replace=False)
        g = np.full(n, 0.15)
        g[idx] = np.maximum(g_pres[idx], 0.85)
        return np.clip(g, 0.0, 1.0)
    raise ValueError(mode)


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    network_types = ["small_world", "erdos", "barabasi", "modular"]
    tau_d = max(10, int(round(args.tau_d / args.dt)))
    tau_m = max(10, int(round(args.tau_m / args.dt)))

    reference = []
    for kind in network_types:
        for seed in range(args.calib_seeds):
            a_raw, w = make_network(kind, args.n_nodes, args.mean_degree, seed=10_000 + seed)
            theta = euler_simulate(
                w,
                np.ones(args.n_nodes),
                k=float(rng.choice([1.5, 1.8, 2.1, 2.4])),
                global_gain=1.0,
                sigma=0.10,
                t=args.t_seconds,
                dt=args.dt,
                seed=20_000 + seed,
            )
            reference.append(observables(theta, tau_d, tau_m))
    bounds = calibrate(reference, args.alpha)

    labels = ["balanced", "undercoupled", "overcoupled", "high_noise", "uniform_degraded", "backbone_reentry"]
    rows = []
    for i in range(args.n_cases):
        label = labels[i % len(labels)]
        params = scenario_params(label, rng)
        kind = network_types[i % len(network_types)]
        net_seed = args.seed + 1000 + i
        a_raw, _ = make_network(kind, args.n_nodes, args.mean_degree, seed=net_seed)
        _, w_les, g_pres = apply_edge_lesion(a_raw, params["lesion"], seed=net_seed + 1)
        g_local = make_g_vector(g_pres, params["mode"], params["backbone"], seed=net_seed + 2)
        theta = euler_simulate(
            w_les,
            g_local,
            k=params["K"],
            global_gain=params["G"],
            sigma=params["sigma"],
            t=args.t_seconds,
            dt=args.dt,
            seed=net_seed + 3,
        )
        r, d, m = observables(theta, tau_d, tau_m)
        scores = score_metrics(r, d, m, bounds, params["G"])
        rows.append(
            {
                "case": i,
                "network": kind,
                "scenario": label,
                "target_access": int(params["target"]),
                "K": params["K"],
                "sigma": params["sigma"],
                "global_gain": params["G"],
                "lesion_fraction": params["lesion"],
                "backbone_fraction": params["backbone"],
                **scores,
            }
        )

    df = pd.DataFrame(rows)
    metrics = [
        "score_gcc_triad",
        "score_r_only",
        "score_d_only",
        "score_m_only",
        "score_criticality_only",
        "score_global_gain_only",
    ]
    aucs = {}
    for metric in metrics:
        aucs[metric] = float(roc_auc_score(df["target_access"], df[metric]))
    pos = df[df["target_access"] == 1]
    neg = df[df["target_access"] == 0]
    tests = {}
    for metric in metrics:
        u = mannwhitneyu(pos[metric], neg[metric], alternative="two-sided")
        tests[metric] = {"U": float(u.statistic), "p": float(u.pvalue), "pos_mean": float(pos[metric].mean()), "neg_mean": float(neg[metric].mean())}
    meta = {
        "bounds": bounds,
        "n_cases": int(len(df)),
        "network_types": network_types,
        "metrics": metrics,
        "aucs": aucs,
        "tests": tests,
        "parameters": vars(args),
    }
    return df, bounds, meta


def plot_results(df: pd.DataFrame, meta: dict[str, object], outdir: Path) -> None:
    metrics = meta["metrics"]
    labels = ["GCC triad", "R only", "D only", "M only", "Criticality", "Global gain"]
    aucs = [meta["aucs"][m] for m in metrics]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    bars = ax.bar(labels, aucs, color=["#0f766e", "#64748b", "#64748b", "#64748b", "#94a3b8", "#94a3b8"])
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC for designed access-compatible regime")
    ax.set_title("Synthetic benchmark: GCC triad vs baseline scores")
    ax.tick_params(axis="x", rotation=25)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.02, f"{auc:.2f}", ha="center", fontsize=9)
    fig.savefig(outdir / "synthetic_baseline_auc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    order = ["balanced", "backbone_reentry", "uniform_degraded", "undercoupled", "overcoupled", "high_noise"]
    data = [df.loc[df["scenario"] == s, "score_gcc_triad"].to_numpy() for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("GCC triad Pi")
    ax.set_title("Synthetic GCC regime occupancy by designed scenario")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(outdir / "synthetic_scenario_pi.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-cases", type=int, default=240)
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
    df, bounds, meta = run_benchmark(args)
    df.to_csv(outdir / "synthetic_benchmark_cases.csv", index=False)
    (outdir / "synthetic_benchmark_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_results(df, meta, outdir)

    print("Bounds:")
    print(json.dumps(bounds, indent=2))
    print("\nAUCs:")
    for k, v in meta["aucs"].items():
        print(f"{k}: {v:.4f}")
    print("\nMeans/tests:")
    print(json.dumps(meta["tests"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
