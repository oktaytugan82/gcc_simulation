#!/usr/bin/env python
"""
Sleep-EDF validation for the GCC regime observables.

This is intentionally conservative: Sleep-EDF contains only two scalp EEG
channels in this subset, so the analysis tests state separability of the
observable triad rather than claiming full large-scale network reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STATE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage R": "REM",
    "Sleep stage 1": "NREM",
    "Sleep stage 2": "NREM",
    "Sleep stage 3": "NREM",
    "Sleep stage 4": "NREM",
}


def _pair_files(root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for psg in sorted(root.glob("*-PSG.edf")):
        subject = psg.name[:6]
        hypos = sorted(root.glob(subject + "*Hypnogram.edf"))
        if not hypos:
            continue
        pairs.append((psg, hypos[0], subject))
    return pairs


def _bandpass_hilbert(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    analytic = signal.hilbert(filtered, axis=-1)
    return filtered, np.angle(analytic)


def _features(epoch: np.ndarray, sfreq: float, band: tuple[float, float]) -> dict[str, float]:
    filtered, phase = _bandpass_hilbert(epoch, sfreq, band)

    order_t = np.abs(np.mean(np.exp(1j * phase), axis=0))
    r_mean = float(np.mean(order_t))
    m_tau = float(np.var(order_t))

    cov = np.cov(filtered)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    d_eff = float((eig.sum() ** 2) / np.sum(eig**2))

    return {"R": r_mean, "D_eff": d_eff, "M_tau": m_tau}


def _crop_to_sleep_window(raw: mne.io.BaseRaw, annotations: mne.Annotations, margin_s: float = 1800.0) -> mne.io.BaseRaw:
    raw.set_annotations(annotations)
    sleep_rows = [
        (float(onset), float(onset + duration))
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description)
        if desc in STATE_MAP and STATE_MAP[desc] != "Wake"
    ]
    if not sleep_rows:
        return raw
    tmin = max(0.0, min(x[0] for x in sleep_rows) - margin_s)
    tmax = min(raw.times[-1], max(x[1] for x in sleep_rows) + margin_s)
    return raw.crop(tmin=tmin, tmax=tmax)


def extract_epochs(root: Path, band: tuple[float, float]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    event_id = {k: i + 1 for i, k in enumerate(STATE_MAP)}

    for psg, hypno, subject in _pair_files(root):
        annotations = mne.read_annotations(hypno)
        raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
        raw = _crop_to_sleep_window(raw, annotations)
        raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])
        raw.load_data(verbose="ERROR")

        events, id_map = mne.events_from_annotations(
            raw,
            event_id=event_id,
            chunk_duration=30.0,
            verbose="ERROR",
        )
        if len(events) == 0:
            continue

        epochs = mne.Epochs(
            raw,
            events,
            event_id=id_map,
            tmin=0,
            tmax=30.0 - 1.0 / raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose="ERROR",
        )

        inv_id = {v: k for k, v in id_map.items()}
        for idx, ev in enumerate(events):
            desc = inv_id[int(ev[2])]
            state = STATE_MAP[desc]
            feat = _features(epochs.get_data(copy=False)[idx], raw.info["sfreq"], band)
            rows.append(
                {
                    "subject": subject,
                    "annotation": desc,
                    "state": state,
                    "epoch_index": idx,
                    **feat,
                }
            )

    return pd.DataFrame(rows)


def calibrate_from_wake(df: pd.DataFrame, alpha: float = 0.10) -> dict[str, float]:
    wake = df[df["state"] == "Wake"]
    if len(wake) < 10:
        raise ValueError("Need at least 10 Wake epochs for calibration.")
    return {
        "R_min": float(wake["R"].quantile(alpha)),
        "D_min": float(wake["D_eff"].quantile(alpha)),
        "D_max": float(wake["D_eff"].quantile(1.0 - alpha)),
        "M_min": float(wake["M_tau"].quantile(alpha)),
        "M_max": float(wake["M_tau"].quantile(1.0 - alpha)),
    }


def add_pi(df: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    indicators = pd.DataFrame(
        {
            "R_ok": out["R"] >= bounds["R_min"],
            "D_ok": (out["D_eff"] >= bounds["D_min"]) & (out["D_eff"] <= bounds["D_max"]),
            "M_ok": (out["M_tau"] >= bounds["M_min"]) & (out["M_tau"] <= bounds["M_max"]),
        }
    )
    out["Pi"] = indicators.mean(axis=1)
    out["Access_all"] = indicators.all(axis=1).astype(int)
    return out


def _binary_logo_auc(df: pd.DataFrame, positive: str, negative: str, feature_cols: list[str]) -> float | None:
    sub = df[df["state"].isin([positive, negative])].copy()
    if sub["subject"].nunique() < 3 or sub["state"].nunique() < 2:
        return None
    y = (sub["state"] == positive).astype(int).to_numpy()
    x = sub[feature_cols].to_numpy()
    groups = sub["subject"].to_numpy()
    logo = LeaveOneGroupOut()
    scores = np.full(len(sub), np.nan)
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17),
        )
        clf.fit(x[train], y[train])
        scores[test] = clf.predict_proba(x[test])[:, 1]
    mask = np.isfinite(scores)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], scores[mask]))


def _multiclass_logo(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float | None]:
    states = ["Wake", "REM", "NREM"]
    sub = df[df["state"].isin(states)].copy()
    if sub["subject"].nunique() < 3 or sub["state"].nunique() < 3:
        return {"accuracy": None, "macro_f1": None}
    y = pd.Categorical(sub["state"], categories=states).codes
    x = sub[feature_cols].to_numpy()
    groups = sub["subject"].to_numpy()
    pred = np.full(len(sub), -1)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 3:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17),
        )
        clf.fit(x[train], y[train])
        pred[test] = clf.predict(x[test])
    mask = pred >= 0
    if mask.sum() < 20:
        return {"accuracy": None, "macro_f1": None}
    return {
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }


def summarize(df: pd.DataFrame, bounds: dict[str, float]) -> dict:
    state_summary = (
        df.groupby("state")[["R", "D_eff", "M_tau", "Pi", "Access_all"]]
        .agg(["mean", "std", "median", "count"])
        .round(6)
    )
    state_summary.columns = [f"{a}_{b}" for a, b in state_summary.columns]

    subject_state = (
        df.groupby(["subject", "state"])[["R", "D_eff", "M_tau", "Pi", "Access_all"]]
        .mean()
        .round(6)
        .reset_index()
    )
    stats_out: dict[str, float | int | None | dict] = {}
    for col in ["R", "D_eff", "M_tau", "Pi"]:
        groups = [g[col].to_numpy() for _, g in df.groupby("state") if len(g) > 0]
        if len(groups) >= 3:
            h, p = stats.kruskal(*groups)
            stats_out[f"kruskal_{col}_H"] = float(h)
            stats_out[f"kruskal_{col}_p"] = float(p)

    feature_sets = {
        "R_only": ["R"],
        "D_only": ["D_eff"],
        "M_only": ["log_M"],
        "GCC_triad": ["R", "D_eff", "log_M"],
        "Pi_only": ["Pi"],
    }
    df_cv = df.copy()
    df_cv["log_M"] = np.log10(df_cv["M_tau"] + 1e-12)
    cv = {}
    for name, cols in feature_sets.items():
        cv[name] = {
            "wake_vs_nrem_auc": _binary_logo_auc(df_cv, "Wake", "NREM", cols),
            "rem_vs_nrem_auc": _binary_logo_auc(df_cv, "REM", "NREM", cols),
            "multiclass": _multiclass_logo(df_cv, cols),
        }

    return {
        "bounds": bounds,
        "n_epochs": int(len(df)),
        "subjects": sorted(df["subject"].unique().tolist()),
        "state_counts": {k: int(v) for k, v in df["state"].value_counts().to_dict().items()},
        "state_summary": state_summary.to_dict(orient="index"),
        "subject_state_mean": subject_state.to_dict(orient="records"),
        "stats": stats_out,
        "cross_validated_baselines": cv,
    }


def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    colors = {"Wake": "#2B6CB0", "REM": "#D69E2E", "NREM": "#2F855A"}
    order = ["Wake", "REM", "NREM"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col in zip(axes, ["R", "D_eff", "M_tau", "Pi"]):
        data = [df.loc[df["state"] == st, col].to_numpy() for st in order]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, st in zip(parts["bodies"], order):
            body.set_facecolor(colors[st])
            body.set_alpha(0.55)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        ax.set_xticks(range(1, len(order) + 1), order, rotation=25)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    fig.suptitle("Sleep-EDF state differentiation with GCC observables")
    fig.tight_layout()
    fig.savefig(outdir / "sleep_edf_state_observables.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for st in order:
        sub = df[df["state"] == st]
        ax.scatter(sub["R"], sub["D_eff"], s=8, alpha=0.35, label=st, color=colors[st])
    ax.set_xlabel("Coherence R")
    ax.set_ylabel("Effective dimensionality D_eff")
    ax.set_title("Sleep-EDF: observable plane")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "sleep_edf_R_D_plane.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--band", choices=["alpha", "sigma"], default="alpha")
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    band = (8.0, 13.0) if args.band == "alpha" else (12.0, 16.0)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = extract_epochs(args.root, band)
    if df.empty:
        raise RuntimeError("No valid Sleep-EDF epochs extracted.")

    bounds = calibrate_from_wake(df, alpha=args.alpha)
    df = add_pi(df, bounds)

    df.to_csv(args.outdir / f"sleep_edf_{args.band}_epoch_features.csv", index=False)
    summary = summarize(df, bounds)
    summary["band"] = args.band
    summary["band_hz"] = band
    with open(args.outdir / f"sleep_edf_{args.band}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(df, args.outdir)
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
