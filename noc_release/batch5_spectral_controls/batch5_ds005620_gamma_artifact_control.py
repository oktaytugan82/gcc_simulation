#!/usr/bin/env python
"""Gamma-band artifact controls for the DS005620 GCC validation.

This script asks whether the gamma-band GCC sedation effect survives
topographic stress tests and whether it is explainable by high-frequency
power, used here as a conservative EMG-like artifact proxy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.metrics import roc_auc_score


EXCLUDE_CHANNELS = {"VEOG", "HEOG", "EOG", "EMG"}
GAMMA_BAND = (30.0, 45.0)
HF_PROXY_BAND = (70.0, 110.0)
TOTAL_POWER_BAND = (1.0, 110.0)


def parse_recording(vhdr: Path) -> dict[str, str | int | Path | None]:
    name = vhdr.name
    subject = re.search(r"sub-([A-Za-z0-9]+)", name)
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    acq = re.search(r"acq-([A-Za-z0-9]+)", name)
    run = re.search(r"run-(\d+)", name)
    if subject is None or task is None:
        raise ValueError(f"Could not parse BIDS entities from {name}")
    task_name = task.group(1)
    return {
        "subject": subject.group(1),
        "task": task_name,
        "condition": "awake" if task_name == "awake" else task_name,
        "acq": acq.group(1) if acq else "",
        "run": int(run.group(1)) if run else 0,
        "filename": name,
        "path": vhdr,
    }


def list_recordings(root: Path) -> pd.DataFrame:
    rows = [parse_recording(vhdr) for vhdr in sorted(root.rglob("*_eeg.vhdr"))]
    df = pd.DataFrame(rows)
    return df[df["condition"].isin(["awake", "sed", "sed2"])].reset_index(drop=True)


def load_preprocessed(vhdr: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    if crop_s is not None:
        raw.crop(tmin=0.0, tmax=min(float(crop_s), float(raw.times[-1])))
    keep = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
    raw.pick(keep)
    raw.load_data(verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    data = raw.get_data()
    data = data - np.nanmean(data, axis=1, keepdims=True)
    scale = np.nanstd(data, axis=1, keepdims=True)
    data = data / np.where(scale > 1e-12, scale, 1.0)
    return data.astype(np.float64, copy=False), float(raw.info["sfreq"]), list(raw.ch_names)


def channel_sets(ch_names: list[str]) -> dict[str, list[int]]:
    def starts(ch: str, prefixes: tuple[str, ...]) -> bool:
        return any(ch.startswith(prefix) for prefix in prefixes)

    sets = {
        "all_eeg": list(range(len(ch_names))),
        "posterior": [
            i
            for i, ch in enumerate(ch_names)
            if ch == "Iz" or starts(ch, ("O", "PO", "P"))
        ],
        "centro_posterior": [
            i
            for i, ch in enumerate(ch_names)
            if ch == "Iz" or starts(ch, ("O", "PO", "P", "CP", "C"))
        ],
        "frontotemporal": [
            i
            for i, ch in enumerate(ch_names)
            if starts(ch, ("Fp", "AF", "F", "FC", "FT", "T", "TP"))
        ],
    }
    return {name: idx for name, idx in sets.items() if len(idx) >= 4}


def bandpass_phase(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    phase = np.angle(signal.hilbert(filtered, axis=-1))
    return filtered, phase


def effective_dimensionality(x: np.ndarray) -> float:
    cov = np.cov(x)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def compute_observable_series(
    filtered: np.ndarray,
    phase: np.ndarray,
    sfreq: float,
    window_s: float,
    stride_s: float,
) -> pd.DataFrame:
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    if filtered.shape[1] < win:
        return pd.DataFrame(columns=["t", "R", "D_eff", "M_tau"])

    rows = []
    for start in range(0, filtered.shape[1] - win + 1, stride):
        stop = start + win
        phase_window = phase[:, start:stop]
        filt_window = filtered[:, start:stop]
        order_t = np.abs(np.exp(1j * phase_window).mean(axis=0))
        rows.append(
            {
                "t": float((start + stop) / (2.0 * sfreq)),
                "R": float(order_t.mean()),
                "D_eff": effective_dimensionality(filt_window),
                "M_tau": float(np.var(order_t)),
            }
        )
    return pd.DataFrame(rows)


def relative_band_power(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    nperseg = int(min(round(4.0 * sfreq), data.shape[1]))
    if nperseg < 16:
        return float("nan")
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
    )
    mean_psd = np.nanmean(psd, axis=0)
    total_mask = (freqs >= TOTAL_POWER_BAND[0]) & (freqs <= TOTAL_POWER_BAND[1])
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total = float(np.trapezoid(mean_psd[total_mask], freqs[total_mask]))
    part = float(np.trapezoid(mean_psd[band_mask], freqs[band_mask]))
    return part / total if total > 0 else float("nan")


def calibrate_bounds(wake: pd.DataFrame, alpha: float) -> dict[str, float]:
    return {
        "R_min": float(wake["R"].quantile(alpha)),
        "D_min": float(wake["D_eff"].quantile(alpha)),
        "D_max": float(wake["D_eff"].quantile(1.0 - alpha)),
        "M_min": float(wake["M_tau"].quantile(alpha)),
        "M_max": float(wake["M_tau"].quantile(1.0 - alpha)),
    }


def add_score(df: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["R_ok"] = out["R"] >= bounds["R_min"]
    out["D_ok"] = (out["D_eff"] >= bounds["D_min"]) & (out["D_eff"] <= bounds["D_max"])
    out["M_ok"] = (out["M_tau"] >= bounds["M_min"]) & (out["M_tau"] <= bounds["M_max"])
    out["Pi"] = out[["R_ok", "D_ok", "M_ok"]].mean(axis=1)
    out["Access_all"] = out[["R_ok", "D_ok", "M_ok"]].all(axis=1).astype(float)
    return out


def paired_metric(sc: pd.DataFrame, target: str, metric: str, higher_awake: bool = True) -> dict[str, float]:
    wide = sc[sc["condition"].isin(["awake", target])].pivot(
        index="subject", columns="condition", values=metric
    ).dropna()
    if len(wide) < 5:
        return {"n": int(len(wide)), "awake_mean": np.nan, "target_mean": np.nan, "mean_diff": np.nan, "paired_d": np.nan, "p": np.nan}
    awake = wide["awake"].to_numpy()
    target_vals = wide[target].to_numpy()
    diff = awake - target_vals
    _, p = stats.ttest_rel(awake, target_vals, nan_policy="omit")
    sd = float(np.nanstd(diff, ddof=1))
    y = np.r_[np.ones_like(awake), np.zeros_like(target_vals)]
    score = np.r_[awake, target_vals]
    auc = float(roc_auc_score(y, score))
    if not higher_awake:
        auc = max(auc, 1.0 - auc)
    return {
        "n": int(len(wide)),
        "awake_mean": float(np.nanmean(awake)),
        "target_mean": float(np.nanmean(target_vals)),
        "mean_diff_awake_minus_target": float(np.nanmean(diff)),
        "paired_d": float(np.nanmean(diff) / sd) if sd > 0 else np.nan,
        "p": float(p),
        "auc_awake_vs_target": auc,
    }


def delta_correlation(sc: pd.DataFrame, target: str, channel_set: str) -> dict[str, float | str]:
    sub = sc[(sc["channel_set"] == channel_set) & (sc["condition"].isin(["awake", target]))]
    wide_pi = sub.pivot(index="subject", columns="condition", values="Pi").dropna()
    wide_hf = sub.pivot(index="subject", columns="condition", values="hf_rel_power").dropna()
    common = wide_pi.index.intersection(wide_hf.index)
    if len(common) < 5:
        return {"channel_set": channel_set, "comparison": f"awake_vs_{target}", "n": int(len(common)), "spearman_r": np.nan, "spearman_p": np.nan}
    dpi = wide_pi.loc[common, "awake"] - wide_pi.loc[common, target]
    dhf = wide_hf.loc[common, "awake"] - wide_hf.loc[common, target]
    rho, p = stats.spearmanr(dpi, dhf, nan_policy="omit")
    return {
        "channel_set": channel_set,
        "comparison": f"awake_vs_{target}",
        "n": int(len(common)),
        "spearman_r_delta_pi_delta_hf": float(rho),
        "spearman_p": float(p),
        "delta_pi_mean": float(np.nanmean(dpi)),
        "delta_hf_mean": float(np.nanmean(dhf)),
    }


def plot_channelset_effects(stats_df: pd.DataFrame, outdir: Path) -> None:
    sub = stats_df[(stats_df["metric"] == "Pi") & (stats_df["comparison"].isin(["awake_vs_sed", "awake_vs_sed2"]))]
    if sub.empty:
        return
    for comp, g in sub.groupby("comparison"):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        order = ["all_eeg", "posterior", "centro_posterior", "frontotemporal"]
        g = g.set_index("channel_set").reindex(order).dropna(how="all")
        ax.bar(g.index, g["paired_d"], color="#2D6A4F")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("Paired d: awake - sedated")
        ax.set_title(f"Gamma GCC Pi topographic stress test: {comp}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"gamma_artifact_topography_{comp}.png", dpi=180)
        plt.close(fig)


def run(root: Path, outdir: Path, alpha: float, target_sfreq: float, crop_s: float | None, window_s: float, stride_s: float) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    recs = list_recordings(root)
    if recs.empty:
        raise RuntimeError(f"No DS005620 recordings found under {root}")

    all_windows = []
    all_powers = []
    channel_counts = {}

    for i, row in recs.iterrows():
        print(f"[{i + 1}/{len(recs)}] {row['filename']}", flush=True)
        data, sfreq, ch_names = load_preprocessed(Path(row["path"]), target_sfreq, crop_s)
        sets = channel_sets(ch_names)
        channel_counts = {name: len(idx) for name, idx in sets.items()}
        filtered_all, phase_all = bandpass_phase(data, sfreq, GAMMA_BAND)

        for set_name, idx in sets.items():
            subset = data[idx, :]
            filtered = filtered_all[idx, :]
            phase = phase_all[idx, :]
            windows = compute_observable_series(filtered, phase, sfreq, window_s, stride_s)
            if windows.empty:
                continue
            for key in ["subject", "task", "condition", "acq", "run", "filename"]:
                windows[key] = row[key]
            windows["channel_set"] = set_name
            windows["n_channels"] = len(idx)
            all_windows.append(windows)

            all_powers.append(
                {
                    "subject": row["subject"],
                    "task": row["task"],
                    "condition": row["condition"],
                    "acq": row["acq"],
                    "run": row["run"],
                    "filename": row["filename"],
                    "channel_set": set_name,
                    "n_channels": len(idx),
                    "gamma_rel_power": relative_band_power(subset, sfreq, GAMMA_BAND),
                    "hf_rel_power": relative_band_power(subset, sfreq, HF_PROXY_BAND),
                }
            )

    windows_df = pd.concat(all_windows, ignore_index=True)
    powers_df = pd.DataFrame(all_powers)
    windows_df.to_csv(outdir / "ds005620_gamma_artifact_window_features.csv", index=False)
    powers_df.to_csv(outdir / "ds005620_gamma_artifact_recording_powers.csv", index=False)

    scored_parts = []
    bounds_rows = []
    for (subject, set_name), sub in windows_df.groupby(["subject", "channel_set"]):
        wake = sub[sub["condition"] == "awake"]
        if wake.empty:
            continue
        bounds = calibrate_bounds(wake, alpha)
        bounds_rows.append({"subject": subject, "channel_set": set_name, **bounds})
        scored_parts.append(add_score(sub, bounds))
    scored = pd.concat(scored_parts, ignore_index=True)
    scored.to_csv(outdir / "ds005620_gamma_artifact_scored_windows.csv", index=False)
    bounds_df = pd.DataFrame(bounds_rows)
    bounds_df.to_csv(outdir / "ds005620_gamma_artifact_bounds.csv", index=False)

    obs_subject_condition = (
        scored.groupby(["subject", "condition", "channel_set"])[["R", "D_eff", "M_tau", "Pi", "Access_all"]]
        .mean()
        .reset_index()
    )
    power_subject_condition = (
        powers_df.groupby(["subject", "condition", "channel_set"])[["gamma_rel_power", "hf_rel_power"]]
        .mean()
        .reset_index()
    )
    subject_condition = obs_subject_condition.merge(
        power_subject_condition,
        on=["subject", "condition", "channel_set"],
        how="left",
    )
    subject_condition.to_csv(outdir / "ds005620_gamma_artifact_subject_condition.csv", index=False)

    stats_rows = []
    for set_name in sorted(subject_condition["channel_set"].unique()):
        sub = subject_condition[subject_condition["channel_set"] == set_name]
        for target in ["sed", "sed2"]:
            for metric in ["Pi", "Access_all", "R", "D_eff", "M_tau", "gamma_rel_power", "hf_rel_power"]:
                res = paired_metric(sub, target, metric)
                stats_rows.append(
                    {
                        "channel_set": set_name,
                        "comparison": f"awake_vs_{target}",
                        "metric": metric,
                        **res,
                    }
                )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(outdir / "ds005620_gamma_artifact_paired_stats.csv", index=False)

    corr_rows = []
    for set_name in sorted(subject_condition["channel_set"].unique()):
        for target in ["sed", "sed2"]:
            corr_rows.append(delta_correlation(subject_condition, target, set_name))
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(outdir / "ds005620_gamma_artifact_delta_correlations.csv", index=False)
    plot_channelset_effects(stats_df, outdir)

    compact = {
        "n_recordings": int(len(recs)),
        "n_subjects": int(recs["subject"].nunique()),
        "channel_counts": channel_counts,
        "alpha": alpha,
        "target_sfreq": target_sfreq,
        "crop_s": crop_s,
        "window_s": window_s,
        "stride_s": stride_s,
        "pi_stats": stats_df[stats_df["metric"] == "Pi"].to_dict(orient="records"),
        "hf_stats": stats_df[stats_df["metric"] == "hf_rel_power"].to_dict(orient="records"),
        "delta_correlations": corr_df.to_dict(orient="records"),
    }
    with open(outdir / "ds005620_gamma_artifact_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(json.dumps(compact, indent=2)[:12000])
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--crop-s", type=float, default=120.0)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    args = parser.parse_args()
    run(
        root=args.root,
        outdir=args.outdir,
        alpha=args.alpha,
        target_sfreq=args.target_sfreq,
        crop_s=args.crop_s,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )


if __name__ == "__main__":
    main()
