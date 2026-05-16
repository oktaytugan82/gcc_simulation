#!/usr/bin/env python
"""Volume-conduction-robust DS005620 control using a wPLI-style GCC variant.

This script replaces the standard Kuramoto order parameter R by a global
weighted phase-lag index proxy. wPLI suppresses zero-lag coupling and is
therefore less sensitive to volume conduction than channel-level synchrony.

The goal is not to define a new final GCC index, but to stress-test whether the
regime logic remains state-sensitive when the coherence observable is made more
conservative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal, stats


EXCLUDE_CHANNELS = {"VEOG", "HEOG", "EOG", "EMG"}
BANDS = {
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 45.0),
}


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


def sampled_pairs(n_channels: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(n_channels, k=1)
    if max_pairs <= 0 or max_pairs >= len(ii):
        return ii, jj
    rng = np.random.default_rng(seed + n_channels)
    idx = rng.choice(len(ii), size=max_pairs, replace=False)
    return ii[idx], jj[idx]


def wpli_observable_series(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    window_s: float,
    stride_s: float,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    filtered, phase = bandpass_phase(data, sfreq, band)
    pair_i, pair_j = sampled_pairs(phase.shape[0], max_pairs=max_pairs, seed=seed)
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    rows = []

    for start in range(0, filtered.shape[1] - win + 1, stride):
        stop = start + win
        phase_window = phase[:, start:stop]
        filt_window = filtered[:, start:stop]

        # wPLI-style global lagged coupling. Zero phase-lag contributes little.
        imag = np.sin(phase_window[pair_i] - phase_window[pair_j])
        denom = np.mean(np.abs(imag), axis=1)
        wpli_pairs = np.abs(np.mean(imag, axis=1)) / np.where(denom > 1e-12, denom, np.nan)
        r_wpli = float(np.nanmean(wpli_pairs))

        # Temporal stability of lagged coupling within the same window.
        lagged_coupling_t = np.nanmean(np.abs(imag), axis=0)
        m_tau = float(np.nanvar(lagged_coupling_t))

        rows.append(
            {
                "R_wpli": r_wpli,
                "D_eff": effective_dimensionality(filt_window),
                "M_tau": m_tau,
            }
        )
    return pd.DataFrame(rows)


def calibrate_bounds(wake: pd.DataFrame, alpha: float) -> dict[str, float]:
    return {
        "R_min": float(wake["R_wpli"].quantile(alpha)),
        "D_min": float(wake["D_eff"].quantile(alpha)),
        "D_max": float(wake["D_eff"].quantile(1.0 - alpha)),
        "M_min": float(wake["M_tau"].quantile(alpha)),
        "M_max": float(wake["M_tau"].quantile(1.0 - alpha)),
    }


def score(series: pd.DataFrame, bounds: dict[str, float]) -> tuple[float, float]:
    r_ok = series["R_wpli"] >= bounds["R_min"]
    d_ok = (series["D_eff"] >= bounds["D_min"]) & (series["D_eff"] <= bounds["D_max"])
    m_ok = (series["M_tau"] >= bounds["M_min"]) & (series["M_tau"] <= bounds["M_max"])
    access_all = r_ok & d_ok & m_ok
    pi = pd.concat([r_ok, d_ok, m_ok], axis=1).mean(axis=1).mean()
    return float(pi), float(access_all.mean())


def auc_from_pairs(awake: np.ndarray, target: np.ndarray) -> float:
    # Probability that a random awake value exceeds a random target value.
    wins = 0.0
    total = 0
    for a in awake:
        wins += np.sum(a > target) + 0.5 * np.sum(a == target)
        total += len(target)
    return float(wins / total) if total else float("nan")


def paired_summary(df: pd.DataFrame, target: str) -> dict[str, float]:
    wide = df[df["condition"].isin(["awake", target])].pivot(index="subject", columns="condition", values="Pi").dropna()
    awake = wide["awake"].to_numpy()
    sed = wide[target].to_numpy()
    diff = awake - sed
    if len(diff) < 5:
        return {"n": int(len(diff))}
    _, p_t = stats.ttest_rel(awake, sed)
    p_w = stats.wilcoxon(diff, alternative="greater").pvalue
    sd = np.std(diff, ddof=1)
    lo, hi = np.quantile(
        [
            np.mean(np.random.default_rng(1000 + b).choice(diff, size=len(diff), replace=True))
            for b in range(5000)
        ],
        [0.025, 0.975],
    )
    return {
        "n": int(len(diff)),
        "awake_mean": float(np.mean(awake)),
        "target_mean": float(np.mean(sed)),
        "mean_delta_awake_minus_target": float(np.mean(diff)),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "paired_d": float(np.mean(diff) / sd) if sd > 0 else np.nan,
        "ttest_p": float(p_t),
        "wilcoxon_greater_p": float(p_w),
        "auc_awake_gt_target": auc_from_pairs(awake, sed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--closest-runs", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--band", choices=sorted(BANDS), default="gamma")
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=90.0)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    closest = pd.read_csv(args.closest_runs)
    closest = closest[closest["band"] == args.band].copy()
    subjects = sorted(closest["subject"].astype(str).unique())
    rows = []
    bounds_rows = []

    for subject in subjects:
        sub = closest[closest["subject"].astype(str) == subject]
        awake_file = sub[sub["condition"] == "awake"]["filename"].iloc[0]
        awake_path = next(args.local_root.rglob(awake_file))
        awake_data, sfreq, ch_names = load_preprocessed(awake_path, args.target_sfreq, args.crop_s)
        wake_series = wpli_observable_series(
            awake_data,
            sfreq,
            BANDS[args.band],
            args.window_s,
            args.stride_s,
            args.max_pairs,
            args.seed,
        )
        bounds = calibrate_bounds(wake_series, args.alpha)
        bounds_rows.append({"subject": subject, "band": args.band, "n_channels": len(ch_names), **bounds})
        pi, access = score(wake_series, bounds)
        rows.append(
            {
                "subject": subject,
                "condition": "awake",
                "band": args.band,
                "filename": awake_file,
                "n_channels": len(ch_names),
                "n_windows": len(wake_series),
                "R_wpli_mean": float(wake_series["R_wpli"].mean()),
                "D_eff_mean": float(wake_series["D_eff"].mean()),
                "M_tau_mean": float(wake_series["M_tau"].mean()),
                "Pi": pi,
                "Access_all": access,
            }
        )

        for _, target_row in sub[sub["condition"].isin(["sed", "sed2"])].iterrows():
            condition = target_row["condition"]
            filename = target_row["filename"]
            target_path = next(args.local_root.rglob(filename))
            data, _, ch_names = load_preprocessed(target_path, args.target_sfreq, args.crop_s)
            series = wpli_observable_series(
                data,
                sfreq,
                BANDS[args.band],
                args.window_s,
                args.stride_s,
                args.max_pairs,
                args.seed,
            )
            pi, access = score(series, bounds)
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "band": args.band,
                    "filename": filename,
                    "n_channels": len(ch_names),
                    "n_windows": len(series),
                    "R_wpli_mean": float(series["R_wpli"].mean()),
                    "D_eff_mean": float(series["D_eff"].mean()),
                    "M_tau_mean": float(series["M_tau"].mean()),
                    "Pi": pi,
                    "Access_all": access,
                }
            )
        print(f"processed subject {subject}", flush=True)

    out = pd.DataFrame(rows)
    out_path = args.outdir / f"ds005620_{args.band}_wpli_gcc.csv"
    out.to_csv(out_path, index=False)
    pd.DataFrame(bounds_rows).to_csv(args.outdir / f"ds005620_{args.band}_wpli_bounds.csv", index=False)

    stats_rows = []
    for target in ["sed", "sed2"]:
        stats_rows.append({"band": args.band, "target": target, **paired_summary(out, target)})
    stats_df = pd.DataFrame(stats_rows)
    stats_path = args.outdir / f"ds005620_{args.band}_wpli_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "band": args.band,
        "n_subjects": len(subjects),
        "target_sfreq": args.target_sfreq,
        "crop_s": args.crop_s,
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "max_pairs": args.max_pairs,
        "stats": stats_df.to_dict(orient="records"),
    }
    with open(args.outdir / f"ds005620_{args.band}_wpli_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
