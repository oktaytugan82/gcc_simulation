#!/usr/bin/env python
"""Phase-randomized surrogate control for DS005620 gamma GCC.

The surrogate preserves the univariate power spectrum of each channel but
destroys temporal phase structure and cross-channel phase relationships. If the
GCC Pi sedation effect were only a spectral-power effect, surrogate-calibrated
Pi should remain close to the original effect.
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
GAMMA_BAND = (30.0, 45.0)


def load_preprocessed(vhdr: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float]:
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
    return data.astype(np.float64, copy=False), float(raw.info["sfreq"])


def phase_randomize(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve each channel's power spectrum while randomizing phase."""
    n = data.shape[1]
    spec = np.fft.rfft(data, axis=1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.shape)
    phases[:, 0] = 0.0
    if n % 2 == 0:
        phases[:, -1] = 0.0
    randomized = np.abs(spec) * np.exp(1j * phases)
    sur = np.fft.irfft(randomized, n=n, axis=1)
    sur = sur - np.nanmean(sur, axis=1, keepdims=True)
    scale = np.nanstd(sur, axis=1, keepdims=True)
    return sur / np.where(scale > 1e-12, scale, 1.0)


def bandpass_phase(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, GAMMA_BAND, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    phase = np.angle(signal.hilbert(filtered, axis=-1))
    return filtered, phase


def effective_dimensionality(x: np.ndarray) -> float:
    cov = np.cov(x)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def observable_series(data: np.ndarray, sfreq: float, window_s: float, stride_s: float) -> pd.DataFrame:
    filtered, phase = bandpass_phase(data, sfreq)
    win = int(round(window_s * sfreq))
    stride = int(round(stride_s * sfreq))
    rows = []
    for start in range(0, filtered.shape[1] - win + 1, stride):
        stop = start + win
        phase_window = phase[:, start:stop]
        filt_window = filtered[:, start:stop]
        order_t = np.abs(np.exp(1j * phase_window).mean(axis=0))
        rows.append(
            {
                "R": float(order_t.mean()),
                "D_eff": effective_dimensionality(filt_window),
                "M_tau": float(np.var(order_t)),
            }
        )
    return pd.DataFrame(rows)


def calibrate_bounds(wake: pd.DataFrame, alpha: float) -> dict[str, float]:
    return {
        "R_min": float(wake["R"].quantile(alpha)),
        "D_min": float(wake["D_eff"].quantile(alpha)),
        "D_max": float(wake["D_eff"].quantile(1.0 - alpha)),
        "M_min": float(wake["M_tau"].quantile(alpha)),
        "M_max": float(wake["M_tau"].quantile(1.0 - alpha)),
    }


def score(series: pd.DataFrame, bounds: dict[str, float]) -> float:
    r_ok = series["R"] >= bounds["R_min"]
    d_ok = (series["D_eff"] >= bounds["D_min"]) & (series["D_eff"] <= bounds["D_max"])
    m_ok = (series["M_tau"] >= bounds["M_min"]) & (series["M_tau"] <= bounds["M_max"])
    return float(pd.concat([r_ok, d_ok, m_ok], axis=1).mean(axis=1).mean())


def paired_summary(df: pd.DataFrame, target: str, metric: str) -> dict[str, float]:
    wide = df[df["condition"].isin(["awake", target])].pivot(index="subject", columns="condition", values=metric).dropna()
    awake = wide["awake"].to_numpy()
    sed = wide[target].to_numpy()
    diff = awake - sed
    if len(diff) < 5:
        return {"n": int(len(diff))}
    _, p_t = stats.ttest_rel(awake, sed)
    p_w = stats.wilcoxon(diff, alternative="greater").pvalue
    sd = np.std(diff, ddof=1)
    return {
        "n": int(len(diff)),
        "awake_mean": float(np.mean(awake)),
        "target_mean": float(np.mean(sed)),
        "mean_delta_awake_minus_target": float(np.mean(diff)),
        "paired_d": float(np.mean(diff) / sd) if sd > 0 else np.nan,
        "ttest_p": float(p_t),
        "wilcoxon_greater_p": float(p_w),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--closest-runs", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-surrogates", type=int, default=3)
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--crop-s", type=float, default=120.0)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    closest = pd.read_csv(args.closest_runs)
    closest = closest[closest["band"] == "gamma"].copy()
    subjects = sorted(closest["subject"].astype(str).unique())
    rows = []
    rng = np.random.default_rng(20260513)

    for subject in subjects:
        sub = closest[closest["subject"].astype(str) == subject]
        awake_file = sub[sub["condition"] == "awake"]["filename"].iloc[0]
        awake_path = next(args.local_root.rglob(awake_file))
        awake_data, sfreq = load_preprocessed(awake_path, args.target_sfreq, args.crop_s)

        target_files = {
            row["condition"]: row["filename"]
            for _, row in sub[sub["condition"].isin(["sed", "sed2"])].iterrows()
        }
        target_data = {}
        for condition, filename in target_files.items():
            target_path = next(args.local_root.rglob(filename))
            data, _ = load_preprocessed(target_path, args.target_sfreq, args.crop_s)
            target_data[condition] = data

        # Original calibration for reference.
        wake_series = observable_series(awake_data, sfreq, args.window_s, args.stride_s)
        bounds = calibrate_bounds(wake_series, args.alpha)
        rows.append({"subject": subject, "condition": "awake", "kind": "original", "surrogate": -1, "Pi": score(wake_series, bounds)})
        for condition, data in target_data.items():
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "kind": "original",
                    "surrogate": -1,
                    "Pi": score(observable_series(data, sfreq, args.window_s, args.stride_s), bounds),
                }
            )

        for s in range(args.n_surrogates):
            wake_sur = phase_randomize(awake_data, rng)
            wake_sur_series = observable_series(wake_sur, sfreq, args.window_s, args.stride_s)
            sur_bounds = calibrate_bounds(wake_sur_series, args.alpha)
            rows.append({"subject": subject, "condition": "awake", "kind": "phase_surrogate", "surrogate": s, "Pi": score(wake_sur_series, sur_bounds)})
            for condition, data in target_data.items():
                sur = phase_randomize(data, rng)
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "kind": "phase_surrogate",
                        "surrogate": s,
                        "Pi": score(observable_series(sur, sfreq, args.window_s, args.stride_s), sur_bounds),
                    }
                )
        print(f"processed subject {subject}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "ds005620_phase_surrogate_pi.csv", index=False)

    # Average surrogates per subject-condition before paired inference.
    avg = out.groupby(["subject", "condition", "kind"])["Pi"].mean().reset_index()
    stats_rows = []
    for kind, sub in avg.groupby("kind"):
        for target in ["sed", "sed2"]:
            stats_rows.append({"kind": kind, "target": target, **paired_summary(sub, target, "Pi")})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(args.outdir / "ds005620_phase_surrogate_stats.csv", index=False)

    summary = {"n_surrogates": args.n_surrogates, "stats": stats_df.to_dict(orient="records")}
    with open(args.outdir / "ds005620_phase_surrogate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
