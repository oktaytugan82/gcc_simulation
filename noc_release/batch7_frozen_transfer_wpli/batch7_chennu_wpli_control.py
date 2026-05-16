#!/usr/bin/env python
"""Chennu propofol wPLI-GCC control.

This mirrors the DS005620 wPLI stress test but uses the Chennu EEG `.set`
recordings and the existing Chennu GCC summary CSVs as the file manifest.
The output allows the manuscript to present lagged-coupling GCC as a robustness
analysis across both independent propofol datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats


BANDS = {
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 45.0),
}
EXCLUDE_TOKENS = ("EOG", "EMG", "ECG", "EXG", "M1", "M2", "A1", "A2")


def _h5_read_string(f: h5py.File, dataset) -> str:
    arr = np.array(dataset).flatten()
    if arr.dtype.kind in ("i", "u"):
        try:
            return "".join(chr(int(c)) for c in arr if c > 0)
        except (ValueError, OverflowError):
            return ""
    if arr.dtype.kind in ("S", "O"):
        try:
            return b"".join(arr.tolist()).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def load_eeglab_v73(set_path: Path):
    """Load MATLAB v7.3 EEGLAB `.set` files as MNE RawArray."""
    with h5py.File(str(set_path), "r") as f:
        if "EEG" not in f:
            raise ValueError(f"No EEG group in {set_path}")
        eeg = f["EEG"]
        srate = float(np.array(eeg["srate"]).flatten()[0])
        nbchan = int(np.array(eeg["nbchan"]).flatten()[0])
        pnts = int(np.array(eeg["pnts"]).flatten()[0])
        trials = int(np.array(eeg["trials"]).flatten()[0])

        data = None
        data_field = eeg["data"]
        if data_field.dtype.kind == "f":
            data = np.array(data_field)
        else:
            data_fname = _h5_read_string(f, data_field)
            if data_fname and not data_fname.endswith(".set"):
                fdt_path = set_path.parent / data_fname
                if not fdt_path.exists():
                    fdt_path = set_path.with_suffix(".fdt")
                if fdt_path.exists():
                    raw_bytes = np.fromfile(str(fdt_path), dtype=np.float32)
                    data = raw_bytes.reshape(pnts * trials, nbchan).T
        if data is None:
            raise ValueError(f"Could not extract data from {set_path.name}")
        if data.shape[0] != nbchan:
            if data.shape[1] == nbchan:
                data = data.T
            else:
                raise ValueError(f"Data shape {data.shape} does not match nbchan={nbchan}")

        ch_names = []
        if "chanlocs" in eeg and "labels" in eeg["chanlocs"]:
            labels_refs = np.array(eeg["chanlocs"]["labels"]).flatten()
            for ref in labels_refs:
                try:
                    ch_names.append(_h5_read_string(f, f[ref]))
                except Exception:
                    ch_names.append("")
        if not ch_names or len(ch_names) != nbchan or all(name == "" for name in ch_names):
            ch_names = [f"EEG{i:03d}" for i in range(nbchan)]

    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose="ERROR")


def smart_load_set(set_path: Path):
    try:
        return mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    except (TypeError, NotImplementedError, ValueError) as exc:
        msg = str(exc)
        if "v7.3" in msg or "HDF" in msg or "h5py" in msg:
            return load_eeglab_v73(set_path)
        raise


def load_preprocessed(set_path: Path, target_sfreq: float, crop_s: float | None) -> tuple[np.ndarray, float, list[str]]:
    raw = smart_load_set(set_path)
    if crop_s is not None:
        raw.crop(tmin=0.0, tmax=min(float(crop_s), float(raw.times[-1])))
    keep = [
        ch
        for ch in raw.ch_names
        if not any(tok.lower() in ch.lower() for tok in EXCLUDE_TOKENS)
    ]
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
        imag = np.sin(phase_window[pair_i] - phase_window[pair_j])
        denom = np.mean(np.abs(imag), axis=1)
        wpli_pairs = np.abs(np.mean(imag, axis=1)) / np.where(denom > 1e-12, denom, np.nan)
        r_wpli = float(np.nanmean(wpli_pairs))
        lagged_coupling_t = np.nanmean(np.abs(imag), axis=0)
        rows.append(
            {
                "R_wpli": r_wpli,
                "D_eff": effective_dimensionality(filt_window),
                "M_tau": float(np.nanvar(lagged_coupling_t)),
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
    wins = 0.0
    total = 0
    for a in awake:
        wins += np.sum(a > target) + 0.5 * np.sum(a == target)
        total += len(target)
    return float(wins / total) if total else float("nan")


def paired_summary(df: pd.DataFrame, target: str) -> dict[str, float]:
    wide = df[df["condition"].isin(["baseline", target])].pivot(index="subject", columns="condition", values="Pi").dropna()
    awake = wide["baseline"].to_numpy()
    sed = wide[target].to_numpy()
    diff = awake - sed
    if len(diff) < 5:
        return {"n": int(len(diff))}
    _, p_t = stats.ttest_rel(awake, sed)
    p_w = stats.wilcoxon(diff, alternative="greater").pvalue
    sd = np.std(diff, ddof=1)
    rng = np.random.default_rng(20260514)
    boot = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(5000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n": int(len(diff)),
        "baseline_mean": float(np.mean(awake)),
        "target_mean": float(np.mean(sed)),
        "mean_delta_baseline_minus_target": float(np.mean(diff)),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "paired_d": float(np.mean(diff) / sd) if sd > 0 else np.nan,
        "ttest_p": float(p_t),
        "wilcoxon_greater_p": float(p_w),
        "auc_baseline_gt_target": auc_from_pairs(awake, sed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--band", choices=sorted(BANDS), default="gamma")
    parser.add_argument("--target-sfreq", type=float, default=125.0)
    parser.add_argument("--crop-s", type=float, default=90.0)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.summary_csv)
    rows = []
    bounds_rows = []

    for subject, sub in manifest.groupby("subject", sort=True):
        sub = sub.copy()
        baseline = sub[sub["level"] == "baseline"]
        if baseline.empty:
            continue
        base_file = str(baseline["filename"].iloc[0])
        base_path = args.data_root / base_file
        if not base_path.exists():
            matches = list(args.data_root.rglob(base_file))
            if not matches:
                raise FileNotFoundError(base_file)
            base_path = matches[0]
        base_data, sfreq, ch_names = load_preprocessed(base_path, args.target_sfreq, args.crop_s)
        base_series = wpli_observable_series(
            base_data, sfreq, BANDS[args.band], args.window_s, args.stride_s, args.max_pairs, args.seed
        )
        bounds = calibrate_bounds(base_series, args.alpha)
        bounds_rows.append({"subject": str(subject), "band": args.band, "n_channels": len(ch_names), **bounds})
        pi, access = score(base_series, bounds)
        rows.append(
            {
                "subject": str(subject),
                "condition": "baseline",
                "band": args.band,
                "filename": base_file,
                "n_channels": len(ch_names),
                "n_windows": len(base_series),
                "R_wpli_mean": float(base_series["R_wpli"].mean()),
                "D_eff_mean": float(base_series["D_eff"].mean()),
                "M_tau_mean": float(base_series["M_tau"].mean()),
                "Pi": pi,
                "Access_all": access,
            }
        )
        for _, row in sub[sub["level"].isin(["mild", "moderate", "recovery"])].iterrows():
            condition = str(row["level"])
            filename = str(row["filename"])
            path = args.data_root / filename
            if not path.exists():
                matches = list(args.data_root.rglob(filename))
                if not matches:
                    raise FileNotFoundError(filename)
                path = matches[0]
            data, _, ch_names = load_preprocessed(path, args.target_sfreq, args.crop_s)
            series = wpli_observable_series(
                data, sfreq, BANDS[args.band], args.window_s, args.stride_s, args.max_pairs, args.seed
            )
            pi, access = score(series, bounds)
            rows.append(
                {
                    "subject": str(subject),
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
    out.to_csv(args.outdir / f"chennu_{args.band}_wpli_gcc.csv", index=False)
    pd.DataFrame(bounds_rows).to_csv(args.outdir / f"chennu_{args.band}_wpli_bounds.csv", index=False)

    stats_rows = []
    for target in ["mild", "moderate", "recovery"]:
        stats_rows.append({"band": args.band, "target": target, **paired_summary(out, target)})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(args.outdir / f"chennu_{args.band}_wpli_stats.csv", index=False)

    summary = {
        "band": args.band,
        "n_subjects": int(out["subject"].nunique()),
        "target_sfreq": args.target_sfreq,
        "crop_s": args.crop_s,
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "max_pairs": args.max_pairs,
        "stats": stats_df.to_dict(orient="records"),
    }
    with open(args.outdir / f"chennu_{args.band}_wpli_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
