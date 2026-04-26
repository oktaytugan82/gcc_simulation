from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = Path(os.environ.get("DS004504_ROOT", REPO_ROOT / "data" / "ds004504-main"))
RESULTS_DIR = REPO_ROOT / "results"
OUT_FEATURES = RESULTS_DIR / "ds004504_gcc_features_by_band.csv"
OUT_WINDOWS = RESULTS_DIR / "ds004504_gcc_window_features.csv"
OUT_WPLI = RESULTS_DIR / "ds004504_wpli_matrices.npz"
OUT_SUMMARY = RESULTS_DIR / "ds004504_feature_extraction_summary.json"

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}

TARGET_SFREQ = 125.0
WINDOW_S = 10.0
STEP_S = 5.0
EPS = 1e-12


def eeg_path(subject_id: str) -> Path:
    return ROOT / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"


def resample_to_target(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, float]:
    if abs(sfreq - TARGET_SFREQ) < 1e-6:
        return data, sfreq
    ratio = sfreq / TARGET_SFREQ
    if abs(ratio - round(ratio)) < 1e-6:
        return signal.resample_poly(data, up=1, down=int(round(ratio)), axis=1), TARGET_SFREQ
    n_samples = int(round(data.shape[1] * TARGET_SFREQ / sfreq))
    return signal.resample(data, n_samples, axis=1), TARGET_SFREQ


def bandpass(data: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def effective_dimensionality(segment: np.ndarray) -> float:
    # Correlation participation ratio. Constant channels are guarded against.
    centered = segment - segment.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    z = centered / np.maximum(scale, EPS)
    corr = np.corrcoef(z)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    denom = float(np.sum(eigvals**2))
    if denom <= EPS:
        return float("nan")
    return float((np.sum(eigvals) ** 2) / denom)


def wpli_matrix(analytic: np.ndarray) -> np.ndarray:
    n_channels = analytic.shape[0]
    out = np.eye(n_channels, dtype=float)
    for i in range(n_channels):
        zi = analytic[i]
        for j in range(i + 1, n_channels):
            im = np.imag(zi * np.conj(analytic[j]))
            denom = np.mean(np.abs(im))
            value = 0.0 if denom <= EPS else abs(float(np.mean(im))) / float(denom)
            out[i, j] = value
            out[j, i] = value
    return out


def upper_mean(matrix: np.ndarray) -> float:
    idx = np.triu_indices_from(matrix, k=1)
    return float(np.mean(matrix[idx]))


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_q10": math.nan,
            f"{prefix}_q50": math.nan,
            f"{prefix}_q90": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_q10": float(np.quantile(arr, 0.10)),
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
    }


def main() -> None:
    participants = pd.read_csv(ROOT / "participants.tsv", sep="\t")
    subject_rows: list[dict] = []
    window_rows: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    channel_names: list[str] | None = None

    win = int(round(WINDOW_S * TARGET_SFREQ))
    step = int(round(STEP_S * TARGET_SFREQ))

    for index, participant in participants.iterrows():
        subject_id = str(participant["participant_id"])
        path = eeg_path(subject_id)
        print(f"[{index + 1:02d}/{len(participants)}] {subject_id} load")
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        data = raw.get_data().astype(np.float64, copy=False)
        data, sfreq = resample_to_target(data, float(raw.info["sfreq"]))
        if channel_names is None:
            channel_names = list(raw.ch_names)

        for band_name, (lo, hi) in BANDS.items():
            filtered = bandpass(data, sfreq, lo, hi)
            analytic = signal.hilbert(filtered, axis=1)
            phase = np.angle(analytic)
            r_t = np.abs(np.mean(np.exp(1j * phase), axis=0))
            wpli = wpli_matrix(analytic)
            matrices[f"{subject_id}__{band_name}"] = wpli.astype(np.float32)

            r_values: list[float] = []
            d_values: list[float] = []
            m_values: list[float] = []
            power_values: list[float] = []
            n_windows = 0

            for start in range(0, filtered.shape[1] - win + 1, step):
                stop = start + win
                segment = filtered[:, start:stop]
                r_mean = float(np.mean(r_t[start:stop]))
                d_eff = effective_dimensionality(segment)
                metastability = float(np.std(r_t[start:stop], ddof=1))
                power = float(np.mean(segment**2))
                n_windows += 1

                r_values.append(r_mean)
                d_values.append(d_eff)
                m_values.append(metastability)
                power_values.append(power)

                window_rows.append(
                    {
                        "participant_id": subject_id,
                        "group": participant["Group"],
                        "gender": participant["Gender"],
                        "age": int(participant["Age"]),
                        "mmse": int(participant["MMSE"]),
                        "band": band_name,
                        "window_index": n_windows - 1,
                        "window_start_s": float(start / sfreq),
                        "window_stop_s": float(stop / sfreq),
                        "R": r_mean,
                        "D_eff": d_eff,
                        "M_tau": metastability,
                        "band_power": power,
                    }
                )

            row = {
                "participant_id": subject_id,
                "group": participant["Group"],
                "gender": participant["Gender"],
                "age": int(participant["Age"]),
                "mmse": int(participant["MMSE"]),
                "band": band_name,
                "sfreq_hz": sfreq,
                "n_channels": data.shape[0],
                "n_samples": data.shape[1],
                "duration_s": float(data.shape[1] / sfreq),
                "n_windows": n_windows,
                "mean_wpli": upper_mean(wpli),
            }
            row.update(summarize(r_values, "R"))
            row.update(summarize(d_values, "D_eff"))
            row.update(summarize(m_values, "M_tau"))
            row.update(summarize(power_values, "band_power"))
            subject_rows.append(row)
            print(
                f"    {band_name}: windows={n_windows} "
                f"R={row['R_mean']:.3f} D={row['D_eff_mean']:.2f} "
                f"M={row['M_tau_mean']:.3f} wPLI={row['mean_wpli']:.3f}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subject_rows).to_csv(OUT_FEATURES, index=False)
    pd.DataFrame(window_rows).to_csv(OUT_WINDOWS, index=False)
    np.savez_compressed(OUT_WPLI, **matrices)

    summary = {
        "dataset_root": str(ROOT),
        "n_subjects": int(len(participants)),
        "n_subject_band_rows": int(len(subject_rows)),
        "n_window_rows": int(len(window_rows)),
        "bands": BANDS,
        "target_sfreq_hz": TARGET_SFREQ,
        "window_s": WINDOW_S,
        "step_s": STEP_S,
        "channels": channel_names,
        "outputs": {
            "subject_band_features": str(OUT_FEATURES),
            "window_features": str(OUT_WINDOWS),
            "wpli_matrices": str(OUT_WPLI),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
