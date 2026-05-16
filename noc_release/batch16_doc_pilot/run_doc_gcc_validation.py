from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMMON_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]
BANDS = {"alpha": (8.0, 13.0), "gamma": (35.0, 45.0)}
SPECTRAL_COLS = [
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "alpha_gamma_ratio",
    "spectral_entropy",
    "aperiodic_slope",
]


def label_from_name(name: str) -> str:
    if re.search(r"MCS\+", name):
        return "MCS+"
    if re.search(r"MCS-", name):
        return "MCS-"
    if re.search(r"VS", name):
        return "VS"
    return "UNK"


def label_score(label: str) -> int:
    return {"VS": 0, "MCS-": 1, "MCS+": 2}.get(label, -1)


def bandpass_phase(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return np.angle(signal.hilbert(filtered, axis=-1))


def graph_participation_from_weights(n_channels: int, pair_i: np.ndarray, pair_j: np.ndarray, weights: np.ndarray) -> float:
    adj = np.zeros((n_channels, n_channels), dtype=np.float64)
    valid = np.isfinite(weights)
    adj[pair_i[valid], pair_j[valid]] = weights[valid]
    adj[pair_j[valid], pair_i[valid]] = weights[valid]
    deg = adj.sum(axis=1)
    keep = deg > 1e-10
    if int(np.sum(keep)) < 2:
        return 1.0
    a = adj[np.ix_(keep, keep)]
    d = deg[keep]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    norm_adj = a * inv_sqrt[:, None] * inv_sqrt[None, :]
    lap = np.eye(norm_adj.shape[0]) - norm_adj
    eig = np.linalg.eigvalsh(lap)
    eig = np.clip(eig, 1e-12, None)
    return float((eig.sum() ** 2) / np.sum(eig**2))


def spectral_features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    nperseg = min(data.shape[1], int(round(4.0 * sfreq)))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_psd = np.nanmean(psd, axis=0)
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total = float(np.trapezoid(mean_psd[total_mask], freqs[total_mask])) if np.any(total_mask) else np.nan

    def rel_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask) or not np.isfinite(total) or total <= 0:
            return np.nan
        return float(np.trapezoid(mean_psd[mask], freqs[mask]) / total)

    theta = rel_power(4.0, 8.0)
    alpha = rel_power(8.0, 13.0)
    beta = rel_power(13.0, 30.0)
    gamma = rel_power(30.0, 45.0)
    p = mean_psd[total_mask].astype(float)
    p = p / np.nansum(p) if np.nansum(p) > 0 else np.full_like(p, np.nan)
    entropy = float(-np.nansum(p * np.log(p + 1e-12)) / np.log(len(p))) if len(p) > 1 else np.nan
    slope_mask = (freqs >= 2.0) & (freqs <= 45.0) & (mean_psd > 0)
    slope = float(np.polyfit(np.log10(freqs[slope_mask]), np.log10(mean_psd[slope_mask]), 1)[0]) if np.sum(slope_mask) >= 5 else np.nan
    return {
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "gamma_power": gamma,
        "alpha_gamma_ratio": float(alpha / (gamma + 1e-12)) if np.isfinite(alpha) and np.isfinite(gamma) else np.nan,
        "spectral_entropy": entropy,
        "aperiodic_slope": slope,
    }


def robust_standardize(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    for i in range(out.shape[0]):
        med = np.nanmedian(out[i])
        mad = np.nanmedian(np.abs(out[i] - med))
        scale = 1.4826 * mad if mad > 1e-12 else np.nanstd(out[i])
        out[i] = (out[i] - med) / (scale + 1e-12)
    return out


def epoch_keep_mask(data: np.ndarray, sfreq: float, epoch_s: float, max_abs_z: float, keep_quantile: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    win = int(round(epoch_s * sfreq))
    epochs = [(s, s + win) for s in range(0, data.shape[1] - win + 1, win)]
    if not epochs:
        return np.array([], dtype=bool), []
    scores = []
    for start, stop in epochs:
        ep = data[:, start:stop]
        scores.append(float(np.nanmax(np.abs(ep))))
    scores_arr = np.asarray(scores)
    cutoff = min(max_abs_z, float(np.nanquantile(scores_arr, keep_quantile)))
    keep = np.isfinite(scores_arr) & (scores_arr <= cutoff)
    if int(np.sum(keep)) < max(20, int(0.25 * len(keep))):
        cutoff = float(np.nanquantile(scores_arr, 0.90))
        keep = np.isfinite(scores_arr) & (scores_arr <= cutoff)
    return keep, epochs


def window_features(data: np.ndarray, sfreq: float, band: tuple[float, float], epoch_s: float, keep: np.ndarray, epochs: list[tuple[int, int]]) -> pd.DataFrame:
    phase = bandpass_phase(data, sfreq, band)
    pair_i, pair_j = np.triu_indices(data.shape[0], k=1)
    rows = []
    for idx, (start, stop) in enumerate(epochs):
        if idx >= len(keep) or not bool(keep[idx]):
            continue
        ph = phase[:, start:stop]
        imag = np.sin(ph[pair_i] - ph[pair_j])
        denom = np.mean(np.abs(imag), axis=1)
        w = np.abs(np.mean(imag, axis=1)) / np.where(denom > 1e-12, denom, np.nan)
        lagged_t = np.nanmean(np.abs(imag), axis=0)
        spec = spectral_features(data[:, start:stop], sfreq)
        rows.append(
            {
                "epoch_index": idx,
                "t": float((start + stop) / (2.0 * sfreq)),
                "R": float(np.nanmean(w)),
                "D_eff": graph_participation_from_weights(data.shape[0], pair_i, pair_j, w),
                "M_tau": float(np.nanvar(lagged_t)),
                **spec,
            }
        )
    return pd.DataFrame(rows)


def summarize_windows(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col in ["R", "D_eff", "M_tau", *SPECTRAL_COLS]:
        if col not in df:
            continue
        vals = df[col].to_numpy(float)
        out[f"{col}_mean"] = float(np.nanmean(vals))
        out[f"{col}_median"] = float(np.nanmedian(vals))
        out[f"{col}_q25"] = float(np.nanquantile(vals, 0.25))
        out[f"{col}_q75"] = float(np.nanquantile(vals, 0.75))
    out["n_epochs"] = int(len(df))
    return out


def load_common_edf(path: Path, target_sfreq: float, analysis_s: float) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    missing = [ch for ch in COMMON_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"missing common channels: {missing}")
    raw.pick(COMMON_CHANNELS)
    raw.crop(tmin=0, tmax=min(float(raw.times[-1]), analysis_s), include_tmax=False)
    raw.load_data(verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    data = robust_standardize(raw.get_data())
    return data, float(raw.info["sfreq"])


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 5000, seed: int = 20260514) -> tuple[float, float, float]:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    rng = np.random.default_rng(seed)
    vals = []
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return auc, np.nan, np.nan
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
        vals.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return auc, float(lo), float(hi)


def loo_model_scores(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> np.ndarray:
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    valid = ~x.isna().any(axis=1).to_numpy()
    scores = np.full(len(df), np.nan)
    loo = LeaveOneOut()
    for train, test in loo.split(x):
        train = train[valid[train]]
        if not valid[test[0]] or len(np.unique(y[train])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=20260514))
        clf.fit(x.iloc[train], y[train])
        scores[test] = clf.predict_proba(x.iloc[test])[:, 1]
    return scores


def add_cv_pi_scores(window_df: pd.DataFrame, record_df: pd.DataFrame, contrast: str, band: str, alpha: float) -> np.ndarray:
    if contrast == "MCSplus_vs_VS":
        positive_labels = {"MCS+"}
        eligible = {"MCS+", "VS"}
    else:
        positive_labels = {"MCS+", "MCS-"}
        eligible = {"MCS+", "MCS-", "VS"}
    records = record_df[(record_df["band"] == band) & (record_df["label"].isin(eligible))].reset_index(drop=True)
    scores = []
    for _, test_row in records.iterrows():
        train_files = set(records.loc[records["filename"] != test_row["filename"], "filename"])
        pos_train_files = set(records[(records["filename"].isin(train_files)) & (records["label"].isin(positive_labels))]["filename"])
        train_pos = window_df[(window_df["band"] == band) & (window_df["filename"].isin(pos_train_files))]
        test_w = window_df[(window_df["band"] == band) & (window_df["filename"] == test_row["filename"])]
        if len(train_pos) < 50 or test_w.empty:
            scores.append(np.nan)
            continue
        bounds = {
            "R_min": float(train_pos["R"].quantile(alpha)),
            "D_min": float(train_pos["D_eff"].quantile(alpha)),
            "D_max": float(train_pos["D_eff"].quantile(1.0 - alpha)),
            "M_min": float(train_pos["M_tau"].quantile(alpha)),
            "M_max": float(train_pos["M_tau"].quantile(1.0 - alpha)),
        }
        ok = (
            (test_w["R"] >= bounds["R_min"])
            & (test_w["D_eff"] >= bounds["D_min"])
            & (test_w["D_eff"] <= bounds["D_max"])
            & (test_w["M_tau"] >= bounds["M_min"])
            & (test_w["M_tau"] <= bounds["M_max"])
        )
        scores.append(float(ok.mean()))
    return np.asarray(scores, dtype=float)


def evaluate(record_df: pd.DataFrame, window_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    feature_sets = {
        "R_only": ["R_mean"],
        "D_only": ["D_eff_mean"],
        "M_only": ["log_M_mean"],
        "GCC_triad": ["R_mean", "D_eff_mean", "log_M_mean"],
        "GCC_triad_quantiles": ["R_mean", "R_q75", "D_eff_mean", "D_eff_q25", "D_eff_q75", "log_M_mean", "log_M_q75"],
        "spectral_all": [f"{c}_mean" for c in SPECTRAL_COLS],
        "spectral_plus_GCC": [f"{c}_mean" for c in SPECTRAL_COLS] + ["R_mean", "D_eff_mean", "log_M_mean"],
    }
    for band, bdf0 in record_df.groupby("band", sort=True):
        bdf = bdf0.copy().reset_index(drop=True)
        bdf["log_M_mean"] = np.log10(bdf["M_tau_mean"] + 1e-12)
        bdf["log_M_q75"] = np.log10(bdf["M_tau_q75"] + 1e-12)
        for contrast, eligible, positive in [
            ("MCSplus_vs_VS", {"MCS+", "VS"}, {"MCS+"}),
            ("MCSany_vs_VS", {"MCS+", "MCS-", "VS"}, {"MCS+", "MCS-"}),
        ]:
            sub = bdf[bdf["label"].isin(eligible)].reset_index(drop=True)
            y = sub["label"].isin(positive).astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            pi_records = sub[["filename", "label", "band"]].copy()
            pi_score = add_cv_pi_scores(window_df, pi_records, contrast, band, alpha)
            auc, lo, hi = auc_ci(y, pi_score)
            pred = (pi_score >= np.nanmedian(pi_score)).astype(int)
            rows.append(
                {
                    "band": band,
                    "contrast": contrast,
                    "model": "CV_GCC_access_Pi",
                    "n": int(np.sum(np.isfinite(pi_score))),
                    "auc": auc,
                    "auc_ci_low": lo,
                    "auc_ci_high": hi,
                    "balanced_accuracy_median_cut": float(balanced_accuracy_score(y[np.isfinite(pi_score)], pred[np.isfinite(pi_score)])) if np.any(np.isfinite(pi_score)) else np.nan,
                }
            )
            for model, cols in feature_sets.items():
                missing = [c for c in cols if c not in sub.columns]
                if missing:
                    continue
                score = loo_model_scores(sub, cols, y)
                auc, lo, hi = auc_ci(y, score)
                pred = (score >= 0.5).astype(int)
                mask = np.isfinite(score)
                rows.append(
                    {
                        "band": band,
                        "contrast": contrast,
                        "model": model,
                        "n": int(np.sum(mask)),
                        "auc": auc,
                        "auc_ci_low": lo,
                        "auc_ci_high": hi,
                        "balanced_accuracy_median_cut": float(balanced_accuracy_score(y[mask], pred[mask])) if np.any(mask) else np.nan,
                    }
                )
        # Severity monotonicity is a second, non-classifier endpoint.
        severity = bdf["label"].map(label_score).to_numpy()
        for col in ["R_mean", "D_eff_mean", "log_M_mean", "R_q75", "D_eff_q25"]:
            rho, p = stats.spearmanr(severity, bdf[col], nan_policy="omit")
            rows.append({"band": band, "contrast": "VS_to_MCSminus_to_MCSplus", "model": f"severity_spearman_{col}", "n": int(len(bdf)), "auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan, "balanced_accuracy_median_cut": np.nan, "spearman_rho": float(rho), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def write_report(outdir: Path, audit: dict, metrics: pd.DataFrame, record_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Batch 16 Public DoC GCC Validation\n")
    lines.append("Date: 2026-05-14\n")
    lines.append("## Dataset\n")
    lines.append(
        "Public Mendeley Data dataset 10.17632/6wx4n25h4v.1, 42 polysomnographic EDF recordings from chronic disorders of consciousness. Labels are parsed from filenames: VS, MCS-, MCS+. The frozen common montage uses F3, F4, C3, C4, O1, and O2, present in all files.\n"
    )
    lines.append("## Audit\n")
    lines.append(f"- Files analyzed: {audit['n_files']}\n")
    lines.append(f"- Label counts: {audit['labels']}\n")
    lines.append(f"- Sampling rate: {audit['sfreq_counts']}\n")
    lines.append(f"- Analysis duration per file: first {audit['analysis_s']} s\n")
    lines.append("## Cross-Validated Clinical Endpoints\n")
    lines.append(md_table(metrics.sort_values(["contrast", "band", "auc"], ascending=[True, True, False])))
    lines.append("\n## Label Means\n")
    means = (
        record_df.groupby(["band", "label"], as_index=False)[["R_mean", "D_eff_mean", "M_tau_mean", "n_epochs"]]
        .mean()
        .sort_values(["band", "label"])
    )
    lines.append(md_table(means))
    lines.append("\n## Interpretation Rule\n")
    lines.append(
        "A spectacular GCC result would require GCC_triad or CV_GCC_access_Pi to outperform spectral_all on VS/MCS classification with non-overlapping or clearly shifted bootstrap CIs, and preferably a monotonic severity trend VS < MCS- < MCS+. If spectral_all dominates, the DoC dataset still supports state sensitivity but not a unique GCC clinical biomarker claim.\n"
    )
    (outdir / "BATCH16_DOC_GCC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if not np.isfinite(v) else format(float(v), floatfmt))
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--analysis-s", type=float, default=14400.0)
    parser.add_argument("--epoch-s", type=float, default=30.0)
    parser.add_argument("--max-abs-z", type=float, default=10.0)
    parser.add_argument("--keep-quantile", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    record_rows = []
    window_frames = []
    failures = []
    files = sorted(args.root.glob("*.edf"))
    for idx, path in enumerate(files, start=1):
        label = label_from_name(path.name)
        try:
            data, sfreq = load_common_edf(path, args.target_sfreq, args.analysis_s)
            keep, epochs = epoch_keep_mask(data, sfreq, args.epoch_s, args.max_abs_z, args.keep_quantile)
            for band_name, band in BANDS.items():
                win = window_features(data, sfreq, band, args.epoch_s, keep, epochs)
                if win.empty:
                    continue
                win["filename"] = path.name
                win["label"] = label
                win["band"] = band_name
                window_frames.append(win)
                record_rows.append({"filename": path.name, "label": label, "band": band_name, **summarize_windows(win)})
            print(f"[{idx}/{len(files)}] analyzed {path.name} keep={int(np.sum(keep))}/{len(keep)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"filename": path.name, "error": str(exc)})
            print(f"[{idx}/{len(files)}] FAILED {path.name}: {exc}", flush=True)

    record_df = pd.DataFrame(record_rows)
    window_df = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    metrics = evaluate(record_df, window_df, args.alpha) if not record_df.empty and not window_df.empty else pd.DataFrame()

    record_df.to_csv(args.outdir / "doc_gcc_record_features.csv", index=False)
    window_df.to_csv(args.outdir / "doc_gcc_window_features.csv", index=False)
    metrics.to_csv(args.outdir / "doc_gcc_cv_metrics.csv", index=False)
    audit = {
        "n_files": len(files),
        "failures": failures,
        "labels": {k: int(v) for k, v in pd.Series([label_from_name(p.name) for p in files]).value_counts().to_dict().items()},
        "sfreq_counts": {"256_raw_to_128": len(files)},
        "analysis_s": args.analysis_s,
    }
    (args.outdir / "doc_gcc_run_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(args.outdir, audit, metrics, record_df)
    print(json.dumps({"record_rows": len(record_df), "window_rows": len(window_df), "metric_rows": len(metrics), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
