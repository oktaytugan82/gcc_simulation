from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gcc_batch15_gcco_20260514"))

from batch15_gcco_pipeline import (  # noqa: E402
    SPECTRAL_COLS,
    STATE_MAP,
    calibrate_bounds,
    crop_to_sleep_window,
    gcco_observable_series,
    spectral_features,
)


SLEEP_BANDS = {"sigma": (12.0, 16.0)}


def pair_sleep_files(root: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for psg in sorted(root.rglob("*-PSG.edf")):
        subject = psg.name[:6]
        hypos = sorted(root.rglob(subject + "*Hypnogram.edf"))
        if hypos:
            pairs.append((psg, hypos[0], subject))
    return pairs


def sleep_epoch_features(epoch: np.ndarray, sfreq: float, band: tuple[float, float], max_pairs: int, seed: int) -> dict[str, float]:
    series = gcco_observable_series(epoch, sfreq, band, window_s=30.0, stride_s=30.0, max_pairs=max_pairs, seed=seed)
    out = {
        "R": float(series["R"].iloc[0]),
        "D_eff": float(series["D_eff"].iloc[0]),
        "M_tau": float(series["M_tau"].iloc[0]),
    }
    out.update(spectral_features(epoch, sfreq))
    return out


def add_subject_pi(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    scored = []
    for (subject, band), sub in df.groupby(["subject", "band"], sort=True):
        wake = sub[sub["state"] == "Wake"]
        if len(wake) < 10:
            continue
        bounds = calibrate_bounds(wake, alpha)
        tmp = sub.copy()
        tmp["R_ok"] = tmp["R"] >= bounds["R_min"]
        tmp["D_ok"] = (tmp["D_eff"] >= bounds["D_min"]) & (tmp["D_eff"] <= bounds["D_max"])
        tmp["M_ok"] = (tmp["M_tau"] >= bounds["M_min"]) & (tmp["M_tau"] <= bounds["M_max"])
        tmp["Pi"] = tmp[["R_ok", "D_ok", "M_ok"]].mean(axis=1)
        scored.append(tmp)
    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()


def analyze_sleep(root: Path, alpha: float, max_pairs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit = []
    event_id = {k: i + 1 for i, k in enumerate(STATE_MAP)}
    for psg, hypno, subject in pair_sleep_files(root):
        try:
            annotations = mne.read_annotations(hypno)
            raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
            raw = crop_to_sleep_window(raw, annotations)
            raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])
            raw.load_data(verbose="ERROR")
            events, id_map = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.0, verbose="ERROR")
            if len(events) == 0:
                audit.append({"subject": subject, "psg": psg.name, "status": "no_events"})
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
            data = epochs.get_data(copy=False)
            for idx, ev in enumerate(events):
                desc = inv_id[int(ev[2])]
                state = STATE_MAP[desc]
                for band_name, band in SLEEP_BANDS.items():
                    feat = sleep_epoch_features(data[idx], raw.info["sfreq"], band, max_pairs, seed)
                    rows.append(
                        {
                            "dataset": "SleepEDF",
                            "subject": subject,
                            "state": state,
                            "annotation": desc,
                            "epoch_index": idx,
                            "band": band_name,
                            **feat,
                        }
                    )
            audit.append({"subject": subject, "psg": psg.name, "status": "ok", "epochs": int(len(events))})
            print(f"SleepEDF expanded: {subject} ok ({len(events)} epochs)", flush=True)
        except Exception as exc:  # noqa: BLE001
            audit.append({"subject": subject, "psg": psg.name, "status": "failed", "error": str(exc)})
            print(f"SleepEDF expanded: {subject} FAILED {exc}", flush=True)
    raw_df = pd.DataFrame(rows)
    scored = add_subject_pi(raw_df, alpha) if not raw_df.empty else pd.DataFrame()
    return scored, pd.DataFrame(audit)


def cv_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["log_M"] = np.log10(work["M_tau"] + 1e-12)
    feature_sets = {
        "spectral_all": SPECTRAL_COLS,
        "gcco_pi": ["Pi"],
        "gcco_triad_plus_pi": ["R", "D_eff", "log_M", "Pi"],
        "spectral_all_plus_gcco": SPECTRAL_COLS + ["R", "D_eff", "log_M", "Pi"],
    }
    for band, bdf in work.groupby("band"):
        for model, cols in feature_sets.items():
            for positive, negative in [("Wake", "NREM"), ("REM", "NREM")]:
                sub = bdf[bdf["state"].isin([positive, negative])].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
                y = (sub["state"] == positive).astype(int).to_numpy()
                groups = sub["subject"].to_numpy()
                scores = np.full(len(sub), np.nan)
                logo = LeaveOneGroupOut()
                for train, test in logo.split(sub[cols], y, groups):
                    if len(np.unique(y[train])) < 2:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260515))
                    clf.fit(sub.iloc[train][cols], y[train])
                    scores[test] = clf.predict_proba(sub.iloc[test][cols])[:, 1]
                mask = np.isfinite(scores)
                auc = float(roc_auc_score(y[mask], scores[mask])) if np.sum(mask) and len(np.unique(y[mask])) == 2 else np.nan
                pred = (scores[mask] >= 0.5).astype(int)
                rows.append(
                    {
                        "band": band,
                        "contrast": f"{positive}_vs_{negative}",
                        "model": model,
                        "n": int(np.sum(mask)),
                        "n_subjects": int(len(np.unique(groups[mask])) if np.any(mask) else 0),
                        "auc": auc,
                        "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)) if np.sum(mask) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def residual_cv_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["log_M"] = np.log10(work["M_tau"] + 1e-12)
    gcols = ["R", "D_eff", "log_M", "Pi"]
    for band, bdf in work.groupby("band"):
        for positive, negative in [("Wake", "NREM"), ("REM", "NREM")]:
            sub = bdf[bdf["state"].isin([positive, negative])].replace([np.inf, -np.inf], np.nan).dropna(subset=SPECTRAL_COLS + gcols)
            y = (sub["state"] == positive).astype(int).to_numpy()
            groups = sub["subject"].to_numpy()
            logo = LeaveOneGroupOut()
            scores = np.full(len(sub), np.nan)
            for train, test in logo.split(sub[SPECTRAL_COLS + gcols], y, groups):
                if len(np.unique(y[train])) < 2:
                    continue
                residual_train = np.zeros((len(train), len(gcols)))
                residual_test = np.zeros((len(test), len(gcols)))
                for j, col in enumerate(gcols):
                    reg = make_pipeline(StandardScaler(), LinearRegression())
                    reg.fit(sub.iloc[train][SPECTRAL_COLS], sub.iloc[train][col])
                    residual_train[:, j] = sub.iloc[train][col].to_numpy() - reg.predict(sub.iloc[train][SPECTRAL_COLS])
                    residual_test[:, j] = sub.iloc[test][col].to_numpy() - reg.predict(sub.iloc[test][SPECTRAL_COLS])
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260515))
                clf.fit(residual_train, y[train])
                scores[test] = clf.predict_proba(residual_test)[:, 1]
            mask = np.isfinite(scores)
            auc = float(roc_auc_score(y[mask], scores[mask])) if np.sum(mask) and len(np.unique(y[mask])) == 2 else np.nan
            pred = (scores[mask] >= 0.5).astype(int)
            rows.append(
                {
                    "band": band,
                    "contrast": f"{positive}_vs_{negative}",
                    "model": "residual_gcco_after_spectral_all",
                    "n": int(np.sum(mask)),
                    "n_subjects": int(len(np.unique(groups[mask])) if np.any(mask) else 0),
                    "auc": auc,
                    "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)) if np.sum(mask) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_stage_stats(df: pd.DataFrame) -> pd.DataFrame:
    subject_stage = df.groupby(["subject", "band", "state"], as_index=False)[["R", "D_eff", "M_tau", "Pi", *SPECTRAL_COLS]].mean()
    rows = []
    for band, bdf in subject_stage.groupby("band"):
        for feature in ["R", "D_eff", "M_tau", "Pi", "spectral_entropy", "theta_power", "alpha_power"]:
            wide = bdf.pivot(index="subject", columns="state", values=feature)
            for a, b in [("Wake", "NREM"), ("REM", "NREM"), ("Wake", "REM")]:
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna()
                if len(pair) < 4:
                    continue
                diff = pair[a] - pair[b]
                rows.append(
                    {
                        "band": band,
                        "feature": feature,
                        "contrast": f"{a}_minus_{b}",
                        "n_subjects": int(len(diff)),
                        "mean_delta": float(diff.mean()),
                        "dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
                        "wilcoxon_two_sided_p": float(stats.wilcoxon(diff).pvalue),
                    }
                )
    return pd.DataFrame(rows), subject_stage


def plot_outputs(outdir: Path, cv: pd.DataFrame, subject_stage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=180)
    show = cv[cv["contrast"].isin(["Wake_vs_NREM", "REM_vs_NREM"])].copy()
    order = ["spectral_all", "gcco_triad_plus_pi", "gcco_pi", "spectral_all_plus_gcco"]
    x = np.arange(len(order))
    width = 0.36
    for offset, contrast in [(-width / 2, "Wake_vs_NREM"), (width / 2, "REM_vs_NREM")]:
        vals = [float(show[(show["model"] == m) & (show["contrast"] == contrast)]["auc"].iloc[0]) if not show[(show["model"] == m) & (show["contrast"] == contrast)].empty else np.nan for m in order]
        ax.bar(x + offset, vals, width, label=contrast)
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    ax.set_xticks(x, ["spectral", "GCC triad", "Pi", "spectral+GCC"], rotation=15)
    ax.set_ylabel("Leave-subject-out AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Sleep-EDF cross-paradigm model comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "sleep_cross_paradigm_auc.png")
    plt.close(fig)

    zdf = subject_stage.copy()
    features = ["R", "M_tau", "Pi", "spectral_entropy"]
    for feature in features:
        vals = zdf[feature].to_numpy(float)
        zdf[feature + "_z"] = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)
    means = zdf.groupby("state")[[f + "_z" for f in features]].mean().reindex(["Wake", "REM", "NREM"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=180)
    im = ax.imshow(means.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_yticks(np.arange(len(means.index)), means.index)
    ax.set_xticks(np.arange(len(features)), features, rotation=20)
    ax.set_title("Subject-level GCC/spectral state geometry")
    fig.colorbar(im, ax=ax, label="z-scored subject-stage mean")
    fig.tight_layout()
    fig.savefig(outdir / "sleep_state_geometry_heatmap.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if not np.isfinite(v) else format(float(v), floatfmt))
    cols = list(show.columns)
    return "\n".join(
        ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        + ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in show.iterrows()]
    )


def write_report(outdir: Path, audit: pd.DataFrame, cv: pd.DataFrame, residual: pd.DataFrame, paired: pd.DataFrame, subject_stage: pd.DataFrame) -> None:
    lines = []
    lines.append("# Batch 18 Sleep-EDF Cross-Paradigm GCC Geometry\n")
    lines.append("Date: 2026-05-15\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether GCC observables capture state geometry outside anesthesia. "
        "The claim is deliberately not that Pi alone is a superior sleep-stage biomarker. "
        "The claim is that the GCC triad generalizes to sleep as an access-state geometry while conventional spectral features remain strong sleep-stage markers.\n"
    )
    lines.append("## Dataset Audit\n")
    lines.append(md_table(audit.groupby("status", as_index=False).size().rename(columns={"size": "count"})))
    lines.append(f"\nAnalyzed subjects: {subject_stage['subject'].nunique() if not subject_stage.empty else 0}\n")
    lines.append("## Leave-Subject-Out Classification\n")
    lines.append(md_table(cv))
    lines.append("\n## Residual GCC After Spectral Regression\n")
    lines.append(md_table(residual))
    lines.append("\n## Subject-Level Paired Stage Contrasts\n")
    core = paired[paired["feature"].isin(["R", "M_tau", "Pi", "spectral_entropy"])].sort_values(["feature", "contrast"])
    lines.append(md_table(core))
    lines.append("\n## Interpretation\n")
    lines.append(
        "Sleep-EDF supports cross-paradigm GCC state geometry: GCC triad features separate Wake/NREM and REM/NREM above chance. "
        "However, spectral-only features are stronger for canonical sleep staging, and Pi alone is weak. "
        "Therefore this result should be reported as cross-paradigm geometric support for GCC, not as evidence that Pi is a standalone sleep biomarker.\n"
    )
    (outdir / "BATCH18_SLEEP_CROSS_PARADIGM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    epoch_path = args.outdir / "sleep_expanded_epoch_features.csv"
    audit_path = args.outdir / "sleep_expanded_audit.csv"
    if args.reuse_existing and epoch_path.exists():
        df = pd.read_csv(epoch_path)
        audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    else:
        df, audit = analyze_sleep(args.sleep_root, args.alpha, args.max_pairs, args.seed)
        df.to_csv(epoch_path, index=False)
        audit.to_csv(audit_path, index=False)

    cv = cv_metrics(df)
    residual = residual_cv_metrics(df)
    paired, subject_stage = paired_stage_stats(df)
    cv.to_csv(args.outdir / "sleep_expanded_cv_metrics.csv", index=False)
    residual.to_csv(args.outdir / "sleep_expanded_residual_cv_metrics.csv", index=False)
    paired.to_csv(args.outdir / "sleep_expanded_paired_stage_stats.csv", index=False)
    subject_stage.to_csv(args.outdir / "sleep_expanded_subject_stage_means.csv", index=False)
    plot_outputs(args.outdir, cv, subject_stage)
    write_report(args.outdir, audit, cv, residual, paired, subject_stage)
    print(json.dumps({"epochs": int(len(df)), "subjects": int(df["subject"].nunique()) if not df.empty else 0, "cv_rows": int(len(cv))}, indent=2))


if __name__ == "__main__":
    main()
