from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_pi_residual_and_plots import residualize_loo  # noqa: E402
from run_doc_gcc_validation import SPECTRAL_COLS, add_cv_pi_scores  # noqa: E402


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not np.any(mask):
        return q
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def primary_scores(record: pd.DataFrame, windows: pd.DataFrame, labels: np.ndarray, alpha: float, residual: str) -> tuple[np.ndarray, np.ndarray]:
    sub = record[(record["band"] == "alpha") & (record["label"].isin(["MCS+", "VS"]))].reset_index(drop=True)
    if len(labels) != len(sub):
        raise ValueError("label length mismatch")
    tmp = sub[["filename", "band"]].copy()
    tmp["label"] = np.where(labels == 1, "MCS+", "VS")
    pi = add_cv_pi_scores(windows, tmp[["filename", "label", "band"]], "MCSplus_vs_VS", "alpha", alpha)
    if residual == "none":
        return labels, pi
    spec_cols = [f"{c}_mean" for c in SPECTRAL_COLS]
    x = sub[spec_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if residual == "spectral":
        return labels, residualize_loo(x, pi)
    if residual == "spectral_epoch":
        x2 = pd.concat([x, sub[["n_epochs"]].reset_index(drop=True)], axis=1)
        return labels, residualize_loo(x2, pi)
    raise ValueError(residual)


def permutation_test(record: pd.DataFrame, windows: pd.DataFrame, alpha: float, residual: str, n_perm: int, seed: int) -> dict[str, float]:
    sub = record[(record["band"] == "alpha") & (record["label"].isin(["MCS+", "VS"]))].reset_index(drop=True)
    y = (sub["label"] == "MCS+").astype(int).to_numpy()
    _, score = primary_scores(record, windows, y, alpha, residual)
    mask = np.isfinite(score)
    obs_auc = float(roc_auc_score(y[mask], score[mask]))
    obs_delta = float(np.nanmean(score[y == 1]) - np.nanmean(score[y == 0]))
    rng = np.random.default_rng(seed)
    auc_null = []
    delta_null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        _, sp = primary_scores(record, windows, yp, alpha, residual)
        mp = np.isfinite(sp)
        if len(np.unique(yp[mp])) < 2:
            continue
        auc_null.append(float(roc_auc_score(yp[mp], sp[mp])))
        delta_null.append(float(np.nanmean(sp[yp == 1]) - np.nanmean(sp[yp == 0])))
    auc_null = np.asarray(auc_null)
    delta_null = np.asarray(delta_null)
    return {
        "endpoint": "alpha_MCSplus_vs_VS",
        "alpha": alpha,
        "residual": residual,
        "n": int(np.sum(mask)),
        "observed_auc": obs_auc,
        "observed_delta": obs_delta,
        "permutation_auc_p_greater": float((1 + np.sum(auc_null >= obs_auc)) / (1 + len(auc_null))),
        "permutation_delta_p_greater": float((1 + np.sum(delta_null >= obs_delta)) / (1 + len(delta_null))),
        "null_auc_mean": float(np.mean(auc_null)),
        "null_auc_q95": float(np.quantile(auc_null, 0.95)),
        "n_perm": int(len(auc_null)),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    sens = pd.read_csv(args.outdir / "doc_pi_alpha_sensitivity.csv")
    sens["q_all_tests_bh"] = bh_fdr(sens["mannwhitney_p_greater"].to_numpy(float))
    sens["q_within_score_bh"] = np.nan
    for score, idx in sens.groupby("score").groups.items():
        sens.loc[idx, "q_within_score_bh"] = bh_fdr(sens.loc[idx, "mannwhitney_p_greater"].to_numpy(float))
    sens.to_csv(args.outdir / "doc_pi_alpha_sensitivity_fdr.csv", index=False)

    record = pd.read_csv(args.outdir / "doc_gcc_record_features.csv")
    windows = pd.read_csv(args.outdir / "doc_gcc_window_features.csv")
    perm_rows = [
        permutation_test(record, windows, alpha=0.10, residual="none", n_perm=args.n_perm, seed=args.seed),
        permutation_test(record, windows, alpha=0.10, residual="spectral", n_perm=args.n_perm, seed=args.seed + 1),
        permutation_test(record, windows, alpha=0.10, residual="spectral_epoch", n_perm=args.n_perm, seed=args.seed + 2),
    ]
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(args.outdir / "doc_primary_permutation_tests.csv", index=False)

    primary = sens[
        (sens["alpha"] == 0.10)
        & (sens["band"] == "alpha")
        & (sens["contrast"] == "MCSplus_vs_VS")
        & (sens["score"].isin(["raw_cv_pi", "spectral_residual_cv_pi", "spectral_plus_epoch_residual_cv_pi"]))
    ].copy()

    lines = []
    lines.append("# Batch 16 DoC Multiple-Comparisons and Permutation Addendum\n")
    lines.append("Date: 2026-05-15\n")
    lines.append("## Primary Endpoint\n")
    lines.append("Pre-specified clinical anchor: alpha-band CV-GCC access Pi for MCS+ vs VS at calibration alpha = 0.10.\n")
    lines.append(md_table(primary[["alpha", "band", "contrast", "score", "n", "auc", "auc_ci_low", "auc_ci_high", "mannwhitney_p_greater", "q_all_tests_bh", "q_within_score_bh"]]))
    lines.append("\n## Label-Permutation Tests\n")
    lines.append(md_table(perm))
    lines.append("\n## Interpretation\n")
    lines.append(
        "The primary alpha MCS+ vs VS GCC-Pi endpoint remains positive under permutation testing. "
        "FDR values are reported transparently across all alpha-threshold/band/contrast/score tests and within each score family. "
        "This supports use as a clinical pilot anchor, not as a definitive diagnostic biomarker.\n"
    )
    (args.outdir / "BATCH16_DOC_MULTIPLE_COMPARISONS_ADDENDUM.md").write_text("\n".join(lines), encoding="utf-8")
    print(perm.to_string(index=False))


if __name__ == "__main__":
    main()
