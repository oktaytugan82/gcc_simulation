from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_doc_gcc_validation import SPECTRAL_COLS, add_cv_pi_scores, auc_ci  # noqa: E402


def residualize_loo(x: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    resid = np.full(len(score), np.nan)
    valid_all = np.isfinite(score) & ~x.replace([np.inf, -np.inf], np.nan).isna().any(axis=1).to_numpy()
    for i in range(len(score)):
        train = np.ones(len(score), dtype=bool)
        train[i] = False
        train &= valid_all
        if not valid_all[i] or np.sum(train) < 5:
            continue
        model = LinearRegression()
        model.fit(x.iloc[train], score[train])
        resid[i] = score[i] - float(model.predict(x.iloc[[i]])[0])
    return resid


def mann_whitney_effect(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {}
    u = stats.mannwhitneyu(pos, neg, alternative="greater")
    d = (float(np.mean(pos) - np.mean(neg)) / float(np.std(np.r_[pos, neg], ddof=1))) if len(score) > 2 else np.nan
    return {
        "pos_mean": float(np.mean(pos)),
        "neg_mean": float(np.mean(neg)),
        "mean_delta": float(np.mean(pos) - np.mean(neg)),
        "cohens_d_pooled": d,
        "mannwhitney_p_greater": float(u.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    record = pd.read_csv(args.outdir / "doc_gcc_record_features.csv")
    windows = pd.read_csv(args.outdir / "doc_gcc_window_features.csv")

    rows = []
    plot_data = []
    for band in sorted(record["band"].unique()):
        for contrast, eligible, positive in [
            ("MCSplus_vs_VS", {"MCS+", "VS"}, {"MCS+"}),
            ("MCSany_vs_VS", {"MCS+", "MCS-", "VS"}, {"MCS+", "MCS-"}),
        ]:
            sub = record[(record["band"] == band) & (record["label"].isin(eligible))].reset_index(drop=True)
            y = sub["label"].isin(positive).astype(int).to_numpy()
            pi_records = sub[["filename", "label", "band"]].copy()
            pi = add_cv_pi_scores(windows, pi_records, contrast, band, args.alpha)
            spec_cols = [f"{c}_mean" for c in SPECTRAL_COLS]
            x = sub[spec_cols].replace([np.inf, -np.inf], np.nan)
            pi_resid = residualize_loo(x, pi)
            x_artifact = pd.concat([x, sub[["n_epochs"]].reset_index(drop=True)], axis=1)
            pi_resid_artifact = residualize_loo(x_artifact, pi)

            for name, score in [
                ("raw_cv_pi", pi),
                ("spectral_residual_cv_pi", pi_resid),
                ("spectral_plus_epoch_residual_cv_pi", pi_resid_artifact),
            ]:
                auc, lo, hi = auc_ci(y, score)
                eff = mann_whitney_effect(y, score)
                rows.append({"band": band, "contrast": contrast, "score": name, "n": int(np.sum(np.isfinite(score))), "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi, **eff})
                for idx, val in enumerate(score):
                    plot_data.append({"band": band, "contrast": contrast, "score": name, "filename": sub.loc[idx, "filename"], "label": sub.loc[idx, "label"], "y": int(y[idx]), "value": val})

            if band == "alpha" and contrast == "MCSplus_vs_VS":
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160)
                pdata = pd.DataFrame({"label": sub["label"], "Pi": pi, "Pi_residual": pi_resid})
                labels = ["VS", "MCS+"]
                axes[0].boxplot([pdata[pdata["label"] == lab]["Pi"].dropna() for lab in labels], labels=labels, showfliers=False)
                axes[0].scatter(np.repeat([1, 2], [sum(pdata["label"] == lab) for lab in labels]), pdata.sort_values("label")["Pi"], s=14, alpha=0.7)
                axes[0].set_title("Alpha CV-GCC access Pi")
                axes[0].set_ylabel("cross-validated access-window fraction")
                axes[1].boxplot([pdata[pdata["label"] == lab]["Pi_residual"].dropna() for lab in labels], labels=labels, showfliers=False)
                axes[1].scatter(np.repeat([1, 2], [sum(pdata["label"] == lab) for lab in labels]), pdata.sort_values("label")["Pi_residual"], s=14, alpha=0.7)
                axes[1].set_title("After spectral residualization")
                axes[1].set_ylabel("LOO residual Pi")
                fig.tight_layout()
                fig.savefig(args.outdir / "doc_alpha_mcsplus_vs_vs_pi_residual.png")
                plt.close(fig)

                fpr, tpr, _ = roc_curve(y[np.isfinite(pi)], pi[np.isfinite(pi)])
                fpr_r, tpr_r, _ = roc_curve(y[np.isfinite(pi_resid)], pi_resid[np.isfinite(pi_resid)])
                fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=160)
                ax.plot(fpr, tpr, label="CV-GCC Pi")
                ax.plot(fpr_r, tpr_r, label="Spectral residual Pi")
                ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.set_title("MCS+ vs VS, alpha")
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(args.outdir / "doc_alpha_mcsplus_vs_vs_roc.png")
                plt.close(fig)

    pd.DataFrame(rows).to_csv(args.outdir / "doc_pi_residual_metrics.csv", index=False)
    pd.DataFrame(plot_data).to_csv(args.outdir / "doc_pi_subject_scores.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
