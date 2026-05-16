from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_pi_residual_and_plots import mann_whitney_effect, residualize_loo  # noqa: E402
from run_doc_gcc_validation import SPECTRAL_COLS, add_cv_pi_scores, auc_ci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()

    record = pd.read_csv(args.outdir / "doc_gcc_record_features.csv")
    windows = pd.read_csv(args.outdir / "doc_gcc_window_features.csv")
    rows = []
    for alpha in args.alphas:
        for band in sorted(record["band"].unique()):
            for contrast, eligible, positive in [
                ("MCSplus_vs_VS", {"MCS+", "VS"}, {"MCS+"}),
                ("MCSany_vs_VS", {"MCS+", "MCS-", "VS"}, {"MCS+", "MCS-"}),
            ]:
                sub = record[(record["band"] == band) & (record["label"].isin(eligible))].reset_index(drop=True)
                y = sub["label"].isin(positive).astype(int).to_numpy()
                pi_records = sub[["filename", "label", "band"]].copy()
                pi = add_cv_pi_scores(windows, pi_records, contrast, band, alpha)
                x = sub[[f"{c}_mean" for c in SPECTRAL_COLS]].replace([np.inf, -np.inf], np.nan)
                pi_resid = residualize_loo(x, pi)
                pi_resid_artifact = residualize_loo(pd.concat([x, sub[["n_epochs"]].reset_index(drop=True)], axis=1), pi)
                for score_name, score in [
                    ("raw_cv_pi", pi),
                    ("spectral_residual_cv_pi", pi_resid),
                    ("spectral_plus_epoch_residual_cv_pi", pi_resid_artifact),
                ]:
                    auc, lo, hi = auc_ci(y, score, n_boot=3000)
                    rows.append(
                        {
                            "alpha": alpha,
                            "band": band,
                            "contrast": contrast,
                            "score": score_name,
                            "n": int(np.sum(np.isfinite(score))),
                            "auc": auc,
                            "auc_ci_low": lo,
                            "auc_ci_high": hi,
                            **mann_whitney_effect(y, score),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "doc_pi_alpha_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
