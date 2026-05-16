from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\oktay\OneDrive\Dokumente\New project")
OUT = ROOT / "gcc_phase4_noc_20260515"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def panel_source(ax):
    stats = load_csv(ROOT / "gcc_batch17_source_chennu_20260515" / "source_gcco_paired_stats.csv")
    labels = [b.upper() for b in stats["band"]]
    vals = stats["mean_delta_baseline_minus_moderate"].to_numpy(float)
    lo = stats["delta_ci_low"].to_numpy(float)
    hi = stats["delta_ci_high"].to_numpy(float)
    yerr = np.vstack([vals - lo, hi - vals])
    x = np.arange(len(vals))
    ax.bar(x, vals, color=["#4977a3", "#2f6f5e"], width=0.58)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1.2)
    for i, row in stats.iterrows():
        p = float(row["wilcoxon_greater_p"])
        dz = float(row["paired_dz"])
        ax.text(i, vals[i] + yerr[1, i] + 0.004, f"dz={dz:.2f}\np={p:.3g}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Baseline - moderate Delta Pi")
    ax.set_ylim(0, max(hi) + 0.035)
    ax.set_title("A. Chennu source-level robustness", loc="left", fontweight="bold")
    ax.text(0.02, 0.96, "fsaverage template, 68 cortical ROIs, n=20",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    style_axes(ax)


def panel_transfer(ax):
    frozen = load_csv(ROOT / "gcc_batch7_noc_20260514" / "noc_frozen_benchmark_results.csv")
    rows = [
        ("No-fit Pi sign\npooled", "GCC_Pi_sign_rule", "pooled_sign_rule", "gamma"),
        ("Pi delta\nCH->DS", "GCC_Pi_delta", "train_chennu_test_ds", "gamma"),
        ("Pi delta\nDS->CH", "GCC_Pi_delta", "train_ds_test_chennu", "gamma"),
        ("wPLI triad+Pi\nCH->DS", "wPLI_triad_plus_Pi", "train_chennu_test_ds", "gamma"),
        ("wPLI triad+Pi\nDS->CH", "wPLI_triad_plus_Pi", "train_ds_test_chennu", "gamma"),
        ("Spectral\nCH->DS", "spectral_bandpowers", "train_chennu_test_ds", "gamma"),
        ("Spectral\nDS->CH", "spectral_bandpowers", "train_ds_test_chennu", "gamma"),
    ]
    vals = []
    for _, model, direction, band in rows:
        match = frozen[(frozen["model"] == model) & (frozen["direction"] == direction) & (frozen["band"] == band)]
        if match.empty:
            raise ValueError((model, direction, band))
        vals.append(float(match.iloc[0]["auc"]))
    x = np.arange(len(vals))
    colors = ["#2f6f5e", "#2f6f5e", "#2f6f5e", "#77a6b6", "#77a6b6", "#7f8794", "#7f8794"]
    ax.bar(x, vals, color=colors, width=0.68)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("AUC")
    ax.set_xticks(x, [r[0] for r in rows], rotation=28, ha="right")
    ax.set_title("B. Propofol cross-dataset transfer", loc="left", fontweight="bold")
    ax.text(0.02, 0.96, "Gamma band; no-fit and train-on-one/test-on-other checks",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    style_axes(ax)


def panel_sleep(ax):
    cv = load_csv(ROOT / "gcc_batch18_sleep_20260515" / "sleep_expanded_cv_metrics.csv")
    residual = load_csv(ROOT / "gcc_batch18_sleep_20260515" / "sleep_expanded_residual_cv_metrics.csv")
    combined = pd.concat([cv, residual], ignore_index=True, sort=False)
    contrasts = ["Wake_vs_NREM", "REM_vs_NREM"]
    models = [
        ("spectral_all", "Spectral", "#7f8794"),
        ("gcco_triad_plus_pi", "GCC triad+Pi", "#2f6f5e"),
        ("gcco_pi", "Pi only", "#a37b39"),
        ("residual_gcco_after_spectral_all", "Residual GCC", "#77a6b6"),
    ]
    width = 0.18
    x = np.arange(len(contrasts))
    for j, (model, label, color) in enumerate(models):
        vals = []
        for contrast in contrasts:
            match = combined[(combined["contrast"] == contrast) & (combined["model"] == model)]
            vals.append(float(match.iloc[0]["auc"]) if not match.empty else np.nan)
        ax.bar(x + (j - 1.5) * width, vals, width=width, label=label, color=color)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Leave-recording-out AUC")
    ax.set_xticks(x, ["Wake vs NREM", "REM vs NREM"])
    ax.set_title("C. Sleep-EDF cross-paradigm geometry", loc="left", fontweight="bold")
    ax.text(0.02, 0.96, "22 recordings, 23,235 epochs; sigma-band two-channel EEG",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="lower left")
    style_axes(ax)


def panel_doc(ax):
    metrics = load_csv(ROOT / "gcc_batch16_doc_20260514" / "doc_pi_residual_metrics.csv")
    perm = load_csv(ROOT / "gcc_batch16_doc_20260514" / "doc_primary_permutation_tests.csv")
    rows = [
        ("Raw Pi", "raw_cv_pi", "none", "#2f6f5e"),
        ("Residual\nspectral", "spectral_residual_cv_pi", "spectral", "#77a6b6"),
        ("Residual\nspectral+epochs", "spectral_plus_epoch_residual_cv_pi", "spectral_epoch", "#9c6b4e"),
    ]
    vals, lows, highs, pvals = [], [], [], []
    for _, score, residual, _ in rows:
        m = metrics[(metrics["band"] == "alpha") & (metrics["contrast"] == "MCSplus_vs_VS") & (metrics["score"] == score)].iloc[0]
        p = perm[(perm["endpoint"] == "alpha_MCSplus_vs_VS") & (perm["alpha"].astype(float) == 0.1) & (perm["residual"] == residual)].iloc[0]
        vals.append(float(m["auc"]))
        lows.append(float(m["auc_ci_low"]))
        highs.append(float(m["auc_ci_high"]))
        pvals.append(float(p["permutation_auc_p_greater"]))
    vals = np.asarray(vals)
    yerr = np.vstack([vals - np.asarray(lows), np.asarray(highs) - vals])
    x = np.arange(len(vals))
    ax.bar(x, vals, color=[r[3] for r in rows], width=0.62)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0.38, 1.02)
    ax.set_ylabel("AUC: MCS+ vs VS")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_title("D. Public DoC pilot anchor", loc="left", fontweight="bold")
    ax.text(0.02, 0.96, "Mendeley EEG/PSG, alpha band, n=38 endpoint",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for i, p in enumerate(pvals):
        ax.text(i, vals[i] + yerr[1, i] + 0.02, f"perm p={p:.3f}",
                ha="center", va="bottom", fontsize=8)
    style_axes(ax)


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    panel_source(axes[0, 0])
    panel_transfer(axes[0, 1])
    panel_sleep(axes[1, 0])
    panel_doc(axes[1, 1])
    fig.suptitle("GCC evidence matrix for NoC submission", fontsize=16, fontweight="bold")
    png = OUT / "gcc_noc_evidence_summary.png"
    pdf = OUT / "gcc_noc_evidence_summary.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
