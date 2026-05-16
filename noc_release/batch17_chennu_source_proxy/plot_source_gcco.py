from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.outdir / "source_gcco_subject_condition_means.csv")
    stats = pd.read_csv(args.outdir / "source_gcco_paired_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), dpi=180, sharey=True)
    for ax, band in zip(axes, ["alpha", "gamma"], strict=True):
        sub = df[df["band"] == band].pivot(index="subject", columns="condition", values="Pi").dropna()
        x0 = np.ones(len(sub))
        x1 = np.ones(len(sub)) * 2
        for _, row in sub.iterrows():
            ax.plot([1, 2], [row["baseline"], row["moderate"]], color="0.72", linewidth=1.0, zorder=1)
        ax.scatter(x0, sub["baseline"], s=26, color="#164a7a", label="baseline", zorder=2)
        ax.scatter(x1, sub["moderate"], s=26, color="#b24a3b", label="moderate", zorder=2)
        st = stats[stats["band"] == band].iloc[0]
        ax.set_title(f"{band}: dz={float(st['paired_dz']):.2f}, p={float(st['wilcoxon_greater_p']):.3g}")
        ax.set_xticks([1, 2], ["baseline", "moderate"])
        ax.set_xlim(0.6, 2.4)
        ax.set_ylim(0.45, 0.9)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    axes[0].set_ylabel("source-space GCC-O access Pi")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Chennu fsaverage source-space robustness", y=1.03)
    fig.tight_layout()
    fig.savefig(args.outdir / "source_gcco_chennu_paired_pi.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
