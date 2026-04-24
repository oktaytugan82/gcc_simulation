"""
V4 Ensemble Test: Degradation shifts and narrows the access region.

Replaces the old 2-seed sweep with a proper 20-seed ensemble per lesion
fraction to obtain robust estimates of K_c^op and max_K Pi(K) with
confidence intervals.

This addresses Codex moderate finding #4: the non-monotonic max Pi values
in Table 1 (0.84 -> 0.97 -> 0.80 -> 0.47) were artifacts of the limited
seed count.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from gcc_simulator import build_network, simulate, observables, apply_random_lesion
from calibrate import in_access_region

N_SEEDS = 10


def sweep_K_lesioned_ensemble(K_values, lesion_fraction, n_seeds=N_SEEDS,
                               N=100, T=8.0, dt=0.01, sigma=0.1,
                               tau_D_steps=100, tau_M_steps=200):
    """Returns arrays of shape (n_K, n_seeds) for R_mean, Pi per K×seed."""
    n_K = len(K_values)
    R_means = np.zeros((n_K, n_seeds))
    D_means = np.zeros((n_K, n_seeds))
    Pi_vals = np.zeros((n_K, n_seeds))
    for ki, K in enumerate(K_values):
        for si in range(n_seeds):
            A_raw, _ = build_network(N=N, mean_degree=10, seed=si)
            A_new, W, g = apply_random_lesion(A_raw, lesion_fraction,
                                                seed=si + 50)
            theta, _ = simulate(W, g, K=K, G=1.0, sigma=sigma,
                                T=T, dt=dt, seed=si + 2000)
            R, D, M = observables(theta, tau_D_steps, tau_M_steps)
            skip = tau_D_steps
            R_k, D_k, M_k = R[skip:-skip], D[skip:-skip], M[skip:-skip]
            R_means[ki, si] = R_k.mean()
            D_means[ki, si] = D_k.mean()
            bounds = pickle.load(open("../results/calibration.pkl", "rb"))["bounds"]
            acc = in_access_region(R_k, D_k, M_k, bounds)
            Pi_vals[ki, si] = acc.mean()
        print(f"  K={K:.2f}, f={lesion_fraction}: "
              f"<R>={R_means[ki].mean():.3f}, <Pi>={Pi_vals[ki].mean():.3f} ± {Pi_vals[ki].std():.3f}")
    return R_means, D_means, Pi_vals


def find_kcop(K_values, R_means_per_seed, R_min):
    """Find K_c^op (first K where R exceeds R_min) per seed, via dR/dK midpoint."""
    n_seeds = R_means_per_seed.shape[1]
    kcops = np.zeros(n_seeds)
    for si in range(n_seeds):
        R_seed = R_means_per_seed[:, si]
        dR = np.diff(R_seed)
        dK = np.diff(K_values)
        slope = dR / dK
        # K_c^op: K at which R crosses R_min for the first time
        above = R_seed > R_min
        if not above.any():
            kcops[si] = np.nan
            continue
        # first True
        first_idx = np.argmax(above)
        if first_idx == 0:
            kcops[si] = K_values[0]
        else:
            # linear interpolation between K[first_idx-1] and K[first_idx]
            R1, R2 = R_seed[first_idx-1], R_seed[first_idx]
            K1, K2 = K_values[first_idx-1], K_values[first_idx]
            if R2 == R1:
                kcops[si] = K1
            else:
                kcops[si] = K1 + (R_min - R1) * (K2 - K1) / (R2 - R1)
    return kcops


if __name__ == "__main__":
    K_values = np.linspace(0.4, 2.8, 9)
    lesion_fractions = [0.0, 0.15, 0.30, 0.45]
    bounds = pickle.load(open("../results/calibration.pkl", "rb"))["bounds"]
    R_min = bounds['R_min']

    print(f"Running V2/V3/V4 ensemble sweep: {N_SEEDS} seeds x "
          f"{len(lesion_fractions)} lesion levels x {len(K_values)} K values")
    print(f"R_min = {R_min:.3f}\n")

    all_R_means = {}  # lesion -> (n_K, n_seeds)
    all_Pi = {}
    all_kcop = {}
    all_max_Pi = {}

    for lf in lesion_fractions:
        print(f"\n=== Lesion fraction f = {lf} ===")
        R_means, D_means, Pi_vals = sweep_K_lesioned_ensemble(K_values, lf)
        all_R_means[lf] = R_means
        all_Pi[lf] = Pi_vals
        # K_c^op per seed
        kcops = find_kcop(K_values, R_means, R_min)
        all_kcop[lf] = kcops
        # max_K Pi per seed
        max_pi = Pi_vals.max(axis=0)
        all_max_Pi[lf] = max_pi

    print("\n\n=== Ensemble Summary ===")
    print(f"{'Lesion f':>10} {'<K_c^op>':>12} {'95% CI':>22} "
          f"{'<max Pi>':>12} {'95% CI':>22}")
    print("-" * 82)
    summary = {}
    for lf in lesion_fractions:
        kcops = all_kcop[lf]
        max_pi = all_max_Pi[lf]
        # Remove NaNs
        kcops_clean = kcops[~np.isnan(kcops)]
        kcop_mean = kcops_clean.mean()
        kcop_ci = (np.quantile(kcops_clean, 0.025), np.quantile(kcops_clean, 0.975))
        maxpi_mean = max_pi.mean()
        maxpi_ci = (np.quantile(max_pi, 0.025), np.quantile(max_pi, 0.975))
        print(f"{lf:>10.2f} {kcop_mean:>12.3f} "
              f"[{kcop_ci[0]:.3f}, {kcop_ci[1]:.3f}]    "
              f"{maxpi_mean:>12.3f} [{maxpi_ci[0]:.3f}, {maxpi_ci[1]:.3f}]")
        summary[lf] = {
            'kcop_mean': kcop_mean, 'kcop_sd': kcops_clean.std(),
            'kcop_ci95': kcop_ci, 'kcop_per_seed': kcops,
            'max_pi_mean': maxpi_mean, 'max_pi_sd': max_pi.std(),
            'max_pi_ci95': maxpi_ci, 'max_pi_per_seed': max_pi,
        }

    # Test monotonicity via paired comparisons
    print("\n=== Monotonicity tests (paired across seeds) ===")
    from scipy import stats
    lf_pairs = [(0.00, 0.15), (0.15, 0.30), (0.30, 0.45), (0.00, 0.45)]
    for lf1, lf2 in lf_pairs:
        maxpi1 = all_max_Pi[lf1]
        maxpi2 = all_max_Pi[lf2]
        diff = maxpi1 - maxpi2  # positive = f1 gives higher max Pi (no deg is better)
        t, p = stats.wilcoxon(maxpi1, maxpi2)
        print(f"  max_Pi({lf1}) vs max_Pi({lf2}): "
              f"mean diff = {diff.mean():+.3f}, Wilcoxon p = {p:.4g}")
        kcop1 = all_kcop[lf1]; kcop2 = all_kcop[lf2]
        mask = ~np.isnan(kcop1) & ~np.isnan(kcop2)
        if mask.sum() > 5:
            t2, p2 = stats.wilcoxon(kcop1[mask], kcop2[mask])
            print(f"  K_cop({lf1}) vs K_cop({lf2}):   "
                  f"mean diff = {(kcop2[mask] - kcop1[mask]).mean():+.3f}, "
                  f"Wilcoxon p = {p2:.4g}")

    # Save
    out = {
        'K_values': K_values,
        'lesion_fractions': lesion_fractions,
        'all_R_means': all_R_means,
        'all_Pi': all_Pi,
        'all_kcop': all_kcop,
        'all_max_Pi': all_max_Pi,
        'summary': summary,
        'n_seeds': N_SEEDS,
        'R_min': R_min,
    }
    with open("../results/v2_v3_ensemble_results.pkl", "wb") as f:
        pickle.dump(out, f)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['black', 'tab:blue', 'tab:orange', 'tab:red']
    labels = [f'f = {lf}' for lf in lesion_fractions]

    # Panel 1: R vs K, ensemble mean ± SD per lesion fraction
    for lf, col, lbl in zip(lesion_fractions, colors, labels):
        R_ens = all_R_means[lf]
        axes[0].plot(K_values, R_ens.mean(axis=1), '-o', color=col, label=lbl)
        axes[0].fill_between(K_values,
                              R_ens.mean(axis=1) - R_ens.std(axis=1),
                              R_ens.mean(axis=1) + R_ens.std(axis=1),
                              alpha=0.2, color=col)
    axes[0].axhline(R_min, color='green', linestyle=':', label='$R_{min}$')
    axes[0].set_xlabel('$K$'); axes[0].set_ylabel(r'$\langle R \rangle$')
    axes[0].set_title(f'Ensemble R(K) per lesion level (N={N_SEEDS} seeds)')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Panel 2: K_c^op per lesion fraction with error bars
    lfs = np.array(lesion_fractions)
    kcop_means = [summary[lf]['kcop_mean'] for lf in lesion_fractions]
    kcop_sds = [summary[lf]['kcop_sd'] for lf in lesion_fractions]
    axes[1].errorbar(lfs, kcop_means, yerr=kcop_sds, fmt='o-', capsize=5,
                       color='tab:blue', markersize=8)
    axes[1].set_xlabel('Lesion fraction $f$')
    axes[1].set_ylabel(r'$\langle K_c^{op}\rangle \pm $ SD')
    axes[1].set_title(r'$K_c^{op}$ shift (via R-crossing)')
    axes[1].grid(alpha=0.3)

    # Panel 3: max Pi per lesion fraction with error bars
    maxpi_means = [summary[lf]['max_pi_mean'] for lf in lesion_fractions]
    maxpi_sds = [summary[lf]['max_pi_sd'] for lf in lesion_fractions]
    axes[2].errorbar(lfs, maxpi_means, yerr=maxpi_sds, fmt='o-', capsize=5,
                       color='tab:red', markersize=8)
    axes[2].set_xlabel('Lesion fraction $f$')
    axes[2].set_ylabel(r'$\langle \max_K \Pi(K) \rangle \pm $ SD')
    axes[2].set_title(r'Access region shrinks under degradation')
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('../figures/fig_v2_v3_ensemble.png', dpi=140, bbox_inches='tight')
    print("\nSaved fig_v2_v3_ensemble.png and v2_v3_ensemble_results.pkl")
