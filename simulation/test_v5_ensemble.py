"""
V5 Ensemble Test: Gain-mediated re-entry, N_seeds-averaged statistics.

Re-runs the V5 test with multiple seeds to obtain proper variance
estimates, confidence intervals, and a paired test on the A-vs-B contrast.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from gcc_simulator import build_network, observables
from test_v5_v2 import simulate_timevar_g, observables_on_subset
from calibrate import in_access_region

N_SEEDS = 20


if __name__ == "__main__":
    N, K, sigma, dt = 100, 1.6, 0.1, 0.01
    T_total = 20.0
    n_steps = int(T_total / dt)
    t_grid = np.arange(n_steps) * dt

    p1 = (t_grid < 4.0)
    p2 = (t_grid >= 4.0) & (t_grid < 9.0)
    p3 = (t_grid >= 9.0) & (t_grid < 14.0)
    p4 = (t_grid >= 14.0)

    # Backbone S (same as original)
    rng_S = np.random.default_rng(7)
    S_size = 30
    S_indices = rng_S.choice(N, size=S_size, replace=False)
    S = np.zeros(N, dtype=bool); S[S_indices] = True

    # g-schedules
    g_A = np.ones((n_steps, N))
    g_B = np.ones((n_steps, N))
    for t_idx, t in enumerate(t_grid):
        if p1[t_idx]:
            pass
        elif p2[t_idx]:
            alpha = (t - 4.0) / 5.0
            val = 1.0 - 0.5 * alpha
            g_A[t_idx] = val; g_B[t_idx] = val
        elif p3[t_idx]:
            alpha = (t - 9.0) / 5.0
            g_A[t_idx] = 0.5 - 0.2 * alpha
            non_S_val = 0.5 - 0.4 * alpha
            S_val = 0.5 + 0.3 * alpha
            row = np.full(N, non_S_val); row[S] = S_val
            g_B[t_idx] = row
        else:
            alpha = (t - 14.0) / 6.0
            g_A[t_idx] = 0.3 - 0.2 * alpha
            S_val = 0.8 - 0.7 * alpha
            row = np.full(N, 0.1); row[S] = S_val
            g_B[t_idx] = row
    g_A = np.clip(g_A, 0, 1); g_B = np.clip(g_B, 0, 1)

    tau_D, tau_M = 100, 200
    bounds = pickle.load(open("../results/calibration.pkl", "rb"))["bounds"]

    phases = [("P1 healthy", p1), ("P2 decline", p2),
              ("P3 re-entry", p3), ("P4 final", p4)]

    # Per-seed storage
    pi_A_full = {ph[0]: [] for ph in phases}
    pi_B_full = {ph[0]: [] for ph in phases}
    pi_A_S = {ph[0]: [] for ph in phases}
    pi_B_S = {ph[0]: [] for ph in phases}

    for seed in range(N_SEEDS):
        A_raw, W = build_network(N=N, mean_degree=10, seed=seed)
        theta_A = simulate_timevar_g(W, g_A, K, G=1.0, sigma=sigma,
                                       dt=dt, seed=seed + 1000)
        theta_B = simulate_timevar_g(W, g_B, K, G=1.0, sigma=sigma,
                                       dt=dt, seed=seed + 1000)

        R_A_full, D_A_full, M_A_full = observables(theta_A, tau_D, tau_M)
        R_B_full, D_B_full, M_B_full = observables(theta_B, tau_D, tau_M)
        R_A_S, D_A_S, M_A_S = observables_on_subset(theta_A, S, tau_D, tau_M)
        R_B_S, D_B_S, M_B_S = observables_on_subset(theta_B, S, tau_D, tau_M)

        acc_A_full = in_access_region(R_A_full, D_A_full, M_A_full, bounds)
        acc_B_full = in_access_region(R_B_full, D_B_full, M_B_full, bounds)
        acc_A_S = in_access_region(R_A_S, D_A_S, M_A_S, bounds)
        acc_B_S = in_access_region(R_B_S, D_B_S, M_B_S, bounds)

        for ph_name, m in phases:
            pi_A_full[ph_name].append(acc_A_full[m].mean())
            pi_B_full[ph_name].append(acc_B_full[m].mean())
            pi_A_S[ph_name].append(acc_A_S[m].mean())
            pi_B_S[ph_name].append(acc_B_S[m].mean())

        p3_A_S = acc_A_S[p3].mean()
        p3_B_S = acc_B_S[p3].mean()
        print(f"  seed {seed+1}/{N_SEEDS}: P3 Pi_S A={p3_A_S:.3f}, B={p3_B_S:.3f}, "
              f"diff={p3_B_S - p3_A_S:+.3f}")

    # Convert to arrays
    for d in [pi_A_full, pi_B_full, pi_A_S, pi_B_S]:
        for k in d: d[k] = np.array(d[k])

    print("\n=== V5 Ensemble Results ===")
    print(f"{'Phase':<16} {'Pi A full':>18} {'Pi B full':>18} {'Pi A on S':>18} {'Pi B on S':>18}")
    print("-" * 92)
    summary = {}
    for ph_name, _ in phases:
        aA = pi_A_full[ph_name]; aB = pi_B_full[ph_name]
        sA = pi_A_S[ph_name]; sB = pi_B_S[ph_name]
        summary[ph_name] = {
            'pi_A_full_mean': aA.mean(), 'pi_A_full_sd': aA.std(),
            'pi_B_full_mean': aB.mean(), 'pi_B_full_sd': aB.std(),
            'pi_A_S_mean': sA.mean(), 'pi_A_S_sd': sA.std(),
            'pi_B_S_mean': sB.mean(), 'pi_B_S_sd': sB.std(),
        }
        print(f"{ph_name:<16} "
              f"{aA.mean():>7.3f} ± {aA.std():<7.3f} "
              f"{aB.mean():>7.3f} ± {aB.std():<7.3f} "
              f"{sA.mean():>7.3f} ± {sA.std():<7.3f} "
              f"{sB.mean():>7.3f} ± {sB.std():<7.3f}")

    # Central test: P3 Pi_S B > P3 Pi_S A (paired across seeds)
    p3_A = pi_A_S["P3 re-entry"]
    p3_B = pi_B_S["P3 re-entry"]
    diff = p3_B - p3_A
    print("\n=== Central test: P3 re-entry on backbone S ===")
    print(f"N_seeds: {N_SEEDS}")
    print(f"Pi_S^A (P3): mean = {p3_A.mean():.4f}, SD = {p3_A.std():.4f}")
    print(f"Pi_S^B (P3): mean = {p3_B.mean():.4f}, SD = {p3_B.std():.4f}")
    print(f"Difference (B - A): mean = {diff.mean():+.4f}, SD = {diff.std():.4f}")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(p3_B, p3_A)
    w_stat, w_p = stats.wilcoxon(p3_B, p3_A)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {t_p:.4g}")
    print(f"Wilcoxon: W = {w_stat:.1f}, p = {w_p:.4g}")

    # Bootstrap CI on relative increase
    np.random.seed(42)
    boot_rel = []
    for _ in range(10000):
        idx = np.random.choice(N_SEEDS, size=N_SEEDS, replace=True)
        a_samp = p3_A[idx].mean(); b_samp = p3_B[idx].mean()
        if a_samp > 0.001:
            boot_rel.append((b_samp - a_samp) / a_samp * 100)
    boot_rel = np.array(boot_rel)
    rel_ci = (np.quantile(boot_rel, 0.025), np.quantile(boot_rel, 0.975))
    rel_point = (p3_B.mean() - p3_A.mean()) / p3_A.mean() * 100 if p3_A.mean() > 0 else np.nan
    print(f"Relative increase: {rel_point:+.1f}%  (bootstrap 95% CI: "
          f"[{rel_ci[0]:+.1f}%, {rel_ci[1]:+.1f}%])")

    # Cohen's d
    d_cohen = diff.mean() / diff.std() if diff.std() > 0 else np.nan
    print(f"Cohen's d (paired): {d_cohen:.3f}")

    out = {
        'n_seeds': N_SEEDS,
        'pi_A_full': pi_A_full, 'pi_B_full': pi_B_full,
        'pi_A_S': pi_A_S, 'pi_B_S': pi_B_S,
        'summary': summary,
        'p3_stats': {
            't_stat': t_stat, 't_p': t_p,
            'w_stat': w_stat, 'w_p': w_p,
            'rel_increase_pct': rel_point,
            'rel_ci_pct': rel_ci,
            'cohen_d': d_cohen,
            'mean_diff': diff.mean(), 'sd_diff': diff.std(),
        },
        'bounds': bounds,
    }
    with open("../results/v5_ensemble_results.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved v5_ensemble_results.pkl")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: bar plot per phase, Pi on S
    phase_labels = [ph[0] for ph in phases]
    means_A = [pi_A_S[p].mean() for p in phase_labels]
    means_B = [pi_B_S[p].mean() for p in phase_labels]
    sds_A = [pi_A_S[p].std() for p in phase_labels]
    sds_B = [pi_B_S[p].std() for p in phase_labels]

    x = np.arange(len(phase_labels))
    w = 0.35
    axes[0].bar(x - w/2, means_A, w, yerr=sds_A, capsize=5,
                 label='A (uniform)', color='steelblue', alpha=0.8)
    axes[0].bar(x + w/2, means_B, w, yerr=sds_B, capsize=5,
                 label='B (selective)', color='darkorange', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phase_labels, rotation=20, ha='right')
    axes[0].set_ylabel(r'$\Pi$ on backbone $S$')
    axes[0].set_title(f'V5 ensemble (N={N_SEEDS}): per-phase access on $S$')
    axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

    # Panel 2: paired dots for P3
    axes[1].plot([0, 1], [p3_A, p3_B], 'k-', alpha=0.3, lw=0.5)
    axes[1].scatter([0]*N_SEEDS, p3_A, color='steelblue', s=40,
                     alpha=0.7, label='A (uniform)')
    axes[1].scatter([1]*N_SEEDS, p3_B, color='darkorange', s=40,
                     alpha=0.7, label='B (selective)')
    axes[1].hlines([p3_A.mean()], -0.15, 0.15, colors='steelblue', lw=3)
    axes[1].hlines([p3_B.mean()], 0.85, 1.15, colors='darkorange', lw=3)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['A', 'B'])
    axes[1].set_ylabel(r'$\Pi_S$ in P3 (re-entry)')
    axes[1].set_title(f'Paired comparison (per seed)\n'
                       f'p = {w_p:.3g}, relative increase: {rel_point:+.1f}% '
                       f'[{rel_ci[0]:+.0f}, {rel_ci[1]:+.0f}]')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_v5_ensemble.png', dpi=140, bbox_inches='tight')
    print("Saved fig_v5_ensemble.png")
