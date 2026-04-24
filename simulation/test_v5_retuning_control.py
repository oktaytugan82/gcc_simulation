"""
V6 Retuning Control: Test whether the V6 effect survives when the baseline K
is already in the optimal access region (so uniform decline cannot improve Pi).

Codex moderate finding #3: In the original V6 test (K_baseline = 1.6), the
condition-A uniform-decline Pi RISES from P1 (healthy) 0.53 to P2 (decline)
0.69, indicating the system was suboptimally coupled at baseline. The +81%
re-entry effect could partly be a retuning artifact.

This test repeats V6 at K_baseline = 2.5 (well within the healthy access
regime per H_full calibration). If the re-entry effect persists, it cannot
be explained by retuning alone.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from gcc_simulator import build_network, observables
from test_v5_v2 import simulate_timevar_g, observables_on_subset
from calibrate import in_access_region

N_SEEDS = 20


def run_v6_at_K(K_baseline, label_suffix=""):
    N, sigma, dt = 100, 0.1, 0.01
    T_total = 20.0
    n_steps = int(T_total / dt)
    t_grid = np.arange(n_steps) * dt

    p1 = (t_grid < 4.0)
    p2 = (t_grid >= 4.0) & (t_grid < 9.0)
    p3 = (t_grid >= 9.0) & (t_grid < 14.0)
    p4 = (t_grid >= 14.0)

    rng_S = np.random.default_rng(7)
    S_size = 30
    S_indices = rng_S.choice(N, size=S_size, replace=False)
    S = np.zeros(N, dtype=bool); S[S_indices] = True

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

    pi_A_full = {ph[0]: [] for ph in phases}
    pi_B_full = {ph[0]: [] for ph in phases}
    pi_A_S = {ph[0]: [] for ph in phases}
    pi_B_S = {ph[0]: [] for ph in phases}

    for seed in range(N_SEEDS):
        A_raw, W = build_network(N=N, mean_degree=10, seed=seed)
        theta_A = simulate_timevar_g(W, g_A, K_baseline, G=1.0, sigma=sigma,
                                       dt=dt, seed=seed + 1000)
        theta_B = simulate_timevar_g(W, g_B, K_baseline, G=1.0, sigma=sigma,
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

    for d in [pi_A_full, pi_B_full, pi_A_S, pi_B_S]:
        for k in d: d[k] = np.array(d[k])

    print(f"\n=== V6 Ensemble at K_baseline = {K_baseline}{label_suffix} ===")
    print(f"{'Phase':<16} {'Pi A full':>14} {'Pi B full':>14} "
          f"{'Pi A on S':>14} {'Pi B on S':>14}")
    for ph_name, _ in phases:
        aA = pi_A_full[ph_name]; aB = pi_B_full[ph_name]
        sA = pi_A_S[ph_name]; sB = pi_B_S[ph_name]
        print(f"{ph_name:<16} {aA.mean():>7.3f}±{aA.std():<5.3f} "
              f"{aB.mean():>7.3f}±{aB.std():<5.3f} "
              f"{sA.mean():>7.3f}±{sA.std():<5.3f} "
              f"{sB.mean():>7.3f}±{sB.std():<5.3f}")

    # Central test
    p3_A = pi_A_S["P3 re-entry"]
    p3_B = pi_B_S["P3 re-entry"]
    diff = p3_B - p3_A
    print(f"\nP3 re-entry on S: A={p3_A.mean():.3f}±{p3_A.std():.3f}, "
          f"B={p3_B.mean():.3f}±{p3_B.std():.3f}")
    print(f"Difference (B-A): {diff.mean():+.3f} ± {diff.std():.3f}")
    try:
        t_stat, t_p = stats.ttest_rel(p3_B, p3_A)
        w_stat, w_p = stats.wilcoxon(p3_B, p3_A)
        print(f"paired t: t={t_stat:.2f}, p={t_p:.4g}")
        print(f"Wilcoxon: W={w_stat:.1f}, p={w_p:.4g}")
    except Exception as e:
        print(f"Test failed: {e}")
        t_stat=t_p=w_stat=w_p=np.nan
    if p3_A.mean() > 0.001:
        rel = (p3_B.mean() - p3_A.mean()) / p3_A.mean() * 100
    else:
        rel = np.inf
    print(f"Relative increase: {rel:+.1f}%")
    d_cohen = diff.mean()/diff.std() if diff.std()>0 else np.nan
    print(f"Cohen's d (paired): {d_cohen:.3f}")

    # Also check the retuning effect: P1 -> P2 Pi increase in condition A (full)
    retuning = pi_A_full["P2 decline"].mean() - pi_A_full["P1 healthy"].mean()
    print(f"\nRetuning effect (P2 Pi_full - P1 Pi_full in condition A): {retuning:+.3f}")

    return {
        'K_baseline': K_baseline,
        'pi_A_full': pi_A_full, 'pi_B_full': pi_B_full,
        'pi_A_S': pi_A_S, 'pi_B_S': pi_B_S,
        'p3_diff': diff,
        't_stat': t_stat, 't_p': t_p, 'w_stat': w_stat, 'w_p': w_p,
        'rel_pct': rel, 'cohen_d': d_cohen,
        'retuning_effect': retuning,
    }


if __name__ == "__main__":
    # Test at multiple K values to map the retuning-vs-reentry relationship
    results = {}
    for K in [1.6, 2.0, 2.5, 3.0]:
        results[K] = run_v6_at_K(K)

    # Summary table
    print("\n\n=== Summary across K_baseline values ===")
    print(f"{'K':>5} {'Retuning':>10} {'<Pi_S^A>':>12} {'<Pi_S^B>':>12} "
          f"{'Rel incr':>10} {'p-value':>10}")
    for K in sorted(results.keys()):
        r = results[K]
        print(f"{K:>5.1f} {r['retuning_effect']:>+10.3f} "
              f"{r['pi_A_S']['P3 re-entry'].mean():>12.3f} "
              f"{r['pi_B_S']['P3 re-entry'].mean():>12.3f} "
              f"{r['rel_pct']:>+9.1f}% "
              f"{r['w_p']:>10.4g}")

    with open("../results/v5_retuning_control.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved v5_retuning_control.pkl")

    # Plot
    Ks = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: retuning vs re-entry effect
    retunings = [results[K]['retuning_effect'] for K in Ks]
    diffs = [results[K]['p3_diff'].mean() for K in Ks]
    diff_sds = [results[K]['p3_diff'].std() for K in Ks]
    axes[0].plot(Ks, retunings, 'o-', color='gray', label='Retuning (P2 $-$ P1 in A)',
                   markersize=10, lw=2)
    axes[0].errorbar(Ks, diffs, yerr=diff_sds, fmt='D-', color='tab:red',
                       label='Re-entry effect (B $-$ A in P3)', markersize=10,
                       capsize=5, lw=2)
    axes[0].axhline(0, color='black', lw=0.5)
    axes[0].set_xlabel('$K_{baseline}$')
    axes[0].set_ylabel(r'$\Delta \Pi$')
    axes[0].set_title('Retuning vs.\\ re-entry effect across $K_{baseline}$')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Right: P1 vs P2 Pi in both conditions per K
    p1s = [results[K]['pi_A_full']['P1 healthy'].mean() for K in Ks]
    p2s = [results[K]['pi_A_full']['P2 decline'].mean() for K in Ks]
    p1_sds = [results[K]['pi_A_full']['P1 healthy'].std() for K in Ks]
    p2_sds = [results[K]['pi_A_full']['P2 decline'].std() for K in Ks]
    axes[1].errorbar(Ks, p1s, yerr=p1_sds, fmt='o-', color='tab:blue',
                       label='P1 (healthy)', markersize=10, capsize=5, lw=2)
    axes[1].errorbar(Ks, p2s, yerr=p2_sds, fmt='s-', color='tab:orange',
                       label='P2 (uniform decline)', markersize=10, capsize=5, lw=2)
    axes[1].set_xlabel('$K_{baseline}$')
    axes[1].set_ylabel(r'$\langle \Pi \rangle$ (full system)')
    axes[1].set_title('Baseline vs.\\ decline phase $\\Pi$ across $K$')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_v5_retuning_control.png', dpi=140, bbox_inches='tight')
    print("Saved fig_v5_retuning_control.png")
