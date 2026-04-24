"""
V4 Ensemble Test: Anesthesia trajectory averaged over N_seeds runs.

Replaces single-seed v4 test with a proper ensemble to resolve the
issue where a single stochastic run can lie outside the quantile-based
access region purely due to initialization fluctuations in M_tau.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from gcc_simulator import build_network, observables
from calibrate import in_access_region

N_SEEDS = 20


def simulate_Gt(W, g_const, K, G_schedule, sigma=0.1, dt=0.01, seed=0,
                burn_in_steps=500):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    omega = rng.normal(0.0, 0.5, size=N)
    phi = np.zeros_like(W)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    for _ in range(burn_in_steps):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - phi
        coupling = (W * np.sin(diff)).sum(axis=1)
        drift = omega + K * 1.0 * g_const * coupling
        noise = sigma * np.sqrt(dt) * rng.standard_normal(N)
        theta = theta + drift * dt + noise
    n = len(G_schedule)
    hist = np.zeros((n, N))
    for t in range(n):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - phi
        coupling = (W * np.sin(diff)).sum(axis=1)
        drift = omega + K * G_schedule[t] * g_const * coupling
        noise = sigma * np.sqrt(dt) * rng.standard_normal(N)
        theta = theta + drift * dt + noise
        hist[t] = theta
    return hist


if __name__ == "__main__":
    N, K, sigma, dt = 100, 1.6, 0.1, 0.01
    T_total = 24.0
    n_steps = int(T_total / dt)
    t_grid = np.arange(n_steps) * dt

    # Same G schedule as v4
    G_schedule = np.ones(n_steps)
    for i, t in enumerate(t_grid):
        if t < 3.0:
            G_schedule[i] = 1.0
        elif t < 9.0:
            alpha = (t - 3.0) / 6.0
            G_schedule[i] = 1.0 - 0.8 * (0.5 - 0.5 * np.cos(np.pi * alpha))
        elif t < 13.0:
            G_schedule[i] = 0.2
        elif t < 19.0:
            alpha = (t - 13.0) / 6.0
            G_schedule[i] = 0.2 + 0.8 * (0.5 - 0.5 * np.cos(np.pi * alpha))
        else:
            G_schedule[i] = 1.0

    g_const = np.ones(N)
    tau_D, tau_M = 100, 200
    bounds = pickle.load(open("../results/calibration.pkl", "rb"))["bounds"]

    phases_mask = {
        "baseline (t<3)": t_grid < 3.0,
        "induction (3-9)": (t_grid >= 3.0) & (t_grid < 9.0),
        "deep (9-13)": (t_grid >= 9.0) & (t_grid < 13.0),
        "emergence (13-19)": (t_grid >= 13.0) & (t_grid < 19.0),
        "recovery (t>=19)": t_grid >= 19.0,
    }

    # Run ensemble
    per_seed_pi = {ph: [] for ph in phases_mask}
    per_seed_R = {ph: [] for ph in phases_mask}
    per_seed_D = {ph: [] for ph in phases_mask}
    per_seed_M = {ph: [] for ph in phases_mask}
    all_R, all_D, all_M, all_acc = [], [], [], []

    for seed in range(N_SEEDS):
        A_raw, W = build_network(N=N, mean_degree=10, seed=seed)
        theta = simulate_Gt(W, g_const, K, G_schedule, sigma=sigma,
                            dt=dt, seed=seed + 1000)
        R, D, M = observables(theta, tau_D, tau_M)
        acc = in_access_region(R, D, M, bounds)

        all_R.append(R); all_D.append(D); all_M.append(M); all_acc.append(acc)

        for ph, m in phases_mask.items():
            per_seed_pi[ph].append(acc[m].mean())
            per_seed_R[ph].append(R[m].mean())
            per_seed_D[ph].append(D[m].mean())
            per_seed_M[ph].append(M[m].mean())

        print(f"  seed {seed+1}/{N_SEEDS}: overall Pi = {acc.mean():.3f}")

    all_R = np.array(all_R); all_D = np.array(all_D)
    all_M = np.array(all_M); all_acc = np.array(all_acc)

    print("\n=== V4 Ensemble Results ===")
    print(f"{'Phase':<24} {'<R>':>8} {'<D>':>8} {'<M>':>12} {'<Pi>':>8} {'SD(Pi)':>8}")
    print("-" * 72)
    summary = {}
    for ph in phases_mask:
        pis = np.array(per_seed_pi[ph])
        Rs = np.array(per_seed_R[ph])
        Ds = np.array(per_seed_D[ph])
        Ms = np.array(per_seed_M[ph])
        summary[ph] = {
            'R_mean': Rs.mean(), 'R_sd': Rs.std(),
            'D_mean': Ds.mean(), 'D_sd': Ds.std(),
            'M_mean': Ms.mean(), 'M_sd': Ms.std(),
            'Pi_mean': pis.mean(), 'Pi_sd': pis.std(),
            'Pi_ci95': (np.quantile(pis, 0.025), np.quantile(pis, 0.975)),
            'n_seeds': len(pis),
        }
        print(f"{ph:<24} {Rs.mean():>8.3f} {Ds.mean():>8.2f} "
              f"{Ms.mean():>12.4e} {pis.mean():>8.3f} {pis.std():>8.3f}")

    # Save
    out = {
        't_grid': t_grid, 'G_schedule': G_schedule,
        'all_R': all_R, 'all_D': all_D, 'all_M': all_M,
        'all_acc': all_acc, 'bounds': bounds,
        'phases': phases_mask, 'summary': summary,
        'n_seeds': N_SEEDS,
    }
    with open("../results/v4_ensemble_results.pkl", "wb") as f:
        pickle.dump(out, f)

    # Ensemble-averaged trajectory plot
    R_mean = all_R.mean(axis=0); R_sd = all_R.std(axis=0)
    D_mean = all_D.mean(axis=0); D_sd = all_D.std(axis=0)
    acc_mean = all_acc.astype(float).mean(axis=0)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_grid, G_schedule, 'k-', lw=2)
    ax1.set_ylabel('$G(t)$'); ax1.set_ylim(0, 1.1)
    ax1.set_title(f'V4 Ensemble (N={N_SEEDS}): Anesthesia trajectory')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.fill_between(t_grid, R_mean - R_sd, R_mean + R_sd, alpha=0.3, color='blue')
    ax2.plot(t_grid, R_mean, 'b-', lw=1.5, label=r'$\langle R(t)\rangle \pm $ SD')
    ax2.axhline(bounds['R_min'], color='green', linestyle=':', alpha=0.6,
                label='$R_{min}$')
    ax2.set_ylabel('$R(t)$'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.fill_between(t_grid, 0, acc_mean, alpha=0.5, color='green', step='pre')
    ax3.set_xlabel('Time'); ax3.set_ylabel(r'$\langle\Pi(t)\rangle$')
    ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[:, 1])
    sc = ax4.scatter(R_mean, D_mean, c=t_grid, cmap='viridis', s=3, alpha=0.6)
    ax4.axvline(bounds['R_min'], color='green', linestyle=':', alpha=0.7,
                label='$R_{min}$')
    ax4.axhline(bounds['D_min'], color='orange', linestyle=':', alpha=0.7,
                label='$D_{min}$')
    ax4.axhline(bounds['D_max'], color='red', linestyle=':', alpha=0.7,
                label='$D_{max}$')
    ax4.axvspan(bounds['R_min'], 1.0, alpha=0.1, color='green')
    ax4.axhspan(bounds['D_min'], bounds['D_max'], alpha=0.1, color='green')
    ax4.set_xlabel('$R(t)$'); ax4.set_ylabel('$D_{eff}(t)$')
    ax4.set_title(f'Ensemble-averaged trajectory (N={N_SEEDS})')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax4, label='Time')

    plt.tight_layout()
    plt.savefig('../figures/fig_v4_ensemble.png', dpi=140, bbox_inches='tight')
    print("\nSaved fig_v4_ensemble.png")
    print("Saved v4_ensemble_results.pkl")
