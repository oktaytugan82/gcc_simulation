"""
V5 Test (revised): Gain-mediated re-entry through differential degradation.

Paper mechanism (§9.2): "Terminal lucidity corresponds ... to a short-term
functional concentration of remaining efficacy on a residual backbone
structure."

Design:
  - Start healthy (all g_i = 1)
  - Phase 2: global decline — all g_i decrease in parallel
  - Phase 3: DIFFERENTIAL decline — non-S nodes collapse to zero,
    S nodes retain moderate g_i. This models competing subnetworks
    failing first, backbone briefly surviving.
  - Phase 4: S also collapses.

The central test: during phase 3, does S re-enter the access region
after the full system has lost it?

Crucially, we compare two conditions:
  (A) UNIFORM decline: all nodes decline together (no selective preservation)
  (B) DIFFERENTIAL decline: non-S collapses first, S preserved
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from gcc_simulator import (build_network, apply_random_lesion,
                            observables, order_parameter,
                            effective_dimensionality, metastability)
from calibrate import in_access_region


def _step(theta, W, g, K, G, omega, sigma, phi, dt, rng):
    diff = theta[np.newaxis, :] - theta[:, np.newaxis] - phi
    coupling = (W * np.sin(diff)).sum(axis=1)
    drift = omega + K * G * g * coupling
    noise = sigma * np.sqrt(dt) * rng.standard_normal(len(theta))
    return theta + drift * dt + noise


def simulate_timevar_g(W, g_schedule, K, G=1.0, sigma=0.1,
                        dt=0.01, seed=0, burn_in_steps=500):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    omega = rng.normal(0.0, 0.5, size=N)
    phi = np.zeros_like(W)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    for _ in range(burn_in_steps):
        theta = _step(theta, W, g_schedule[0], K, G, omega, sigma, phi, dt, rng)
    n = len(g_schedule)
    hist = np.zeros((n, N))
    for t in range(n):
        theta = _step(theta, W, g_schedule[t], K, G, omega, sigma, phi, dt, rng)
        hist[t] = theta
    return hist


def observables_on_subset(theta_hist, subset_mask, tau_D_steps, tau_M_steps,
                           ridge_lambda=1e-3):
    sub = theta_hist[:, subset_mask]
    if sub.shape[1] < 2:
        return None, None, None
    R = order_parameter(sub)
    D = effective_dimensionality(sub, tau_D_steps, ridge_lambda=ridge_lambda)
    M = metastability(R, tau_M_steps)
    return R, D, M


if __name__ == "__main__":
    N = 100
    K = 1.6       # in access regime for healthy network
    sigma = 0.1
    dt = 0.01

    # Build healthy network
    A_raw, W = build_network(N=N, mean_degree=10, seed=0)
    print(f"Network: N={N}, K={K}, sigma={sigma}")

    # Choose backbone S: a random subset (could be "hub-rich" — here random for simplicity)
    rng = np.random.default_rng(7)
    S_size = 30
    S_indices = rng.choice(N, size=S_size, replace=False)
    S = np.zeros(N, dtype=bool)
    S[S_indices] = True
    print(f"Residual backbone S: {S.sum()} randomly chosen nodes")

    # Time axes
    T_total = 20.0
    n_steps = int(T_total / dt)
    t_grid = np.arange(n_steps) * dt

    p1 = (t_grid < 4.0)                        # healthy
    p2 = (t_grid >= 4.0) & (t_grid < 9.0)      # global decline
    p3 = (t_grid >= 9.0) & (t_grid < 14.0)     # re-entry window
    p4 = (t_grid >= 14.0)                      # final collapse

    # Build g(t) schedules for both conditions
    # Condition B (differential): everyone declines 1 -> 0.5 in P2,
    #   then in P3: non-S drops 0.5 -> 0.1 quickly, S retained at 0.8
    #   in P4: S also drops 0.8 -> 0.1
    # Condition A (uniform): everyone declines together -
    #   in P2: 1 -> 0.5, in P3: 0.5 -> 0.3, in P4: 0.3 -> 0.1
    g_A = np.ones((n_steps, N))
    g_B = np.ones((n_steps, N))

    for t_idx, t in enumerate(t_grid):
        if p1[t_idx]:
            g_A[t_idx] = 1.0
            g_B[t_idx] = 1.0
        elif p2[t_idx]:
            # linear decline from 1 to 0.5
            alpha = (t - 4.0) / 5.0
            val = 1.0 - 0.5 * alpha
            g_A[t_idx] = val
            g_B[t_idx] = val
        elif p3[t_idx]:
            # Condition A: uniform decline 0.5 -> 0.3
            alpha = (t - 9.0) / 5.0
            g_A[t_idx] = 0.5 - 0.2 * alpha
            # Condition B: non-S drops fast (0.5 -> 0.1), S retained high
            non_S_val = 0.5 - 0.4 * alpha
            S_val = 0.5 + 0.3 * alpha   # S rises from 0.5 to 0.8
            row = np.full(N, non_S_val)
            row[S] = S_val
            g_B[t_idx] = row
        else:  # p4
            alpha = (t - 14.0) / 6.0
            g_A[t_idx] = 0.3 - 0.2 * alpha
            # Condition B: S drops 0.8 -> 0.1, non-S stays at 0.1
            S_val = 0.8 - 0.7 * alpha
            non_S_val = 0.1
            row = np.full(N, non_S_val)
            row[S] = S_val
            g_B[t_idx] = row

    # Clip to [0, 1]
    g_A = np.clip(g_A, 0, 1)
    g_B = np.clip(g_B, 0, 1)

    # Simulate
    print("Simulating condition A (uniform decline) ...")
    theta_A = simulate_timevar_g(W, g_A, K, G=1.0, sigma=sigma,
                                  dt=dt, seed=100)
    print("Simulating condition B (differential decline, S preserved) ...")
    theta_B = simulate_timevar_g(W, g_B, K, G=1.0, sigma=sigma,
                                  dt=dt, seed=100)

    # Observables
    tau_D, tau_M = 100, 200
    bounds = pickle.load(open("calibration.pkl", "rb"))["bounds"]
    print("Bounds:", {k: f"{v:.3f}" for k, v in bounds.items()})

    R_A_full, D_A_full, M_A_full = observables(theta_A, tau_D, tau_M)
    R_B_full, D_B_full, M_B_full = observables(theta_B, tau_D, tau_M)
    R_A_S, D_A_S, M_A_S = observables_on_subset(theta_A, S, tau_D, tau_M)
    R_B_S, D_B_S, M_B_S = observables_on_subset(theta_B, S, tau_D, tau_M)

    acc_A_full = in_access_region(R_A_full, D_A_full, M_A_full, bounds)
    acc_B_full = in_access_region(R_B_full, D_B_full, M_B_full, bounds)
    acc_A_S = in_access_region(R_A_S, D_A_S, M_A_S, bounds)
    acc_B_S = in_access_region(R_B_S, D_B_S, M_B_S, bounds)

    def mean_per_phase(arr, m):
        return float(arr[m].mean())

    print("\n=== V5 detailed phase-wise results ===")
    print(f"{'Phase':<20} {'R_A_full':>10} {'R_B_full':>10} "
          f"{'R_A_S':>10} {'R_B_S':>10}")
    for lbl, m in [("P1 healthy", p1), ("P2 decline", p2),
                   ("P3 re-entry", p3), ("P4 final", p4)]:
        print(f"{lbl:<20} "
              f"{mean_per_phase(R_A_full, m):>10.3f} "
              f"{mean_per_phase(R_B_full, m):>10.3f} "
              f"{mean_per_phase(R_A_S, m):>10.3f} "
              f"{mean_per_phase(R_B_S, m):>10.3f}")

    print(f"\n=== V5 access-region fractions Pi(t) ===")
    print(f"{'Phase':<20} {'A full':>10} {'B full':>10} {'A on S':>10} {'B on S':>10}")
    for lbl, m in [("P1 healthy", p1), ("P2 decline", p2),
                   ("P3 re-entry", p3), ("P4 final", p4)]:
        print(f"{lbl:<20} "
              f"{mean_per_phase(acc_A_full.astype(int), m):>10.3f} "
              f"{mean_per_phase(acc_B_full.astype(int), m):>10.3f} "
              f"{mean_per_phase(acc_A_S.astype(int), m):>10.3f} "
              f"{mean_per_phase(acc_B_S.astype(int), m):>10.3f}")

    # Save
    out = dict(t_grid=t_grid, p1=p1, p2=p2, p3=p3, p4=p4,
               g_A_mean=g_A.mean(axis=1), g_B_mean=g_B.mean(axis=1),
               g_B_S=g_B[:, S].mean(axis=1), g_B_nonS=g_B[:, ~S].mean(axis=1),
               R_A_full=R_A_full, R_B_full=R_B_full,
               R_A_S=R_A_S, R_B_S=R_B_S,
               acc_A_full=acc_A_full, acc_B_full=acc_B_full,
               acc_A_S=acc_A_S, acc_B_S=acc_B_S,
               bounds=bounds, S=S)
    with open("v5_results_v2.pkl", "wb") as f:
        pickle.dump(out, f)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    # Panel 1: g(t) schedules
    axes[0].plot(t_grid, g_A.mean(axis=1), 'b-',
                 label='A uniform: mean $g_i$')
    axes[0].plot(t_grid, g_B[:, S].mean(axis=1), 'r-',
                 label='B diff: mean $g_i$ on S (backbone)')
    axes[0].plot(t_grid, g_B[:, ~S].mean(axis=1), 'r--',
                 label='B diff: mean $g_i$ outside S')
    axes[0].set_ylabel('Preservation $g_i(t)$')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title('V5: Differential degradation produces transient re-entry on backbone')

    # Panel 2: R(t) full system
    axes[1].plot(t_grid, R_A_full, 'b-', alpha=0.7, label='A uniform: R full')
    axes[1].plot(t_grid, R_B_full, 'r-', alpha=0.7, label='B diff: R full')
    axes[1].axhline(bounds['R_min'], color='green', linestyle=':',
                    alpha=0.7, label='$R_{min}$')
    axes[1].set_ylabel(r'$R(t)$ full system')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: R(t) on S
    axes[2].plot(t_grid, R_A_S, 'b-', alpha=0.7, label='A uniform: R on S')
    axes[2].plot(t_grid, R_B_S, 'r-', alpha=0.9, label='B diff: R on S (preserved)')
    axes[2].axhline(bounds['R_min'], color='green', linestyle=':',
                    alpha=0.7, label='$R_{min}$')
    axes[2].set_ylabel(r'$R(t)$ on $S$')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    # Panel 4: access membership
    axes[3].fill_between(t_grid, 0, acc_A_S.astype(int), alpha=0.4,
                         color='blue', step='pre', label='A on S in access')
    axes[3].fill_between(t_grid, 0, -acc_B_S.astype(int), alpha=0.6,
                         color='red', step='pre', label='B on S in access')
    axes[3].set_ylabel('Access on S\n(A: +, B: −)')
    axes[3].set_xlabel('Time')
    axes[3].set_yticks([-1, 0, 1])
    axes[3].set_yticklabels(['B', '', 'A'])
    axes[3].legend(fontsize=9)
    axes[3].grid(alpha=0.3)

    # Add phase shading
    for ax in axes:
        ax.axvspan(0, 4, alpha=0.08, color='green')
        ax.axvspan(4, 9, alpha=0.08, color='orange')
        ax.axvspan(9, 14, alpha=0.15, color='red')
        ax.axvspan(14, 20, alpha=0.12, color='black')
    axes[0].text(2, 1.05, 'P1: healthy', ha='center', fontsize=9)
    axes[0].text(6.5, 1.05, 'P2: decline', ha='center', fontsize=9)
    axes[0].text(11.5, 1.05, 'P3: re-entry window', ha='center', fontsize=9,
                 fontweight='bold', color='darkred')
    axes[0].text(17, 1.05, 'P4: final', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig_v5.png', dpi=140, bbox_inches='tight')
    print("\nSaved fig_v5.png")
