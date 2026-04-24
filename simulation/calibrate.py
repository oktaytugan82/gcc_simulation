"""
Calibrate regime boundaries R_min, D_min, D_max, M_min, M_max from healthy
reference ensemble H_full via the quantile method (Eq. 13 of paper).

Generates healthy systems across a range of K and aggregates observables.
"""

import numpy as np
from gcc_simulator import build_network, simulate, observables


def generate_reference_ensemble(
    N=100, mean_degree=10,
    K_values=(1.2, 1.4, 1.6, 1.8),
    sigma=0.1,
    T=40.0, dt=0.01,
    tau_D_steps=100, tau_M_steps=200,
    n_seeds=3,
):
    """
    Generate H_full: healthy (g_i = 1 for all i) networks across K values.
    Returns dict of observable arrays concatenated across the ensemble.
    """
    all_R, all_D, all_M = [], [], []
    for K in K_values:
        for seed in range(n_seeds):
            A_raw, W = build_network(N=N, mean_degree=mean_degree, seed=seed)
            g = np.ones(N)
            theta, _ = simulate(W, g, K=K, G=1.0, sigma=sigma,
                                T=T, dt=dt, seed=seed + 100)
            R, D, M = observables(theta, tau_D_steps, tau_M_steps)
            # Drop first window-length to avoid edge effects
            skip = max(tau_D_steps, tau_M_steps)
            all_R.append(R[skip:-skip])
            all_D.append(D[skip:-skip])
            all_M.append(M[skip:-skip])
    return {
        "R": np.concatenate(all_R),
        "D": np.concatenate(all_D),
        "M": np.concatenate(all_M),
    }


def calibrate_bounds(ensemble, alpha=0.1):
    """Quantile-based calibration per Eq. (13)."""
    R, D, M = ensemble["R"], ensemble["D"], ensemble["M"]
    bounds = {
        "R_min": float(np.quantile(R, alpha)),
        "D_min": float(np.quantile(D, alpha)),
        "D_max": float(np.quantile(D, 1 - alpha)),
        "M_min": float(np.quantile(M, alpha)),
        "M_max": float(np.quantile(M, 1 - alpha)),
    }
    return bounds


def in_access_region(R, D, M, bounds):
    """Indicator per Eq. (10)."""
    return ((R > bounds["R_min"]) &
            (D > bounds["D_min"]) & (D < bounds["D_max"]) &
            (M > bounds["M_min"]) & (M < bounds["M_max"]))


def pi_index(R, D, M, bounds):
    """Fraction of time inside access region."""
    return float(in_access_region(R, D, M, bounds).mean())


if __name__ == "__main__":
    print("Generating reference ensemble H_full ...")
    ensemble = generate_reference_ensemble(n_seeds=3, T=30.0)
    print(f"Ensemble size: {len(ensemble['R'])} samples")
    print(f"R:  mean={ensemble['R'].mean():.3f}, "
          f"q10={np.quantile(ensemble['R'], 0.1):.3f}, "
          f"q90={np.quantile(ensemble['R'], 0.9):.3f}")
    print(f"D:  mean={ensemble['D'].mean():.2f}, "
          f"q10={np.quantile(ensemble['D'], 0.1):.2f}, "
          f"q90={np.quantile(ensemble['D'], 0.9):.2f}")
    print(f"M:  mean={ensemble['M'].mean():.5f}, "
          f"q10={np.quantile(ensemble['M'], 0.1):.5f}, "
          f"q90={np.quantile(ensemble['M'], 0.9):.5f}")
    bounds = calibrate_bounds(ensemble, alpha=0.1)
    print("Calibrated bounds:")
    for k, v in bounds.items():
        print(f"  {k}: {v:.4f}")

    import pickle
    with open("../results/calibration.pkl", "wb") as f:
        pickle.dump({"ensemble": ensemble, "bounds": bounds}, f)
    print("Saved calibration.pkl")
