"""
GCC-R Simulator — Numerical integration of the phase-oscillator SDE (Eq. 1 of paper).

Implements:
  - Row-normalized residual topology W_ij
  - Global gain G(t)
  - Population-specific preservation g_i(t)
  - Noise sigma_i
  - Phase delays phi_ij

Observables: R (global coherence), D_eff (participation ratio of windowed covariance),
M_tau (temporal variability of R).
"""

import numpy as np
import networkx as nx


# -------------- Network generation --------------

def build_network(N=100, mean_degree=10, seed=0):
    """Watts-Strogatz small-world network, row-normalized to W."""
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(N, k=mean_degree, p=0.1, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    # Store raw A for later m_ref
    A_raw = A.copy()
    # Row-normalize to W
    row_sums = A.sum(axis=1)
    W = np.zeros_like(A)
    for i in range(N):
        if row_sums[i] > 0:
            W[i] = A[i] / row_sums[i]
    return A_raw, W


def apply_random_lesion(A_raw, lesion_fraction, seed=0):
    """
    Generic random edge-removal lesion.
    Returns new (A_lesioned, W_lesioned, g_i vector).
    g_i = m_i(t) / m_i_ref preserves loss of afferent efficacy per Eq. (3).
    """
    rng = np.random.default_rng(seed)
    N = A_raw.shape[0]
    m_ref = A_raw.sum(axis=1).copy()

    if lesion_fraction <= 0:
        A_new = A_raw.copy()
    else:
        # remove fraction of edges uniformly at random
        edges = np.argwhere(A_raw > 0)
        n_remove = int(lesion_fraction * len(edges))
        if n_remove > 0:
            idx = rng.choice(len(edges), size=n_remove, replace=False)
            A_new = A_raw.copy()
            for e_idx in idx:
                i, j = edges[e_idx]
                A_new[i, j] = 0.0
        else:
            A_new = A_raw.copy()

    # Row-normalize survivor
    m_curr = A_new.sum(axis=1)
    W_new = np.zeros_like(A_new)
    for i in range(N):
        if m_curr[i] > 0:
            W_new[i] = A_new[i] / m_curr[i]

    # g_i = m_i / m_ref, in [0,1]; zero where m_ref was zero (isolates)
    g = np.zeros(N)
    for i in range(N):
        if m_ref[i] > 0:
            g[i] = min(m_curr[i] / m_ref[i], 1.0)
    return A_new, W_new, g


# -------------- SDE integration --------------

def simulate(W, g, K, G=1.0, omega=None, sigma=0.1, phi=None,
             T=60.0, dt=0.01, seed=0, burn_in_steps=500):
    """
    Integrate Eq. (1) via Euler-Maruyama.
    Returns: theta_history (T_steps, N) after burn-in.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    if omega is None:
        omega = rng.normal(0.0, 0.5, size=N)
    if phi is None:
        phi = np.zeros_like(W)

    n_steps = int(T / dt)
    theta = rng.uniform(0, 2*np.pi, size=N)

    # Burn-in
    for _ in range(burn_in_steps):
        theta = _euler_step(theta, W, g, K, G, omega, sigma, phi, dt, rng)

    hist = np.zeros((n_steps, N))
    for t in range(n_steps):
        theta = _euler_step(theta, W, g, K, G, omega, sigma, phi, dt, rng)
        hist[t] = theta
    return hist, dt


def _euler_step(theta, W, g, K, G, omega, sigma, phi, dt, rng):
    N = len(theta)
    # Pair differences: theta_j - theta_i - phi_ij
    diff = theta[np.newaxis, :] - theta[:, np.newaxis] - phi
    coupling = (W * np.sin(diff)).sum(axis=1)  # sum over j
    drift = omega + K * G * g * coupling
    noise = sigma * np.sqrt(dt) * rng.standard_normal(N)
    return theta + drift * dt + noise


# -------------- Observables --------------

def order_parameter(theta_hist):
    """R(t) per Eq. (4)."""
    N = theta_hist.shape[1]
    z = np.exp(1j * theta_hist)
    return np.abs(z.mean(axis=1))


def effective_dimensionality(theta_hist, window_steps, ridge_lambda=1e-3, eps_floor=1e-6):
    """
    D_eff^lambda(t) per Eq. (8). Returns trajectory over windowed times.
    Uses cos(theta) as centered activity; ridge regularization as per Eq. (6).
    """
    X = np.cos(theta_hist)  # (T, N)
    T_len, N = X.shape
    half = window_steps // 2
    D_eff = np.zeros(T_len)
    for t in range(T_len):
        lo = max(0, t - half)
        hi = min(T_len, t + half)
        if hi - lo < 2:
            D_eff[t] = 1.0
            continue
        Xw = X[lo:hi] - X[lo:hi].mean(axis=0, keepdims=True)
        C = (Xw.T @ Xw) / max(1, Xw.shape[0] - 1)
        eps = max(ridge_lambda * np.trace(C) / N, eps_floor)
        Cl = C + eps * np.eye(N)
        tr = np.trace(Cl)
        tr2 = np.trace(Cl @ Cl)
        if tr2 > 0:
            D_eff[t] = (tr ** 2) / tr2
        else:
            D_eff[t] = 1.0
    return D_eff


def metastability(R_series, window_steps):
    """M_tau(t) per Eq. (9): variance of R in a window."""
    T_len = len(R_series)
    half = window_steps // 2
    M = np.zeros(T_len)
    for t in range(T_len):
        lo = max(0, t - half)
        hi = min(T_len, t + half)
        if hi - lo < 2:
            M[t] = 0.0
        else:
            M[t] = np.var(R_series[lo:hi])
    return M


def observables(theta_hist, tau_D_steps, tau_M_steps, ridge_lambda=1e-3):
    """Compute R, D_eff, M_tau trajectories for the full theta_hist."""
    R = order_parameter(theta_hist)
    D = effective_dimensionality(theta_hist, tau_D_steps, ridge_lambda=ridge_lambda)
    M = metastability(R, tau_M_steps)
    return R, D, M


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    A_raw, W = build_network(N=100, mean_degree=10, seed=0)
    g = np.ones(100)
    print("Network built. Mean row sum W:", W.sum(axis=1).mean())
    theta, dt = simulate(W, g, K=1.5, T=10.0, dt=0.01, seed=0)
    R, D, M = observables(theta, tau_D_steps=50, tau_M_steps=100)
    print(f"R mean: {R.mean():.3f}, D_eff mean: {D.mean():.1f}, M_tau mean: {M.mean():.4f}")
