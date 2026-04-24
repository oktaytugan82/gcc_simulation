"""
GCC Pilot — Observables R, D_eff, M_tau computed on EEG phase matrices.

These are structurally identical to the functions in gcc_simulator.py from
Part A, but adapted for:
  - Real EEG sampling rates (250-1000 Hz, not simulation dt=0.01)
  - Time windows specified in seconds, converted to sample counts
  - Potentially longer recordings (minutes of data, not seconds)

Paper equations:
  R(t)     = Eq. (4)  — magnitude of mean complex exponential of phases
  D_eff(t) = Eq. (8)  — participation ratio of windowed covariance matrix
  M_tau(t) = Eq. (9)  — variance of R in a window
"""

import numpy as np


def order_parameter(phases):
    """
    R(t) per Eq. (4). phases: (n_ch, n_samples). Returns: (n_samples,).
    """
    z = np.exp(1j * phases)
    return np.abs(z.mean(axis=0))


def effective_dimensionality(phases, window_samples, ridge_lambda=1e-3,
                              eps_floor=1e-6, stride=None):
    """
    D_eff^lambda per Eq. (8) with ridge regularization per Eq. (6).

    Parameters
    ----------
    phases : (n_ch, n_samples)
    window_samples : int
    ridge_lambda : float
    eps_floor : float
    stride : int or None
        If given, compute D_eff at every `stride` samples (sparse grid) to
        speed up on long recordings. Returns values at those grid points.

    Returns
    -------
    If stride is None: (n_samples,) trajectory, filled by forward-fill from
    centered windows where valid.
    If stride given: dict with keys 'times_idx' (indices where computed) and
    'values' (D_eff at those indices).
    """
    X = np.cos(phases)  # (n_ch, n_samples) - centered activity proxy
    n_ch, n_t = X.shape
    half = window_samples // 2

    if stride is None:
        # Compute at every sample — expensive but complete
        D = np.zeros(n_t)
        for t in range(n_t):
            lo = max(0, t - half)
            hi = min(n_t, t + half)
            if hi - lo < 2:
                D[t] = 1.0
                continue
            Xw = X[:, lo:hi] - X[:, lo:hi].mean(axis=1, keepdims=True)
            C = (Xw @ Xw.T) / max(1, Xw.shape[1] - 1)
            eps = max(ridge_lambda * np.trace(C) / n_ch, eps_floor)
            Cl = C + eps * np.eye(n_ch)
            tr = np.trace(Cl)
            tr2 = np.trace(Cl @ Cl)
            D[t] = (tr ** 2) / tr2 if tr2 > 0 else 1.0
        return D
    else:
        # Sparse grid
        times_idx = np.arange(half, n_t - half, stride)
        vals = np.zeros(len(times_idx))
        for i, t in enumerate(times_idx):
            lo, hi = t - half, t + half
            Xw = X[:, lo:hi] - X[:, lo:hi].mean(axis=1, keepdims=True)
            C = (Xw @ Xw.T) / max(1, Xw.shape[1] - 1)
            eps = max(ridge_lambda * np.trace(C) / n_ch, eps_floor)
            Cl = C + eps * np.eye(n_ch)
            tr = np.trace(Cl)
            tr2 = np.trace(Cl @ Cl)
            vals[i] = (tr ** 2) / tr2 if tr2 > 0 else 1.0
        return {'times_idx': times_idx, 'values': vals}


def metastability(R_series, window_samples):
    """
    M_tau(t) per Eq. (9): windowed variance of R.
    """
    n_t = len(R_series)
    half = window_samples // 2
    M = np.zeros(n_t)
    for t in range(n_t):
        lo = max(0, t - half)
        hi = min(n_t, t + half)
        M[t] = np.var(R_series[lo:hi]) if hi - lo >= 2 else 0.0
    return M


def compute_observables(phases, fs, tau_D_s=0.2, tau_M_s=0.5,
                         ridge_lambda=1e-3, stride_s=0.05):
    """
    Main entry point: compute R, D_eff, M_tau on an EEG phase matrix.

    Parameters
    ----------
    phases : (n_ch, n_samples)
    fs : float — sampling rate in Hz
    tau_D_s : float — window for D_eff in seconds
    tau_M_s : float — window for M_tau in seconds
    ridge_lambda : float
    stride_s : float — sparse grid stride in seconds for D_eff

    Returns
    -------
    dict with keys:
        't'       : (n_samples,) time axis in seconds
        'R'       : (n_samples,) order parameter
        'D_times' : (n_grid,) times at which D_eff was computed (sparse grid)
        'D'       : (n_grid,) D_eff values at those times
        'M'       : (n_samples,) metastability
    """
    n_ch, n_samples = phases.shape
    tau_D_samples = int(tau_D_s * fs)
    tau_M_samples = int(tau_M_s * fs)
    stride_samples = max(1, int(stride_s * fs))

    t = np.arange(n_samples) / fs

    R = order_parameter(phases)
    D_sparse = effective_dimensionality(
        phases, tau_D_samples, ridge_lambda=ridge_lambda, stride=stride_samples)
    M = metastability(R, tau_M_samples)

    return {
        't': t,
        'R': R,
        'D_times': D_sparse['times_idx'] / fs,
        'D': D_sparse['values'],
        'M': M,
    }


def access_region_indicator(obs, bounds):
    """
    Compute Pi(t) — fraction of time in access region per Eq. (10).

    bounds: dict with keys R_min, D_min, D_max, M_min, M_max.

    Returns a dict with sample-wise indicator arrays and summary.
    """
    # R and M are dense; D is sparse. Interpolate D to dense grid.
    D_dense = np.interp(obs['t'], obs['D_times'], obs['D'])

    in_R = obs['R'] > bounds['R_min']
    in_D = (D_dense > bounds['D_min']) & (D_dense < bounds['D_max'])
    in_M = (obs['M'] > bounds.get('M_min', 0.0)) & (obs['M'] < bounds['M_max'])

    in_all = in_R & in_D & in_M
    return {
        'access': in_all,
        'in_R': in_R, 'in_D': in_D, 'in_M': in_M,
        'fraction': float(in_all.mean()),
        'D_dense': D_dense,
    }


def calibrate_from_baseline(obs, alpha=0.1):
    """
    Quantile-based calibration from a 'baseline' segment of observables,
    per Eq. (13). The caller provides `obs` already restricted to the
    baseline time window.
    """
    # M may have near-zero values; we set M_min = 0 as in Part A
    R, D, M = obs['R'], obs['D'], obs['M']
    bounds = {
        'R_min': float(np.quantile(R, alpha)),
        'D_min': float(np.quantile(D, alpha)),
        'D_max': float(np.quantile(D, 1 - alpha)),
        'M_min': 0.0,  # as in Part A; M_min degenerates to ~0
        'M_max': float(np.quantile(M, 1 - alpha)),
    }
    return bounds


if __name__ == "__main__":
    print("Observables module loaded.")
    print("  order_parameter(phases) -> R(t)")
    print("  effective_dimensionality(phases, window_samples) -> D_eff(t)")
    print("  metastability(R_series, window_samples) -> M_tau(t)")
    print("  compute_observables(phases, fs) -> dict")
    print("  access_region_indicator(obs, bounds) -> dict")
    print("  calibrate_from_baseline(obs, alpha) -> bounds dict")
