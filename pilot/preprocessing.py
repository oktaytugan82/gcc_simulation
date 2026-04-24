"""
GCC Pilot — Preprocessing and phase extraction for EEG data.

Pipeline:
  1. Band-pass filter (default 1-45 Hz for broadband, or 30-45 Hz for gamma only)
  2. Notch filter at 50 Hz (European power line) or 60 Hz (US)
  3. Re-reference to average reference
  4. Hilbert transform on gamma-band (30-45 Hz) to extract analytic phase
  5. Return phase matrix (n_channels x n_samples) — structurally identical to
     the theta matrix from Part A

The analytic phase from Hilbert transform is a standard proxy for
oscillator phase in the EEG literature (Lachaux et al. 1999,
Stam et al. 2005). It is NOT identical to the model's theoretical
oscillator phase, but it is the conventionally accepted operationalization.
"""

import numpy as np
import mne
from scipy.signal import hilbert


def preprocess_raw(raw, l_freq=1.0, h_freq=45.0, notch_freq=50.0,
                    resample_to=None, rereference=True, verbose=False):
    """
    Standard preprocessing: band-pass, notch, optional resample, rereference.
    """
    raw = raw.copy()
    # Pick only EEG channels (exclude EOG, ECG, etc. if present)
    raw.pick('eeg')
    # Band-pass
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               fir_design='firwin', verbose='ERROR')
    # Notch
    if notch_freq is not None:
        raw.notch_filter(freqs=notch_freq, verbose='ERROR')
    # Resample if requested
    if resample_to is not None and raw.info['sfreq'] != resample_to:
        raw.resample(resample_to, verbose='ERROR')
    # Average reference
    if rereference:
        raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    if verbose:
        print(f"  Preprocessed: {len(raw.ch_names)} channels, "
              f"{raw.info['sfreq']} Hz, "
              f"{raw.times[-1]:.1f}s, filter [{l_freq}, {h_freq}] Hz")
    return raw


def extract_gamma_phases(raw, gamma_low=30.0, gamma_high=45.0):
    """
    Extract gamma-band analytic phase via Hilbert transform.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG (broadband).
    gamma_low, gamma_high : float
        Gamma band limits in Hz.

    Returns
    -------
    phases : ndarray (n_channels, n_samples)
        Analytic phase in radians, wrapped to (-pi, pi].
    fs : float
        Sampling rate in Hz.
    ch_names : list
        Channel names corresponding to rows of phases.
    """
    # Band-pass into gamma
    raw_gamma = raw.copy().filter(l_freq=gamma_low, h_freq=gamma_high,
                                    fir_design='firwin', verbose='ERROR')
    data = raw_gamma.get_data()  # (n_ch, n_samples)
    # Hilbert transform along time axis
    analytic = hilbert(data, axis=1)
    phases = np.angle(analytic)
    return phases, raw.info['sfreq'], raw.ch_names


def segment_by_events(raw, tmin=0.0, tmax=None, event_id=None):
    """
    If the dataset has event markers, return segmented data per event code.

    Returns dict of {event_label: (data, times)}.
    """
    events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
    if len(events) == 0:
        return None
    segments = {}
    for label, code in event_dict.items():
        ep = mne.Epochs(raw, events, event_id={label: code},
                        tmin=tmin, tmax=tmax if tmax else 10.0,
                        baseline=None, preload=True, verbose='ERROR')
        segments[label] = ep
    return segments


if __name__ == "__main__":
    # Sanity check with MNE sample data if available; else just verify imports
    print("Preprocessing module loaded OK.")
    print("  preprocess_raw(): band-pass + notch + re-reference")
    print("  extract_gamma_phases(): Hilbert on 30-45 Hz band -> phase matrix")
    print("  segment_by_events(): optional epoching if event markers present")
