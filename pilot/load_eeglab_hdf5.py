"""
Load EEGLAB .set files that are stored in MATLAB v7.3 (HDF5) format.

Standard MNE readers use scipy.io.loadmat, which fails on v7.3 files with
the error "Please use HDF reader for matlab v7.3 files". This module
provides a fallback that uses h5py to open the file and then constructs
a standard MNE Raw object from the extracted arrays.

This is adapted from the approach used in eeglabio and similar toolkits,
specialized for the needs of this pilot analysis.
"""

import numpy as np
from pathlib import Path
import mne


def _h5_to_value(obj):
    """Dereference h5py objects — HDF5 stores MATLAB strings/cells as refs."""
    import h5py
    if isinstance(obj, h5py.Reference):
        return None  # caller must resolve
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj


def _h5_read_string(f, dataset):
    """Read a MATLAB char array stored as uint16 into a Python string."""
    arr = np.array(dataset).flatten()
    if arr.dtype.kind in ('i', 'u'):
        # MATLAB chars are uint16 code points
        try:
            return ''.join(chr(int(c)) for c in arr if c > 0)
        except (ValueError, OverflowError):
            return ''
    return ''


def _resolve_cell_array(f, cell_dataset):
    """MATLAB cell arrays appear as arrays of HDF5 references. Resolve each."""
    results = []
    refs = np.array(cell_dataset).flatten()
    for ref in refs:
        try:
            sub = f[ref]
            results.append(_h5_read_string(f, sub))
        except Exception:
            results.append('')
    return results


def load_eeglab_v73(set_path, preload=True):
    """
    Load a MATLAB v7.3 .set file as MNE Raw.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file. The corresponding .fdt file (if used) must
        be in the same directory.
    preload : bool
        Load all data into memory (required for filtering).

    Returns
    -------
    raw : mne.io.RawArray
    """
    import h5py

    set_path = Path(set_path)

    with h5py.File(str(set_path), 'r') as f:
        # EEGLAB stores the dataset struct under 'EEG'
        if 'EEG' not in f:
            raise ValueError(f"No 'EEG' group in {set_path.name}")
        EEG = f['EEG']

        # Sampling rate
        srate = float(np.array(EEG['srate']).flatten()[0])

        # Number of channels and samples
        nbchan = int(np.array(EEG['nbchan']).flatten()[0])
        pnts = int(np.array(EEG['pnts']).flatten()[0])
        trials = int(np.array(EEG['trials']).flatten()[0])

        # Data — either stored in the .set file itself under EEG/data, or
        # in a separate .fdt file (32-bit float, column-major)
        data = None
        # Case 1: data is a reference to an external file (string)
        if 'data' in EEG:
            data_field = EEG['data']
            # If it's a numeric array -> inline data
            if data_field.dtype.kind == 'f':
                data = np.array(data_field)
            else:
                # It's a string reference to the .fdt file
                data_fname = _h5_read_string(f, data_field)
                if data_fname and not data_fname.endswith('.set'):
                    fdt_path = set_path.parent / data_fname
                    if not fdt_path.exists():
                        # Try same stem with .fdt
                        fdt_path = set_path.with_suffix('.fdt')
                    if fdt_path.exists():
                        raw_bytes = np.fromfile(str(fdt_path), dtype=np.float32)
                        # EEGLAB uses column-major layout: (nbchan, pnts, trials)
                        # Reshaping as (pnts*trials, nbchan) and transposing gets
                        # us (nbchan, pnts*trials)
                        data = raw_bytes.reshape(pnts * trials, nbchan).T
                    else:
                        raise FileNotFoundError(
                            f"Cannot find .fdt data file for {set_path.name}")

        if data is None:
            raise ValueError(f"Could not extract data from {set_path.name}")

        # Ensure shape (nbchan, n_samples)
        if data.shape[0] != nbchan:
            if data.shape[1] == nbchan:
                data = data.T
            else:
                raise ValueError(
                    f"Data shape {data.shape} does not match nbchan={nbchan}")

        # Channel names — stored as a cell array of strings under chanlocs
        ch_names = []
        if 'chanlocs' in EEG:
            chanlocs = EEG['chanlocs']
            if 'labels' in chanlocs:
                labels_refs = np.array(chanlocs['labels']).flatten()
                for ref in labels_refs:
                    try:
                        ch_names.append(_h5_read_string(f, f[ref]))
                    except Exception:
                        ch_names.append('')
        # Fallback if we couldn't read names
        if not ch_names or len(ch_names) != nbchan or all(n == '' for n in ch_names):
            ch_names = [f'EEG{i:03d}' for i in range(nbchan)]

        # Build MNE Raw
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
        # EEGLAB data is in microvolts; MNE expects volts
        raw = mne.io.RawArray(data * 1e-6, info, verbose='ERROR')

        return raw


def smart_load_set(set_path, preload=True):
    """
    Try standard MNE loader first; on v7.3 failure, fall back to HDF5 loader.
    """
    try:
        return mne.io.read_raw_eeglab(str(set_path), preload=preload,
                                       verbose='ERROR')
    except (TypeError, NotImplementedError, ValueError) as e:
        msg = str(e)
        if 'v7.3' in msg or 'HDF' in msg or 'h5py' in msg:
            return load_eeglab_v73(set_path, preload=preload)
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_eeglab_hdf5.py <path_to_set_file>")
        sys.exit(1)
    raw = smart_load_set(sys.argv[1])
    print(f"Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, "
          f"{raw.times[-1]:.1f} s")
    print(f"First 5 channels: {raw.ch_names[:5]}")
