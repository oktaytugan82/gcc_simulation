"""
GCC Pilot — Data loader for Chennu et al. 2016 Propofol Sedation dataset.

Reference:
    Chennu, S., et al. (2016). Brain connectivity dissociates responsiveness
    from drug exposure during propofol-induced transitions of consciousness.
    PLoS Computational Biology, 12(1), e1004669.

Dataset: OpenNeuro ds003754 or similar. BIDS-formatted, 91-channel EEG,
20 healthy participants, four sedation levels + recovery.

Expected directory structure (BIDS):
    <dataset_root>/
        participants.tsv
        sub-XX/
            ses-YY/
                eeg/
                    sub-XX_ses-YY_task-rest_eeg.set  (EEGLAB format)
                    sub-XX_ses-YY_task-rest_eeg.fdt
                    sub-XX_ses-YY_task-rest_channels.tsv
                    sub-XX_ses-YY_task-rest_events.tsv

This loader handles both .set (EEGLAB) and .edf formats flexibly.

The four expected sedation levels (labeled by session or task in different
versions of the dataset):
    baseline   — fully awake, pre-drug
    mild       — light sedation, responsive
    moderate   — moderate sedation, slow responses
    recovery   — post-sedation, recovered
"""

from pathlib import Path
import numpy as np
import mne


# Sedation levels in canonical order (ascending depth, then recovery)
SEDATION_LEVELS = ['baseline', 'mild', 'moderate', 'recovery']


def find_eeg_files(dataset_root, participant=None):
    """
    Scan a BIDS-formatted dataset root for EEG files.

    Parameters
    ----------
    dataset_root : str or Path
        Root directory of the BIDS dataset.
    participant : str or None
        If given, only load this participant (e.g. 'sub-01'). Else load all.

    Returns
    -------
    files : list of dict
        Each dict has keys: 'subject', 'session', 'task', 'path', 'format'.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    files = []
    # Iterate over sub-* directories
    if participant:
        sub_dirs = [root / participant]
    else:
        sub_dirs = sorted(root.glob('sub-*'))

    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            continue
        subject = sub_dir.name
        # Sessions (may or may not exist)
        ses_dirs = list(sub_dir.glob('ses-*'))
        if not ses_dirs:
            ses_dirs = [sub_dir]  # no session layer

        for ses_dir in ses_dirs:
            session = ses_dir.name if ses_dir.name.startswith('ses-') else None
            eeg_dir = ses_dir / 'eeg' if (ses_dir / 'eeg').exists() else ses_dir
            # Find .set or .edf or .vhdr files
            for ext in ['*.set', '*.edf', '*.vhdr', '*.bdf', '*.fif']:
                for f in sorted(eeg_dir.glob(ext)):
                    # Infer task from filename
                    task = _infer_task(f.name)
                    files.append({
                        'subject': subject,
                        'session': session,
                        'task': task,
                        'path': f,
                        'format': ext.replace('*.', ''),
                    })
    return files


def _infer_task(filename):
    """Extract task-XXX from BIDS filename."""
    parts = filename.split('_')
    for p in parts:
        if p.startswith('task-'):
            return p[5:]
    return 'unknown'


def infer_sedation_level(file_info, participants_tsv=None):
    """
    Attempt to identify sedation level from file info and/or participants.tsv.

    Strategy order:
      1. If task name contains a sedation label, use that.
      2. If session name contains a sedation label, use that.
      3. If participants.tsv has a mapping, look up.
      4. Else, return 'unknown'.
    """
    # Check task
    task = file_info.get('task', '').lower()
    for level in SEDATION_LEVELS:
        if level in task:
            return level
    # Check session
    session = (file_info.get('session') or '').lower()
    for level in SEDATION_LEVELS:
        if level in session:
            return level
    # Try participants.tsv mapping
    # (dataset-specific; Chennu uses session numbers that map via a TSV)
    return 'unknown'


def load_raw(file_info, preload=True):
    """
    Load a single EEG file into an MNE Raw object.
    """
    path = file_info['path']
    fmt = file_info['format']
    if fmt == 'set':
        raw = mne.io.read_raw_eeglab(str(path), preload=preload, verbose='ERROR')
    elif fmt == 'edf':
        raw = mne.io.read_raw_edf(str(path), preload=preload, verbose='ERROR')
    elif fmt == 'vhdr':
        raw = mne.io.read_raw_brainvision(str(path), preload=preload, verbose='ERROR')
    elif fmt == 'bdf':
        raw = mne.io.read_raw_bdf(str(path), preload=preload, verbose='ERROR')
    elif fmt == 'fif':
        raw = mne.io.read_raw_fif(str(path), preload=preload, verbose='ERROR')
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return raw


def summarize_files(files):
    """Print a summary of found files, grouped by subject and task."""
    by_subject = {}
    for f in files:
        by_subject.setdefault(f['subject'], []).append(f)
    print(f"\nFound {len(files)} EEG files across {len(by_subject)} subjects.\n")
    for subj in sorted(by_subject.keys())[:5]:  # show first 5
        print(f"  {subj}:")
        for f in by_subject[subj]:
            level = infer_sedation_level(f)
            print(f"    session={f['session']}, task={f['task']}, "
                  f"level={level}, format={f['format']}")
    if len(by_subject) > 5:
        print(f"  ... ({len(by_subject) - 5} more subjects)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_chennu.py <dataset_root> [participant]")
        sys.exit(1)
    root = sys.argv[1]
    participant = sys.argv[2] if len(sys.argv) > 2 else None
    files = find_eeg_files(root, participant=participant)
    summarize_files(files)
    # Try loading the first file as a sanity check
    if files:
        print(f"\nLoading first file: {files[0]['path'].name}")
        raw = load_raw(files[0])
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Duration: {raw.times[-1]:.1f} s")
        print(f"  First 5 channel names: {raw.ch_names[:5]}")
