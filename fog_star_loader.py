"""
fog_star_loader.py
------------------


Agent-to-sensor mapping:
    Node 1 (back):  back_acc_*, back_gyro_*
    Node 2 (wrist): wrist_acc_*, wrist_gyro_*   (most-affected wrist)
    Node 3 (ankle): ankleR_acc_*, ankleR_gyro_*
    ankleL excluded -- has NaN values for some participants, retained as
    held-out validation channel only.

Pre-processing applied (matching authors Analytics notebook):
    - Rows with activity == 0 are removed (unlabelled periods)
    - Rows with NaN in any agent sensor column are removed
    - Data sorted by subjectID then timestamp

Real dwell statistics from Analytics notebook (across all 22 subjects):
    Shuffling: 33 events, mean=3.0s, median=2.4s, max=9.9s
    Trembling: 40 events, mean=5.8s, median=4.9s, max=24.1s
    Akinesia:  37 events, mean=20.0s, median=12.4s, max=108.7s

    Class balance (post activity-filter):
    Non-FoG: 80.1%  |  FoG: 19.9%
    Within FoG: Shuffling=9.3%, Trembling=21.7%, Akinesia=69.0%

States:
    0 = Healthy   (fog == 0)
    1 = Shuffling (fog == 1, fog_severity == 1)
    2 = Trembling (fog == 1, fog_severity == 2)
    3 = Akinesia  (fog == 1, fog_severity == 3)

States:
    0 = Healthy   (fog == 0)
    1 = Shuffling (fog == 1, fog_severity == 1)
    2 = Trembling (fog == 1, fog_severity == 2)
    3 = Akinesia  (fog == 1, fog_severity == 3)
"""

import os; os.environ.setdefault("PYTHONIOENCODING", "utf-8"); import sys; sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── constants ─────────────────────────────────────────────────────────────────

FS            = 60          # sampling rate Hz
WINDOW        = 128         # feature extraction window (≈2.1 s)
BOUNDARY_PAD  = 20          # timesteps to discard either side of a state transition
LOCO_LOW      = 0.5         # locomotor band lower bound Hz
LOCO_HIGH     = 3.0         # locomotor band upper bound Hz
FREEZE_LOW    = 3.0         # freeze band lower bound Hz
FREEZE_HIGH   = 8.0         # freeze band upper bound Hz

# State encoding
STATE_NAMES   = {0: 'Healthy', 1: 'Shuffling', 2: 'Trembling', 3: 'Akinesia'}
N_STATES      = 4
N_NODES       = 3           # back, wrist, ankle (in that order)
N_FEATURES    = 4           # FI, RMS_acc, RMS_gyr, delta_FI
NODE_NAMES    = ['back', 'wrist', 'ankle']
FEATURE_NAMES = ['FI', 'RMS_acc', 'RMS_gyr', 'delta_FI']

# Column names matching the actual FoG-STAR sensor_data.csv
#
# Ankle selection: ankleR is used as Node 3 (primary ankle agent).
# Justification: the Analytics notebook shows ankleL has NaN values for
# some participants (e.g. subjectID=9, taskID=7) while ankleR remains
# valid throughout. ankleL is retained as a held-out validation channel.
NODE_COLUMNS = {
    'back':  {'acc': ['back_acc_x',   'back_acc_y',   'back_acc_z'],
               'gyr': ['back_gyro_x',  'back_gyro_y',  'back_gyro_z']},
    'wrist': {'acc': ['wrist_acc_x',  'wrist_acc_y',  'wrist_acc_z'],
               'gyr': ['wrist_gyro_x', 'wrist_gyro_y', 'wrist_gyro_z']},
    'ankle': {'acc': ['ankleR_acc_x', 'ankleR_acc_y', 'ankleR_acc_z'],
               'gyr': ['ankleR_gyro_x','ankleR_gyro_y','ankleR_gyro_z']},
}

# Left ankle columns -- retained as held-out validation channel only.
# Note: ankleL has NaN values for some participants and tasks.
ANKLE_L_COLUMNS = {
    'acc': ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z'],
    'gyr': ['ankleL_gyro_x','ankleL_gyro_y','ankleL_gyro_z'],
}

# Label and ID column names
COL_SUBJECT   = 'subjectID'
COL_SESSION   = 'sessionID'
COL_TASK      = 'taskID'
COL_ACTIVITY  = 'activity'       # 1-7; rows with activity==0 are unlabelled
COL_FOG       = 'fog'
COL_SEVERITY  = 'fog_severity'
COL_TIMESTAMP = 'timestamp'      # sensor timestamp in ms at 60 Hz


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ParticipantData:
    """Clean feature data for one participant."""
    pid:         str
    states:      np.ndarray   # shape (T,)       integer state label per timestep
    features:    np.ndarray   # shape (T, 3, 4)  features per timestep per node
    valid_mask:  np.ndarray   # shape (T,)       True where features are reliable
    task_ids:    np.ndarray   # shape (T,)       taskID per timestep (1-7)


@dataclass
class DatasetStats:
    """Summary statistics computed after loading."""
    n_participants:    int
    n_timesteps_total: int
    state_counts:      Dict[int, int]   # timesteps per state
    state_durations:   Dict[int, List[float]]  # run durations in seconds per state
    feature_ranges:    Dict[str, Tuple[float, float]]  # min/max per feature


# ── feature extraction ────────────────────────────────────────────────────────

def compute_freeze_index(acc_mag: np.ndarray, window: int = WINDOW,
                          fs: int = FS) -> np.ndarray:
    """
    Compute Freeze Index at each timestep using a sliding window FFT.

    FI(t) = P_freeze(t) / P_loco(t)
    where P_freeze = integral of PSD over [3, 8] Hz
          P_loco   = integral of PSD over [0.5, 3] Hz

    This version is fully vectorised using numpy stride tricks.
    Processes data in batches of 8192 windows to keep memory use below 50 MB
    per call regardless of signal length.

    Parameters
    ----------
    acc_mag : (T,) accelerometer magnitude signal
    window  : sliding window length in samples (default 128 = 2.13 s at 60 Hz)
    fs      : sampling rate in Hz

    Returns
    -------
    fi : (T,) Freeze Index, NaN for the first (window-1) timesteps
    """
    from numpy.lib.stride_tricks import sliding_window_view

    T  = len(acc_mag)
    fi = np.full(T, np.nan)

    if T < window:
        return fi

    freqs       = np.fft.rfftfreq(window, d=1.0 / fs)
    loco_mask   = (freqs >= LOCO_LOW)   & (freqs <  LOCO_HIGH)
    freeze_mask = (freqs >= FREEZE_LOW) & (freqs <= FREEZE_HIGH)
    hann        = np.hanning(window)

    n_windows  = T - window + 1
    fi_vals    = np.empty(n_windows)
    batch_size = 8192   # keeps each batch under ~50 MB RAM

    for start in range(0, n_windows, batch_size):
        end  = min(start + batch_size, n_windows)
        seg  = acc_mag[start : end + window - 1]

        # sliding_window_view returns a VIEW (no copy) of shape (B, window)
        wins    = sliding_window_view(seg, window) * hann
        spectra = np.abs(np.fft.rfft(wins, axis=1)) ** 2

        p_loco   = spectra[:, loco_mask].sum(axis=1)
        p_freeze = spectra[:, freeze_mask].sum(axis=1)

        fi_vals[start:end] = np.where(
            p_loco < 1e-12,
            0.0,
            p_freeze / np.maximum(p_loco, 1e-12)
        )

    fi[window - 1:] = fi_vals
    return fi


def compute_rms(signal: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """
    Compute RMS of a signal magnitude using a sliding window.

    Parameters
    ----------
    signal : (T,) signal magnitude (e.g. ||acc|| or ||gyr||)
    window : sliding window length in samples

    Returns
    -------
    rms : (T,) RMS values, NaN for the first (window-1) timesteps
    """
    T   = len(signal)
    rms = np.full(T, np.nan)

    # Use cumulative sum trick for efficiency
    sq  = signal ** 2
    cs  = np.cumsum(sq)
    # cs[t] - cs[t-window] = sum of sq over window ending at t
    rms[window-1:] = np.sqrt(
        (cs[window-1:] - np.concatenate([[0], cs[:-(window)]]))  / window
    )
    return rms


def extract_features(acc: np.ndarray, gyr: np.ndarray,
                      window: int = WINDOW, fs: int = FS) -> np.ndarray:
    """
    Extract the four features from raw triaxial IMU signals.

    Parameters
    ----------
    acc : (T, 3) triaxial accelerometer signal in g
    gyr : (T, 3) triaxial gyroscope signal in deg/s
    window : sliding window length in samples
    fs     : sampling rate in Hz

    Returns
    -------
    features : (T, 4) array of [FI, RMS_acc, RMS_gyr, delta_FI]
               NaN for the first (window-1) timesteps
    """
    # Compute signal magnitudes
    acc_mag = np.linalg.norm(acc, axis=1)   # (T,)
    gyr_mag = np.linalg.norm(gyr, axis=1)   # (T,)

    fi      = compute_freeze_index(acc_mag, window, fs)   # (T,)
    rms_acc = compute_rms(acc_mag, window)                # (T,)
    rms_gyr = compute_rms(gyr_mag, window)                # (T,)

    # delta_FI: first difference of FI
    # NaN where FI is NaN or at the boundary after NaN ends
    delta_fi        = np.full_like(fi, np.nan)
    valid           = ~np.isnan(fi)
    delta_fi[valid] = np.concatenate([[np.nan], np.diff(fi[valid])])

    features = np.stack([fi, rms_acc, rms_gyr, delta_fi], axis=1)  # (T, 4)
    return features


# ── state labelling ───────────────────────────────────────────────────────────

def derive_states(fog: np.ndarray, fog_severity: np.ndarray) -> np.ndarray:
    """
    Combine fog and fog_severity columns into a single integer state label.

    0 = Healthy   (fog == 0, fog_severity == 0)
    1 = Shuffling (fog == 1, fog_severity == 1)
    2 = Trembling (fog == 1, fog_severity == 2)
    3 = Akinesia  (fog == 1, fog_severity == 3)

    In FoG-STAR, fog_severity == 0 when fog == 0 (healthy rows).
    Timesteps where fog == 1 but fog_severity is not in {1, 2, 3}
    are marked as -1 and excluded from all analysis.
    """
    states   = np.zeros(len(fog), dtype=int)
    fog_mask = fog == 1

    states[fog_mask & (fog_severity == 1)] = 1
    states[fog_mask & (fog_severity == 2)] = 2
    states[fog_mask & (fog_severity == 3)] = 3
    # Invalid: fog active but fog_severity is not a recognised phase
    states[fog_mask & ~np.isin(fog_severity, [1, 2, 3])] = -1

    return states


def compute_boundary_mask(states: np.ndarray,
                           pad: int = BOUNDARY_PAD) -> np.ndarray:
    """
    Build a boolean mask that is False within `pad` timesteps of any
    state transition. These boundary regions are excluded from emission
    model fitting because the signal is mid-transition.

    Parameters
    ----------
    states : (T,) integer state labels
    pad    : number of timesteps to exclude on each side of a transition

    Returns
    -------
    mask : (T,) True where data is reliable (away from boundaries)
    """
    T    = len(states)
    mask = np.ones(T, dtype=bool)

    # Find all transition points
    transitions = np.where(np.diff(states) != 0)[0]  # index just before change

    for t in transitions:
        lo = max(0,   t - pad + 1)
        hi = min(T-1, t + pad)
        mask[lo:hi+1] = False

    # Also mask invalid state labels
    mask[states == -1] = False

    return mask


# ── main loader ───────────────────────────────────────────────────────────────

def load_participant(df_pid: pd.DataFrame) -> ParticipantData:
    """
    Process one participant's data into features and state labels.

    Parameters
    ----------
    df_pid : rows from sensor_data.csv for one participant,
             sorted by timestamp

    Returns
    -------
    ParticipantData with features shape (T, 3, 4)
    """
    pid = str(df_pid[COL_SUBJECT].iloc[0])

    # ── state labels ──────────────────────────────────────────────────────────
    fog          = df_pid[COL_FOG].values.astype(int)
    fog_severity = df_pid[COL_SEVERITY].values.astype(int)
    states       = derive_states(fog, fog_severity)

    # ── activity mask ─────────────────────────────────────────────────────────
    # The Freeze Index is only meaningful during locomotor activities.
    # Activities 1=Walk, 6=Turn-right, 7=Turn-left all produce stepping motion
    # where FI distinguishes normal gait from FoG.
    #
    # Turning was trialled as an exclusion to reduce FI inflation in Healthy gait.
    # However FoG-STAR's FoG episodes occur predominantly during turning tasks.
    # Restricting to activity==1 alone reduces Shuffling to only 38 valid
    # timesteps across 22 patients -- insufficient to fit any statistical model.
    # Turning is therefore retained. The minor FI elevation during healthy
    # turning is acknowledged as a source of variance in the emission model.
    #
    # Activities 2=Sit, 3=Stand, 4=Sit-to-stand, 5=Stand-to-sit are excluded
    # because they have no walking cadence and produce meaningless FI values.
    activity        = df_pid[COL_ACTIVITY].values.astype(int)
    locomotor_mask  = np.isin(activity, [1, 6, 7])

    # ── features per node ─────────────────────────────────────────────────────
    T         = len(states)
    all_feats = np.full((T, N_NODES, N_FEATURES), np.nan)

    for ni, node in enumerate(NODE_NAMES):
        acc_cols = NODE_COLUMNS[node]['acc']
        gyr_cols = NODE_COLUMNS[node]['gyr']

        acc = df_pid[acc_cols].values.astype(float)  # (T, 3)
        gyr = df_pid[gyr_cols].values.astype(float)  # (T, 3)

        all_feats[:, ni, :] = extract_features(acc, gyr)

    # ── valid mask ────────────────────────────────────────────────────────────
    # Timestep is valid for emission model fitting if ALL of:
    #   1. state label is valid (not -1)
    #   2. not within BOUNDARY_PAD of a state transition
    #   3. features are not NaN (first WINDOW-1 timesteps)
    #   4. activity is locomotor (Walk, Turn-right, Turn-left)
    #      -- FI is only meaningful during stepping activities
    boundary_ok  = compute_boundary_mask(states)
    features_ok  = ~np.any(np.isnan(all_feats), axis=(1, 2))
    valid_mask   = boundary_ok & features_ok & (states >= 0) & locomotor_mask

    return ParticipantData(
        pid        = pid,
        states     = states,
        features   = all_feats,
        valid_mask = valid_mask,
        task_ids   = df_pid[COL_TASK].values.astype(int),
    )


def load_fogstar(csv_path: str) -> Tuple[List[ParticipantData], DatasetStats]:
    """
    Load the complete FoG-STAR dataset.

    Parameters
    ----------
    csv_path : path to sensor_data.csv

    Returns
    -------
    participants : list of ParticipantData, one per participant
    stats        : DatasetStats summary
    """
    print(f"Loading FoG-STAR from {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"  Raw rows: {len(df):,}")

    # Remove unlabelled rows where activity==0.
    # The authors' Analytics notebook applies this filter first:
    #   fog_star = fog_star[fog_star.activity > 0]
    df = df[df[COL_ACTIVITY] > 0].copy()
    print(f"  After activity>0 filter: {len(df):,} rows")

    # Remove rows where any of the three agent sensor nodes has NaN values.
    # The notebook shows ankleL has NaN for some participants and tasks;
    # ankleR, back, and wrist are checked here to ensure complete agent data.
    agent_cols = (
        NODE_COLUMNS['back']['acc']  + NODE_COLUMNS['back']['gyr'] +
        NODE_COLUMNS['wrist']['acc'] + NODE_COLUMNS['wrist']['gyr'] +
        NODE_COLUMNS['ankle']['acc'] + NODE_COLUMNS['ankle']['gyr']
    )
    rows_before = len(df)
    df = df.dropna(subset=agent_cols).copy()
    print(f"  After dropping NaN sensor rows: {len(df):,} rows "
          f"({rows_before - len(df):,} removed)")

    # Sort by subjectID then timestamp for strict temporal order.
    df = df.sort_values([COL_SUBJECT, COL_TIMESTAMP]).reset_index(drop=True)

    pids         = df[COL_SUBJECT].unique()
    participants = []

    for pid in pids:
        df_pid = df[df[COL_SUBJECT] == pid].copy()
        pdata  = load_participant(df_pid)
        participants.append(pdata)
        n_valid = pdata.valid_mask.sum()
        n_total = len(pdata.states)
        print(f"  Subject {pid}: {n_total} timesteps, "
              f"{n_valid} valid ({100*n_valid/n_total:.1f}%)")

    stats = compute_dataset_stats(participants)
    print_stats(stats)

    return participants, stats


# ── statistics ────────────────────────────────────────────────────────────────

def get_run_durations(states: np.ndarray,
                       valid_mask: np.ndarray) -> Dict[int, List[float]]:
    """
    Extract durations of contiguous runs of each state.
    Used for fitting semi-Markov dwell distributions.

    A run is included if it exists in the state sequence regardless of
    the valid_mask. The valid_mask is used only to confirm the run is
    not entirely invalid (e.g. all NaN or all boundary padding).

    Note: requiring 100% validity within a run would exclude all runs
    because boundary padding always marks the edges of every run invalid.
    Instead we require at least 50% of the run to be valid.
    """
    durations = {s: [] for s in range(N_STATES)}
    T         = len(states)
    t         = 0

    while t < T:
        s = states[t]
        if s < 0:
            t += 1
            continue

        # Find end of this run
        t_end = t
        while t_end < T and states[t_end] == s:
            t_end += 1

        run_len = t_end - t
        # Include run if at least 50% of timesteps are valid
        valid_frac = valid_mask[t:t_end].mean()
        if valid_frac >= 0.5 and s >= 0 and run_len >= 2:
            duration_sec = run_len / FS
            durations[s].append(duration_sec)

        t = t_end

    return durations


def compute_dataset_stats(participants: List[ParticipantData]) -> DatasetStats:
    """Compute summary statistics across all participants."""
    state_counts   = {s: 0 for s in range(N_STATES)}
    state_durations = {s: [] for s in range(N_STATES)}
    all_features   = []

    for pdata in participants:
        vm = pdata.valid_mask

        # Count valid timesteps per state
        for s in range(N_STATES):
            state_counts[s] += int(np.sum((pdata.states == s) & vm))

        # Collect run durations
        durs = get_run_durations(pdata.states, vm)
        for s in range(N_STATES):
            state_durations[s].extend(durs[s])

        # Collect features for range computation
        valid_feats = pdata.features[vm]   # (N_valid, 3, 4)
        all_features.append(valid_feats.reshape(-1, N_FEATURES))

    all_features = np.concatenate(all_features, axis=0)

    feature_ranges = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        col = all_features[:, fi]
        col = col[~np.isnan(col)]
        feature_ranges[fname] = (float(np.min(col)), float(np.max(col)))

    return DatasetStats(
        n_participants    = len(participants),
        n_timesteps_total = sum(state_counts.values()),
        state_counts      = state_counts,
        state_durations   = state_durations,
        feature_ranges    = feature_ranges,
    )


def print_stats(stats: DatasetStats):
    """Print a readable summary of dataset statistics."""
    print("\n── Dataset Statistics ─────────────────────────────────────────")
    print(f"  Participants:     {stats.n_participants}")
    print(f"  Valid timesteps:  {stats.n_timesteps_total:,}")
    print()
    print("  Timesteps per state:")
    for s, name in STATE_NAMES.items():
        count = stats.state_counts[s]
        pct   = 100 * count / max(stats.n_timesteps_total, 1)
        durs  = stats.state_durations[s]
        if durs:
            mean_d = np.mean(durs)
            std_d  = np.std(durs)
            n_runs = len(durs)
            print(f"    {name:12s}: {count:8,} timesteps ({pct:5.1f}%)  "
                  f"|  {n_runs} runs, mean duration {mean_d:.2f}s ± {std_d:.2f}s")
        else:
            print(f"    {name:12s}: {count:8,} timesteps ({pct:5.1f}%)")
    print()
    print("  Feature ranges (across all valid timesteps and nodes):")
    for fname, (lo, hi) in stats.feature_ranges.items():
        print(f"    {fname:12s}: [{lo:.4f}, {hi:.4f}]")
    print("───────────────────────────────────────────────────────────────\n")


# ── feature collection helpers ────────────────────────────────────────────────

def collect_features_by_state(
        participants: List[ParticipantData]
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Collect all valid feature vectors grouped by state and node.

    Returns
    -------
    data[state][node] = np.ndarray shape (N_samples, 4)

    This is the direct input to the emission model fitting in Task 3.
    """
    data = {s: {n: [] for n in range(N_NODES)} for s in range(N_STATES)}

    for pdata in participants:
        for s in range(N_STATES):
            mask = (pdata.states == s) & pdata.valid_mask
            if mask.sum() == 0:
                continue
            feats = pdata.features[mask]   # (N, 3, 4)
            for n in range(N_NODES):
                data[s][n].append(feats[:, n, :])

    # Concatenate across participants
    for s in range(N_STATES):
        for n in range(N_NODES):
            if data[s][n]:
                data[s][n] = np.concatenate(data[s][n], axis=0)
            else:
                data[s][n] = np.empty((0, N_FEATURES))

    # Print collection summary
    print("Feature collection summary:")
    for s in range(N_STATES):
        counts = [data[s][n].shape[0] for n in range(N_NODES)]
        print(f"  {STATE_NAMES[s]:12s}: "
              + "  ".join(f"{NODE_NAMES[n]}={counts[n]:6,}" for n in range(N_NODES)))

    return data


def collect_state_sequences(
        participants: List[ParticipantData]
) -> List[np.ndarray]:
    """
    Return list of state sequences for HMM transition matrix fitting.

    Sequences are split at two kinds of boundary:
        1. Genuinely invalid label timesteps (state == -1)
        2. Task boundaries -- where taskID changes between timesteps

    Why task boundaries matter:
        FoG-STAR recordings are structured as 7 sequential tasks. Between
        tasks the patient stops, sits down, receives instructions, and then
        starts the next task. At the boundary between task k and task k+1
        the recording may end with the patient in Healthy state and begin
        the next task already in a FoG state. The transition counter would
        see Healthy → Akinesia and count it as a real clinical transition.
        In reality the patient entered Akinesia independently at the start
        of the new task -- there was no Healthy → Akinesia transition.

        Splitting at task boundaries removes these artefact transitions and
        gives a clinically accurate transition matrix where Healthy almost
        always transitions to Shuffling first, matching clinical reality.

    Locomotor filter is NOT applied here:
        The valid_mask includes the locomotor filter which splits at every
        non-walking period. This would destroy real clinical transitions
        that happen within a task. Task boundary splitting is sufficient.
    """
    sequences   = []
    n_artefacts = 0

    for pdata in participants:
        states   = pdata.states
        task_ids = pdata.task_ids
        T        = len(states)
        current  = []

        for t in range(T):
            s = states[t]

            # Check for split conditions
            split = False
            if s < 0:
                # Invalid label -- always split
                split = True
            elif t > 0 and task_ids[t] != task_ids[t - 1]:
                # Task boundary -- split to avoid artefact transitions
                split = True
                if len(current) > 1:
                    n_artefacts += 1  # count prevented artefacts

            if split:
                if len(current) > 1:
                    sequences.append(np.array(current))
                current = []
                if s >= 0:
                    current.append(s)
            else:
                current.append(s)

        if len(current) > 1:
            sequences.append(np.array(current))

    print(f"Extracted {len(sequences)} state sequences "
          f"across {len(participants)} participants")
    print(f"  Task boundary splits prevented {n_artefacts} "
          f"potential artefact cross-task transitions")
    return sequences


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'sensor_data.csv'

    participants, stats = load_fogstar(csv_path)

    # Collect features ready for Task 2 and Task 3
    feature_data   = collect_features_by_state(participants)
    state_sequences = collect_state_sequences(participants)

    # Save for downstream tasks
    import pickle
    with open('fogstar_processed.pkl', 'wb') as f:
        pickle.dump({
            'participants':    participants,
            'stats':           stats,
            'feature_data':    feature_data,
            'state_sequences': state_sequences,
        }, f)
    print("Saved to fogstar_processed.pkl")