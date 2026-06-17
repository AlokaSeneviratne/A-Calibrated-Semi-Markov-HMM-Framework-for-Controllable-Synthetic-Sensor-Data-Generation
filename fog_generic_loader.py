"""
fog_generic_loader.py
---------------------
Generalised loader: accepts any wearable IMU dataset (raw acc/gyr columns)
and produces ParticipantData objects compatible with fog_model_fitter.py.

Supports
    1 to 3 sensor nodes (always padded to 3 internal slots; unused are NaN)
    Binary FoG labels (0/1)  -> 2-state model (Healthy + FoG only)
    Severity labels (1/2/3)  -> 4-state model (Healthy, Shuffling, Trembling, Akinesia)
    Configurable sampling rate (window and boundary pad scale automatically)
    Optional activity column for locomotor filtering
    Optional task/session column for transition boundary splitting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fog_star_loader import (
    BOUNDARY_PAD as _BOUNDARY_PAD_DEFAULT,
    FS           as _FS_DEFAULT,
    N_FEATURES,
    N_NODES,
    N_STATES,
    ParticipantData,
    compute_boundary_mask,
    compute_dataset_stats,
    derive_states,
    extract_features,
)


# ── Column mapping dataclass ───────────────────────────────────────────────────

@dataclass
class ColumnMapping:
    """Declares which CSV column serves which pipeline role."""

    pid:               str                      # participant ID column
    timestamp:         str                      # timestamp column (numeric)
    fog:               str                      # binary FoG column (0/1)
    nodes:             List[Dict[str, str]]      # sensor node definitions (1-3)
    fog_severity:      Optional[str]   = None   # severity column (optional)
    task_id:           Optional[str]   = None   # task/session ID (optional)
    activity:          Optional[str]   = None   # activity label (optional)
    locomotor_values:  Optional[List[Any]] = None  # valid activity values
    fs:                int             = 60     # sampling rate in Hz


# ── Scaling helpers ────────────────────────────────────────────────────────────

def _window_for_fs(fs: int) -> int:
    """FFT window size approximately 2 seconds, rounded to nearest power of 2."""
    exp = int(np.ceil(np.log2(max(2.0 * fs, 32))))
    return 2 ** exp


def _boundary_pad_for_fs(fs: int) -> int:
    """Scale the default 0.33 s boundary pad to the given sampling rate."""
    return max(5, int(round(_BOUNDARY_PAD_DEFAULT * fs / _FS_DEFAULT)))


# ── State derivation ───────────────────────────────────────────────────────────

def derive_states_generic(fog: np.ndarray,
                           fog_severity: Optional[np.ndarray],
                           has_severity: bool) -> np.ndarray:
    """
    Combine fog and optional severity arrays into integer state labels.

    With severity:    0=Healthy, 1=Shuffling, 2=Trembling, 3=Akinesia
    Without severity: 0=Healthy, 1=FoG (all FoG timesteps treated as state 1)
    """
    if has_severity and fog_severity is not None:
        return derive_states(fog, fog_severity)
    states = np.zeros(len(fog), dtype=int)
    states[fog.astype(int) == 1] = 1
    return states


# ── Per-participant processing ─────────────────────────────────────────────────

def _load_participant(df_p: pd.DataFrame,
                       col_map: ColumnMapping,
                       n_active: int) -> ParticipantData:
    pid          = str(df_p[col_map.pid].iloc[0])
    has_severity = col_map.fog_severity is not None
    fs           = col_map.fs

    fog = df_p[col_map.fog].values.astype(int)
    sev = (df_p[col_map.fog_severity].values.astype(int)
           if has_severity else None)
    states = derive_states_generic(fog, sev, has_severity)

    T      = len(states)
    window = _window_for_fs(fs)
    feats  = np.full((T, N_NODES, N_FEATURES), np.nan)

    for ni, node in enumerate(col_map.nodes[:N_NODES]):
        acc = df_p[[node['acc_x'], node['acc_y'], node['acc_z']]].values.astype(float)
        gyr = df_p[[node['gyr_x'], node['gyr_y'], node['gyr_z']]].values.astype(float)
        feats[:, ni, :] = extract_features(acc, gyr, window=window, fs=fs)

    pad         = _boundary_pad_for_fs(fs)
    boundary_ok = compute_boundary_mask(states, pad=pad)
    feat_ok     = ~np.any(np.isnan(feats[:, :n_active, :]), axis=(1, 2))
    state_ok    = states >= 0

    if col_map.activity is not None and col_map.locomotor_values is not None:
        act_ok     = np.isin(df_p[col_map.activity].values, col_map.locomotor_values)
        valid_mask = boundary_ok & feat_ok & state_ok & act_ok
    else:
        valid_mask = boundary_ok & feat_ok & state_ok

    task_ids = (df_p[col_map.task_id].values.astype(int)
                if col_map.task_id else np.zeros(T, dtype=int))

    return ParticipantData(
        pid        = pid,
        states     = states,
        features   = feats,
        valid_mask = valid_mask,
        task_ids   = task_ids,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def load_from_dataframe(
        df: pd.DataFrame,
        col_map: ColumnMapping,
        progress_cb=None,           # optional callable(float) for 0-1 progress
) -> Tuple[List[ParticipantData], object]:
    """
    Convert a raw IMU DataFrame into ParticipantData objects.

    Parameters
    ----------
    df         : full dataset as a pandas DataFrame
    col_map    : ColumnMapping describing column roles
    progress_cb: optional callback(float) called with progress 0.0 to 1.0

    Returns
    -------
    participants, stats
    """
    n_active = min(len(col_map.nodes), N_NODES)

    if col_map.activity is not None and col_map.locomotor_values is not None:
        df = df[df[col_map.activity].isin(col_map.locomotor_values)].copy()

    sensor_cols = []
    for nd in col_map.nodes[:N_NODES]:
        sensor_cols += [nd['acc_x'], nd['acc_y'], nd['acc_z'],
                        nd['gyr_x'], nd['gyr_y'], nd['gyr_z']]
    df = df.dropna(subset=sensor_cols).copy()
    df = df.sort_values([col_map.pid, col_map.timestamp]).reset_index(drop=True)

    pids         = df[col_map.pid].unique()
    participants = []
    for i, pid in enumerate(pids):
        pdata = _load_participant(df[df[col_map.pid] == pid].copy(),
                                   col_map, n_active)
        participants.append(pdata)
        if progress_cb:
            progress_cb((i + 1) / len(pids))

    stats = compute_dataset_stats(participants)
    return participants, stats


# ── Validation ────────────────────────────────────────────────────────────────

def validate_mapping(df: pd.DataFrame,
                      col_map: ColumnMapping) -> List[str]:
    """
    Check that all mapped columns exist and contain plausible values.

    Returns a list of error strings (empty list means all OK).
    """
    errors = []
    cols   = set(df.columns)

    def need(role: str, col: str):
        if col not in cols:
            errors.append(f"Column not found: '{col}' (mapped as {role})")

    need('participant ID', col_map.pid)
    need('timestamp',      col_map.timestamp)
    need('FoG label',      col_map.fog)

    if col_map.fog_severity:
        need('FoG severity', col_map.fog_severity)
    if col_map.task_id:
        need('task ID',      col_map.task_id)
    if col_map.activity:
        need('activity',     col_map.activity)

    if not col_map.nodes:
        errors.append("At least one sensor node must be defined.")
    for i, nd in enumerate(col_map.nodes[:N_NODES]):
        for role in ('acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'):
            if not nd.get(role):
                errors.append(f"Node {i + 1}: '{role}' column not assigned.")
            else:
                need(f'node {i + 1} {role}', nd[role])

    if col_map.fog in cols:
        fog_uniq = set(df[col_map.fog].dropna().unique())
        if not fog_uniq.issubset({0, 1, 0.0, 1.0}):
            errors.append(
                f"FoG column '{col_map.fog}' contains non-binary values: "
                f"{sorted(fog_uniq)[:6]}"
            )

    n_pids = df[col_map.pid].nunique() if col_map.pid in cols else 0
    if n_pids < 2:
        errors.append(
            f"Only {n_pids} participant(s) detected. "
            f"At least 2 participants are required for reliable fitting."
        )

    if col_map.fs < 10 or col_map.fs > 500:
        errors.append(
            f"Sampling rate {col_map.fs} Hz seems implausible. "
            f"Expected 10-500 Hz."
        )

    return errors
