"""
fog_app.py
----------
Streamlit app: FoG IMU Synthetic Data Generator

Workflow
    1. Upload a raw wearable IMU CSV
    2. Map columns (participant ID, sensor axes, FoG labels)
    3. Fit the semi-Markov HMM on the uploaded data
    4. Adjust parameters with interactive knobs
    5. Preview synthetic episodes with live statistical validation
    6. Export labelled datasets with optional validation report PDF

Run with:
    streamlit run fog_app.py

Required files in the same directory:
    fog_generic_loader.py
    fog_star_loader.py
    fog_model_fitter.py
    fog_episode_generator.py
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import pickle
import re
import tempfile
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

from fog_star_loader import (
    FS as FS_DEFAULT,
    N_FEATURES, N_NODES, N_STATES,
    FEATURE_NAMES, STATE_NAMES,
    collect_features_by_state,
    collect_state_sequences,
    compute_dataset_stats,
)
from fog_model_fitter import (
    HMMParams, DwellDistribution, EmissionParams, PreprocessParams,
    clip_pooled_features,
    fit_ar1_coefficient,
    fit_dwell_distributions,
    fit_emission_model,
    fit_initial_distribution,
    fit_transition_matrix,
    preprocess_features,
)
from fog_episode_generator import FogEpisodeGenerator, Episode
from fog_generic_loader import ColumnMapping, load_from_dataframe, validate_mapping

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_NODES        = 3
MAX_PARTICIPANTS = 30
MAX_EPISODES     = 10
STATE_COLORS     = {0: '#4CAF50', 1: '#FFD700', 2: '#FF8C00', 3: '#DC143C'}
STATE_HEX        = ['#4CAF50', '#FFD700', '#FF8C00', '#DC143C']
SYNTH_COLOR      = '#DC143C'
REAL_COLOR       = '#4472C4'

# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        'phase':        'upload',
        'df':           None,
        'col_names':    [],
        'params':       None,
        'node_names':   [],
        'has_severity': False,
        'n_nodes':      1,
        'fit_log':      [],
        # real feature data stored at fit time for validation comparisons
        'real_feats_by_state': None,   # dict: state -> (T, N_NODES, N_FEATURES)
        'real_dwell_samples':  None,   # dict: state -> list of durations (s)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── DwellDistribution helpers ─────────────────────────────────────────────────

def _mean_dwell(d: DwellDistribution) -> float:
    return float(np.exp(d.mu + d.sigma ** 2 / 2))


def _set_mean_dwell(d: DwellDistribution, target_s: float):
    target_s = max(target_s, d.min_sec + 0.05)
    d.mu     = np.log(target_s) - d.sigma ** 2 / 2
    d.mu_sec = target_s


# ── Knob application ──────────────────────────────────────────────────────────

def apply_knobs(base: HMMParams,
                fog_burden: float,
                phase_w: List[float],
                dwell_means: Dict[int, float],
                rhos: Dict[int, float]) -> HMMParams:
    p = deepcopy(base)

    mean_fog = sum(
        dwell_means[s] * phase_w[s - 1]
        for s in range(1, N_STATES)
    )
    mean_fog = max(mean_fog, 0.5)

    if 0.01 < fog_burden < 0.99:
        target_healthy = mean_fog * (1.0 - fog_burden) / fog_burden
    else:
        target_healthy = _mean_dwell(base.dwell[0])
    _set_mean_dwell(p.dwell[0], target_healthy)

    total_h2fog = max(p.T_mat[0, 1] + p.T_mat[0, 2] + p.T_mat[0, 3], 1e-9)
    w = np.array(phase_w, dtype=float)
    w = w / w.sum()
    p.T_mat[0, 1] = total_h2fog * w[0]
    p.T_mat[0, 2] = total_h2fog * w[1]
    p.T_mat[0, 3] = total_h2fog * w[2]

    for s in range(1, N_STATES):
        _set_mean_dwell(p.dwell[s], dwell_means[s])

    for s in range(N_STATES):
        p.rho[s] = float(rhos[s])

    return p


# ── Generator construction ────────────────────────────────────────────────────

def _make_generator(params: HMMParams,
                     seed: int = 0,
                     max_seconds: float = 300.0) -> FogEpisodeGenerator:
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as fh:
        pickle.dump(params, fh)
        tmp = fh.name
    gen = FogEpisodeGenerator(tmp, seed=seed, max_seconds=max_seconds)
    os.unlink(tmp)
    return gen


def _store_defaults(params: HMMParams):
    mean_fog = sum(_mean_dwell(params.dwell[s]) for s in range(1, N_STATES)) / 3
    mean_h   = _mean_dwell(params.dwell[0])
    default_burden = round(mean_fog / max(mean_h + mean_fog, 1e-6), 2)
    st.session_state['default_knobs'] = {
        'knob_fog_burden': float(np.clip(default_burden, 0.01, 0.60)),
        'knob_wt_s':       float(round(params.T_mat[0, 1], 3)),
        'knob_wt_t':       float(round(params.T_mat[0, 2], 3)),
        'knob_wt_a':       float(round(params.T_mat[0, 3], 3)),
        'knob_dwell_1':    float(round(_mean_dwell(params.dwell[1]), 1)),
        'knob_dwell_2':    float(round(_mean_dwell(params.dwell[2]), 1)),
        'knob_dwell_3':    float(round(_mean_dwell(params.dwell[3]), 1)),
        'knob_rho_0':      float(round(params.rho[0], 3)),
        'knob_rho_1':      float(round(params.rho[1], 3)),
        'knob_rho_2':      float(round(params.rho[2], 3)),
        'knob_rho_3':      float(round(params.rho[3], 3)),
    }


# ── Post-processing ───────────────────────────────────────────────────────────

def _postprocess(ep: Episode,
                  noise_std: float,
                  active_nodes: List[bool],
                  sampling_rate: int,
                  orig_fs: int) -> Episode:
    ep = deepcopy(ep)
    rng = np.random.default_rng()
    if noise_std > 0:
        for n in range(N_NODES):
            if active_nodes[n]:
                ep.obs[:, n, :] += rng.normal(0, noise_std, (ep.T, N_FEATURES))
    for n in range(N_NODES):
        if not active_nodes[n]:
            ep.obs[:, n, :] = np.nan
    if sampling_rate < orig_fs:
        step       = max(1, int(orig_fs // sampling_rate))
        ep.obs     = ep.obs[::step]
        ep.states  = ep.states[::step]
        ep.T       = len(ep.states)
    return ep


# ── Episode plot ──────────────────────────────────────────────────────────────

def _plot_episode(ep: Episode,
                   active_nodes: List[bool],
                   node_names: List[str],
                   fs: int) -> plt.Figure:
    n_active = sum(active_nodes)
    if n_active == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No active nodes', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(n_active, 1,
                              figsize=(12, 2.5 * n_active), sharex=True)
    if n_active == 1:
        axes = [axes]
    t_sec = np.arange(ep.T) / fs
    ai = 0
    for n in range(N_NODES):
        if not active_nodes[n]:
            continue
        ax   = axes[ai]
        col  = ep.obs[:, n, 0]
        ax.plot(t_sec, col, color='steelblue', linewidth=0.8)

        prev_t, prev_s = 0, int(ep.states[0])
        for t in range(1, ep.T):
            s = int(ep.states[t])
            if s != prev_s or t == ep.T - 1:
                if prev_s > 0:
                    ax.axvspan(t_sec[prev_t], t_sec[t],
                               alpha=0.25, color=STATE_COLORS[prev_s])
                prev_t, prev_s = t, s

        label = node_names[n] if n < len(node_names) else f'node {n+1}'
        ax.set_ylabel(f'{label}\nFI (z-score)')
        ax.grid(alpha=0.3)
        ai += 1

    patches = [
        mpatches.Patch(color=STATE_COLORS[s], alpha=0.5,
                        label=STATE_NAMES[s])
        for s in range(1, N_STATES)
    ]
    axes[0].legend(handles=patches, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    return fig


# ── Episode stats ─────────────────────────────────────────────────────────────

def _episode_stats(ep: Episode, fs: int) -> dict:
    T        = ep.T
    fog_mask = ep.states > 0
    fog_frac = float(fog_mask.sum() / T) if T else 0.0
    n_ev     = 0
    in_fog   = False
    durs     = []
    start    = 0
    for t, s in enumerate(ep.states):
        if s > 0 and not in_fog:
            in_fog, n_ev, start = True, n_ev + 1, t
        elif s == 0 and in_fog:
            in_fog = False
            durs.append((t - start) / fs)
    if in_fog:
        durs.append((T - start) / fs)
    return {
        'duration_s':    T / fs,
        'fog_fraction':  fog_frac,
        'n_fog_events':  n_ev,
        'mean_fog_dur':  float(np.mean(durs)) if durs else 0.0,
    }


# ── Validation: extract synthetic features from episodes ─────────────────────

def _synth_feats_by_state(episodes: List[Episode]) -> Dict[int, np.ndarray]:
    """Stack synthetic obs by state label. Returns dict state -> (T,N,F)."""
    buckets = {s: [] for s in range(N_STATES)}
    for ep in episodes:
        for s in range(N_STATES):
            mask = ep.states == s
            if mask.any():
                buckets[s].append(ep.obs[mask])
    out = {}
    for s in range(N_STATES):
        if buckets[s]:
            out[s] = np.concatenate(buckets[s], axis=0)
        else:
            out[s] = np.empty((0, N_NODES, N_FEATURES))
    return out


def _reshape_real_feats(feat_data: Dict) -> Dict[int, np.ndarray]:
    """
    Convert collect_features_by_state output into validation-ready format.

    Input:  feat_data[state][node] = np.ndarray (T, N_FEATURES)
    Output: out[state]             = np.ndarray (T, N_NODES, N_FEATURES)

    Rows are aligned across nodes by taking the minimum count so the array
    is rectangular. In practice all nodes have the same count because the
    valid mask is applied jointly across all nodes.
    """
    out = {}
    for s in range(N_STATES):
        node_arrays = [feat_data[s][n] for n in range(N_NODES)]
        min_t = min(a.shape[0] for a in node_arrays)
        if min_t == 0:
            out[s] = np.empty((0, N_NODES, N_FEATURES))
        else:
            # Stack to (N_NODES, T, N_FEATURES) then transpose to (T, N_NODES, N_FEATURES)
            stacked = np.stack([a[:min_t] for a in node_arrays], axis=0)
            out[s]  = stacked.transpose(1, 0, 2)
    return out


def _synth_dwell_samples(episodes: List[Episode], fs: int) -> Dict[int, List[float]]:
    """Extract per-state dwell durations (seconds) from episodes."""
    dwells = {s: [] for s in range(N_STATES)}
    for ep in episodes:
        for state, t0, t1 in ep.dwell_info:
            dwells[state].append((t1 - t0) / fs)
    return dwells


# ── Validation plots ──────────────────────────────────────────────────────────

def _fig_marginals(real_feats: Dict, synth_feats: Dict,
                   node_idx: int, node_name: str) -> plt.Figure:
    """4×4 grid: rows=states, cols=features. Real vs synth histograms."""
    fig, axes = plt.subplots(N_STATES, N_FEATURES,
                             figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f'Marginal feature distributions — {node_name} node\n'
                 f'real (blue) vs synthetic (red outline)', fontsize=11)

    for s in range(N_STATES):
        r_data = real_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))
        sy_data = synth_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))

        for f in range(N_FEATURES):
            ax = axes[s, f]
            r_vals  = r_data[:, node_idx, f] if r_data.shape[0] > 0 else np.array([])
            sy_vals = sy_data[:, node_idx, f] if sy_data.shape[0] > 0 else np.array([])

            r_vals  = r_vals[np.isfinite(r_vals)]
            sy_vals = sy_vals[np.isfinite(sy_vals)]

            ks = np.nan
            if len(r_vals) > 1 and len(sy_vals) > 1:
                ks, _ = ks_2samp(r_vals, sy_vals)
                all_v = np.concatenate([r_vals, sy_vals])
                bins  = np.linspace(np.percentile(all_v, 1),
                                    np.percentile(all_v, 99), 40)
                ax.hist(r_vals,  bins=bins, density=True,
                        alpha=0.5, color=REAL_COLOR,  label='real')
                ax.hist(sy_vals, bins=bins, density=True,
                        histtype='step', linewidth=1.4,
                        color=SYNTH_COLOR, label='synth')

            col_ok = '#2e7d32' if (np.isfinite(ks) and ks < 0.2) else \
                     '#f57c00' if (np.isfinite(ks) and ks < 0.4) else '#c62828'
            ks_str = f'KS={ks:.3f}' if np.isfinite(ks) else 'KS=N/A'
            ax.set_title(f'{STATE_NAMES[s]} / {FEATURE_NAMES[f]}\n{ks_str}',
                         fontsize=8, color=col_ok)
            ax.tick_params(labelsize=7)
            if s == 0 and f == 0:
                ax.legend(fontsize=7)

    return fig


def _fig_dwell_cdfs(real_dwells: Dict, synth_dwells: Dict) -> plt.Figure:
    """2×2 grid of empirical CDF plots per state."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle('Dwell time distributions per state\nreal vs synthetic', fontsize=11)
    axes_flat = axes.flatten()

    for s in range(N_STATES):
        ax = axes_flat[s]
        r  = np.sort(np.array(real_dwells.get(s, [])))
        sy = np.sort(np.array(synth_dwells.get(s, [])))

        ks_str = ''
        if len(r) > 1 and len(sy) > 1:
            ks, _ = ks_2samp(r, sy)
            ks_str = f'  KS={ks:.3f}'

        for vals, col, lbl in [(r, REAL_COLOR, 'real'), (sy, SYNTH_COLOR, 'synth')]:
            if len(vals) > 0:
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, cdf, color=col, label=f'{lbl} n={len(vals)}', linewidth=1.4)

        ax.set_title(f'{STATE_NAMES[s]}{ks_str}', fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('dwell duration (s)', fontsize=8)
        ax.set_ylabel('empirical CDF', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    return fig


def _fig_acf(real_feats: Dict, synth_feats: Dict,
             node_idx: int = 2, feat_idx: int = 0,
             max_lag: int = 60) -> plt.Figure:
    """2×2 ACF plots per state for one feature at one node."""
    feat_name = FEATURE_NAMES[feat_idx]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(f'Autocorrelation of {feat_name} — node index {node_idx}\n'
                 f'real vs synthetic', fontsize=11)
    axes_flat = axes.flatten()

    def _acf_series(arr, max_lag):
        arr = arr[np.isfinite(arr)]
        if len(arr) < max_lag + 2:
            return None
        arr = arr - arr.mean()
        denom = np.dot(arr, arr)
        if denom < 1e-12:
            return None
        return np.array([np.dot(arr[:len(arr)-k], arr[k:]) / denom
                         for k in range(max_lag + 1)])

    for s in range(N_STATES):
        ax = axes_flat[s]
        r_data  = real_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))
        sy_data = synth_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))

        for data, col, lbl in [(r_data, REAL_COLOR, 'real'),
                                (sy_data, SYNTH_COLOR, 'synth')]:
            if data.shape[0] > max_lag + 2:
                series = data[:, node_idx, feat_idx]
                acf = _acf_series(series, max_lag)
                if acf is not None:
                    ax.plot(range(max_lag + 1), acf, color=col,
                            label=lbl, linewidth=1.4)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(STATE_NAMES[s], fontsize=9)
        ax.set_xlabel('lag (timesteps)', fontsize=8)
        ax.set_ylabel('ACF', fontsize=8)
        ax.set_ylim(-0.3, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    return fig


def _fig_internode(real_feats: Dict, synth_feats: Dict) -> plt.Figure:
    """Inter-node Pearson correlation matrices for FI, pooled across states."""
    node_labels = ['node 0', 'node 1', 'node 2']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle('Inter-node FI correlation  (pooled all states)\nreal vs synthetic',
                 fontsize=11)

    for ax, feats, title in [
        (axes[0], real_feats,  'real'),
        (axes[1], synth_feats, 'synth'),
    ]:
        blocks = [feats.get(s, np.empty((0, N_NODES, N_FEATURES)))
                  for s in range(N_STATES)]
        all_data = np.concatenate([b for b in blocks if b.shape[0] > 0], axis=0)

        if all_data.shape[0] > 10:
            mat = np.corrcoef(all_data[:, :, 0].T)  # FI only, shape (N,N)
        else:
            mat = np.eye(N_NODES)

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(N_NODES)); ax.set_xticklabels(node_labels, fontsize=8)
        ax.set_yticks(range(N_NODES)); ax.set_yticklabels(node_labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(N_NODES):
            for j in range(N_NODES):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(mat[i,j]) > 0.5 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)

    return fig


def _fig_ks_summary(real_feats: Dict, synth_feats: Dict) -> plt.Figure:
    """Heatmap of KS distances across (state × feature), averaged over nodes."""
    ks_grid = np.full((N_STATES, N_FEATURES), np.nan)
    for s in range(N_STATES):
        r  = real_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))
        sy = synth_feats.get(s, np.empty((0, N_NODES, N_FEATURES)))
        for f in range(N_FEATURES):
            vals = []
            for n in range(N_NODES):
                rv  = r[:, n, f][np.isfinite(r[:, n, f])] if r.shape[0] > 0 else np.array([])
                sv  = sy[:, n, f][np.isfinite(sy[:, n, f])] if sy.shape[0] > 0 else np.array([])
                if len(rv) > 1 and len(sv) > 1:
                    ks, _ = ks_2samp(rv, sv)
                    vals.append(ks)
            if vals:
                ks_grid[s, f] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    im = ax.imshow(ks_grid, cmap='RdYlGn_r', vmin=0, vmax=0.6, aspect='auto')
    ax.set_xticks(range(N_FEATURES)); ax.set_xticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_yticks(range(N_STATES));   ax.set_yticklabels(STATE_NAMES, fontsize=9)
    ax.set_title('KS distance summary (averaged over nodes)\n'
                 'Green < 0.2 = good   Red > 0.4 = structural mismatch',
                 fontsize=10)
    for s in range(N_STATES):
        for f in range(N_FEATURES):
            v = ks_grid[s, f]
            if np.isfinite(v):
                ax.text(f, s, f'{v:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if v > 0.45 else 'black')
    fig.colorbar(im, ax=ax, label='KS distance')
    return fig


def _fig_episode_summary(episodes: List[Episode], fs: int) -> plt.Figure:
    """FoG fraction distribution and phase composition for synthetic episodes."""
    fog_fracs   = []
    n_events    = []
    phase_times = {s: 0 for s in range(1, N_STATES)}

    for ep in episodes:
        stats = _episode_stats(ep, fs)
        fog_fracs.append(stats['fog_fraction'])
        n_events.append(stats['n_fog_events'])
        for s in range(1, N_STATES):
            phase_times[s] += int((ep.states == s).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig.suptitle('Synthetic episode summary', fontsize=11)

    axes[0].hist(fog_fracs, bins=20, color=SYNTH_COLOR, alpha=0.7, edgecolor='white')
    axes[0].set_xlabel('FoG fraction per episode')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'FoG burden  (mean={np.mean(fog_fracs):.1%})')
    axes[0].axvline(np.mean(fog_fracs), color='black', linestyle='--', linewidth=1)

    total_fog = sum(phase_times.values())
    if total_fog > 0:
        fracs  = [phase_times[s] / total_fog for s in range(1, N_STATES)]
        colors = [STATE_COLORS[s] for s in range(1, N_STATES)]
        labels = [f'{STATE_NAMES[s]}\n{fracs[s-1]:.1%}' for s in range(1, N_STATES)]
        axes[1].pie(fracs, labels=labels, colors=colors,
                    autopct='', startangle=90, textprops={'fontsize': 9})
        axes[1].set_title('Within-FoG phase composition')
    else:
        axes[1].text(0.5, 0.5, 'No FoG in episodes',
                     ha='center', va='center', transform=axes[1].transAxes)

    return fig


def _build_all_validation_figs(episodes: List[Episode],
                                real_feats: Dict,
                                real_dwells: Dict,
                                fs: int,
                                node_names: List[str]) -> Dict[str, plt.Figure]:
    """Compute all six validation figures. Returns dict of name -> fig."""
    synth_feats  = _synth_feats_by_state(episodes)
    synth_dwells = _synth_dwell_samples(episodes, fs)
    node0_name   = node_names[0] if node_names else 'node 0'

    figs = {}
    figs['KS distance summary']           = _fig_ks_summary(real_feats, synth_feats)
    figs['Marginal distributions (node 0)'] = _fig_marginals(real_feats, synth_feats, 0, node0_name)
    figs['Dwell time CDFs']               = _fig_dwell_cdfs(real_dwells, synth_dwells)
    figs['Autocorrelation (FI)']          = _fig_acf(real_feats, synth_feats)
    figs['Inter-node correlation']        = _fig_internode(real_feats, synth_feats)
    figs['Episode summary']               = _fig_episode_summary(episodes, fs)
    return figs


def _render_validation_panel(figs: Dict[str, plt.Figure]):
    """Show all validation figures in the Streamlit UI as an expander."""
    with st.expander('Statistical validation against real data', expanded=True):
        st.caption(
            'Comparing synthetic episodes against your uploaded dataset. '
            'KS < 0.2 (green) = good distributional match. '
            'KS > 0.4 (red) = model-family limitation, not a fitting error.'
        )
        # KS heatmap first as the summary
        st.pyplot(figs['KS distance summary'], use_container_width=True)
        plt.close(figs['KS distance summary'])

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(figs['Dwell time CDFs'], use_container_width=True)
            plt.close(figs['Dwell time CDFs'])
        with col2:
            st.pyplot(figs['Autocorrelation (FI)'], use_container_width=True)
            plt.close(figs['Autocorrelation (FI)'])

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(figs['Inter-node correlation'], use_container_width=True)
            plt.close(figs['Inter-node correlation'])
        with col4:
            st.pyplot(figs['Episode summary'], use_container_width=True)
            plt.close(figs['Episode summary'])

        st.pyplot(figs['Marginal distributions (node 0)'], use_container_width=True)
        plt.close(figs['Marginal distributions (node 0)'])


def _figs_to_pdf_bytes(figs: Dict[str, plt.Figure]) -> bytes:
    """Render all validation figures into a single multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for title, fig in figs.items():
            fig.suptitle(title, fontsize=12, y=1.01)
            pdf.savefig(fig, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


# ── CSV dataset generation ────────────────────────────────────────────────────

def _generate_csv_gz(params: HMMParams,
                      n_participants: int,
                      n_episodes: int,
                      noise_std: float,
                      active_nodes: List[bool],
                      sampling_rate: int,
                      orig_fs: int,
                      diversity: bool,
                      variability: float,
                      node_names: List[str],
                      progress_bar,
                      max_seconds: float = 300.0,
                      base_seed: int = 0) -> tuple[bytes, List[Episode]]:
    """Returns (csv_gz_bytes, list_of_all_episodes)."""
    eff_fs   = sampling_rate
    buf      = io.StringIO()
    writer   = csv.writer(buf)
    all_eps  = []

    header = ['participant_id', 'episode_id', 'timestep', 'time_s', 'state', 'fog']
    for n in range(N_NODES):
        lbl = node_names[n] if n < len(node_names) else f'node{n+1}'
        for f in FEATURE_NAMES:
            header.append(f'{lbl}_{f}')
    writer.writerow(header)

    for pid in range(n_participants):
        p_params = deepcopy(params)
        if diversity and variability > 0:
            rng_d = np.random.default_rng(base_seed + pid)
            for s in range(N_STATES):
                for n in range(N_NODES):
                    sd = np.sqrt(np.diag(params.emission[s][n].Sigma))
                    p_params.emission[s][n].mu = (
                        params.emission[s][n].mu
                        + rng_d.normal(0, variability * sd)
                    )

        gen = _make_generator(p_params, seed=base_seed + pid * 997,
                               max_seconds=max_seconds)
        for eid in range(n_episodes):
            ep = gen.sample_episode()
            ep = _postprocess(ep, noise_std, active_nodes, sampling_rate, orig_fs)
            all_eps.append(ep)
            for t in range(ep.T):
                row = [pid, eid, t, round(t / eff_fs, 4),
                       int(ep.states[t]), int(ep.states[t] > 0)]
                for n in range(N_NODES):
                    for fi in range(N_FEATURES):
                        v = ep.obs[t, n, fi]
                        row.append('' if np.isnan(v) else round(float(v), 6))
                writer.writerow(row)

        progress_bar.progress((pid + 1) / n_participants)

    return gzip.compress(buf.getvalue().encode('utf-8'), compresslevel=6), all_eps


# ── Fitting pipeline ──────────────────────────────────────────────────────────

def _run_fitting(participants, stats, log_fn) -> HMMParams:
    log_fn('Preprocessing features (log + z-score)...')
    processed, prep = preprocess_features(participants)

    log_fn('Collecting per-state features...')
    feat_data = collect_features_by_state(processed)
    feat_data = clip_pooled_features(feat_data, percentile=99.0)

    log_fn('Extracting state sequences...')
    sequences = collect_state_sequences(participants)

    log_fn('Fitting transition matrix...')
    T_raw, T_mat = fit_transition_matrix(sequences)
    pi_0         = fit_initial_distribution(sequences)

    log_fn('Fitting dwell distributions...')
    dwell = fit_dwell_distributions(stats.state_durations)

    log_fn('Fitting emission model...')
    emission = fit_emission_model(processed, weighting='equal', clip_pct=99.0)

    log_fn('Fitting AR(1) coefficients...')
    rho = fit_ar1_coefficient(processed)

    log_fn('Done.')
    return HMMParams(
        T_mat         = T_mat,
        T_mat_raw     = T_raw,
        pi_0          = pi_0,
        dwell         = dwell,
        emission      = emission,
        preprocess    = prep,
        rho           = rho,
        state_counts  = stats.state_counts,
        dwell_samples = stats.state_durations,
    ), feat_data, stats.state_durations


# ── UI phases ─────────────────────────────────────────────────────────────────

def _phase_upload():
    st.subheader('Step 1: Upload your dataset')
    st.markdown(
        'Upload a CSV file containing raw IMU sensor data with FoG annotations. '
        'Each row should represent one timestep. The app will guide you through '
        'column assignment on the next screen.'
    )

    with st.expander('Required CSV format', expanded=False):
        st.markdown(
            '**Required columns:** participant ID, timestamp, binary FoG label (0/1)\n\n'
            '**Per sensor node (up to 3):** acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z\n\n'
            '**Optional:** FoG severity (1=Shuffling, 2=Trembling, 3=Akinesia), '
            'task/session ID, activity label\n\n'
            'Signals should be in raw physical units (g for acc, deg/s for gyr). '
            'The app applies log-transform and z-score normalisation internally.'
        )

    uploaded = st.file_uploader(
        'Choose CSV file', type=['csv'],
        help='Large files (>100 MB) may take time to parse.'
    )

    if uploaded is not None:
        with st.spinner('Parsing CSV...'):
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f'Failed to parse CSV: {exc}')
                return

        st.success(f'Loaded {len(df):,} rows, {len(df.columns)} columns.')
        st.dataframe(df.head(5), use_container_width=True)

        if st.button('Continue to column mapping', type='primary',
                      use_container_width=True):
            st.session_state['df']        = df
            st.session_state['col_names'] = list(df.columns)
            st.session_state['phase']     = 'mapping'
            st.rerun()


# ── Column auto-mapping ───────────────────────────────────────────────────────

# Token patterns per non-sensor field, ordered best first.
_FIELD_PATTERNS = {
    'pid':       ['participant_id', 'participant', 'subject_id', 'subject',
                  'patient_id', 'patient', 'pid', 'subj', 'id'],
    'timestamp': ['timestamp', 'time_s', 'time', 'ts', 'seconds', 'sec',
                  'sample', 'frame', 't'],
    'fog':       ['fog_label', 'fog', 'freezing', 'freeze', 'frz', 'is_fog',
                  'label'],
    'severity':  ['fog_severity', 'severity', 'fog_type', 'fog_class', 'sev',
                  'grade'],
    'task':      ['task_id', 'task', 'session_id', 'session', 'trial', 'run',
                  'block'],
    'activity':  ['activity', 'gait', 'locomotion', 'movement', 'motion',
                  'behaviour', 'behavior'],
}


def _score_column(col_lower: str, patterns: List[str]) -> int:
    """Score one column name against an ordered pattern list. Higher is better."""
    best = 0
    n = len(patterns)
    for rank, pat in enumerate(patterns):
        weight = n - rank  # earlier patterns weigh more
        if col_lower == pat:
            best = max(best, 300 + weight)
        elif col_lower.startswith(pat) or col_lower.endswith(pat):
            best = max(best, 200 + weight)
        elif pat in col_lower:
            best = max(best, 100 + weight)
    return best


def _best_field_match(options: List[str], field: str, offset: int = 0):
    """
    Best (index, score) for `field` among `options`, scoring only entries from
    `offset` onward (used to skip a leading '(none)' sentinel). When nothing
    matches, index falls back to 0 so '(none)' or the first column is selected.
    """
    patterns = _FIELD_PATTERNS.get(field, [field])
    best_i, best_score = 0, 0
    for i in range(offset, len(options)):
        score = _score_column(options[i].lower(), patterns)
        if score > best_score:
            best_score, best_i = score, i
    return best_i, best_score


def _guess_field_index(options: List[str], field: str, offset: int = 0) -> int:
    return _best_field_match(options, field, offset)[0]


def _classify_channel(col: str):
    """Return (sensor, axis) for an IMU column; each element is None if unclear."""
    cl = col.lower()
    if 'gyr' in cl or 'gyro' in cl:
        sensor = 'gyr'
    elif 'acc' in cl or 'accel' in cl:
        sensor = 'acc'
    else:
        sensor = None
    axis = None
    m = re.search(r'(?:^|[_\-\s\.])([xyz])(?:$|[_\-\s\.0-9])', cl)
    if m:
        axis = m.group(1)
    else:
        m2 = re.search(r'[XYZ]', col)
        if m2:
            axis = m2.group(0).lower()
    return sensor, axis


def _channel_candidates(col_names: List[str], sensor: str, axis: str) -> List[int]:
    """Indices in col_names matching the sensor type and axis, in file order."""
    out = []
    for i, c in enumerate(col_names):
        s, a = _classify_channel(c)
        if s == sensor and a == axis:
            out.append(i)
    return out


def _guess_node_channel_index(col_names: List[str], node_i: int,
                               channel: str) -> int:
    """
    Index in col_names for `channel` (e.g. 'acc_x') of node `node_i` (0-based).
    Repeated channels are spread across nodes in file order, so columns like
    back_acc_x / wrist_acc_x / ankle_acc_x land on nodes 1 / 2 / 3.
    """
    sensor, axis = channel.split('_')
    cands = _channel_candidates(col_names, sensor, axis)
    if node_i < len(cands):
        return cands[node_i]
    return 0


def _guess_node_label(col_names: List[str], node_i: int) -> Optional[str]:
    """Infer a node location name from its acc_x column, stripping sensor/axis tokens."""
    cands = _channel_candidates(col_names, 'acc', 'x')
    if node_i >= len(cands):
        return None
    raw = col_names[cands[node_i]]
    tokens = re.split(r'[_\-\s\.]+', raw)
    drop = {'acc', 'accel', 'accelerometer', 'gyr', 'gyro', 'gyroscope',
            'x', 'y', 'z', 'imu', 'sensor'}
    keep = [t for t in tokens if t and t.lower() not in drop and not t.isdigit()]
    return '_'.join(keep).lower() if keep else None


def _phase_mapping():
    df       = st.session_state['df']
    all_cols = ['(none)'] + st.session_state['col_names']
    num_cols = ['(none)'] + [
        c for c in st.session_state['col_names']
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    st.subheader('Step 2: Map your columns')
    st.caption('Columns are pre-filled by matching the closest column names. '
               'Review each one and change any that are wrong.')
    cols = st.session_state['col_names']

    with st.expander('Dataset basics', expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            pid_col = st.selectbox('Participant ID column',
                                    cols, key='map_pid',
                                    index=_guess_field_index(cols, 'pid'))
        with c2:
            ts_col = st.selectbox('Timestamp column',
                                   cols, key='map_ts',
                                   index=_guess_field_index(cols, 'timestamp'))
        with c3:
            fs_val = st.number_input('Sampling rate (Hz)', min_value=10,
                                      max_value=500, value=60, step=1, key='map_fs')
        c4, c5 = st.columns(2)
        with c4:
            task_col = st.selectbox('Task/session ID column (optional)',
                                     all_cols, key='map_task',
                                     index=_guess_field_index(all_cols, 'task', offset=1))
        with c5:
            act_col = st.selectbox('Activity column (optional)',
                                    all_cols, key='map_act',
                                    index=_guess_field_index(all_cols, 'activity', offset=1))
        loco_vals = []
        if act_col != '(none)':
            unique_acts = sorted(df[act_col].dropna().unique().tolist())
            loco_vals = st.multiselect('Which activity values represent locomotion?',
                                        unique_acts, default=unique_acts, key='map_loco')

    with st.expander('FoG state labels', expanded=True):
        fog_col = st.selectbox('FoG label column (0 = no FoG, 1 = FoG)',
                                cols, key='map_fog',
                                index=_guess_field_index(cols, 'fog'))
        _sev_present = _best_field_match(cols, 'severity')[1] >= 200
        has_sev = st.checkbox(
            'Dataset has FoG severity labels (Shuffling/Trembling/Akinesia)',
            value=_sev_present, key='map_has_sev')
        sev_col = None
        if has_sev:
            sev_col = st.selectbox(
                'FoG severity column (1=Shuffling, 2=Trembling, 3=Akinesia)',
                cols, key='map_sev',
                index=_guess_field_index(cols, 'severity'))

    with st.expander('Sensor nodes', expanded=True):
        col_add, col_rem = st.columns([1, 1])
        with col_add:
            if st.button('Add node', disabled=st.session_state['n_nodes'] >= MAX_NODES):
                st.session_state['n_nodes'] += 1
                st.rerun()
        with col_rem:
            if st.button('Remove node', disabled=st.session_state['n_nodes'] <= 1):
                st.session_state['n_nodes'] -= 1
                st.rerun()

        node_defs = []
        for ni in range(st.session_state['n_nodes']):
            st.markdown(f'**Node {ni + 1}**')
            _label = _guess_node_label(cols, ni) or f'node_{ni+1}'
            name = st.text_input('Node name', value=_label,
                                  key=f'node_{ni}_name')
            c1, c2 = st.columns(2)
            with c1:
                ax = st.selectbox('acc_x', cols, key=f'node_{ni}_acc_x',
                                  index=_guess_node_channel_index(cols, ni, 'acc_x'))
                ay = st.selectbox('acc_y', cols, key=f'node_{ni}_acc_y',
                                  index=_guess_node_channel_index(cols, ni, 'acc_y'))
                az = st.selectbox('acc_z', cols, key=f'node_{ni}_acc_z',
                                  index=_guess_node_channel_index(cols, ni, 'acc_z'))
            with c2:
                gx = st.selectbox('gyr_x', cols, key=f'node_{ni}_gyr_x',
                                  index=_guess_node_channel_index(cols, ni, 'gyr_x'))
                gy = st.selectbox('gyr_y', cols, key=f'node_{ni}_gyr_y',
                                  index=_guess_node_channel_index(cols, ni, 'gyr_y'))
                gz = st.selectbox('gyr_z', cols, key=f'node_{ni}_gyr_z',
                                  index=_guess_node_channel_index(cols, ni, 'gyr_z'))
            node_defs.append({
                'name':  name,
                'acc_x': ax, 'acc_y': ay, 'acc_z': az,
                'gyr_x': gx, 'gyr_y': gy, 'gyr_z': gz,
            })
            st.divider()

    c_back, c_fit = st.columns([1, 3])
    with c_back:
        if st.button('Back', use_container_width=True):
            st.session_state['phase'] = 'upload'
            st.rerun()
    with c_fit:
        if st.button('Validate and fit model', type='primary',
                      use_container_width=True):
            col_map = ColumnMapping(
                pid              = pid_col,
                timestamp        = ts_col,
                fog              = fog_col,
                nodes            = node_defs,
                fog_severity     = sev_col if has_sev else None,
                task_id          = task_col if task_col != '(none)' else None,
                activity         = act_col  if act_col  != '(none)' else None,
                locomotor_values = loco_vals if loco_vals else None,
                fs               = int(fs_val),
            )
            errors = validate_mapping(df, col_map)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state['col_map']      = col_map
                st.session_state['node_names']   = [nd['name'] for nd in node_defs]
                st.session_state['has_severity'] = has_sev
                st.session_state['phase']        = 'fitting'
                st.rerun()


def _phase_fitting():
    st.subheader('Step 3: Fitting the HMM')
    col_map = st.session_state['col_map']
    df      = st.session_state['df']

    progress = st.progress(0, text='Loading data...')
    log_box  = st.empty()
    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(msg)
        log_box.code('\n'.join(log_lines), language=None)

    try:
        log('Extracting features from raw IMU signals...')
        n_pids = df[col_map.pid].nunique()

        def prog_cb(frac):
            progress.progress(int(frac * 40),
                               text=f'Processing participants ({int(frac*n_pids)}/{n_pids})...')

        participants, stats = load_from_dataframe(df, col_map, progress_cb=prog_cb)
        log(f'Loaded {len(participants)} participants, '
            f'{stats.n_timesteps_total:,} valid timesteps.')
        progress.progress(40, text='Fitting model parameters...')

        params, feat_data, dwell_samples = _run_fitting(
            participants, stats, log)

        progress.progress(100, text='Complete.')

        st.session_state['params']               = params
        st.session_state['orig_fs']              = col_map.fs
        st.session_state['real_feats_by_state']  = _reshape_real_feats(feat_data)
        st.session_state['real_dwell_samples']   = dwell_samples
        st.session_state['phase']                = 'ready'
        _store_defaults(params)
        st.success('Model fitted successfully. Proceed to the generator.')
        time.sleep(0.8)
        st.rerun()

    except Exception as exc:
        st.error(f'Fitting failed: {exc}')
        if st.button('Back to column mapping'):
            st.session_state['phase'] = 'mapping'
            st.rerun()


def _sidebar_knobs(base: HMMParams):
    defaults = st.session_state.get('default_knobs', {})

    def _default(key, fallback):
        return st.session_state.get(key, defaults.get(key, fallback))

    with st.sidebar:
        st.header('Population')
        fog_burden = st.slider(
            'FoG burden', 0.01, 0.60,
            _default('knob_fog_burden', 0.20),
            step=0.01, key='knob_fog_burden',
            help='Fraction of episode time the patient spends in FoG.'
        )
        st.caption('Initial FoG phase mix (relative weights)')
        c1, c2, c3 = st.columns(3)
        with c1:
            wt_s = st.number_input('Shuf', 0.01, 5.0,
                                    _default('knob_wt_s', float(round(base.T_mat[0, 1], 3))),
                                    0.01, key='knob_wt_s')
        with c2:
            wt_t = st.number_input('Trem', 0.01, 5.0,
                                    _default('knob_wt_t', float(round(base.T_mat[0, 2], 3))),
                                    0.01, key='knob_wt_t')
        with c3:
            wt_a = st.number_input('Akin', 0.01, 5.0,
                                    _default('knob_wt_a', float(round(base.T_mat[0, 3], 3))),
                                    0.01, key='knob_wt_a')

        st.divider()
        st.header('Dynamics')
        st.caption('Mean dwell per state')
        dwell_means = {
            1: st.slider('Shuffling (s)', 0.5, 15.0,
                          _default('knob_dwell_1', float(round(_mean_dwell(base.dwell[1]), 1))),
                          0.1, key='knob_dwell_1'),
            2: st.slider('Trembling (s)', 0.5, 30.0,
                          _default('knob_dwell_2', float(round(_mean_dwell(base.dwell[2]), 1))),
                          0.1, key='knob_dwell_2'),
            3: st.slider('Akinesia (s)',  1.0, 120.0,
                          _default('knob_dwell_3', float(round(_mean_dwell(base.dwell[3]), 1))),
                          1.0, key='knob_dwell_3'),
        }
        st.caption('AR(1) temporal coherence per state')
        rhos = {
            0: st.slider('rho Healthy',   0.50, 0.99,
                          _default('knob_rho_0', float(round(base.rho[0], 3))),
                          0.01, key='knob_rho_0'),
            1: st.slider('rho Shuffling', 0.50, 0.99,
                          _default('knob_rho_1', float(round(base.rho[1], 3))),
                          0.01, key='knob_rho_1'),
            2: st.slider('rho Trembling', 0.50, 0.99,
                          _default('knob_rho_2', float(round(base.rho[2], 3))),
                          0.01, key='knob_rho_2'),
            3: st.slider('rho Akinesia',  0.50, 0.99,
                          _default('knob_rho_3', float(round(base.rho[3], 3))),
                          0.01, key='knob_rho_3'),
        }

        st.divider()
        st.header('Sensor')
        orig_fs = st.session_state.get('orig_fs', FS_DEFAULT)
        sr_opts = sorted({s for s in [orig_fs, orig_fs // 2, orig_fs // 4, 4]
                          if 4 <= s <= orig_fs}, reverse=True)
        sampling_rate = st.selectbox(
            'Output sampling rate', sr_opts,
            format_func=lambda x: f'{x} Hz'
        )
        noise_std = st.slider('Additive sensor noise std', 0.0, 3.0, 0.0, 0.05,
                               key='knob_noise')
        st.caption('Active nodes')
        node_names = st.session_state.get('node_names',
                                           [f'node_{i+1}' for i in range(N_NODES)])
        active_nodes = []
        cols = st.columns(min(N_NODES, 3))
        for n in range(N_NODES):
            label = node_names[n] if n < len(node_names) else f'node {n+1}'
            with cols[n]:
                active_nodes.append(
                    st.checkbox(label,
                                 value=(n < st.session_state.get('n_nodes', 1)),
                                 key=f'active_{n}')
                )

        st.divider()
        def _reset_knobs():
            defaults = st.session_state.get('default_knobs', {})
            for k, v in defaults.items():
                st.session_state[k] = v

        st.button('Reset to fitted defaults', use_container_width=True,
                   help='Restore all knobs to the values fitted from your dataset.',
                   on_click=_reset_knobs)

    return (fog_burden, [wt_s, wt_t, wt_a], dwell_means, rhos,
            noise_std, active_nodes, sampling_rate)


def _phase_ready():
    base    = st.session_state['params']
    orig_fs = st.session_state.get('orig_fs', FS_DEFAULT)

    (fog_burden, phase_w, dwell_means, rhos,
     noise_std, active_nodes, sampling_rate) = _sidebar_knobs(base)

    current    = apply_knobs(base, fog_burden, phase_w, dwell_means, rhos)
    node_names = st.session_state.get('node_names',
                                       [f'node_{i+1}' for i in range(N_NODES)])
    real_feats  = st.session_state.get('real_feats_by_state') or {}
    real_dwells = st.session_state.get('real_dwell_samples') or {}

    tab_prev, tab_export, tab_params = st.tabs(
        ['Episode preview', 'Export dataset', 'Parameter summary']
    )

    # ── Preview ───────────────────────────────────────────────────────────────
    with tab_prev:
        st.subheader('Single Episode Preview')
        dur_col, s_col, b_col = st.columns([2, 2, 1])
        with dur_col:
            max_dur = st.slider('Max episode duration (s)', 30, 600, 300, 30,
                                 key='prev_duration')
        with s_col:
            seed = st.number_input('Random seed', 0, 9999, 42, 1, key='prev_seed')
        with b_col:
            st.write('')
            st.write('')
            go = st.button('Generate episode', type='primary',
                            use_container_width=True)

        prev_div_col, _ = st.columns([2, 3])
        with prev_div_col:
            prev_variability = st.slider(
                'Participant diversity',
                0.0, 1.0, 0.0, 0.05,
                key='prev_variability',
                help='Adds a random per-participant offset to emission means, '
                     'simulating between-patient variability. '
                     '0 = all participants identical. '
                     'This value is carried over to the Export tab automatically.'
            )

        if go:
            with st.spinner('Generating...'):
                # Apply participant diversity offset to the params for this preview
                prev_params = deepcopy(current)
                if prev_variability > 0:
                    rng_d = np.random.default_rng(int(seed))
                    for s in range(N_STATES):
                        for n in range(N_NODES):
                            sd = np.sqrt(np.diag(
                                current.emission[s][n].Sigma))
                            prev_params.emission[s][n].mu = (
                                current.emission[s][n].mu
                                + rng_d.normal(0, prev_variability * sd)
                            )
                gen = _make_generator(prev_params, seed=int(seed),
                                       max_seconds=float(max_dur))
                ep  = gen.sample_episode()
                ep  = _postprocess(ep, noise_std, active_nodes,
                                    sampling_rate, orig_fs)
                st.session_state['preview_ep'] = ep

        if 'preview_ep' in st.session_state:
            ep    = st.session_state['preview_ep']
            stats = _episode_stats(ep, sampling_rate)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Duration',     f"{stats['duration_s']:.0f} s")
            m2.metric('FoG burden',   f"{stats['fog_fraction']:.1%}")
            m3.metric('FoG events',   stats['n_fog_events'])
            m4.metric('Active nodes', sum(active_nodes))

            if any(active_nodes):
                fig = _plot_episode(ep, active_nodes, node_names, sampling_rate)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning('Enable at least one node to see the episode plot.')

            # ── Live validation panel ─────────────────────────────────────────
            if real_feats:
                st.divider()
                with st.spinner('Computing validation statistics...'):
                    figs = _build_all_validation_figs(
                        [ep], real_feats, real_dwells, sampling_rate, node_names)
                _render_validation_panel(figs)
            else:
                st.info('Validation charts appear here once the model is fitted '
                        'from an uploaded dataset.')

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.subheader('Generate and download labelled dataset')
        e1, e2 = st.columns(2)
        with e1:
            n_pids   = st.slider('Participants', 1, MAX_PARTICIPANTS, 5, key='exp_npids')
            n_eps    = st.slider('Episodes per participant', 1, MAX_EPISODES, 3, key='exp_neps')
            exp_dur  = st.slider('Max episode duration (s)', 30, 600, 300, 30,
                                  key='exp_duration')
        with e2:
            # Autofill seed and variability from preview tab values
            _prev_seed_val = int(st.session_state.get('prev_seed', 42))
            _prev_var_val  = float(st.session_state.get('prev_variability', 0.0))

            exp_seed = st.number_input(
                'Base seed',
                0, 9999, _prev_seed_val, 1,
                key='exp_seed',
                help='Automatically filled from the preview tab seed. '
                     'Change if you want a different random draw.'
            )
            variability = st.slider(
                'Participant diversity',
                0.0, 1.0, _prev_var_val, 0.05,
                key='exp_variability',
                help='Automatically filled from the preview tab diversity. '
                     'Keeping this the same as the preview ensures the '
                     'exported data has similar statistical properties to '
                     'what you saw in the charts.'
            )

            st.divider()
            include_report = st.checkbox(
                'Include statistical validation report (PDF)',
                value=True,
                help='Generates six validation figures comparing the synthetic dataset '
                     'against your uploaded real data: KS distances, dwell CDFs, '
                     'autocorrelation, inter-node correlation, marginal distributions, '
                     'and episode summary. Adds a few seconds to generation time.'
            )

        # Diversity is always on (variability=0 means no effect)
        diversity = True

        est_rows = n_pids * n_eps * int(exp_dur * orig_fs)
        est_mb   = est_rows * 200 / 1024 ** 2 / 12
        st.info(f'Estimated: ~{est_rows:,} rows, compressed download ~{est_mb:.0f} MB')
        if est_mb > 150:
            st.warning('Large output. Generation may take several minutes.')

        if variability > 0:
            st.caption(
                f'Preview used diversity={prev_variability:.2f}, seed={_prev_seed_val}. '
                f'Export is set to diversity={variability:.2f}, seed={int(exp_seed)}. '
                f'Charts will be similar when these match.'
            )

        if st.button('Generate dataset', type='primary', use_container_width=True):
            prog = st.progress(0, text='Generating participants...')
            with st.spinner('Running generator...'):
                data_gz, all_eps = _generate_csv_gz(
                    current, n_pids, n_eps, noise_std,
                    active_nodes, sampling_rate, orig_fs,
                    diversity, variability, node_names, prog,
                    max_seconds=float(exp_dur),
                    base_seed=int(exp_seed)
                )
            prog.empty()
            st.success(f'Done. {n_pids} participants, {n_eps} episodes each.')
            st.session_state['csv_gz']   = data_gz
            st.session_state['all_eps']  = all_eps

            # Build validation figures for the full generated batch
            if include_report and real_feats:
                with st.spinner('Building validation report...'):
                    val_figs = _build_all_validation_figs(
                        all_eps, real_feats, real_dwells, sampling_rate, node_names)
                    st.session_state['val_figs']   = val_figs
                    st.session_state['val_pdf']    = _figs_to_pdf_bytes(val_figs)
            else:
                st.session_state.pop('val_figs', None)
                st.session_state.pop('val_pdf',  None)

        # ── Download buttons ──────────────────────────────────────────────────
        if 'csv_gz' in st.session_state:
            fname_base = (f'fogstar_synth_{n_pids}p_{n_eps}ep'
                          f'_burden{int(fog_burden*100)}')

            if 'val_pdf' in st.session_state:
                # Bundle CSV + PDF into a single ZIP
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(f'{fname_base}.csv.gz',
                                st.session_state['csv_gz'])
                    zf.writestr(f'{fname_base}_validation_report.pdf',
                                st.session_state['val_pdf'])
                zip_buf.seek(0)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        'Download CSV + Validation Report (ZIP)',
                        data=zip_buf.getvalue(),
                        file_name=f'{fname_base}_with_report.zip',
                        mime='application/zip',
                        use_container_width=True,
                        type='primary',
                    )
                with dl_col2:
                    st.download_button(
                        'Download CSV only (.csv.gz)',
                        data=st.session_state['csv_gz'],
                        file_name=f'{fname_base}.csv.gz',
                        mime='application/gzip',
                        use_container_width=True,
                    )
                st.caption('ZIP contains: data CSV and a 6-page validation PDF. '
                            'Load CSV in Python: `pd.read_csv("file.csv.gz")`')

                # Show the validation figures inline after generation
                if 'val_figs' in st.session_state:
                    st.divider()
                    _render_validation_panel(st.session_state['val_figs'])

            else:
                st.download_button(
                    'Download CSV.GZ',
                    data=st.session_state['csv_gz'],
                    file_name=f'{fname_base}.csv.gz',
                    mime='application/gzip',
                    use_container_width=True,
                )
                st.caption('Load in Python: `pd.read_csv("file.csv.gz")`')

    # ── Parameter summary ─────────────────────────────────────────────────────
    with tab_params:
        st.subheader('Current vs default parameters')
        rows = []
        for s in range(N_STATES):
            rows.append({
                'State':             STATE_NAMES[s],
                'Default dwell (s)': f"{_mean_dwell(base.dwell[s]):.1f}",
                'Current dwell (s)': f"{_mean_dwell(current.dwell[s]):.1f}",
                'Default rho':       f"{base.rho[s]:.3f}",
                'Current rho':       f"{current.rho[s]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader('Healthy transition distribution')
        tot = (current.T_mat[0, 1] + current.T_mat[0, 2] + current.T_mat[0, 3])
        tot = max(tot, 1e-9)
        tc1, tc2, tc3 = st.columns(3)
        for col, s in zip([tc1, tc2, tc3], [1, 2, 3]):
            col.metric(f'H to {STATE_NAMES[s]}',
                        f"{current.T_mat[0, s] / tot:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='FoG IMU Synthetic Data Generator',
    page_icon=':brain:',
    layout='wide',
)
_init_state()

st.title('FoG IMU Synthetic Data Generator')
st.caption(
    'A calibrated semi-Markov HMM generator for Freezing of Gait wearable IMU data. '
    'Upload a labelled dataset, fit the model, and generate unlimited synthetic episodes '
    'with controllable clinical parameters.'
)

phase = st.session_state['phase']

if phase == 'upload':
    _phase_upload()
elif phase == 'mapping':
    _phase_mapping()
elif phase == 'fitting':
    _phase_fitting()
elif phase == 'ready':
    _phase_ready()