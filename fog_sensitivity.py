"""
fog_sensitivity.py
------------------
Chapter 8 sensitivity studies for:
    "A Calibrated Semi-Markov HMM Generator for Synthetic Freezing of Gait
    Sensor Data"

Three studies, one figure each:

Study 1: FoG burden
    Sweep target FoG fraction: [0.05, 0.10, 0.20, 0.35, 0.50]
    Knob: Healthy dwell mean (back-calculated from target fraction)
    Clinical meaning: how much of the episode is spent in FoG states

Study 2: Akinesia mean dwell
    Sweep Akinesia mean dwell: [5, 10, 20, 40, 80] seconds
    Knob: Akinesia DwellDistribution.mu_sec and .mu (log-normal parameter)
    Structural meaning: how long each Akinesia episode lasts

Study 3: AR(1) rho for Shuffling
    Sweep Shuffling rho: [0.50, 0.65, 0.80, 0.90, 0.99]
    Signal meaning: temporal coherence of the Shuffling state signal
    (0.50 = fast noisy, 0.99 = slow smooth)

For each study: generate 100 episodes per knob value, compute output metrics,
produce one figure with three panels (one metric per panel).

Output metrics per episode batch:
    fog_fraction    : fraction of timesteps in FoG states (1+2+3)
    ks_ankle_fi     : 2-sample KS distance between synthetic and real
                      ankle FI distributions (Shuffling state only, headline)
    ks_back_fi      : same KS distance at the back node
    ks_wrist_fi     : same KS distance at the wrist node
    mean_dwell_fog  : mean duration of FoG runs in seconds

Figures produced:
    fig_8_sens_fog_burden.png
    fig_8_sens_akinesia_dwell.png
    fig_8_sens_shuffling_rho.png

Tables produced:
    table_8_sens_fog_burden.csv
    table_8_sens_akinesia_dwell.csv
    table_8_sens_shuffling_rho.csv

Usage
-----
    python fog_sensitivity.py
        [--processed  fogstar_processed.pkl]
        [--params     fogstar_model_params.pkl]
        [--n-episodes 100]
        [--seed       42]
        [--outdir     validation]
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ks_2samp

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fog_star_loader import (
    FS, N_STATES, N_NODES, N_FEATURES,
    STATE_NAMES, NODE_NAMES, FEATURE_NAMES,
    ParticipantData,
)
from fog_model_fitter import (
    HMMParams, DwellDistribution, PreprocessParams,
    preprocess_features,
)
from fog_episode_generator import FogEpisodeGenerator, Episode


# ── constants ─────────────────────────────────────────────────────────────────

HEALTHY   = 0
SHUFFLING = 1
TREMBLING = 2
AKINESIA  = 3

ANKLE = 2
BACK  = 0
WRIST = 1
FI    = 0   # feature index

# Sweep values
BURDEN_LEVELS        = [0.05, 0.10, 0.20, 0.35, 0.50]
AKINESIA_DWELL_SECS  = [5.0, 10.0, 20.0, 40.0, 80.0]
SHUFFLING_RHO_VALS   = [0.50, 0.65, 0.80, 0.90, 0.99]


# ── loading ───────────────────────────────────────────────────────────────────

def load_params(params_path: str) -> HMMParams:
    import fog_model_fitter as _fmf  # noqa

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'fog_model_fitter'
            return super().find_class(module, name)

    with open(params_path, 'rb') as f:
        return _Unpickler(f).load()


def load_real_shuffling_fi(processed_path: str) -> Dict[int, np.ndarray]:
    """
    Load real Shuffling-state FI values per node, in z-score space, for use as
    the KS reference distributions.

    Returns
    -------
    {node_idx: 1d array of real Shuffling FI z-scores} for every node.
    """
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'fog_star_loader'
            return super().find_class(module, name)

    with open(processed_path, 'rb') as f:
        data = _Unpickler(f).load()
    raw_participants = data['participants']
    processed, _     = preprocess_features(raw_participants)

    out = {n: [] for n in range(N_NODES)}
    for p in processed:
        mask = p.valid_mask & (p.states == SHUFFLING)
        if mask.sum() == 0:
            continue
        for n in range(N_NODES):
            fi_vals = p.features[mask, n, FI]
            fi_vals = fi_vals[~np.isnan(fi_vals)]
            out[n].extend(fi_vals.tolist())
    return {n: np.array(out[n]) for n in range(N_NODES)}


# ── param mutation helpers ────────────────────────────────────────────────────

def burden_to_healthy_dwell(params: HMMParams,
                              target_fog_fraction: float) -> float:
    T_mat       = params.T_mat
    fog_states  = [SHUFFLING, TREMBLING, AKINESIA]
    weights     = np.array([T_mat[HEALTHY, s] for s in fog_states])
    fog_mu_secs = np.array([params.dwell[s].mu_sec for s in fog_states])
    w_sum       = weights.sum()
    mu_fog      = float(np.dot(weights, fog_mu_secs) / w_sum) if w_sum > 0 else 10.0
    f           = float(np.clip(target_fog_fraction, 0.01, 0.99))
    return mu_fog * (1.0 - f) / f


def mutate_healthy_dwell(params: HMMParams, mu_h_target: float) -> HMMParams:
    mutated  = copy.deepcopy(params)
    d        = mutated.dwell[HEALTHY]
    sig      = d.sigma
    mu_log   = np.log(max(mu_h_target, 0.5)) - 0.5 * sig ** 2
    d.mu     = float(mu_log)
    d.mu_sec = float(mu_h_target)
    original_cv = (params.dwell[HEALTHY].std_sec
                   / max(params.dwell[HEALTHY].mu_sec, 1e-6))
    d.std_sec   = float(mu_h_target * original_cv)
    return mutated


def mutate_akinesia_dwell(params: HMMParams,
                           mu_sec_target: float) -> HMMParams:
    """
    Return deep copy with Akinesia mean dwell set to mu_sec_target seconds.
    Moment-matching: mu_log = log(mu_sec) - 0.5 * sigma_log^2
    """
    mutated  = copy.deepcopy(params)
    d        = mutated.dwell[AKINESIA]
    sig      = d.sigma
    mu_log   = np.log(max(mu_sec_target, 0.5)) - 0.5 * sig ** 2
    d.mu     = float(mu_log)
    d.mu_sec = float(mu_sec_target)
    original_cv = (params.dwell[AKINESIA].std_sec
                   / max(params.dwell[AKINESIA].mu_sec, 1e-6))
    d.std_sec   = float(mu_sec_target * original_cv)
    return mutated


def mutate_shuffling_rho(params: HMMParams, rho: float) -> HMMParams:
    """Return deep copy with Shuffling AR(1) rho set to rho."""
    mutated              = copy.deepcopy(params)
    mutated.rho[SHUFFLING] = float(np.clip(rho, 0.0, 0.9999))
    return mutated


# ── episode generation ────────────────────────────────────────────────────────

def generate_episodes(params: HMMParams,
                       n_episodes: int,
                       seed: int) -> List[Episode]:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp = f.name
    with open(tmp, 'wb') as f:
        pickle.dump(params, f)
    gen = FogEpisodeGenerator(tmp, seed=seed,
                               min_fog=1, max_seconds=300.0)
    os.unlink(tmp)
    return [gen.sample_episode() for _ in range(n_episodes)]


# ── output metrics ────────────────────────────────────────────────────────────

def compute_metrics(episodes: List[Episode],
                     real_shuf_fi: Dict[int, np.ndarray]
                     ) -> Dict[str, float]:
    """
    Compute output metrics from a batch of synthetic episodes.

    Returns
    -------
    fog_fraction   : fraction of all timesteps in FoG states
    ks_ankle_fi    : 2-sample KS, synthetic vs real Shuffling ankle FI (headline)
    ks_back_fi     : same KS at the back node
    ks_wrist_fi    : same KS at the wrist node
    mean_dwell_fog : mean duration of FoG runs in seconds; nan if none found

    Each per-node KS is nan if that node has fewer than 10 synthetic or real
    Shuffling samples.
    """
    all_states = np.concatenate([e.states for e in episodes])

    # FoG fraction
    fog_fraction = float(np.mean(all_states > 0))

    # KS distance per node: synthetic Shuffling FI vs real Shuffling FI
    ks = {}
    for n in range(N_NODES):
        synth_fi      = np.concatenate([e.obs[:, n, FI] for e in episodes])
        synth_shuf_fi = synth_fi[all_states == SHUFFLING]
        real_fi       = real_shuf_fi.get(n, np.array([]))
        if len(synth_shuf_fi) > 10 and len(real_fi) > 10:
            ks[n] = float(ks_2samp(synth_shuf_fi, real_fi).statistic)
        else:
            ks[n] = float('nan')

    # Mean FoG run duration in seconds
    fog_run_lengths = []
    for ep in episodes:
        s      = ep.states
        in_fog = False
        run    = 0
        for state in s:
            if state > 0:
                in_fog = True
                run   += 1
            else:
                if in_fog and run > 0:
                    fog_run_lengths.append(run / FS)
                in_fog = False
                run    = 0
        if in_fog and run > 0:
            fog_run_lengths.append(run / FS)

    mean_dwell_fog = float(np.mean(fog_run_lengths)) if fog_run_lengths else float('nan')

    return dict(
        fog_fraction   = fog_fraction,
        ks_ankle_fi    = ks[ANKLE],
        ks_back_fi     = ks[BACK],
        ks_wrist_fi    = ks[WRIST],
        mean_dwell_fog = mean_dwell_fog,
    )


# ── figure ────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    'fog_fraction':   'FoG fraction',
    'ks_ankle_fi':    'KS distance\n(Shuffling ankle FI vs real)',
    'mean_dwell_fog': 'Mean FoG run duration (s)',
}

METRIC_COLOURS = {
    'fog_fraction':   '#4C72B0',
    'ks_ankle_fi':    '#C44E52',
    'mean_dwell_fog': '#55A868',
}


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))


def fig_sensitivity(x_vals:    List[float],
                     x_label:   str,
                     x_default: float,
                     results:   List[Dict[str, float]],
                     title:     str,
                     outpath:   Path):
    """
    Three-panel figure: FoG fraction, KS distance, and mean FoG run duration
    against the swept knob value. Vertical dashed line marks the fitted default.

    The KS panel overlays all three nodes. Ankle is the headline (bold solid
    line, larger markers); back and wrist are drawn alongside as secondary
    dashed lines so the reader can see whether the trend holds across nodes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=12, y=1.01)

    # ── Panel 1: FoG fraction ──────────────────────────────────────────────
    ax = axes[0]
    y  = [r['fog_fraction'] for r in results]
    ax.plot(x_vals, y, 'o-', color=METRIC_COLOURS['fog_fraction'],
            linewidth=1.8, markersize=7)
    ax.axvline(x_default, color='#888888', linewidth=1.1, linestyle='--',
               label=f'Fitted default\n({x_default})')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(METRIC_LABELS['fog_fraction'], fontsize=10)
    ax.legend(fontsize=8)
    _despine(ax)

    # ── Panel 2: KS distance per node (ankle headline + back/wrist) ────────
    ax = axes[1]
    node_styles = [
        ('ks_ankle_fi', 'ankle (headline)', '#C44E52', 2.0, 'o-',  8),
        ('ks_back_fi',  'back',             '#8172B3', 1.2, 's--', 5),
        ('ks_wrist_fi', 'wrist',            '#937860', 1.2, '^--', 5),
    ]
    for key, lbl, col, lw, style, ms in node_styles:
        y = [r[key] for r in results]
        ax.plot(x_vals, y, style, color=col, linewidth=lw, markersize=ms,
                label=lbl)
    ax.axvline(x_default, color='#888888', linewidth=1.1, linestyle='--')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('KS distance\n(Shuffling FI vs real)', fontsize=10)
    ax.legend(fontsize=7, title='node', title_fontsize=7)
    _despine(ax)

    # ── Panel 3: mean FoG run duration ─────────────────────────────────────
    ax = axes[2]
    y  = [r['mean_dwell_fog'] for r in results]
    ax.plot(x_vals, y, 'o-', color=METRIC_COLOURS['mean_dwell_fog'],
            linewidth=1.8, markersize=7)
    ax.axvline(x_default, color='#888888', linewidth=1.1, linestyle='--',
               label=f'Fitted default\n({x_default})')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(METRIC_LABELS['mean_dwell_fog'], fontsize=10)
    ax.legend(fontsize=8)
    _despine(ax)

    # Mark NaN knob values on the single-metric panels
    for ax, key in ((axes[0], 'fog_fraction'), (axes[2], 'mean_dwell_fog')):
        yv    = [r[key] for r in results]
        nan_x = [x for x, y in zip(x_vals, yv) if np.isnan(y)]
        for nx in nan_x:
            ax.axvline(nx, color='#DDDDDD', linewidth=2, zorder=0)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── table writer ──────────────────────────────────────────────────────────────

def write_sensitivity_table(x_vals:    List[float],
                              x_col:     str,
                              results:   List[Dict[str, float]],
                              outpath:   Path):
    def _fmt(v, nd=4):
        return f'{v:.{nd}f}' if not np.isnan(v) else 'nan'

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([x_col, 'fog_fraction',
                         'ks_ankle_fi', 'ks_back_fi', 'ks_wrist_fi',
                         'mean_dwell_fog_s'])
        for x, r in zip(x_vals, results):
            writer.writerow([
                f'{x}',
                f'{r["fog_fraction"]:.4f}',
                _fmt(r['ks_ankle_fi']),
                _fmt(r['ks_back_fi']),
                _fmt(r['ks_wrist_fi']),
                _fmt(r['mean_dwell_fog'], 3),
            ])
    print(f"  Saved {outpath}")


# ── fitted default estimation ─────────────────────────────────────────────────

def estimate_fitted_fog_fraction(params: HMMParams) -> float:
    fog_states  = [SHUFFLING, TREMBLING, AKINESIA]
    T_mat       = params.T_mat
    weights     = np.array([T_mat[HEALTHY, s] for s in fog_states])
    fog_mu_secs = np.array([params.dwell[s].mu_sec for s in fog_states])
    w_sum       = weights.sum()
    mu_fog      = float(np.dot(weights, fog_mu_secs) / w_sum) if w_sum > 0 else 10.0
    mu_h        = params.dwell[HEALTHY].mu_sec
    return mu_fog / (mu_h + mu_fog)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sensitivity studies for FoG HMM generator (Ch 8)')
    parser.add_argument('--processed',   default='fogstar_processed.pkl')
    parser.add_argument('--params',      default='fogstar_model_params.pkl')
    parser.add_argument('--n-episodes',  type=int,   default=100)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--outdir',      default='validation')
    args = parser.parse_args()

    fig_dir = Path(args.outdir) / 'figures'
    tbl_dir = Path(args.outdir) / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading params from {args.params} ...")
    params = load_params(args.params)

    print(f"Loading real Shuffling FI per node from {args.processed} ...")
    real_shuf_fi = load_real_shuffling_fi(args.processed)
    for n in range(N_NODES):
        print(f"  Real Shuffling {NODE_NAMES[n]} FI samples: {len(real_shuf_fi[n])}")

    fitted_fog_fraction = estimate_fitted_fog_fraction(params)
    fitted_akinesia_dwell = params.dwell[AKINESIA].mu_sec
    fitted_shuffling_rho  = params.rho[SHUFFLING]

    print(f"\n  Fitted defaults:")
    print(f"    FoG fraction:       {fitted_fog_fraction*100:.1f}%")
    print(f"    Akinesia dwell:     {fitted_akinesia_dwell:.2f}s")
    print(f"    Shuffling rho:      {fitted_shuffling_rho:.3f}")

    # ── Study 1: FoG burden ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STUDY 1: FoG burden sweep")
    print(f"  Levels: {[f'{b*100:.0f}%' for b in BURDEN_LEVELS]}")
    print(f"  Episodes per level: {args.n_episodes}")

    results_s1 = []
    for i, burden in enumerate(BURDEN_LEVELS):
        mu_h    = burden_to_healthy_dwell(params, burden)
        mutated = mutate_healthy_dwell(params, mu_h)
        print(f"\n  Burden {burden*100:.0f}%: Healthy dwell={mu_h:.1f}s "
              f"(fitted={params.dwell[HEALTHY].mu_sec:.1f}s)")
        eps     = generate_episodes(mutated, args.n_episodes,
                                     seed=args.seed + i)
        metrics = compute_metrics(eps, real_shuf_fi)
        results_s1.append(metrics)
        print(f"    fog_fraction={metrics['fog_fraction']:.3f}  "
              f"ks_fi(ankle/back/wrist)={metrics['ks_ankle_fi']:.3f}/"
              f"{metrics['ks_back_fi']:.3f}/{metrics['ks_wrist_fi']:.3f}  "
              f"mean_dwell_fog={metrics['mean_dwell_fog']:.2f}s")

    fig_sensitivity(
        x_vals    = [b * 100 for b in BURDEN_LEVELS],
        x_label   = 'FoG Burden (%)',
        x_default = fitted_fog_fraction * 100,
        results   = results_s1,
        title     = 'Study 1: Sensitivity to FoG Burden',
        outpath   = fig_dir / 'fig_8_sens_fog_burden.png',
    )
    write_sensitivity_table(
        x_vals  = [b * 100 for b in BURDEN_LEVELS],
        x_col   = 'fog_burden_pct',
        results = results_s1,
        outpath = tbl_dir / 'table_8_sens_fog_burden.csv',
    )

    # ── Study 2: Akinesia mean dwell ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STUDY 2: Akinesia mean dwell sweep")
    print(f"  Levels: {AKINESIA_DWELL_SECS} seconds")
    print(f"  Episodes per level: {args.n_episodes}")

    results_s2 = []
    for i, dwell_s in enumerate(AKINESIA_DWELL_SECS):
        mutated = mutate_akinesia_dwell(params, dwell_s)
        print(f"\n  Akinesia dwell={dwell_s:.0f}s "
              f"(fitted={params.dwell[AKINESIA].mu_sec:.1f}s)")
        eps     = generate_episodes(mutated, args.n_episodes,
                                     seed=args.seed + 100 + i)
        metrics = compute_metrics(eps, real_shuf_fi)
        results_s2.append(metrics)
        print(f"    fog_fraction={metrics['fog_fraction']:.3f}  "
              f"ks_fi(ankle/back/wrist)={metrics['ks_ankle_fi']:.3f}/"
              f"{metrics['ks_back_fi']:.3f}/{metrics['ks_wrist_fi']:.3f}  "
              f"mean_dwell_fog={metrics['mean_dwell_fog']:.2f}s")

    fig_sensitivity(
        x_vals    = AKINESIA_DWELL_SECS,
        x_label   = 'Akinesia Mean Dwell (s)',
        x_default = fitted_akinesia_dwell,
        results   = results_s2,
        title     = 'Study 2: Sensitivity to Akinesia Mean Dwell',
        outpath   = fig_dir / 'fig_8_sens_akinesia_dwell.png',
    )
    write_sensitivity_table(
        x_vals  = AKINESIA_DWELL_SECS,
        x_col   = 'akinesia_dwell_s',
        results = results_s2,
        outpath = tbl_dir / 'table_8_sens_akinesia_dwell.csv',
    )

    # ── Study 3: Shuffling AR(1) rho ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STUDY 3: Shuffling AR(1) rho sweep")
    print(f"  Levels: {SHUFFLING_RHO_VALS}")
    print(f"  Episodes per level: {args.n_episodes}")

    results_s3 = []
    for i, rho in enumerate(SHUFFLING_RHO_VALS):
        mutated = mutate_shuffling_rho(params, rho)
        print(f"\n  Shuffling rho={rho:.2f} "
              f"(fitted cap={params.rho[SHUFFLING]:.3f})")
        eps     = generate_episodes(mutated, args.n_episodes,
                                     seed=args.seed + 200 + i)
        metrics = compute_metrics(eps, real_shuf_fi)
        results_s3.append(metrics)
        print(f"    fog_fraction={metrics['fog_fraction']:.3f}  "
              f"ks_fi(ankle/back/wrist)={metrics['ks_ankle_fi']:.3f}/"
              f"{metrics['ks_back_fi']:.3f}/{metrics['ks_wrist_fi']:.3f}  "
              f"mean_dwell_fog={metrics['mean_dwell_fog']:.2f}s")

    fig_sensitivity(
        x_vals    = SHUFFLING_RHO_VALS,
        x_label   = 'Shuffling AR(1) rho',
        x_default = fitted_shuffling_rho,
        results   = results_s3,
        title     = 'Study 3: Sensitivity to Shuffling AR(1) rho',
        outpath   = fig_dir / 'fig_8_sens_shuffling_rho.png',
    )
    write_sensitivity_table(
        x_vals  = SHUFFLING_RHO_VALS,
        x_col   = 'shuffling_rho',
        results = results_s3,
        outpath = tbl_dir / 'table_8_sens_shuffling_rho.csv',
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SENSITIVITY STUDIES COMPLETE")
    print(f"\nStudy 1 (FoG burden) summary:")
    print(f"  {'Burden':>8}  {'FoG frac':>9}  {'KS ankle':>9}  "
          f"{'KS back':>9}  {'KS wrist':>9}  {'Dwell(s)':>9}")
    for b, r in zip(BURDEN_LEVELS, results_s1):
        print(f"  {b*100:>7.0f}%  {r['fog_fraction']:>9.3f}  "
              f"{r['ks_ankle_fi']:>9.3f}  {r['ks_back_fi']:>9.3f}  "
              f"{r['ks_wrist_fi']:>9.3f}  {r['mean_dwell_fog']:>9.2f}")

    print(f"\nStudy 2 (Akinesia dwell) summary:")
    print(f"  {'Dwell(s)':>8}  {'FoG frac':>9}  {'KS ankle':>9}  "
          f"{'KS back':>9}  {'KS wrist':>9}  {'Dwell(s)':>9}")
    for d, r in zip(AKINESIA_DWELL_SECS, results_s2):
        print(f"  {d:>8.0f}  {r['fog_fraction']:>9.3f}  "
              f"{r['ks_ankle_fi']:>9.3f}  {r['ks_back_fi']:>9.3f}  "
              f"{r['ks_wrist_fi']:>9.3f}  {r['mean_dwell_fog']:>9.2f}")

    print(f"\nStudy 3 (Shuffling rho) summary:")
    print(f"  {'Rho':>8}  {'FoG frac':>9}  {'KS ankle':>9}  "
          f"{'KS back':>9}  {'KS wrist':>9}  {'Dwell(s)':>9}")
    for rho, r in zip(SHUFFLING_RHO_VALS, results_s3):
        print(f"  {rho:>8.2f}  {r['fog_fraction']:>9.3f}  "
              f"{r['ks_ankle_fi']:>9.3f}  {r['ks_back_fi']:>9.3f}  "
              f"{r['ks_wrist_fi']:>9.3f}  {r['mean_dwell_fog']:>9.2f}")

    print(f"\nAll outputs written to: {Path(args.outdir).resolve()}")
    print(f"  Figures : {fig_dir}")
    print(f"  Tables  : {tbl_dir}")


if __name__ == '__main__':
    main()