"""
fog_validate.py
---------------
Statistical validation of the calibrated semi-Markov HMM generator against
real FoG-STAR data. Produces every figure and table required for Chapter 6
of the thesis ("A Calibrated Semi-Markov HMM Generator for Synthetic
Freezing of Gait Sensor Data").

Outputs written to <outdir>/figures and <outdir>/tables.

Figures
    fig_6_1_marginals_ankle.png       Marginal feature distributions at ankle (AR(1) output)
    fig_6_2_dwell_distributions.png   Per-state dwell time CDFs
    fig_6_3_acf_ankle_fi.png          ACF of ankle FI per state
    fig_6_4_pca_ankle.png             2D PCA projection at ankle, per state
    fig_6_5_internode_correlation.png Inter-node correlation per feature
    fig_6_6_episode_summary.png       Episode-level summary statistics
    fig_6_x_iid_emission_ankle.png    Emission model validation: iid N(mu,Sigma) vs real
    fig_6_y_mahalanobis_ankle.png     Mahalanobis distance distributions at ankle
    fig_a_1_marginals_back.png        Appendix: marginals at back node (AR(1) output)
    fig_a_2_marginals_wrist.png       Appendix: marginals at wrist node (AR(1) output)
    fig_a_3_iid_emission_back.png     Appendix: iid emission validation at back node
    fig_a_4_iid_emission_wrist.png    Appendix: iid emission validation at wrist node
    fig_a_7_acf_back_fi.png           Appendix: ACF of back FI per state
    fig_a_8_acf_wrist_fi.png          Appendix: ACF of wrist FI per state
    fig_a_9_pca_back.png              Appendix: 2D PCA projection at back, per state
    fig_a_10_pca_wrist.png            Appendix: 2D PCA projection at wrist, per state

Tables (CSV)
    table_6_1_ks_summary.csv          KS distance, averaged across nodes
    table_6_1_ks_per_node.csv         KS distance per (state, node, feature)
    table_6_2_dwell_stats.csv         Dwell time statistics
    table_6_3_ar1.csv                 Fitted vs empirical AR(1) coefficient

Usage
    python fog_validate.py [--processed fogstar_processed.pkl]
                           [--params    fogstar_model_params.pkl]
                           [--gmm]
                           [--n-synth   100]
                           [--seed      0]
                           [--outdir    validation]

    Pass --gmm to use the GMM emission model instead of the single Gaussian.
    This loads fogstar_model_params_gmm.pkl by default and uses
    FogEpisodeGeneratorGMM. The Mahalanobis KS is computed from the pooled
    GMM covariance (law of total variance across components).
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from fog_star_loader import (
    FS, N_STATES, N_NODES, N_FEATURES,
    STATE_NAMES, NODE_NAMES, FEATURE_NAMES,
    BOUNDARY_PAD,
    ParticipantData, DatasetStats,
    collect_features_by_state,
)
from fog_model_fitter import (
    HMMParams, DwellDistribution, EmissionParams, PreprocessParams,
    preprocess_features, clip_pooled_features,
)
from fog_episode_generator import FogEpisodeGenerator, Episode

# GMM variants -- imported lazily so the script still runs without gmm pickle
try:
    from fog_episode_generator_gmm import FogEpisodeGeneratorGMM
    from fog_model_fitter_gmm import GMMHMMParams, GMMEmissionParams
    _GMM_AVAILABLE = True
except ImportError:
    _GMM_AVAILABLE = False

# Node indices used in figures
BACK, WRIST, ANKLE = 0, 1, 2

# State indices
HEALTHY, SHUFFLING, TREMBLING, AKINESIA = 0, 1, 2, 3


# ── Data loading ──────────────────────────────────────────────────────────────

def load_real_data(processed_path: str):
    """
    Returns (processed_participants, stats_obj, raw_participants).

    processed_participants are in z-score space (log-then-z per participant)
    and are the right object to compare against the generator's outputs.
    raw_participants are kept for episode-level statistics that do not
    care about feature scale.
    """
    with open(processed_path, 'rb') as f:
        data = pickle.load(f)
    raw_participants = data['participants']
    stats_obj        = data['stats']
    processed, _     = preprocess_features(raw_participants)
    return processed, stats_obj, raw_participants


def sample_synthetic_episodes(gen: FogEpisodeGenerator, n: int):
    eps = [gen.sample_episode() for _ in range(n)]
    total = sum(e.T for e in eps)
    print(f"Sampled {n} synthetic episodes ({total:,} timesteps)")
    return eps


def pool_features_by_state(processed_participants, episodes):
    """
    Build per-state, per-node feature pools for real and synthetic data,
    applying identical filtering so the two distributions are comparable.

    Real pool
    ---------
    collect_features_by_state already applies valid_mask, which enforces:
      - locomotor activity only (activity in {1, 6, 7})
      - BOUNDARY_PAD=20 timesteps excluded either side of state transitions
      - no NaN features
      - states >= 0
    The pooled data are then clipped at the 99th percentile per feature,
    matching the clip applied before emission-model fitting.

    Synthetic pool
    --------------
    The same boundary-pad exclusion is applied to each dwell segment:
    the first and last BOUNDARY_PAD timesteps of every dwell are dropped.
    This removes AR(1) transients that would not appear in the real pool.
    The same 99th-percentile clip is then applied.

    real_pool[s][n]  : (N_real,  4)  z-score, clipped
    synth_pool[s][n] : (N_synth, 4)  z-score, clipped
    """
    # Real pool: valid_mask already applied inside collect_features_by_state
    real_pool = collect_features_by_state(processed_participants)
    real_pool = clip_pooled_features(real_pool, percentile=99.0)

    # Synthetic pool: apply boundary-pad exclusion per dwell, then clip
    synth_pool = {s: {n: [] for n in range(N_NODES)} for s in range(N_STATES)}
    for ep in episodes:
        for state, t0, t1 in ep.dwell_info:
            dwell_len = t1 - t0
            # Drop first and last BOUNDARY_PAD timesteps of each dwell,
            # matching the real valid_mask boundary exclusion exactly.
            pad  = min(BOUNDARY_PAD, dwell_len // 3)   # guard very short dwells
            t_lo = t0 + pad
            t_hi = t1 - pad
            if t_hi <= t_lo:
                continue                                 # dwell too short
            feats = ep.obs[t_lo:t_hi]                   # (M, 3, 4)
            s     = int(state)
            for n in range(N_NODES):
                synth_pool[s][n].append(feats[:, n, :])

    for s in range(N_STATES):
        for n in range(N_NODES):
            if synth_pool[s][n]:
                synth_pool[s][n] = np.concatenate(synth_pool[s][n], axis=0)
            else:
                synth_pool[s][n] = np.empty((0, N_FEATURES))
    synth_pool = clip_pooled_features(synth_pool, percentile=99.0)

    # Print pool size summary
    print("  Pool sizes after filtering and clipping:")
    for s in range(N_STATES):
        r_counts  = [real_pool[s][n].shape[0]  for n in range(N_NODES)]
        sy_counts = [synth_pool[s][n].shape[0] for n in range(N_NODES)]
        print(f"    {STATE_NAMES[s]:12s}  real={r_counts}  synth={sy_counts}")

    return real_pool, synth_pool


def extract_runs_real(participants, node: int, feature: int):
    """
    For each state, return a list of contiguous valid-timestep runs as 1D
    feature arrays. Used by ACF and AR(1) computations on real data.
    """
    runs = {s: [] for s in range(N_STATES)}
    for p in participants:
        states = p.states
        vm     = p.valid_mask
        col    = p.features[:, node, feature]
        T      = len(states)
        t      = 0
        while t < T:
            s = states[t]
            if not vm[t] or s < 0:
                t += 1
                continue
            t_end = t
            while t_end < T and states[t_end] == s and vm[t_end]:
                t_end += 1
            if t_end - t >= 5:
                run = col[t:t_end]
                run = run[~np.isnan(run)]
                if len(run) >= 5:
                    runs[int(s)].append(run)
            t = t_end
    return runs


def extract_runs_synth(episodes, node: int, feature: int):
    """Same shape as extract_runs_real but for synthetic episodes."""
    runs = {s: [] for s in range(N_STATES)}
    for ep in episodes:
        states = ep.states
        col    = ep.obs[:, node, feature]
        T      = len(states)
        t      = 0
        while t < T:
            s = int(states[t])
            t_end = t
            while t_end < T and int(states[t_end]) == s:
                t_end += 1
            if t_end - t >= 5:
                runs[s].append(col[t:t_end])
            t = t_end
    return runs


def compute_synth_dwells(episodes):
    """Return dict {state: [durations_seconds]} from synthetic episodes."""
    out = {s: [] for s in range(N_STATES)}
    for ep in episodes:
        for state, t0, t1 in ep.dwell_info:
            out[int(state)].append((t1 - t0) / FS)
    return out


# ── Mahalanobis KS helper ─────────────────────────────────────────────────────

def get_sigma_inv(emission_params) -> np.ndarray:
    """
    Return a (4,4) inverse covariance matrix from either an EmissionParams
    (single Gaussian) or a GMMEmissionParams (mixture of Gaussians).

    For a single Gaussian, Sigma_inv is stored directly.

    For a GMM with K components, the total covariance is computed via the
    law of total variance:

        Sigma_total = sum_k w_k * (Sigma_k + mu_k mu_k^T) - mu_bar mu_bar^T

    where mu_bar = sum_k w_k * mu_k is the mixture mean.  This is the
    correct pooled covariance that accounts for both within-component spread
    and between-component separation.
    """
    # Single Gaussian: Sigma_inv is stored directly
    if hasattr(emission_params, 'Sigma_inv'):
        return emission_params.Sigma_inv

    # GMM: compute pooled covariance via law of total variance
    weights     = emission_params.weights      # (K,)
    means       = emission_params.means        # (K, 4)
    covariances = emission_params.covariances  # (K, 4, 4)

    mu_bar = np.einsum('k,kf->f', weights, means)   # (4,)

    Sigma_total = np.zeros((N_FEATURES, N_FEATURES))
    for k in range(emission_params.n_components):
        diff = means[k] - mu_bar                     # (4,)
        Sigma_total += weights[k] * (
            covariances[k] + np.outer(diff, diff)
        )

    # Regularise slightly to ensure invertibility
    Sigma_total += 1e-6 * np.eye(N_FEATURES)
    return np.linalg.inv(Sigma_total)


def energy_distance_4d(real_4d: np.ndarray,
                       params_emission,
                       rng: np.random.Generator,
                       n_samples: int = 20000,
                       subsample: int = 2000) -> float:
    """
    Multivariate Energy Distance between the real 4D feature distribution
    and iid draws from the fitted emission Gaussian N(mu, Sigma).

    Energy Distance = 2*E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]

    This is a proper multivariate two-sample statistic that uses the full
    4x4 covariance through the sampling step. It does not require any
    projection or inversion and is not bounded to [0,1], but values close
    to zero indicate good fit.

    Returns nan if either pool has fewer than 10 samples.
    """
    from scipy.spatial.distance import cdist
    xr = real_4d[~np.any(np.isnan(real_4d), axis=1)]
    if len(xr) < 10:
        return float('nan')

    # iid draws from N(mu, Sigma)
    if hasattr(params_emission, 'mu'):
        mu    = params_emission.mu
        Sigma = params_emission.Sigma
    else:
        # GMM: use weighted mean and pooled covariance
        mu    = np.einsum('k,kf->f', params_emission.weights, params_emission.means)
        Sigma = np.zeros((N_FEATURES, N_FEATURES))
        for k in range(params_emission.n_components):
            diff = params_emission.means[k] - mu
            Sigma += params_emission.weights[k] * (
                params_emission.covariances[k] + np.outer(diff, diff))

    xii = rng.multivariate_normal(mu, Sigma, size=n_samples)

    # Subsample for speed
    nx = min(len(xr),  subsample)
    ny = min(len(xii), subsample)
    xi = xr[ rng.integers(0, len(xr),  nx)]
    yi = xii[rng.integers(0, len(xii), ny)]

    dxy = cdist(xi, yi).mean()
    dxx = cdist(xi, xi).mean()
    dyy = cdist(yi, yi).mean()
    return float(2*dxy - dxx - dyy)


def mahalanobis_ks(real_4d: np.ndarray,
                   synth_4d: np.ndarray,
                   Sigma_inv: np.ndarray) -> float:
    """
    Project real and synthetic 4D feature vectors onto the Mahalanobis distance
    from the fitted emission mean and run a 2-sample KS test on those scalars.

    This leverages the full 4x4 covariance structure rather than testing each
    feature dimension independently.  The Mahalanobis distance of a point x
    with inverse covariance Sigma_inv is:

        d(x) = sqrt( x^T  Sigma_inv  x )

    Both pools are in the same z-score space the emission model was fitted in.
    The mean offset is captured by the per-feature marginal KS; the covariance
    structure is captured here.

    Returns the KS statistic, or nan if either pool has fewer than 10 samples.
    """
    r = real_4d[~np.any(np.isnan(real_4d), axis=1)]
    s = synth_4d[~np.any(np.isnan(synth_4d), axis=1)]
    if len(r) < 10 or len(s) < 10:
        return float("nan")
    dr = np.sqrt(np.einsum("ni,ij,nj->n", r, Sigma_inv, r))
    ds = np.sqrt(np.einsum("ni,ij,nj->n", s, Sigma_inv, s))
    return float(ks_2samp(dr, ds).statistic)


# ── Figure 6.1, A.1, A.2: marginals ───────────────────────────────────────────

def fig_marginals(real_pool, synth_pool, node: int, outpath: Path,
                  params: HMMParams = None):
    """
    4xN_FEATURES grid of marginal histograms comparing real vs synthetic
    feature distributions in z-score space. Both pools have BOUNDARY_PAD
    exclusion and 99th-percentile clipping applied before this function.

    Each subplot title shows the univariate KS statistic for that feature.
    The state-row label also shows the Mahalanobis KS (Mah-KS) when params
    is provided, which tests distributional match using the full 4x4
    fitted covariance Sigma_inv rather than collapsing to 1D marginals.
    """
    fig, axes = plt.subplots(N_STATES, N_FEATURES, figsize=(13, 11))

    for si in range(N_STATES):
        # Mahalanobis KS: uses full Sigma_inv fitted on this (state, node)
        mah_ks = float('nan')
        if params is not None:
            Sigma_inv = get_sigma_inv(params.emission[si][node])
            mah_ks = mahalanobis_ks(
                real_pool[si][node],
                synth_pool[si][node],
                Sigma_inv,
            )

        for fi in range(N_FEATURES):
            ax = axes[si, fi]
            xr = real_pool[si][node][:, fi]
            xs = synth_pool[si][node][:, fi]
            xr = xr[~np.isnan(xr)]
            xs = xs[~np.isnan(xs)]

            if len(xr) < 10 or len(xs) < 10:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.set_title(f"{STATE_NAMES[si]} / {FEATURE_NAMES[fi]}",
                             fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            lo = min(np.percentile(xr, 1), np.percentile(xs, 1))
            hi = max(np.percentile(xr, 99), np.percentile(xs, 99))
            bins = np.linspace(lo, hi, 40)

            ax.hist(xr, bins=bins, alpha=0.55, color='steelblue',
                    density=True, label='real')
            ax.hist(xs, bins=bins, histtype='step', color='crimson',
                    linewidth=1.4, density=True, label='synth')

            ks_uni = ks_2samp(xr, xs).statistic
            # Show Mahalanobis KS in the leftmost column title for this state
            if fi == 0 and not np.isnan(mah_ks):
                title = (f"{STATE_NAMES[si]}  [Mah-KS={mah_ks:.3f}]\n"
                         f"{FEATURE_NAMES[fi]}  KS={ks_uni:.3f}")
            else:
                title = f"{STATE_NAMES[si]} / {FEATURE_NAMES[fi]}\nKS={ks_uni:.3f}"
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)

    axes[0, 0].legend(loc='upper right', fontsize=7)
    mah_note = ("  |  Mah-KS = Mahalanobis-projected KS (full 4x4 covariance)"
                if params else "")
    fig.suptitle(
        f"Marginal feature distributions at the {NODE_NAMES[node]} node\n"
        f"real vs synthetic (z-score space, boundary-padded){mah_note}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.X: iid emission comparison ──────────────────────────────────────

def fig_iid_emission(real_pool, params, node: int, outpath: Path,
                     n_samples: int = 50000):
    """
    Compare real pooled z-scores against iid draws directly from the fitted
    emission Gaussian N(mu_{s,n}, Sigma_{s,n}), WITHOUT passing through the
    AR(1) filter.

    This isolates the emission model from the AR(1) temporal structure and
    shows whether the fitted Gaussian correctly captures the marginal feature
    distribution of the real data. The AR(1) process intentionally concentrates
    synthetic output near the running state mean (visible in the ACF figure),
    which is why the AR(1) output histograms appear narrower than the real data.
    The iid draws here show what the emission model alone produces.

    Outputs a 4xN_FEATURES grid identical in layout to fig_marginals.
    """
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(N_STATES, N_FEATURES, figsize=(13, 11))

    for si in range(N_STATES):
        # Draw n_samples iid from N(mu, Sigma) for this (state, node)
        ep = params.emission[si][node]
        iid_samples = rng.multivariate_normal(ep.mu, ep.Sigma, size=n_samples)

        for fi in range(N_FEATURES):
            ax = axes[si, fi]
            xr  = real_pool[si][node][:, fi]
            xii = iid_samples[:, fi]
            xr  = xr[~np.isnan(xr)]

            if len(xr) < 10:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.set_title(f"{STATE_NAMES[si]} / {FEATURE_NAMES[fi]}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            lo = min(np.percentile(xr, 1),  np.percentile(xii, 1))
            hi = max(np.percentile(xr, 99), np.percentile(xii, 99))
            bins = np.linspace(lo, hi, 40)

            ax.hist(xr,  bins=bins, alpha=0.55, color='steelblue',
                    density=True, label='real')
            ax.hist(xii, bins=bins, histtype='step', color='darkorange',
                    linewidth=1.4, density=True, label='iid emission')

            ks = ks_2samp(xr, xii).statistic
            ax.set_title(f"{STATE_NAMES[si]} / {FEATURE_NAMES[fi]}\nKS={ks:.3f}",
                         fontsize=8)
            ax.tick_params(labelsize=7)

    axes[0, 0].legend(loc='upper right', fontsize=7)
    fig.suptitle(
        f"Emission model validation at the {NODE_NAMES[node]} node\n"
        f"real (blue) vs iid draws from N(mu, Sigma) without AR(1) (orange)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.Y: Mahalanobis distance distributions ────────────────────────────

def fig_mahalanobis_distance(real_pool, params, node: int, outpath: Path,
                              n_samples: int = 50000):
    """
    For each state, project real and iid-emission samples onto the scalar
    Mahalanobis distance d(x) = sqrt(x^T Sigma_inv x) using the fitted
    inverse covariance Sigma_inv_{s,n}.

    This collapses the full 4D covariance structure to a single 1D scalar
    and provides a direct visualisation of what the Mahalanobis KS statistic
    is measuring. If the two histograms overlap well, the fitted covariance
    captures the shape of the real 4D distribution.

    Layout: 1x4 row of subplots, one per state.
    Real data in blue, iid emission draws in orange.
    """
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, N_STATES, figsize=(14, 4))

    for si in range(N_STATES):
        ax  = axes[si]
        ep  = params.emission[si][node]

        # Real data: 4D vectors from real pool
        xr  = real_pool[si][node]
        xr  = xr[~np.any(np.isnan(xr), axis=1)]

        # iid emission: 50k draws from N(mu, Sigma)
        xii = rng.multivariate_normal(ep.mu, ep.Sigma, size=n_samples)

        if len(xr) < 10:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.set_title(STATE_NAMES[si], fontsize=9)
            continue

        # Compute Sigma_inv from params
        Sigma_inv = ep.Sigma_inv

        # Mahalanobis distance for each sample
        dr  = np.sqrt(np.einsum('ni,ij,nj->n', xr,  Sigma_inv, xr))
        dii = np.sqrt(np.einsum('ni,ij,nj->n', xii, Sigma_inv, xii))

        # KS statistic on the distance distributions
        ks = ks_2samp(dr, dii).statistic

        lo   = 0.0
        hi   = max(np.percentile(dr, 99), np.percentile(dii, 99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(dr,  bins=bins, density=True, alpha=0.55,
                color='steelblue', label='real')
        ax.hist(dii, bins=bins, density=True, histtype='step',
                linewidth=1.6, color='darkorange', label='iid emission')

        ax.set_title(f"{STATE_NAMES[si]}\nKS={ks:.3f}", fontsize=9)
        ax.set_xlabel("Mahalanobis distance d(x)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("density", fontsize=8)
    axes[0].legend(fontsize=7, loc='upper right')

    fig.suptitle(
        f"Mahalanobis distance distributions at the {NODE_NAMES[node]} node\n"
        f"d(x) = sqrt(xᵀ Σ⁻¹ x)  |  real (blue) vs iid N(mu, Σ) (orange)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.2: dwell distributions ───────────────────────────────────────────

def fig_dwell_distributions(real_dwells, synth_dwells, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for si in range(N_STATES):
        ax = axes[si]
        r = np.array(real_dwells.get(si, []), dtype=float)
        s = np.array(synth_dwells.get(si, []), dtype=float)
        r = r[r > 0]; s = s[s > 0]

        if len(r) < 2 or len(s) < 2:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(STATE_NAMES[si], fontsize=10)
            continue

        rs = np.sort(r); ss = np.sort(s)
        rc = np.arange(1, len(rs) + 1) / len(rs)
        sc = np.arange(1, len(ss) + 1) / len(ss)

        ax.step(rs, rc, where='post', color='steelblue', linewidth=1.6,
                label=f'real  n={len(rs)}')
        ax.step(ss, sc, where='post', color='crimson', linewidth=1.6,
                label=f'synth n={len(ss)}')
        ax.set_xscale('log')
        ax.set_xlabel('dwell duration (s)')
        ax.set_ylabel('empirical CDF')

        ks = ks_2samp(r, s).statistic
        ax.set_title(f"{STATE_NAMES[si]}    real mean {r.mean():.1f}s, "
                     f"synth mean {s.mean():.1f}s, KS={ks:.3f}", fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

    fig.suptitle("Dwell time distributions per state", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.3: ACF ───────────────────────────────────────────────────────────

def acf_weighted_average(runs, max_lag: int = 60):
    """Length-weighted mean ACF across a list of 1D arrays."""
    if not runs:
        return np.zeros(max_lag + 1)
    total_w = 0.0
    acf_sum = np.zeros(max_lag + 1)
    for run in runs:
        run = np.asarray(run, dtype=float)
        run = run[~np.isnan(run)]
        if len(run) < max_lag + 5:
            continue
        x = run - run.mean()
        v = (x ** 2).sum()
        if v < 1e-12:
            continue
        acf = np.empty(max_lag + 1)
        for lag in range(max_lag + 1):
            acf[lag] = (x[:len(x) - lag] * x[lag:]).sum() / v
        w = len(run)
        acf_sum += acf * w
        total_w += w
    return acf_sum / max(total_w, 1)


def fig_acf(real_runs, synth_runs, node: int, feature: int, outpath: Path):
    max_lag = 60
    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for si in range(N_STATES):
        ax = axes[si]
        r_acf = acf_weighted_average(real_runs[si],  max_lag)
        s_acf = acf_weighted_average(synth_runs[si], max_lag)

        ax.plot(lags, r_acf, color='steelblue', linewidth=1.6, label='real')
        ax.plot(lags, s_acf, color='crimson',  linewidth=1.6, label='synth')
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xlabel('lag (timesteps at 60 Hz)')
        ax.set_ylabel('ACF')
        ax.set_title(STATE_NAMES[si], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Autocorrelation of {FEATURE_NAMES[feature]} at the "
                 f"{NODE_NAMES[node]} node", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.4: PCA ───────────────────────────────────────────────────────────

def fig_pca(real_pool, synth_pool, node: int, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for si in range(N_STATES):
        ax = axes[si]
        Xr = real_pool[si][node]
        Xs = synth_pool[si][node]
        Xr = Xr[~np.isnan(Xr).any(axis=1)]
        Xs = Xs[~np.isnan(Xs).any(axis=1)]

        if len(Xr) < 10 or len(Xs) < 10:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(STATE_NAMES[si], fontsize=10)
            continue

        mu = Xr.mean(axis=0)
        Xc = Xr - mu
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Vt[:2]    # (2, 4)

        Yr = Xc @ proj.T
        Ys = (Xs - mu) @ proj.T

        if len(Yr) > 3000:
            Yr = Yr[rng.choice(len(Yr), 3000, replace=False)]
        if len(Ys) > 3000:
            Ys = Ys[rng.choice(len(Ys), 3000, replace=False)]

        ax.scatter(Yr[:, 0], Yr[:, 1], s=4, alpha=0.25, color='steelblue',
                   label='real')
        ax.scatter(Ys[:, 0], Ys[:, 1], s=4, alpha=0.25, color='crimson',
                   label='synth')
        var_pct = (S[:2] ** 2).sum() / (S ** 2).sum() * 100
        ax.set_title(f"{STATE_NAMES[si]}    PC1+PC2 = {var_pct:.0f}% real var",
                     fontsize=9)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(alpha=0.3)

    fig.suptitle(f"2D PCA projection of feature space at the {NODE_NAMES[node]} node\n"
                 f"PCA fit on real, both projected into the same axes",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.5: inter-node correlation ────────────────────────────────────────

def build_internode_matrix_real(participants, feature: int) -> np.ndarray:
    """
    Stack per-participant (T_valid, N_NODES) blocks for one feature.
    Drops rows where any node is NaN, preserving cross-node alignment
    per timestep.
    """
    blocks = []
    for p in participants:
        vm    = p.valid_mask
        block = p.features[vm, :, feature]   # (T_valid, 3)
        keep  = ~np.isnan(block).any(axis=1)
        blocks.append(block[keep])
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, N_NODES))


def build_internode_matrix_synth(episodes, feature: int) -> np.ndarray:
    blocks = [ep.obs[:, :, feature] for ep in episodes]
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, N_NODES))


def fig_internode_correlation(processed_participants, episodes, outpath: Path):
    fig, axes = plt.subplots(N_FEATURES, 2, figsize=(8, 14))
    for fi in range(N_FEATURES):
        Mr = build_internode_matrix_real(processed_participants, fi)
        Ms = build_internode_matrix_synth(episodes, fi)
        if Mr.shape[0] < 10 or Ms.shape[0] < 10:
            for ax in axes[fi]:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes)
            continue
        Cr = np.corrcoef(Mr.T)
        Cs = np.corrcoef(Ms.T)

        for ax, C, label in [(axes[fi, 0], Cr, 'real'),
                             (axes[fi, 1], Cs, 'synth')]:
            ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(N_NODES))
            ax.set_xticklabels(NODE_NAMES, fontsize=8)
            ax.set_yticks(range(N_NODES))
            ax.set_yticklabels(NODE_NAMES, fontsize=8)
            ax.set_title(f"{FEATURE_NAMES[fi]}    {label}", fontsize=9)
            for i in range(N_NODES):
                for j in range(N_NODES):
                    ax.text(j, i, f"{C[i, j]:.2f}", ha='center', va='center',
                            color='white' if abs(C[i, j]) > 0.5 else 'black',
                            fontsize=8)

    fig.suptitle("Inter-node correlation per feature\n"
                 "real vs synthetic, pooled over all valid timesteps",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Figure 6.6: episode-level summary ─────────────────────────────────────────

def episode_summary_stats(states_per_episode):
    fog_fraction, n_fog_eps, mean_dur = [], [], []
    pct_S, pct_T, pct_A = [], [], []

    for states in states_per_episode:
        states = np.asarray(states)
        T = len(states)
        if T == 0:
            continue
        fog_mask = states > 0
        fog_fraction.append(fog_mask.sum() / T)

        in_fog, n_eps, durs, start = False, 0, [], 0
        for t in range(T):
            if states[t] > 0 and not in_fog:
                in_fog, n_eps, start = True, n_eps + 1, t
            elif states[t] == 0 and in_fog:
                in_fog = False
                durs.append((t - start) / FS)
        if in_fog:
            durs.append((T - start) / FS)
        n_fog_eps.append(n_eps)
        if durs:
            mean_dur.append(float(np.mean(durs)))

        n_fog = fog_mask.sum()
        if n_fog > 0:
            pct_S.append((states == 1).sum() / n_fog * 100)
            pct_T.append((states == 2).sum() / n_fog * 100)
            pct_A.append((states == 3).sum() / n_fog * 100)

    # Pooled timestep counts across all episodes/participants
    # Used for within-FoG composition bar chart so that participants
    # with more FoG timesteps contribute proportionally, matching
    # how the transition matrix was estimated.
    all_states = np.concatenate([np.asarray(s) for s in states_per_episode
                                 if len(s) > 0]) if states_per_episode else np.array([])
    fog_ts     = all_states[all_states > 0] if len(all_states) > 0 else np.array([])
    n_fog_tot  = len(fog_ts)
    pool_S = float((fog_ts == 1).sum() / n_fog_tot * 100) if n_fog_tot > 0 else 0.0
    pool_T = float((fog_ts == 2).sum() / n_fog_tot * 100) if n_fog_tot > 0 else 0.0
    pool_A = float((fog_ts == 3).sum() / n_fog_tot * 100) if n_fog_tot > 0 else 0.0

    return {
        'fog_fraction': np.array(fog_fraction),
        'n_fog_eps':    np.array(n_fog_eps),
        'mean_dur':     np.array(mean_dur),
        'pct_S':        np.array(pct_S),
        'pct_T':        np.array(pct_T),
        'pct_A':        np.array(pct_A),
        'pool_S':       pool_S,
        'pool_T':       pool_T,
        'pool_A':       pool_A,
    }


def fig_episode_summary(real_stats, synth_stats, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def violin_pair(ax, r, s, title, ylabel):
        if len(r) < 2 or len(s) < 2:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return
        parts = ax.violinplot([r, s], showmeans=True)
        for pc, c in zip(parts['bodies'], ['steelblue', 'crimson']):
            pc.set_facecolor(c); pc.set_alpha(0.5)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['real', 'synth'])
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    violin_pair(axes[0, 0], real_stats['fog_fraction'], synth_stats['fog_fraction'],
                'FoG fraction per episode', 'fraction')
    violin_pair(axes[0, 1], real_stats['n_fog_eps'], synth_stats['n_fog_eps'],
                'FoG events per episode', 'count')
    violin_pair(axes[1, 0], real_stats['mean_dur'], synth_stats['mean_dur'],
                'Mean FoG event duration', 'seconds')

    ax = axes[1, 1]
    # Use pooled timestep fractions so participants with more FoG contribute
    # proportionally, matching the transition matrix estimation method.
    real_comp  = [real_stats['pool_S'],
                  real_stats['pool_T'],
                  real_stats['pool_A']]
    synth_comp = [synth_stats['pool_S'],
                  synth_stats['pool_T'],
                  synth_stats['pool_A']]
    labels = ['Shuffling', 'Trembling', 'Akinesia']
    colors = ['gold', 'orange', 'crimson']
    x = ['real', 'synth']
    bottom = [0.0, 0.0]
    for i, lbl in enumerate(labels):
        vals = [real_comp[i], synth_comp[i]]
        ax.bar(x, vals, bottom=bottom, color=colors[i], label=lbl,
               edgecolor='black')
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel('percent within FoG')
    ax.set_title('Within-FoG phase composition\n(pooled timesteps)', fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("Episode-level summary statistics\n"
                 "(real: locomotor-context FoG only, matching synthetic scope)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath.name}")


# ── Tables ────────────────────────────────────────────────────────────────────

def table_ks_summary(real_pool, synth_pool, outpath_main: Path,
                      outpath_pernode: Path,
                      params: HMMParams = None):
    """
    Write two CSV tables:

    table_6_1_ks_per_node.csv
        Per (state, node, feature) univariate KS distance, plus:
          - ks_mahalanobis : Mahalanobis-projected KS using full Sigma_inv
          - energy_distance: multivariate Energy Distance real vs iid N(mu,Sigma)

    table_6_1_ks_summary.csv
        Univariate KS averaged across nodes per (state, feature), plus
        mah_ks_avg and energy_dist_avg columns.
    """
    extra_col = params is not None
    rng_ed    = np.random.default_rng(0)   # fixed seed for reproducibility

    header_pn = ['state', 'node', 'feature', 'n_real', 'n_synth', 'ks_univariate']
    if extra_col:
        header_pn += ['ks_mahalanobis', 'energy_distance']
    rows_pn = [header_pn]

    summary_uni = {(s, f): [] for s in range(N_STATES) for f in range(N_FEATURES)}
    summary_mah = {s: [] for s in range(N_STATES)}
    summary_ed  = {s: [] for s in range(N_STATES)}

    for s in range(N_STATES):
        for n in range(N_NODES):
            # Per (state, node): Mahalanobis KS and Energy Distance
            mah_ks = float('nan')
            ed     = float('nan')
            if extra_col:
                mah_ks = mahalanobis_ks(
                    real_pool[s][n],
                    synth_pool[s][n],
                    get_sigma_inv(params.emission[s][n]),
                )
                ed = energy_distance_4d(
                    real_pool[s][n],
                    params.emission[s][n],
                    rng_ed,
                )
                if not np.isnan(mah_ks):
                    summary_mah[s].append(mah_ks)
                if not np.isnan(ed):
                    summary_ed[s].append(ed)

            for f in range(N_FEATURES):
                xr = real_pool[s][n][:, f];  xr = xr[~np.isnan(xr)]
                xs = synth_pool[s][n][:, f]; xs = xs[~np.isnan(xs)]
                if len(xr) < 10 or len(xs) < 10:
                    ks = float('nan')
                else:
                    ks = float(ks_2samp(xr, xs).statistic)

                row = [STATE_NAMES[s], NODE_NAMES[n], FEATURE_NAMES[f],
                       len(xr), len(xs),
                       f"{ks:.4f}" if not np.isnan(ks) else ""]
                if extra_col:
                    row.append(f"{mah_ks:.4f}" if not np.isnan(mah_ks) else "")
                    row.append(f"{ed:.4f}"     if not np.isnan(ed)     else "")
                rows_pn.append(row)

                if not np.isnan(ks):
                    summary_uni[(s, f)].append(ks)

    with open(outpath_pernode, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows_pn)

    # Summary table
    header_main = ['state'] + FEATURE_NAMES
    if extra_col:
        header_main += ['mah_ks_avg', 'energy_dist_avg']
    rows_main = [header_main]
    for s in range(N_STATES):
        row = [STATE_NAMES[s]]
        for f in range(N_FEATURES):
            vals = summary_uni[(s, f)]
            row.append(f"{np.mean(vals):.4f}" if vals else "")
        if extra_col:
            mah_vals = summary_mah[s]
            ed_vals  = summary_ed[s]
            row.append(f"{np.mean(mah_vals):.4f}" if mah_vals else "")
            row.append(f"{np.mean(ed_vals):.4f}"  if ed_vals  else "")
        rows_main.append(row)

    with open(outpath_main, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows_main)

    print(f"  saved {outpath_main.name}, {outpath_pernode.name}")


def table_dwell_stats(real_dwells, synth_dwells, outpath: Path):
    rows = [['state', 'n_real', 'real_mean_s', 'real_std_s',
             'n_synth', 'synth_mean_s', 'synth_std_s',
             'ks', 'pct_error_mean']]
    for s in range(N_STATES):
        r = np.array(real_dwells.get(s, []),  dtype=float)
        ss = np.array(synth_dwells.get(s, []), dtype=float)
        r = r[r > 0]; ss = ss[ss > 0]
        if len(r) < 2 or len(ss) < 2:
            rows.append([STATE_NAMES[s], len(r), '', '', len(ss),
                         '', '', '', ''])
            continue
        ks = float(ks_2samp(r, ss).statistic)
        pct_err = 100 * (ss.mean() - r.mean()) / r.mean() if r.mean() > 0 else 0
        rows.append([STATE_NAMES[s], len(r),
                     f"{r.mean():.3f}", f"{r.std():.3f}",
                     len(ss),
                     f"{ss.mean():.3f}", f"{ss.std():.3f}",
                     f"{ks:.4f}", f"{pct_err:+.1f}%"])
    with open(outpath, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f"  saved {outpath.name}")


def fit_ar1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 10:
        return float('nan')
    x = x - x.mean()
    den = (x[:-1] ** 2).sum()
    if den < 1e-12:
        return float('nan')
    return float((x[:-1] * x[1:]).sum() / den)


def table_ar1_comparison(params: HMMParams, synth_runs_ankle_fi, outpath: Path):
    """
    Compare fitted per-state rho against empirical AR(1) computed from
    synthetic ankle-FI runs. Same (node, feature) as the ACF figure for
    consistency.
    """
    rows = [['state', 'fitted_rho', 'empirical_synth_rho_ankleFI',
             'abs_diff', 'n_runs']]
    for s in range(N_STATES):
        fitted = float(params.rho[s])
        rhos = [fit_ar1(r) for r in synth_runs_ankle_fi[s]]
        rhos = [v for v in rhos if not np.isnan(v)]
        if rhos:
            emp = float(np.mean(rhos))
            rows.append([STATE_NAMES[s], f"{fitted:.4f}",
                         f"{emp:.4f}", f"{abs(fitted - emp):.4f}", len(rhos)])
        else:
            rows.append([STATE_NAMES[s], f"{fitted:.4f}", '', '', 0])
    with open(outpath, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f"  saved {outpath.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', default='fogstar_processed.pkl',
                    help='path to processed FoG-STAR pickle')
    ap.add_argument('--params',    default='fogstar_model_params.pkl',
                    help='path to fitted HMMParams pickle')
    ap.add_argument('--gmm',       action='store_true',
                    help='use GMM emission model (fogstar_model_params_gmm.pkl '
                         'and FogEpisodeGeneratorGMM). Overrides --params default.')
    ap.add_argument('--n-synth',   type=int, default=100,
                    help='number of synthetic episodes to sample')
    ap.add_argument('--seed',      type=int, default=0)
    ap.add_argument('--outdir',    default='validation',
                    help='output directory for figures and tables')
    args = ap.parse_args()

    # If --gmm is set, switch to GMM pickle and generator unless user
    # explicitly provided --params on the command line.
    if args.gmm:
        if not _GMM_AVAILABLE:
            raise ImportError(
                'GMM modules not found. Ensure fog_episode_generator_gmm.py '
                'and fog_model_fitter_gmm.py are on the Python path.')
        if args.params == 'fogstar_model_params.pkl':
            args.params = 'fogstar_model_params_gmm.pkl'

    outdir   = Path(args.outdir)
    fig_dir  = outdir / 'figures'
    tab_dir  = outdir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("[1/9] Loading real data and applying log+z preprocessing ...")
    processed, stats_obj, raw = load_real_data(args.processed)

    print("[2/9] Loading generator and sampling synthetic episodes ...")
    if args.gmm:
        gen = FogEpisodeGeneratorGMM(args.params, seed=args.seed, min_fog=1)
        print(f"  Using GMM generator with params: {args.params}")
    else:
        gen = FogEpisodeGenerator(args.params, seed=args.seed, min_fog=1)
    episodes = sample_synthetic_episodes(gen, args.n_synth)

    print("[3/9] Building per-state pools ...")
    real_pool, synth_pool = pool_features_by_state(processed, episodes)

    print("[4/9] Figure 6.1, A.1, A.2: marginals per node (AR(1) output vs real) ...")
    fig_marginals(real_pool, synth_pool, ANKLE, fig_dir / 'fig_6_1_marginals_ankle.png', params=gen.params)
    fig_marginals(real_pool, synth_pool, BACK,  fig_dir / 'fig_a_1_marginals_back.png',  params=gen.params)
    fig_marginals(real_pool, synth_pool, WRIST, fig_dir / 'fig_a_2_marginals_wrist.png', params=gen.params)

    print("[4b/9] Figure 6.X: iid emission comparison (N(mu,Sigma) draws vs real) ...")
    fig_iid_emission(real_pool, gen.params, ANKLE,
                     fig_dir / 'fig_6_x_iid_emission_ankle.png')
    fig_iid_emission(real_pool, gen.params, BACK,
                     fig_dir / 'fig_a_3_iid_emission_back.png')
    fig_iid_emission(real_pool, gen.params, WRIST,
                     fig_dir / 'fig_a_4_iid_emission_wrist.png')

    print("[4c/9] Figure 6.Y: Mahalanobis distance distributions ...")
    fig_mahalanobis_distance(real_pool, gen.params, ANKLE,
                              fig_dir / 'fig_6_y_mahalanobis_ankle.png')
    fig_mahalanobis_distance(real_pool, gen.params, BACK,
                              fig_dir / 'fig_a_5_mahalanobis_back.png')
    fig_mahalanobis_distance(real_pool, gen.params, WRIST,
                              fig_dir / 'fig_a_6_mahalanobis_wrist.png')

    print("[5/9] Figure 6.2: dwell distributions ...")
    real_dwells  = stats_obj.state_durations
    synth_dwells = compute_synth_dwells(episodes)
    fig_dwell_distributions(real_dwells, synth_dwells,
                             fig_dir / 'fig_6_2_dwell_distributions.png')

    print("[6/9] Figure 6.3 + appendix: ACF of FI per node ...")
    # Ankle FI (Figure 6.3, main text). The ankle runs are also reused by the
    # AR(1) self-consistency table below, so they stay named explicitly.
    real_runs_ankle_fi  = extract_runs_real(processed, node=ANKLE, feature=0)
    synth_runs_ankle_fi = extract_runs_synth(episodes,  node=ANKLE, feature=0)
    fig_acf(real_runs_ankle_fi, synth_runs_ankle_fi, ANKLE, 0,
            fig_dir / 'fig_6_3_acf_ankle_fi.png')
    # Back and wrist FI ACF (appendix), identical per-state layout.
    real_runs_back_fi   = extract_runs_real(processed, node=BACK, feature=0)
    synth_runs_back_fi  = extract_runs_synth(episodes,  node=BACK, feature=0)
    fig_acf(real_runs_back_fi, synth_runs_back_fi, BACK, 0,
            fig_dir / 'fig_a_7_acf_back_fi.png')
    real_runs_wrist_fi  = extract_runs_real(processed, node=WRIST, feature=0)
    synth_runs_wrist_fi = extract_runs_synth(episodes,  node=WRIST, feature=0)
    fig_acf(real_runs_wrist_fi, synth_runs_wrist_fi, WRIST, 0,
            fig_dir / 'fig_a_8_acf_wrist_fi.png')

    print("[7/9] Figure 6.4 + appendix: PCA projection per node ...")
    fig_pca(real_pool, synth_pool, ANKLE, fig_dir / 'fig_6_4_pca_ankle.png')
    fig_pca(real_pool, synth_pool, BACK,  fig_dir / 'fig_a_9_pca_back.png')
    fig_pca(real_pool, synth_pool, WRIST, fig_dir / 'fig_a_10_pca_wrist.png')

    print("[8/9] Figure 6.5: inter-node correlation ...")
    fig_internode_correlation(processed, episodes,
                               fig_dir / 'fig_6_5_internode_correlation.png')

    print("[9/9] Figure 6.6: episode-level summary ...")
    # Use processed participants (valid_mask already applied) so the real
    # episode stats reflect locomotor-context FoG only, matching the scope
    # of the synthetic generator which also produces locomotor-context data.
    # Using raw participants without the valid mask includes FoG timesteps
    # from non-locomotor activities, inflating Shuffling and Trembling
    # fractions and making Akinesia appear underrepresented.
    real_states_per_ep  = [p.states[p.valid_mask] for p in processed]
    synth_states_per_ep = [ep.states for ep in episodes]
    real_ep  = episode_summary_stats(real_states_per_ep)
    synth_ep = episode_summary_stats(synth_states_per_ep)
    fig_episode_summary(real_ep, synth_ep,
                         fig_dir / 'fig_6_6_episode_summary.png')

    print("Tables ...")
    table_ks_summary(real_pool, synth_pool,
                      tab_dir / 'table_6_1_ks_summary.csv',
                      tab_dir / 'table_6_1_ks_per_node.csv',
                      params=gen.params)
    table_dwell_stats(real_dwells, synth_dwells,
                       tab_dir / 'table_6_2_dwell_stats.csv')
    table_ar1_comparison(gen.params, synth_runs_ankle_fi,
                          tab_dir / 'table_6_3_ar1.csv')

    print(f"\nDone. Outputs written to {outdir.resolve()}")


if __name__ == '__main__':
    main()