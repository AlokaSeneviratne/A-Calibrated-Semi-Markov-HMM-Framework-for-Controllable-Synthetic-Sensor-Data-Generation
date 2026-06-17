"""
fog_ml_validation.py
--------------------
Chapter 8 ML validation for:
    "A Calibrated Semi-Markov HMM Generator for Synthetic Freezing of Gait
    Sensor Data"

Three experiments:

Experiment 0: Real-data baseline (NEW)
    Train a random forest on real FoG-STAR using leave-one-participant-out CV.
    At each fold: train on the other 21 participants, test on the left-out one.
    This is the fair comparison for the synthetic transfer test: if real-data
    LOPO gives similar F1 to synthetic-trained LOPO, the generator produces
    training data of comparable quality to what 21 real participants provide.
    Produces:
        fig_8_ml_baseline_vs_synth.png  per-fold comparison bar chart
        table_8_ml_baseline.csv         fold-level baseline metrics

Experiment 1: Transfer test
    Train a random forest on 100 synthetic episodes.
    Test on real FoG-STAR using leave-one-participant-out (LOPO) CV.
    Report macro F1 and per-class recall (binary + 4-class layered).
    Produces:
        fig_8_ml_transfer_f1.png      per-fold binary F1 bar chart
        fig_8_ml_confusion.png        averaged 4-class confusion matrix
        table_8_ml_transfer.csv       fold-level metrics

Experiment 2: FoG burden sweep
    Vary target FoG fraction across [0.10, 0.20, 0.40].
    At each level: back-calculate Healthy dwell mean, generate 100 synthetic
    episodes, train random forest, test on real data LOPO.
    Produces:
        fig_8_ml_burden_sweep.png     transfer F1 vs FoG burden
        table_8_ml_burden.csv         burden sweep metrics

Usage
-----
    python fog_ml_validation.py
        [--processed   fogstar_processed.pkl]
        [--params      fogstar_model_params.pkl]
        [--params-gmm  fogstar_model_params_gmm.pkl]   (optional)
        [--classifier  rf|lgbm|both]                   (default: rf)
        [--n-synth     100]
        [--n-trees     100]
        [--seed        42]
        [--outdir      validation]

Classifiers
-----------
    rf   : RandomForestClassifier (sklearn), balanced class weights
    lgbm : LGBMClassifier (LightGBM), is_unbalance=True
    both : run rf and lgbm, produce comparison table and figure

GMM comparison
--------------
    If --params-gmm is provided, Experiment 1 and Experiment 2 are repeated
    with the GMM generator. A side-by-side comparison figure is produced:
        fig_8_ml_gaussian_vs_gmm.png

Design notes
------------
Feature matrix:
    12 features per timestep: 3 nodes x 4 features (FI, RMS_acc, RMS_gyr,
    delta_FI) in z-score space, same space the generator produces.

Labels:
    4-class: states 0-3 (Healthy, Shuffling, Trembling, Akinesia)
    Binary:  0=Healthy, 1=FoG (states 1+2+3 collapsed)

Burden back-calculation:
    Given FoG states have fixed mean dwells (mu_fog_s), and we want fraction
    f_fog of timesteps in FoG, we solve for the Healthy dwell mean mu_h:

        f_fog = mu_fog_total / (mu_h + mu_fog_total)
        => mu_h = mu_fog_total * (1 - f_fog) / f_fog

    where mu_fog_total is the mean total time in FoG per H->FoG->H cycle,
    approximated as the mean dwell of the expected FoG state weighted by
    transition probabilities out of Healthy.

    We then back-calculate the log-normal mu parameter from the target mu_sec:
        mu_log = log(mu_sec) - 0.5 * sigma_log^2
    (moment-matching for log-normal: E[X] = exp(mu + sigma^2/2))
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix, classification_report
)

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    print("WARNING: lightgbm not installed. "
          "Run: pip install lightgbm\n"
          "Falling back to random forest only.")

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from any directory by adding the script's own directory to
# sys.path so that fog_* modules are importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fog_star_loader import (
    FS, N_STATES, N_NODES, N_FEATURES,
    STATE_NAMES, NODE_NAMES, FEATURE_NAMES,
    ParticipantData, DatasetStats,
)
from fog_model_fitter import (
    HMMParams, DwellDistribution, EmissionParams, PreprocessParams,
    preprocess_features,
)
from fog_episode_generator import FogEpisodeGenerator, Episode

try:
    from fog_episode_generator_gmm import FogEpisodeGeneratorGMM
    _GMM_GEN_AVAILABLE = True
except ImportError:
    _GMM_GEN_AVAILABLE = False


# ── constants ─────────────────────────────────────────────────────────────────

HEALTHY    = 0
SHUFFLING  = 1
TREMBLING  = 2
AKINESIA   = 3

CLASS_NAMES_4  = ['Healthy', 'Shuffling', 'Trembling', 'Akinesia']
CLASS_NAMES_BIN = ['Healthy', 'FoG']

BURDEN_LEVELS  = [0.10, 0.20, 0.40]   # target FoG fractions for sweep


# ── data loading ──────────────────────────────────────────────────────────────

def load_real_data(processed_path: str):
    """
    Load and preprocess real FoG-STAR participants.
    Returns list of ParticipantData in z-score space.
    """
    print(f"Loading real data from {processed_path} ...")
    with open(processed_path, 'rb') as f:
        data = pickle.load(f)
    raw_participants = data['participants']
    processed, _     = preprocess_features(raw_participants)
    print(f"  {len(processed)} participants loaded.")
    return processed


def load_params(params_path: str) -> HMMParams:
    """Load fitted HMM parameters, handling __main__ pickle remapping."""
    import fog_model_fitter as _fmf  # noqa: ensures classes registered

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                module = 'fog_model_fitter'
            return super().find_class(module, name)

    with open(params_path, 'rb') as f:
        params = _Unpickler(f).load()
    print(f"Loaded HMM params from {params_path}")
    return params


# ── feature extraction ────────────────────────────────────────────────────────

def participant_to_xy(pdata: ParticipantData,
                      binary: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one participant's data to (X, y) arrays.

    X : (T_valid, 12)  flattened z-score features across 3 nodes x 4 features
    y : (T_valid,)     state labels, 4-class or binary

    Only valid_mask timesteps are included. Timesteps with any NaN are dropped.
    """
    valid  = pdata.valid_mask
    feats  = pdata.features[valid]        # (T_valid, 3, 4)
    states = pdata.states[valid]          # (T_valid,)

    # Flatten nodes and features: (T_valid, 12)
    X = feats.reshape(len(feats), N_NODES * N_FEATURES)

    # Drop rows with NaN
    nan_mask = ~np.any(np.isnan(X), axis=1)
    X      = X[nan_mask]
    states = states[nan_mask]

    if binary:
        y = (states > 0).astype(int)
    else:
        y = states.astype(int)

    return X, y


def episodes_to_xy(episodes: List[Episode],
                   binary: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of synthetic episodes to (X, y).

    X : (T_total, 12)
    y : (T_total,)
    """
    X_list, y_list = [], []
    for ep in episodes:
        X = ep.obs.reshape(ep.T, N_NODES * N_FEATURES)   # (T, 12)
        y = ep.states.astype(int)                          # (T,)
        # Drop NaN rows (should be none in synthetic, but defensive)
        nan_mask = ~np.any(np.isnan(X), axis=1)
        X_list.append(X[nan_mask])
        y_list.append(y[nan_mask])
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    if binary:
        y_all = (y_all > 0).astype(int)
    return X_all, y_all


# ── synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic(params_path: str,
                        n_episodes: int,
                        seed: int,
                        params_override=None,
                        use_gmm: bool = False,
                        ) -> List[Episode]:
    """
    Generate n_episodes synthetic episodes.
    If params_override is given, write to a temp file and load from there.
    If use_gmm is True, use FogEpisodeGeneratorGMM instead of FogEpisodeGenerator.
    """
    import tempfile

    if params_override is not None:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name
        with open(tmp_path, 'wb') as f:
            pickle.dump(params_override, f)
        load_path = tmp_path
        cleanup   = True
    else:
        load_path = params_path
        cleanup   = False

    if use_gmm and _GMM_GEN_AVAILABLE:
        gen = FogEpisodeGeneratorGMM(load_path, seed=seed,
                                      min_fog=1, max_seconds=300.0)
    else:
        gen = FogEpisodeGenerator(load_path, seed=seed,
                                   min_fog=1, max_seconds=300.0)

    if cleanup:
        os.unlink(load_path)

    episodes = [gen.sample_episode() for _ in range(n_episodes)]
    total_t  = sum(e.T for e in episodes)
    print(f"  Generated {n_episodes} synthetic episodes ({total_t:,} timesteps)")
    return episodes


# ── burden back-calculation ───────────────────────────────────────────────────

def burden_to_healthy_dwell(params: HMMParams, target_fog_fraction: float) -> float:
    """
    Given a target fraction of timesteps in FoG states, back-calculate the
    Healthy state mean dwell time (in seconds).

    Derivation
    ----------
    Let mu_h = Healthy mean dwell (seconds).
    Let mu_fog = expected total time in FoG per Healthy->FoG->Healthy cycle.

    In steady state:
        f_fog = mu_fog / (mu_h + mu_fog)
        => mu_h = mu_fog * (1 - f_fog) / f_fog

    mu_fog is approximated as the weighted mean dwell of FoG states, weighted
    by the transition probability from Healthy to each FoG state:

        p_h_to_s = T_mat[HEALTHY, s]  for s in {1, 2, 3}
        mu_fog   = sum(p_h_to_s * dwell[s].mu_sec) / sum(p_h_to_s)

    This is an approximation because it ignores FoG-to-FoG transitions, but
    it is accurate enough for the 3 burden levels used in the sweep.

    Parameters
    ----------
    params              : fitted HMMParams
    target_fog_fraction : desired fraction of timesteps in FoG (0 < f < 1)

    Returns
    -------
    mu_h_target : target Healthy mean dwell in seconds
    """
    T_mat = params.T_mat   # (4, 4), diagonal zeroed

    # Transition weights from Healthy to each FoG state
    fog_states   = [SHUFFLING, TREMBLING, AKINESIA]
    weights      = np.array([T_mat[HEALTHY, s] for s in fog_states])
    fog_mu_secs  = np.array([params.dwell[s].mu_sec for s in fog_states])

    # Weighted mean FoG dwell
    w_sum  = weights.sum()
    mu_fog = float(np.dot(weights, fog_mu_secs) / w_sum) if w_sum > 0 else 10.0

    f = float(np.clip(target_fog_fraction, 0.01, 0.99))
    mu_h_target = mu_fog * (1.0 - f) / f

    print(f"  Burden {f*100:.0f}%: mu_fog={mu_fog:.2f}s  "
          f"=> target Healthy dwell={mu_h_target:.2f}s  "
          f"(fitted={params.dwell[HEALTHY].mu_sec:.2f}s)")
    return mu_h_target


def mutate_healthy_dwell(params: HMMParams, mu_h_target: float) -> HMMParams:
    """
    Return a deep copy of params with Healthy dwell mean set to mu_h_target.

    The log-normal mu parameter is back-calculated using moment-matching:
        E[X] = exp(mu_log + 0.5 * sigma_log^2)
        => mu_log = log(mu_h_target) - 0.5 * sigma_log^2
    """
    mutated = copy.deepcopy(params)
    d       = mutated.dwell[HEALTHY]
    sig     = d.sigma   # sigma_log (unchanged)

    # Moment-matching: mu_log = log(E[X]) - 0.5 * sigma^2
    mu_log_new    = np.log(max(mu_h_target, 0.5)) - 0.5 * sig ** 2
    d.mu          = float(mu_log_new)
    d.mu_sec      = float(mu_h_target)
    # std_sec is approximate; scale proportionally to keep CV constant
    original_cv   = params.dwell[HEALTHY].std_sec / max(params.dwell[HEALTHY].mu_sec, 1e-6)
    d.std_sec     = float(mu_h_target * original_cv)

    return mutated


# ── classifier factory ────────────────────────────────────────────────────────

def build_classifier(clf_type: str, n_trees: int, seed: int):
    """
    Build and return a classifier.

    Parameters
    ----------
    clf_type : 'rf' or 'lgbm'
    n_trees  : number of estimators
    seed     : random seed

    Returns
    -------
    sklearn-compatible classifier (not yet fitted)
    """
    if clf_type == 'lgbm':
        if not _LGBM_AVAILABLE:
            print("  WARNING: lgbm requested but not installed. "
                  "Falling back to rf.")
            return _build_rf(n_trees, seed)
        return LGBMClassifier(
            n_estimators   = n_trees,
            is_unbalance   = True,      # handles class imbalance
            random_state   = seed,
            n_jobs         = -1,
            verbose        = -1,        # suppress lgbm output
        )
    return _build_rf(n_trees, seed)


def _build_rf(n_trees: int, seed: int):
    return RandomForestClassifier(
        n_estimators = n_trees,
        n_jobs       = -1,
        random_state = seed,
        class_weight = 'balanced',
    )


# ── LOPO cross-validation ─────────────────────────────────────────────────────

def lopo_evaluate(processed_participants: List[ParticipantData],
                   X_train: np.ndarray,
                   y_train_4: np.ndarray,
                   n_trees: int,
                   seed: int,
                   clf_type: str = 'rf',
                   verbose: bool = True
                   ) -> Dict:
    """
    Leave-one-participant-out evaluation.

    Trains once on X_train (synthetic), tests on each real participant in turn.

    Returns dict with:
        fold_binary_f1   : list of binary F1 per fold
        fold_macro_f1    : list of 4-class macro F1 per fold
        fold_recall      : list of per-class recall arrays per fold (4-class)
        conf_matrix      : summed 4-class confusion matrix across all folds
        y_true_all       : concatenated true labels (4-class)
        y_pred_all       : concatenated predicted labels (4-class)
    """
    clf = build_classifier(clf_type, n_trees, seed)
    clf.fit(X_train, y_train_4)
    clf_label = clf_type.upper()
    print(f"  {clf_label} trained: {n_trees} estimators, "
          f"{X_train.shape[0]:,} synthetic timesteps, "
          f"{X_train.shape[1]} features")

    fold_binary_f1 = []
    fold_macro_f1  = []
    fold_recall    = []   # per-fold, shape (4,)
    conf_matrix    = np.zeros((4, 4), dtype=int)
    y_true_all     = []
    y_pred_all     = []

    for pi, pdata in enumerate(processed_participants):
        X_test, y_test_4 = participant_to_xy(pdata, binary=False)
        if len(X_test) == 0:
            if verbose:
                print(f"  Participant {pi+1:2d}: skipped (no valid timesteps)")
            continue

        y_pred_4 = clf.predict(X_test)

        # Binary collapse
        y_test_bin = (y_test_4 > 0).astype(int)
        y_pred_bin = (y_pred_4 > 0).astype(int)

        # Metrics
        bin_f1   = f1_score(y_test_bin, y_pred_bin,
                             average='binary', zero_division=0)
        macro_f1 = f1_score(y_test_4, y_pred_4,
                             average='macro', zero_division=0)
        recall_4 = recall_score(y_test_4, y_pred_4,
                                 average=None, labels=[0,1,2,3],
                                 zero_division=0)

        fold_binary_f1.append(bin_f1)
        fold_macro_f1.append(macro_f1)
        fold_recall.append(recall_4)

        cm = confusion_matrix(y_test_4, y_pred_4, labels=[0,1,2,3])
        conf_matrix += cm

        y_true_all.extend(y_test_4.tolist())
        y_pred_all.extend(y_pred_4.tolist())

        if verbose:
            shuf_recall = recall_4[SHUFFLING]
            print(f"  Participant {pi+1:2d}: "
                  f"binary_F1={bin_f1:.3f}  "
                  f"macro_F1={macro_f1:.3f}  "
                  f"Shuffling_recall={shuf_recall:.3f}")

    return dict(
        fold_binary_f1 = fold_binary_f1,
        fold_macro_f1  = fold_macro_f1,
        fold_recall    = fold_recall,
        conf_matrix    = conf_matrix,
        y_true_all     = np.array(y_true_all),
        y_pred_all     = np.array(y_pred_all),
    )



# ── real-data LOPO baseline ───────────────────────────────────────────────────

def lopo_real_baseline(processed_participants: List[ParticipantData],
                        n_trees: int,
                        seed: int,
                        clf_type: str = 'rf',
                        verbose: bool = True
                        ) -> Dict:
    """
    Leave-one-participant-out baseline using only real FoG-STAR data.

    At each fold:
        - Train on all participants except the left-out one
        - Test on the left-out participant

    This is the natural comparison for the synthetic transfer test. If
    synthetic-trained LOPO gives similar F1 to real-data LOPO, the generator
    produces training data of comparable practical quality to 21 real
    participants.

    Returns same dict structure as lopo_evaluate.
    """
    fold_binary_f1 = []
    fold_macro_f1  = []
    fold_recall    = []
    conf_matrix    = np.zeros((4, 4), dtype=int)
    y_true_all     = []
    y_pred_all     = []

    for pi in range(len(processed_participants)):
        # Build train set from all other participants
        train_parts = [p for j, p in enumerate(processed_participants) if j != pi]
        X_parts, y_parts = [], []
        for p in train_parts:
            Xp, yp = participant_to_xy(p, binary=False)
            if len(Xp) > 0:
                X_parts.append(Xp)
                y_parts.append(yp)

        if not X_parts:
            if verbose:
                print(f"  Participant {pi+1:2d}: skipped (no training data)")
            continue

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)

        # Test on left-out participant
        X_test, y_test_4 = participant_to_xy(
            processed_participants[pi], binary=False)
        if len(X_test) == 0:
            if verbose:
                print(f"  Participant {pi+1:2d}: skipped (no test data)")
            continue

        clf = build_classifier(clf_type, n_trees, seed)
        clf.fit(X_train, y_train)

        y_pred_4   = clf.predict(X_test)
        y_test_bin = (y_test_4 > 0).astype(int)
        y_pred_bin = (y_pred_4 > 0).astype(int)

        bin_f1   = f1_score(y_test_bin, y_pred_bin,
                             average='binary', zero_division=0)
        macro_f1 = f1_score(y_test_4, y_pred_4,
                             average='macro', zero_division=0)
        recall_4 = recall_score(y_test_4, y_pred_4,
                                 average=None, labels=[0,1,2,3],
                                 zero_division=0)

        fold_binary_f1.append(bin_f1)
        fold_macro_f1.append(macro_f1)
        fold_recall.append(recall_4)

        cm = confusion_matrix(y_test_4, y_pred_4, labels=[0,1,2,3])
        conf_matrix += cm
        y_true_all.extend(y_test_4.tolist())
        y_pred_all.extend(y_pred_4.tolist())

        if verbose:
            print(f"  Participant {pi+1:2d}: "
                  f"binary_F1={bin_f1:.3f}  "
                  f"macro_F1={macro_f1:.3f}  "
                  f"Shuffling_recall={recall_4[SHUFFLING]:.3f}")

    return dict(
        fold_binary_f1 = fold_binary_f1,
        fold_macro_f1  = fold_macro_f1,
        fold_recall    = fold_recall,
        conf_matrix    = conf_matrix,
        y_true_all     = np.array(y_true_all),
        y_pred_all     = np.array(y_pred_all),
    )


def fig_transfer_f1(results: Dict, outpath: Path):
    """
    Bar chart of binary F1 per participant fold (Experiment 1).
    Horizontal line at mean F1.
    """
    f1s   = results['fold_binary_f1']
    n     = len(f1s)
    mean  = float(np.mean(f1s))
    std   = float(np.std(f1s))
    folds = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#4C72B0' if v >= mean else '#DD8452' for v in f1s]
    bars = ax.bar(folds, f1s, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(mean, color='#333333', linewidth=1.2, linestyle='--',
               label=f'Mean = {mean:.3f} ± {std:.3f}')

    ax.set_xlabel('Participant (left-out fold)', fontsize=11)
    ax.set_ylabel('Binary F1 (FoG vs Healthy)', fontsize=11)
    ax.set_title('Transfer Test: Synthetic-trained RF on Real FoG-STAR (LOPO)',
                 fontsize=12)
    ax.set_xticks(folds)
    ax.set_xticklabels([str(i) for i in folds], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig_confusion(conf_matrix: np.ndarray, outpath: Path):
    """
    Normalised 4-class confusion matrix (Experiment 1).
    """
    cm_norm = conf_matrix.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm /= row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(4):
        for j in range(4):
            val   = cm_norm[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES_4, fontsize=10)
    ax.set_yticklabels(CLASS_NAMES_4, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix (normalised by true class)\nSummed across LOPO folds',
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig_baseline_vs_synth(results_real: Dict,
                           results_synth: Dict,
                           outpath: Path):
    """
    Side-by-side per-fold binary F1: real-data LOPO vs synthetic-trained LOPO.

    Blue bars: real-data baseline (trained on 21 real participants)
    Orange bars: synthetic transfer (trained on 100 synthetic episodes)
    Dashed lines: respective means
    """
    f1_real  = results_real['fold_binary_f1']
    f1_synth = results_synth['fold_binary_f1']
    n        = max(len(f1_real), len(f1_synth))
    folds    = list(range(1, n + 1))

    # Pad shorter list with NaN if lengths differ
    def _pad(lst, length):
        arr = np.full(length, np.nan)
        arr[:len(lst)] = lst
        return arr

    r = _pad(f1_real,  n)
    s = _pad(f1_synth, n)

    mean_r = float(np.nanmean(r))
    mean_s = float(np.nanmean(s))

    x      = np.arange(n)
    width  = 0.38

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, r, width, label=f'Real-data LOPO (mean={mean_r:.3f})',
           color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, s, width, label=f'Synthetic transfer (mean={mean_s:.3f})',
           color='#DD8452', edgecolor='white', linewidth=0.5)

    ax.axhline(mean_r, color='#4C72B0', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(mean_s, color='#DD8452', linewidth=1.2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Participant (left-out fold)', fontsize=11)
    ax.set_ylabel('Binary F1 (FoG vs Healthy)', fontsize=11)
    ax.set_title('Real-data LOPO vs Synthetic Transfer: Binary F1 per Fold',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in folds], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig_gaussian_vs_gmm(results_gauss: Dict,
                         results_gmm:   Dict,
                         outpath: Path):
    """
    Side-by-side per-fold binary F1: Gaussian emission vs GMM emission.
    Both classifiers are synthetic-trained.
    """
    f1_g = results_gauss['fold_binary_f1']
    f1_m = results_gmm['fold_binary_f1']
    n    = max(len(f1_g), len(f1_m))

    def _pad(lst, length):
        arr = np.full(length, np.nan)
        arr[:len(lst)] = lst
        return arr

    g = _pad(f1_g, n)
    m = _pad(f1_m, n)

    mean_g = float(np.nanmean(g))
    mean_m = float(np.nanmean(m))

    x     = np.arange(n)
    width = 0.38
    folds = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, g, width,
           label=f'Gaussian emission (mean={mean_g:.3f})',
           color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, m, width,
           label=f'GMM emission (mean={mean_m:.3f})',
           color='#55A868', edgecolor='white', linewidth=0.5)

    ax.axhline(mean_g, color='#4C72B0', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(mean_m, color='#55A868', linewidth=1.2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Participant (left-out fold)', fontsize=11)
    ax.set_ylabel('Binary F1 (FoG vs Healthy)', fontsize=11)
    ax.set_title('Gaussian vs GMM Emission: Transfer Binary F1 per Fold',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in folds], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig_classifier_comparison(results_rf:   Dict,
                                results_lgbm: Dict,
                                label_rf:   str,
                                label_lgbm: str,
                                outpath: Path):
    """
    Side-by-side per-fold binary F1: RF vs LightGBM.
    """
    f1_r = results_rf['fold_binary_f1']
    f1_l = results_lgbm['fold_binary_f1']
    n    = max(len(f1_r), len(f1_l))

    def _pad(lst, length):
        arr = np.full(length, np.nan)
        arr[:len(lst)] = lst
        return arr

    r = _pad(f1_r, n)
    l = _pad(f1_l, n)

    mean_r = float(np.nanmean(r))
    mean_l = float(np.nanmean(l))

    x     = np.arange(n)
    width = 0.38
    folds = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, r, width,
           label=f'{label_rf} (mean={mean_r:.3f})',
           color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, l, width,
           label=f'{label_lgbm} (mean={mean_l:.3f})',
           color='#C44E52', edgecolor='white', linewidth=0.5)

    ax.axhline(mean_r, color='#4C72B0', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(mean_l, color='#C44E52', linewidth=1.2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Participant (left-out fold)', fontsize=11)
    ax.set_ylabel('Binary F1 (FoG vs Healthy)', fontsize=11)
    ax.set_title(f'Classifier Comparison: {label_rf} vs {label_lgbm}',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in folds], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig_burden_sweep(burdens: List[float],
                      mean_f1s: List[float],
                      std_f1s:  List[float],
                      fitted_burden: float,
                      outpath: Path):
    """
    Line plot of mean binary F1 vs FoG burden (Experiment 2).
    Vertical line at fitted (default) burden.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(
        [b * 100 for b in burdens],
        mean_f1s,
        yerr=std_f1s,
        fmt='o-',
        color='#4C72B0',
        linewidth=1.8,
        markersize=7,
        capsize=4,
        label='Mean binary F1 ± std'
    )
    ax.axvline(fitted_burden * 100, color='#C44E52', linewidth=1.2,
               linestyle='--', label=f'Fitted default ({fitted_burden*100:.0f}%)')

    ax.set_xlabel('Target FoG Burden (%)', fontsize=11)
    ax.set_ylabel('Binary F1 (FoG vs Healthy)', fontsize=11)
    ax.set_title('Effect of FoG Burden on Transfer F1\n(Synthetic train, real LOPO test)',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([b * 100 for b in burdens])
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── table writers ─────────────────────────────────────────────────────────────

def write_transfer_table(results: Dict, outpath: Path):
    f1s      = results['fold_binary_f1']
    macro_f1 = results['fold_macro_f1']
    recalls  = results['fold_recall']   # list of (4,) arrays

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['fold', 'binary_f1', 'macro_f1_4class',
                  'recall_healthy', 'recall_shuffling',
                  'recall_trembling', 'recall_akinesia']
        writer.writerow(header)
        for i, (bf1, mf1, rec) in enumerate(zip(f1s, macro_f1, recalls)):
            writer.writerow([
                i + 1,
                f'{bf1:.4f}',
                f'{mf1:.4f}',
                f'{rec[0]:.4f}',
                f'{rec[1]:.4f}',
                f'{rec[2]:.4f}',
                f'{rec[3]:.4f}',
            ])
        # Summary row
        writer.writerow([
            'mean',
            f'{np.mean(f1s):.4f}',
            f'{np.mean(macro_f1):.4f}',
            f'{np.mean([r[0] for r in recalls]):.4f}',
            f'{np.mean([r[1] for r in recalls]):.4f}',
            f'{np.mean([r[2] for r in recalls]):.4f}',
            f'{np.mean([r[3] for r in recalls]):.4f}',
        ])
        writer.writerow([
            'std',
            f'{np.std(f1s):.4f}',
            f'{np.std(macro_f1):.4f}',
            f'{np.std([r[0] for r in recalls]):.4f}',
            f'{np.std([r[1] for r in recalls]):.4f}',
            f'{np.std([r[2] for r in recalls]):.4f}',
            f'{np.std([r[3] for r in recalls]):.4f}',
        ])
    print(f"  Saved {outpath}")


def write_baseline_table(results: Dict, outpath: Path):
    f1s      = results['fold_binary_f1']
    macro_f1 = results['fold_macro_f1']
    recalls  = results['fold_recall']

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['fold', 'binary_f1', 'macro_f1_4class',
                  'recall_healthy', 'recall_shuffling',
                  'recall_trembling', 'recall_akinesia']
        writer.writerow(header)
        for i, (bf1, mf1, rec) in enumerate(zip(f1s, macro_f1, recalls)):
            writer.writerow([
                i + 1,
                f'{bf1:.4f}',
                f'{mf1:.4f}',
                f'{rec[0]:.4f}',
                f'{rec[1]:.4f}',
                f'{rec[2]:.4f}',
                f'{rec[3]:.4f}',
            ])
        writer.writerow([
            'mean',
            f'{np.mean(f1s):.4f}',
            f'{np.mean(macro_f1):.4f}',
            f'{np.mean([r[0] for r in recalls]):.4f}',
            f'{np.mean([r[1] for r in recalls]):.4f}',
            f'{np.mean([r[2] for r in recalls]):.4f}',
            f'{np.mean([r[3] for r in recalls]):.4f}',
        ])
        writer.writerow([
            'std',
            f'{np.std(f1s):.4f}',
            f'{np.std(macro_f1):.4f}',
            f'{np.std([r[0] for r in recalls]):.4f}',
            f'{np.std([r[1] for r in recalls]):.4f}',
            f'{np.std([r[2] for r in recalls]):.4f}',
            f'{np.std([r[3] for r in recalls]):.4f}',
        ])
    print(f"  Saved {outpath}")


def write_burden_table(burdens: List[float],
                        mean_f1s: List[float],
                        std_f1s:  List[float],
                        mean_shuf: List[float],
                        std_shuf:  List[float],
                        healthy_dwells: List[float],
                        outpath: Path):
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['burden_pct', 'healthy_dwell_s',
                         'mean_binary_f1', 'std_binary_f1',
                         'mean_shuffling_recall', 'std_shuffling_recall'])
        for b, hd, mf, sf, ms, ss in zip(
                burdens, healthy_dwells, mean_f1s, std_f1s, mean_shuf, std_shuf):
            writer.writerow([
                f'{b*100:.0f}',
                f'{hd:.2f}',
                f'{mf:.4f}',
                f'{sf:.4f}',
                f'{ms:.4f}',
                f'{ss:.4f}',
            ])
    print(f"  Saved {outpath}")


# ── fitted burden estimate ────────────────────────────────────────────────────

def estimate_fitted_burden(params: HMMParams) -> float:
    """
    Estimate the FoG fraction implied by the fitted dwell parameters.
    Uses the same approximation as burden_to_healthy_dwell, solved in reverse.
    """
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
        description='ML validation for FoG HMM generator (Ch 8)')
    parser.add_argument('--processed',   default='fogstar_processed.pkl')
    parser.add_argument('--params',      default='fogstar_model_params.pkl')
    parser.add_argument('--params-gmm',  default=None,
                        help='Path to GMM params pkl. If provided, runs '
                             'Gaussian vs GMM comparison.')
    parser.add_argument('--classifier',  default='rf',
                        choices=['rf', 'lgbm', 'both'],
                        help='Classifier to use: rf, lgbm, or both.')
    parser.add_argument('--n-synth',     type=int,   default=100)
    parser.add_argument('--n-trees',     type=int,   default=100)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--outdir',      default='validation')
    args = parser.parse_args()

    # ── output dirs ───────────────────────────────────────────────────────────
    fig_dir = Path(args.outdir) / 'figures'
    tbl_dir = Path(args.outdir) / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ── load data and params ──────────────────────────────────────────────────
    processed = load_real_data(args.processed)
    params    = load_params(args.params)

    fitted_burden = estimate_fitted_burden(params)
    print(f"\nFitted FoG burden (from dwell means): {fitted_burden*100:.1f}%")

    # Determine classifiers to run
    clf_types = ['rf', 'lgbm'] if args.classifier == 'both' else [args.classifier]

    # ── Experiment 0: Real-data LOPO baseline ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 0: Real-data LOPO baseline  (classifier: {args.classifier})")
    print(f"  Training on 21 real participants, testing on left-out one ...")

    results_e0_by_clf = {}
    for clf_type in clf_types:
        print(f"\n  -- Classifier: {clf_type.upper()} --")
        results_e0_by_clf[clf_type] = lopo_real_baseline(
            processed, n_trees=args.n_trees, seed=args.seed,
            clf_type=clf_type, verbose=True)

    # Use first classifier for primary summary display
    results_e0 = results_e0_by_clf[clf_types[0]]

    print(f"\n  ── Experiment 0 Summary ──────────────────────────────────")
    print(f"  Binary F1:   mean={np.mean(results_e0['fold_binary_f1']):.3f}"
          f"  std={np.std(results_e0['fold_binary_f1']):.3f}")
    print(f"  4-class macro F1: mean={np.mean(results_e0['fold_macro_f1']):.3f}"
          f"  std={np.std(results_e0['fold_macro_f1']):.3f}")
    recalls_e0 = results_e0['fold_recall']
    for si, sname in enumerate(CLASS_NAMES_4):
        vals = [r[si] for r in recalls_e0]
        print(f"  {sname:12s} recall: mean={np.mean(vals):.3f}"
              f"  std={np.std(vals):.3f}")

    print(f"\n  Classification report (pooled across all folds):")
    print(classification_report(
        results_e0['y_true_all'], results_e0['y_pred_all'],
        target_names=CLASS_NAMES_4, zero_division=0))

    write_baseline_table(results_e0, tbl_dir / 'table_8_ml_baseline.csv')

    # ── Experiment 1: Transfer test ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Transfer test")
    print(f"  Generating {args.n_synth} synthetic episodes ...")
    synth_episodes = generate_synthetic(
        args.params, args.n_synth, seed=args.seed)

    X_synth, y_synth_4 = episodes_to_xy(synth_episodes, binary=False)
    print(f"  Synthetic set: {X_synth.shape[0]:,} timesteps, "
          f"{X_synth.shape[1]} features")
    for s in range(N_STATES):
        n = int(np.sum(y_synth_4 == s))
        print(f"    {STATE_NAMES[s]:12s}: {n:8,} ({100*n/len(y_synth_4):.1f}%)")

    print(f"\n  Running LOPO evaluation on {len(processed)} participants ...")
    results_e1_by_clf = {}
    for clf_type in clf_types:
        print(f"\n  -- Classifier: {clf_type.upper()} --")
        results_e1_by_clf[clf_type] = lopo_evaluate(
            processed, X_synth, y_synth_4,
            n_trees=args.n_trees, seed=args.seed,
            clf_type=clf_type, verbose=True)

    results_e1 = results_e1_by_clf[clf_types[0]]

    # Print summary for each classifier
    for clf_type in clf_types:
        r = results_e1_by_clf[clf_type]
        print(f"\n  ── Experiment 1 Summary [{clf_type.upper()}] ─────────────────────────────")
        print(f"  Binary F1:   mean={np.mean(r['fold_binary_f1']):.3f}"
              f"  std={np.std(r['fold_binary_f1']):.3f}")
        print(f"  4-class macro F1: mean={np.mean(r['fold_macro_f1']):.3f}"
              f"  std={np.std(r['fold_macro_f1']):.3f}")
        for si, sname in enumerate(CLASS_NAMES_4):
            vals = [rec[si] for rec in r['fold_recall']]
            print(f"  {sname:12s} recall: mean={np.mean(vals):.3f}"
                  f"  std={np.std(vals):.3f}")

    # Full classification report on primary classifier
    print(f"\n  Classification report (pooled, {clf_types[0].upper()}):")
    print(classification_report(
        results_e1['y_true_all'], results_e1['y_pred_all'],
        target_names=CLASS_NAMES_4, zero_division=0))

    # Figures and tables
    fig_transfer_f1(results_e1,
                    fig_dir / 'fig_8_ml_transfer_f1.png')
    fig_confusion(results_e1['conf_matrix'],
                  fig_dir / 'fig_8_ml_confusion.png')
    fig_baseline_vs_synth(results_e0, results_e1,
                          fig_dir / 'fig_8_ml_baseline_vs_synth.png')
    write_transfer_table(results_e1,
                         tbl_dir / 'table_8_ml_transfer.csv')

    # Multi-classifier comparison figure
    if args.classifier == 'both':
        fig_classifier_comparison(
            results_e1_by_clf['rf'],
            results_e1_by_clf['lgbm'],
            label_rf   = 'Random Forest',
            label_lgbm = 'LightGBM',
            outpath    = fig_dir / 'fig_8_ml_rf_vs_lgbm.png')
        write_transfer_table(results_e1_by_clf['lgbm'],
                             tbl_dir / 'table_8_ml_transfer_lgbm.csv')
        print(f"\n  Classifier comparison:")
        rf_f1   = float(np.mean(results_e1_by_clf['rf']['fold_binary_f1']))
        lgbm_f1 = float(np.mean(results_e1_by_clf['lgbm']['fold_binary_f1']))
        print(f"    RF   binary F1: {rf_f1:.3f}")
        print(f"    LGBM binary F1: {lgbm_f1:.3f}")
        print(f"    Difference:    {lgbm_f1 - rf_f1:+.3f}")

    # ── GMM comparison (if params-gmm provided) ───────────────────────────────
    if args.params_gmm and _GMM_GEN_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 1b: GMM emission transfer test")
        print(f"  Generating {args.n_synth} synthetic episodes from GMM params ...")
        synth_gmm = generate_synthetic(
            args.params_gmm, args.n_synth,
            seed=args.seed, use_gmm=True)
        X_gmm, y_gmm = episodes_to_xy(synth_gmm, binary=False)
        print(f"  GMM synthetic set: {X_gmm.shape[0]:,} timesteps")
        for s in range(N_STATES):
            n_s = int(np.sum(y_gmm == s))
            print(f"    {STATE_NAMES[s]:12s}: {n_s:8,} ({100*n_s/len(y_gmm):.1f}%)")

        results_gmm_by_clf = {}
        for clf_type in clf_types:
            print(f"\n  -- Classifier: {clf_type.upper()} --")
            results_gmm_by_clf[clf_type] = lopo_evaluate(
                processed, X_gmm, y_gmm,
                n_trees=args.n_trees, seed=args.seed,
                clf_type=clf_type, verbose=True)

        results_e1_gmm = results_gmm_by_clf[clf_types[0]]

        for clf_type in clf_types:
            r = results_gmm_by_clf[clf_type]
            print(f"\n  ── GMM Summary [{clf_type.upper()}] ───────────────────────────────────")
            print(f"  Binary F1:   mean={np.mean(r['fold_binary_f1']):.3f}"
                  f"  std={np.std(r['fold_binary_f1']):.3f}")
            for si, sname in enumerate(CLASS_NAMES_4):
                vals = [rec[si] for rec in r['fold_recall']]
                print(f"  {sname:12s} recall: mean={np.mean(vals):.3f}")

        fig_gaussian_vs_gmm(results_e1, results_e1_gmm,
                            fig_dir / 'fig_8_ml_gaussian_vs_gmm.png')
        write_transfer_table(results_e1_gmm,
                             tbl_dir / 'table_8_ml_transfer_gmm.csv')

        print(f"\n  Gaussian vs GMM comparison:")
        g_f1 = float(np.mean(results_e1['fold_binary_f1']))
        m_f1 = float(np.mean(results_e1_gmm['fold_binary_f1']))
        print(f"    Gaussian binary F1: {g_f1:.3f}")
        print(f"    GMM      binary F1: {m_f1:.3f}")
        print(f"    Difference:        {m_f1 - g_f1:+.3f}")
    elif args.params_gmm and not _GMM_GEN_AVAILABLE:
        print(f"\n  WARNING: --params-gmm provided but "
              f"fog_episode_generator_gmm.py not found. Skipping GMM comparison.")

    # ── Experiment 2: Burden sweep ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: FoG burden sweep")
    print(f"  Burden levels: {[f'{b*100:.0f}%' for b in BURDEN_LEVELS]}")

    sweep_mean_f1   = []
    sweep_std_f1    = []
    sweep_mean_shuf = []
    sweep_std_shuf  = []
    sweep_h_dwells  = []

    for burden in BURDEN_LEVELS:
        print(f"\n  -- Burden {burden*100:.0f}% --")
        mu_h    = burden_to_healthy_dwell(params, burden)
        mutated = mutate_healthy_dwell(params, mu_h)
        sweep_h_dwells.append(mu_h)

        print(f"  Generating {args.n_synth} episodes ...")
        eps = generate_synthetic(
            args.params, args.n_synth,
            seed=args.seed + int(burden * 1000),
            params_override=mutated)

        X_s, y_s = episodes_to_xy(eps, binary=False)
        fog_frac  = float(np.mean(y_s > 0))
        print(f"  Actual FoG fraction in generated data: {fog_frac*100:.1f}%")

        res = lopo_evaluate(
            processed, X_s, y_s,
            n_trees=args.n_trees, seed=args.seed,
            clf_type=clf_types[0], verbose=False)

        m_f1 = float(np.mean(res['fold_binary_f1']))
        s_f1 = float(np.std(res['fold_binary_f1']))
        shuf  = [r[SHUFFLING] for r in res['fold_recall']]
        m_sh = float(np.mean(shuf))
        s_sh = float(np.std(shuf))

        sweep_mean_f1.append(m_f1)
        sweep_std_f1.append(s_f1)
        sweep_mean_shuf.append(m_sh)
        sweep_std_shuf.append(s_sh)

        print(f"  Binary F1: {m_f1:.3f} ± {s_f1:.3f}  "
              f"Shuffling recall: {m_sh:.3f} ± {s_sh:.3f}")

    print(f"\n  ── Experiment 2 Summary ──────────────────────────────────")
    for b, mf, sf in zip(BURDEN_LEVELS, sweep_mean_f1, sweep_std_f1):
        print(f"  {b*100:.0f}%: binary_F1={mf:.3f} ± {sf:.3f}")

    fig_burden_sweep(BURDEN_LEVELS, sweep_mean_f1, sweep_std_f1,
                     fitted_burden,
                     fig_dir / 'fig_8_ml_burden_sweep.png')
    write_burden_table(BURDEN_LEVELS, sweep_mean_f1, sweep_std_f1,
                       sweep_mean_shuf, sweep_std_shuf, sweep_h_dwells,
                       tbl_dir / 'table_8_ml_burden.csv')

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY: Real-data LOPO vs Synthetic Transfer")
    print(f"  {'Metric':30s}  {'Real LOPO':>10}  {'Synthetic':>10}  {'Diff':>8}")
    print(f"  {'-'*62}")

    e0_bf1 = float(np.mean(results_e0['fold_binary_f1']))
    e1_bf1 = float(np.mean(results_e1['fold_binary_f1']))
    e0_mf1 = float(np.mean(results_e0['fold_macro_f1']))
    e1_mf1 = float(np.mean(results_e1['fold_macro_f1']))

    def _rec(results, si):
        return float(np.mean([r[si] for r in results['fold_recall']]))

    rows = [
        ('Binary F1 (mean)',       e0_bf1, e1_bf1),
        ('4-class macro F1 (mean)', e0_mf1, e1_mf1),
        ('Healthy recall',         _rec(results_e0,0), _rec(results_e1,0)),
        ('Shuffling recall',       _rec(results_e0,1), _rec(results_e1,1)),
        ('Trembling recall',       _rec(results_e0,2), _rec(results_e1,2)),
        ('Akinesia recall',        _rec(results_e0,3), _rec(results_e1,3)),
    ]
    for label, rv, sv in rows:
        diff = sv - rv
        sign = '+' if diff >= 0 else ''
        print(f"  {label:30s}  {rv:10.3f}  {sv:10.3f}  {sign}{diff:7.3f}")

    print(f"\n{'='*60}")
    print(f"All outputs written to: {Path(args.outdir).resolve()}")
    print(f"  Figures : {fig_dir}")
    print(f"  Tables  : {tbl_dir}")


if __name__ == '__main__':
    main()