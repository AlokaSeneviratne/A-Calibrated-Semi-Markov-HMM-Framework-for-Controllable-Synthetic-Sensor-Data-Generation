"""
fog_episode_generator.py
------------------------
Task 4: Episode generator.

Combines the fitted semi-Markov HMM with the Gaussian emission model
to produce synthetic FoG episodes for RL training.

Each call to sample_episode() returns a complete episode containing:
    states    (T,)       ground truth HMM state per timestep
    obs       (T, 3, 4)  feature vectors per timestep per node
    flags     (T, 3)     2-bit severity flag per timestep per node

The generator is the only thing the Dec-POMDP environment needs.
Everything upstream (loading, fitting) is done once offline.
"""


import os; os.environ.setdefault("PYTHONIOENCODING", "utf-8"); import sys; sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

from fog_star_loader import (
    N_STATES, N_NODES, N_FEATURES,
    STATE_NAMES, NODE_NAMES, FEATURE_NAMES,
    FS,
    ParticipantData, DatasetStats   # required for pickle to deserialise fogstar_processed.pkl
)
from fog_model_fitter import (
    HMMParams,
    DwellDistribution, EmissionParams, PreprocessParams
    # All three required for pickle to deserialise fogstar_model_params.pkl
)


# ── flag encoding thresholds (z-score space) ─────────────────────────────────
# Observations are in preprocessed z-score space (log-transformed then
# z-scored per participant). Thresholds must match this space.
#
# From the fitted emission means at the ankle node:
#   Healthy   FI = -0.18    RMS_acc = +0.16
#   Shuffling FI = +1.33    RMS_acc = -0.52
#   Trembling FI = +0.89    RMS_acc = +0.14
#   Akinesia  FI = +0.42    RMS_acc = -0.67
#
# Flag encoding:
#   00 = Healthy
#   01 = Shuffling  (FI elevated, still actively moving)
#   10 = Trembling  (FI highly elevated)
#   11 = Akinesia   (FI moderate but RMS_acc very low -- near-zero movement)
#
# These thresholds are calibrated from the emission means above.
# They are deliberately conservative -- the hub alert logic applies
# its own multi-node voting, so per-node flags can afford to be noisy.

THETA_FOG   =  0.20   # FI z-score: above this suggests FoG activity
THETA_HIGH  =  1.00   # FI z-score: above this suggests Trembling
THETA_QUIET = -0.40   # RMS_acc z-score: below this AND moderate FI -> Akinesia


# ── episode output ────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """One complete simulated episode."""
    states:       np.ndarray   # (T,)      integer state label, ground truth
    obs:          np.ndarray   # (T, 3, 4) feature vector per timestep per node
    flags:        np.ndarray   # (T, 3)    2-bit flag per timestep per node
    dwell_info:   list         # list of (state, start_t, end_t) for each dwell
    T:            int          # total timesteps

    def fog_episode_count(self) -> int:
        """Number of complete FoG episodes in this simulation episode."""
        in_fog  = False
        count   = 0
        for s in self.states:
            if s > 0 and not in_fog:
                in_fog = True
                count += 1
            elif s == 0:
                in_fog = False
        return count


# ── generator ─────────────────────────────────────────────────────────────────

class FogEpisodeGenerator:
    """
    Generates synthetic FoG episodes by combining:
        1. Semi-Markov HMM for state sequence generation
        2. Multivariate Gaussian emission model for feature generation
        3. AR(1) process for temporal coherence within each state

    Usage
    -----
        gen     = FogEpisodeGenerator('fogstar_model_params.pkl')
        episode = gen.sample_episode()

    Parameters
    ----------
    params_path : path to fitted HMMParams pickle
    seed        : optional random seed for reproducibility
    min_fog     : minimum number of FoG episodes to include
                  (generator keeps sampling until this is met)
    max_seconds : maximum episode length in seconds
    min_seconds : minimum episode length in seconds
    """

    def __init__(self,
                 params_path:  str,
                 seed:         Optional[int] = None,
                 min_fog:      int   = 1,
                 max_seconds:  float = 300.0,
                 min_seconds:  float = 30.0):

        # Custom unpickler: remaps __main__ -> fog_model_fitter so that
        # fogstar_model_params.pkl deserialises correctly regardless of
        # whether fog_model_fitter.py was run as __main__ (which causes
        # pickle to store class paths as __main__.HMMParams rather than
        # fog_model_fitter.HMMParams, breaking loading from any other module).
        import fog_model_fitter as _fmf  # noqa: F401 -- ensures classes registered

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__':
                    module = 'fog_model_fitter'
                return super().find_class(module, name)

        with open(params_path, 'rb') as f:
            self.params: HMMParams = _Unpickler(f).load()

        self.rng         = np.random.default_rng(seed)
        self.min_fog     = min_fog
        self.max_steps   = int(max_seconds * FS)
        self.min_steps   = int(min_seconds * FS)

        # Cache emission params as arrays for speed
        # mu[s, n] shape (4,)    Sigma[s, n] shape (4,4)
        self._mu    = np.array([[self.params.emission[s][n].mu
                                  for n in range(N_NODES)]
                                  for s in range(N_STATES)])   # (4, 3, 4)
        self._Sigma = np.array([[self.params.emission[s][n].Sigma
                                  for n in range(N_NODES)]
                                  for s in range(N_STATES)])   # (4, 3, 4, 4)

        # Cache per-state rho as a numpy array for fast indexing
        self._rho = np.array([self.params.rho[s] for s in range(N_STATES)])

        rho_str = "  ".join(f"{STATE_NAMES[s]}={self._rho[s]:.3f}"
                             for s in range(N_STATES))
        print(f"FogEpisodeGenerator ready. min_fog={min_fog}  "
              f"max_episode={max_seconds:.0f}s")
        print(f"  Per-state rho: {rho_str}")

    # ── public API ────────────────────────────────────────────────────────────

    def sample_episode(self) -> Episode:
        """
        Sample one complete episode.
        Retries until at least min_fog FoG episodes are present.
        """
        for attempt in range(100):
            ep = self._generate_episode()
            if ep.fog_episode_count() >= self.min_fog:
                return ep
        # If we never got enough FoG episodes, return last attempt
        return ep

    # ── internal generation ───────────────────────────────────────────────────

    def _generate_episode(self) -> Episode:
        """
        Generate one episode without checking min_fog constraint.

        Vectorised: observations for each dwell period are sampled in a
        single batch call rather than one timestep at a time, giving
        roughly 100x speedup over the per-timestep loop.
        """
        states_list = []
        obs_list    = []
        dwell_info  = []

        s0       = self._sample_initial_state()
        phi_prev = self._mu[s0].copy()   # (3, 4)  AR(1) carry-over
        current  = s0
        t        = 0

        while t < self.max_steps:
            dwell_steps = self.params.dwell[current].sample_timesteps(self.rng)
            dwell_steps = min(dwell_steps, self.max_steps - t)

            dwell_info.append((current, t, t + dwell_steps))

            # ── vectorised dwell generation ───────────────────────────────────
            # Sample all iid Gaussian draws for this dwell at once: (D, 3, 4)
            iid = np.stack([
                self.rng.multivariate_normal(
                    self._mu[current, n],
                    self._Sigma[current, n],
                    size=dwell_steps          # shape (D, 4)
                )
                for n in range(N_NODES)
            ], axis=1)                        # (D, 3, 4)

            # Apply AR(1) recurrence across the dwell dimension.
            # Variance-preserving form: stationary mean mu, stationary covariance
            # Sigma, and lag-1 autocorrelation rho.
            #   carry = mu + rho*(carry - mu) + sqrt(1-rho^2)*(z - mu),  z ~ N(mu, Sigma)
            # The earlier rho*carry + (1-rho)*z form shrank the stationary variance
            # to (1-rho)/(1+rho)*Sigma, collapsing the synthetic spread.
            rho       = self._rho[current]
            a_innov   = np.sqrt(1.0 - rho * rho)   # innovation scale that keeps Var = Sigma
            mu_cur    = self._mu[current]          # (3, 4)
            obs_dwell = np.empty_like(iid)
            carry     = phi_prev
            for d in range(dwell_steps):
                carry        = mu_cur + rho * (carry - mu_cur) + a_innov * (iid[d] - mu_cur)
                obs_dwell[d] = carry
            phi_prev = carry

            states_list.append(np.full(dwell_steps, current, dtype=np.int32))
            obs_list.append(obs_dwell)

            t      += dwell_steps
            next_s  = self._sample_next_state(current)
            phi_prev = self._boundary_reset(phi_prev, next_s)
            current  = next_s

        # Minimum length top-up (rarely needed)
        while sum(len(s) for s in states_list) < self.min_steps:
            iid  = np.stack([
                self.rng.multivariate_normal(self._mu[current, n], self._Sigma[current, n])
                for n in range(N_NODES)
            ])   # (3, 4)
            rho      = self._rho[current]
            a_innov  = np.sqrt(1.0 - rho * rho)
            mu_cur   = self._mu[current]
            carry    = mu_cur + rho * (phi_prev - mu_cur) + a_innov * (iid - mu_cur)
            phi_prev = carry
            states_list.append(np.array([current], dtype=np.int32))
            obs_list.append(carry[np.newaxis])   # (1, 3, 4)

        states = np.concatenate(states_list, axis=0)   # (T,)
        obs    = np.concatenate(obs_list,    axis=0)   # (T, 3, 4)
        flags  = self._compute_flags_batch(obs)         # (T, 3)

        return Episode(
            states     = states,
            obs        = obs,
            flags      = flags,
            dwell_info = dwell_info,
            T          = len(states),
        )

    def _sample_initial_state(self) -> int:
        return int(self.rng.choice(N_STATES, p=self.params.pi_0))

    def _sample_next_state(self, current: int) -> int:
        return int(self.rng.choice(N_STATES, p=self.params.T_mat[current]))

    def _boundary_reset(self,
                         phi_prev:   np.ndarray,
                         next_state: int) -> np.ndarray:
        """
        At a state transition, blend phi_prev toward the new state's mean.
        This prevents a sharp discontinuity in the AR(1) signal at the
        boundary between states.

        Uses a single-step blend: new_phi = 0.5 * phi_prev + 0.5 * mu_new
        """
        mu_new   = self._mu[next_state]   # (3, 4)
        return 0.5 * phi_prev + 0.5 * mu_new

    def _compute_flags_batch(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute 2-bit flags for an entire episode at once.

        Parameters
        ----------
        obs : (T, N_NODES, N_FEATURES) z-scored feature array

        Returns
        -------
        flags : (T, N_NODES) int8 array of values in {0, 1, 2, 3}
        """
        fi      = obs[:, :, 0]   # (T, 3)
        rms_acc = obs[:, :, 1]   # (T, 3)

        flags = np.zeros((len(obs), N_NODES), dtype=np.int8)
        # Order matters: check Akinesia first (special case)
        flags[fi >= THETA_FOG]                             = 1  # Shuffling
        flags[fi >= THETA_HIGH]                            = 2  # Trembling
        flags[(fi < THETA_HIGH) & (rms_acc < THETA_QUIET)] = 3  # Akinesia
        return flags


# ── validation ────────────────────────────────────────────────────────────────

def validate_generator(generator: 'FogEpisodeGenerator',
                        n_episodes: int = 100):
    """
    Validate the generator by running episodes and checking clinical consistency.
    Uses vectorised numpy operations -- no per-timestep Python loops.
    """
    print(f"\n── Generator Validation ({n_episodes} episodes) ───────────────")

    all_states  = []
    all_obs     = []   # ankle FI only -- shape (T,) per episode
    all_flags   = []   # shape (T, N_NODES) per episode
    fog_per_ep  = []

    for _ in range(n_episodes):
        ep = generator.sample_episode()
        fog_per_ep.append(ep.fog_episode_count())
        all_states.append(ep.states)               # (T,)
        all_obs.append(ep.obs[:, 2, 0])            # ankle FI only
        all_flags.append(ep.flags)                  # (T, 3)

    states = np.concatenate(all_states)            # (total_T,)
    ankle_fi = np.concatenate(all_obs)             # (total_T,)
    flags  = np.concatenate(all_flags, axis=0)     # (total_T, 3)

    total_t = len(states)
    print(f"\n  Episodes generated: {n_episodes}  "
          f"Total timesteps: {total_t:,}")
    print(f"  FoG episodes per sim episode: "
          f"mean={np.mean(fog_per_ep):.2f} ± {np.std(fog_per_ep):.2f}  "
          f"min={min(fog_per_ep)}  max={max(fog_per_ep)}")

    print(f"\n  State proportions (FoG-STAR reference: H=75%, S=1%, T=5%, A=19%):")
    for s in range(N_STATES):
        count = int(np.sum(states == s))
        pct   = 100 * count / max(total_t, 1)
        print(f"    {STATE_NAMES[s]:12s}: {count:8,} timesteps ({pct:5.1f}%)")

    print(f"\n  Ankle FI means per state (z-score space):")
    fi_means = {}
    for s in range(N_STATES):
        mask = states == s
        fi_means[s] = float(np.mean(ankle_fi[mask])) if mask.sum() > 0 else float('nan')
        print(f"    {STATE_NAMES[s]:12s}: mean={fi_means[s]:7.3f}")

    print(f"\n  Flag distribution across all nodes and timesteps:")
    flag_labels = ['Healthy(00)', 'Shuffling(01)', 'Trembling(10)', 'Akinesia(11)']
    flat_flags  = flags.flatten()
    total_flags = len(flat_flags)
    for i, label in enumerate(flag_labels):
        count = int(np.sum(flat_flags == i))
        pct   = 100 * count / max(total_flags, 1)
        print(f"    {label:14s}: {count:8,} ({pct:5.1f}%)")

    print(f"\n  Sanity checks:")
    checks = [
        ("At least one FoG episode generated",
         sum(fog_per_ep) > 0),
        ("Ankle FI: Shuffling > Healthy",
         fi_means[1] > fi_means[0]),
        ("Ankle FI: Trembling > Healthy",
         fi_means[2] > fi_means[0]),
        ("Ankle FI: Shuffling > Akinesia",
         fi_means[1] > fi_means[3]),
        ("FoG flags generated (not all Healthy)",
         int(np.sum(flat_flags > 0)) > 0),
    ]
    all_pass = True
    for name, result in checks:
        print(f"    {'PASS' if result else 'FAIL'}  {name}")
        if not result:
            all_pass = False

    if all_pass:
        print("\n  All checks passed. Generator is working correctly.")
    else:
        print("\n  WARNING: One or more checks failed.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    params_path = sys.argv[1] if len(sys.argv) > 1 else 'fogstar_model_params.pkl'

    gen = FogEpisodeGenerator(params_path, seed=0)

    # Quick smoke test
    print("\nSample episode:")
    ep = gen.sample_episode()
    print(f"  Length:        {ep.T} timesteps ({ep.T/FS:.1f}s)")
    print(f"  FoG episodes:  {ep.fog_episode_count()}")
    print(f"  State counts:  "
          + ", ".join(f"{STATE_NAMES[s]}={np.sum(ep.states==s)}"
                      for s in range(N_STATES)))
    print(f"  obs shape:     {ep.obs.shape}")
    print(f"  flags shape:   {ep.flags.shape}")
    print(f"  Dwell periods: {len(ep.dwell_info)}")
    for state, t0, t1 in ep.dwell_info[:5]:
        print(f"    {STATE_NAMES[state]:12s}: t={t0}..{t1} ({(t1-t0)/FS:.2f}s)")

    # Full validation
    validate_generator(gen, n_episodes=100)

    # ── Plot one episode ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t_axis = np.arange(ep.T) / FS   # seconds

    for ni, nname in enumerate(NODE_NAMES):
        axes[ni].plot(t_axis, ep.obs[:, ni, 0],
                      linewidth=0.6, color='steelblue', label='FI')
        axes[ni].set_ylabel(f'{nname}\nFI (z-score)', fontsize=9)
        axes[ni].axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # Shade FoG periods by severity
        colours = {1: 'gold', 2: 'orange', 3: 'red'}
        labels  = {1: 'Shuffling', 2: 'Trembling', 3: 'Akinesia'}
        seen    = set()
        for state, t0, t1 in ep.dwell_info:
            if state > 0:
                lbl = labels[state] if state not in seen else None
                axes[ni].axvspan(t0/FS, t1/FS,
                                 alpha=0.35, color=colours[state], label=lbl)
                seen.add(state)

        if ni == 0:
            axes[ni].legend(loc='upper right', fontsize=8)

    axes[0].set_title('Generated Episode -- Freeze Index per node\n'
                       '(shaded regions = FoG: gold=Shuffling, '
                       'orange=Trembling, red=Akinesia)',
                       fontsize=10)
    axes[2].set_xlabel('Time (seconds)', fontsize=9)
    plt.tight_layout()
    plt.savefig('episode_plot.png', dpi=150, bbox_inches='tight')
    print("\nSaved episode_plot.png")