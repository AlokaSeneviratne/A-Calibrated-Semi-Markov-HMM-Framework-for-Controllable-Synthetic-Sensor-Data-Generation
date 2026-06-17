# A Calibrated Semi-Markov HMM Framework for Controllable Synthetic Sensor Data Generation

Master's thesis, University of Oulu, Electronics and Communications Engineering.
Author: Praveen Aloka Seneviratne. Supervisor: Prof. Sumudu Samarakoon. Second examiner: Dr. Chathuranga Weeraddana.

## Overview

This project is a general-purpose system for generating modifiable synthetic time-series sensor data from any dataset that fits a semi-Markov hidden Markov model. The goal is to let machine-learning researchers produce unlimited labelled training data with controllable parameters, so that scarce or imbalanced real recordings can be supplemented with synthetic episodes whose statistical properties match the source data while remaining tunable.

Freezing of Gait (FoG) detection on the FoG-STAR dataset is used as the proof-of-concept demonstration. It is not the primary subject; it is the worked example that shows the framework reproduces real signal statistics, responds predictably to parameter changes, and yields synthetic data that transfers to a downstream classifier.

## Method

The generative model is a flat semi-Markov HMM with four peer states: Healthy, Shuffling, Trembling, and Akinesia. There is no hierarchy. Each state carries an explicit dwell-time distribution, fitted from the real episodes, so episode duration in each state is sampled directly rather than left to a geometric self-transition.

Within each state, observations are produced by a per-state AR(1) process operating in a global z-score space. Features are standardised per (node, feature) across the pooled population, in log space for the strictly positive features (FI, RMS_acc, RMS_gyr) and in linear space for delta_FI. The AR(1) recurrence is centred on the mean:

```
x_t = mu + rho * (x_{t-1} - mu) + sqrt(1 - rho^2) * (z - mu),   z ~ N(mu, Sigma)
```

This form preserves the stationary mean, the stationary covariance, and the autocorrelation rho^k. The AR(1) coefficients in use are 0.98 for Healthy and Akinesia, 0.70 for Shuffling, and 0.92 for Trembling.

## Dataset

FoG-STAR: 22 Parkinson's disease participants, 60 Hz IMU recordings. Three active sensor nodes are used: back (0), wrist (1), and right ankle (2). The left ankle is excluded from training because of NaN values for some participants. Each node carries four features: FI (0), RMS_acc (1), RMS_gyr (2), and delta_FI (3). The four states are Healthy (0), Shuffling (1), Trembling (2), and Akinesia (3). The pipeline starts from 101 episodes and retains 74 after the valid-mask filter.

## Pipeline

| File | Role |
|------|------|
| `fog_star_loader.py`, `fog_generic_loader.py` | Load and pre-process real recordings into per-participant feature tensors |
| `fog_model_fitter.py` | Fit per-state dwell distributions, emission means and covariances, and AR(1) coefficients |
| `fog_episode_generator.py` | Sample synthetic episodes from the fitted parameters with controllable overrides |
| `fog_validate.py` | Statistical validation: KS, Mahalanobis KS, energy distance, dwell-time and AR(1) recovery |
| `fog_sensitivity.py` | Parameter sweeps measuring controllability against fidelity |
| `fog_ml_validation.py` | Downstream classifier experiments (real baseline vs synthetic transfer) |
| `fog_app.py` | Streamlit app for interactive parameter control and episode inspection |

## Results

### Emission fidelity (per-state, pooled across nodes)

Mahalanobis KS, the joint-distribution agreement measure, stays in a tight band across all four states. Univariate KS is reported per feature.

| State | FI | RMS_acc | RMS_gyr | delta_FI | Mahalanobis KS | Energy dist. |
|-------|----|---------|---------|----------|----------------|--------------|
| Healthy   | 0.034 | 0.170 | 0.076 | 0.240 | 0.169 | 0.031 |
| Shuffling | 0.166 | 0.235 | 0.163 | 0.153 | 0.125 | 0.117 |
| Trembling | 0.243 | 0.253 | 0.102 | 0.179 | 0.144 | 0.188 |
| Akinesia  | 0.031 | 0.300 | 0.230 | 0.248 | 0.133 | 0.205 |

The Mahalanobis KS of 0.13 to 0.17 reflects the gain from moving to a single global z-score per (node, feature) in log space. The known limitation is that synthetic inter-node correlation falls below the real correlation, which follows directly from independent per-node emission and is reported as a finding rather than a defect.

### Dwell-time fidelity

Synthetic dwell-time means track the real means closely. The largest relative error is the Healthy state, which is also the state with the heaviest tail.

| State | Real mean (s) | Synth mean (s) | KS | Mean error |
|-------|---------------|----------------|----|------------|
| Healthy   | 40.29 | 32.94 | 0.119 | -18.2% |
| Shuffling | 2.85  | 2.84  | 0.159 | -0.4%  |
| Trembling | 6.34  | 5.99  | 0.083 | -5.5%  |
| Akinesia  | 20.36 | 18.55 | 0.097 | -8.9%  |

<img width="1525" height="1099" alt="fig_6_2_dwell_distributions" src="https://github.com/user-attachments/assets/bda28e39-47ea-4781-8f32-956b80818da7" />


### AR(1) recovery

The empirical AR(1) coefficient measured back from synthetic ankle-FI traces matches the fitted target to within about 0.02 in every state.

| State | Fitted rho | Synthetic rho | Abs. diff |
|-------|-----------|---------------|-----------|
| Healthy   | 0.980 | 0.973 | 0.007 |
| Shuffling | 0.700 | 0.686 | 0.014 |
| Trembling | 0.920 | 0.902 | 0.018 |
| Akinesia  | 0.980 | 0.968 | 0.012 |

### Controllability (sensitivity sweeps)

The point of the framework is that target parameters move the synthetic output predictably while fidelity holds. The FoG fraction responds monotonically to both the Akinesia dwell target and the FoG burden target, and the per-node KS values stay flat across each sweep.

- Akinesia dwell target swept 5 to 80 s raises the realised FoG fraction from 0.135 to 0.424, with ankle-FI KS holding near 0.16 to 0.18.
- FoG burden target swept 5% to 50% gives realised FoG fractions of 0.065 to 0.503, again with stable KS.
- Shuffling AR(1) coefficient has little effect on FoG fraction, as expected, and only the extreme 0.99 setting noticeably perturbs ankle-FI KS.
  
<img width="703" height="375" alt="app_interface" src="https://github.com/user-attachments/assets/df994616-710d-455a-a61c-a0ecc21b5e6d" />

### Downstream machine learning

A random forest trained only on 100 synthetic episodes and tested on real FoG-STAR with leave-one-participant-out cross-validation outperforms the equivalent real-data-only LOPO baseline on binary FoG detection.

| Experiment | Binary F1 | Macro 4-class F1 | Shuffling recall |
|------------|-----------|------------------|------------------|
| Real-data LOPO baseline | 0.163 | 0.481 | 0.000 |
| Synthetic transfer      | 0.346 | 0.357 | 0.093 |

The binary F1 more than doubles under synthetic training. The macro 4-class figure for the baseline is inflated by trivial all-Healthy folds and is not the meaningful comparison; binary F1 is. Minority-state recall, Shuffling in particular, remains the principal weakness of both settings. A burden sweep confirms the mechanism: raising the synthetic FoG burden from 10% to 40% lifts mean binary F1 from 0.318 to 0.364 and Shuffling recall from 0.027 to 0.120.
<img width="1785" height="586" alt="fig_8_ml_baseline_vs_synth" src="https://github.com/user-attachments/assets/046975bb-09ea-4a56-be80-c91f279f99c2" />

## Limitations

- Independent per-node emission means synthetic inter-node correlation sits below the real correlation.
- Minority states, especially Shuffling, are still hard to recover downstream despite balanced synthetic exposure.
- Heavy-tailed dwell distributions, such as the Healthy state, are reproduced with a modest negative mean bias.

## Future work

The framework was originally motivated by a sensor-scheduling problem framed as a decentralised POMDP and solved with multi-agent reinforcement learning (MAPPO). That direction is retained as future work: the synthetic generator can supply unlimited labelled episodes for training and evaluating such an agent without further real data collection. The continuation is being developed here: [Emergent Communication in Partially Observable Multi-Agent IoT Systems](https://github.com/AlokaSeneviratne/Emergent-Communication-in-Partially-Observable-Multi-Agent-IoT-Systems).
