# Personalized Physiological Forecasting & Anomaly Detection System

A "digital twin" approach to wearable HR/HRV monitoring: learn what is *normal for this person*,
forecast their HR 1/5/10 minutes ahead, and flag meaningful deviations from their own baseline.

This document is the full technical design. The accompanying `src/` package is a working
reference implementation that runs against the Empatica E4 `Dataset3/Raw_data/SXX` files
(HR, IBI, ACC, BVP, EDA, TEMP, tags) and is structured so the same code scales to 15+ days
of continuous single-subject data.

---

## 1. System Architecture Diagram

```mermaid
flowchart TD
    subgraph Sensors["Wearable (Empatica E4 / future smartwatch)"]
        ACC[ACC 32Hz]
        BVP[BVP 64Hz]
        EDA[EDA 4Hz]
        TEMP[TEMP 4Hz]
        HR[HR ~1Hz]
        IBI[IBI irregular]
    end

    ACC & BVP & EDA & TEMP & HR & IBI --> SYNC[Timestamp Sync + Resample to 1Hz grid]
    SYNC --> CLEAN[Signal Cleaning\nartifact removal, missing-value handling]
    CLEAN --> FEAT[Feature Engineering\nHR / HRV / ACC / BVP / TEMP / EDA / time features\n+ circadian baseline]
    FEAT --> STATE[Physiological State Classifier\nsleep/rest/focused_work/walking/exercise/recovery]
    STATE --> RECOV[Recovery Features\nHR-recovery rate after exercise]
    RECOV --> WIN[Windowing\ninput sequence -> multi-horizon targets\n(HR + RMSSD)]

    WIN --> CAL{Calibration\ncomplete?\n(>= N days)}
    CAL -- no --> BASELINE[Bootstrap with population /\ndefault-prior model\n(global pretrain)]
    CAL -- yes --> PERSONAL[Personalized Forecaster\n(TCN/LSTM/GRU/Transformer/XGBoost,\nper-subject fine-tuned)]

    BASELINE -. transfer + fine-tune .-> PERSONAL
    PERSONAL --> PRED[Forecast: HR & RMSSD\n@+1min, +5min, +10min\n+ predictive uncertainty]
    WIN --> ACTUAL[Actual HR / RMSSD stream]
    STATE --> TWIN[Personal Digital Twin\nresting/sleep/walk/run HR,\ncircadian table]

    PRED & ACTUAL --> DEV[Deviation Engine\nresidual, z-score, prediction interval,\ncircadian z-score, digital-twin score]
    TWIN --> DEV
    DEV --> THRESH[Adaptive Thresholds\nEWMA mean/std of residuals]
    THRESH --> SCORE[Combined Anomaly Score\n+ Illness Score]
    SCORE --> ALERT{Score > threshold\nfor >= K samples?}
    ALERT -- yes --> EXPLAIN[Explainable Alert Reason\n+HR above expected, RMSSD drop,\nTemp elevated, Activity state]
    EXPLAIN --> NOTIFY[Alert / Notification]
    ALERT -- no --> LOG[Log: normal]

    PERSONAL -. periodic, small recent window .-> ONLINE[Online Incremental Update\n(scaler partial_fit + low-LR fine-tune)]
    LOG -. feedback .-> ONLINE
    ONLINE --> PERSONAL

    SIM[Simulation Engine\nScenarios A-G] -. synthetic streams .-> SYNC
```

## 2. Data Flow Diagram

```mermaid
flowchart LR
    A[Raw CSVs per subject\nACC/BVP/EDA/TEMP/HR/IBI/tags] --> B[Loader\nparse start-time + Hz from header]
    B --> C[Per-signal DataFrame\nDatetimeIndex]
    C --> D[Resample to common 1Hz grid\n(mean/agg per second)]
    D --> E[Merge into single\nmultivariate DataFrame]
    E --> F[Cleaning\nclip artifacts, interpolate gaps <= 5s,\nflag longer gaps]
    F --> G[Feature Extraction\nrolling windows per signal]
    G --> H[Feature Matrix X(t)\n+ target HR(t+1m), HR(t+5m), HR(t+10m)]
    H --> I[Train/Val/Test split\nwalk-forward, chronological]
    I --> J[Model Training\nTCN / LSTM / GRU / Transformer / Kalman+ML]
    J --> K[Trained personal model\n+ residual statistics]
    K --> L[Inference: streaming window -> forecast + interval]
    L --> M[Anomaly scoring vs actual]
    M --> N[Alerts / Dashboard / Logs]
```

---

## 3. Data Pipeline

### 3.1 Reading raw files
Each `SXX/` folder follows the Empatica E4 export format (see `info.txt`):
- `HR.csv`, `EDA.csv`, `TEMP.csv`: row 1 = Unix start timestamp (UTC), row 2 = sample rate (Hz), then data.
- `ACC.csv`: row 1 = start timestamp (x3, identical), row 2 = sample rate (x3), then `x,y,z` in units of 1/64 g.
- `BVP.csv`: same 2-row header format, 64 Hz.
- `IBI.csv`: header row `start_timestamp, IBI`, then `(t_offset_seconds, ibi_seconds)` pairs — irregular sampling, no fixed Hz.
- `tags_SXX.csv`: one Unix timestamp per line marking task-transition button presses.

`src/data/loader.py` parses each file into a `pandas.Series`/`DataFrame` indexed by `DatetimeIndex` (UTC), using the declared start time and sample rate (or offsets, for IBI).

### 3.2 Timestamp synchronization & resampling
All signals are resampled onto a **common 1 Hz grid** (configurable — `1S` is the default,
chosen because HR is natively ~1 Hz and this is fine enough for minute-scale forecasting
while being cheap enough for on-device streaming):

- **ACC (32 Hz)** → per-second aggregates: mean magnitude, std, max magnitude (captures activity intensity within the second).
- **BVP (64 Hz)** → per-second aggregates: peak count, amplitude stats (pulse features computed first at native rate, then aggregated).
- **EDA / TEMP (4 Hz)** → per-second mean (simple downsampling; these are slow signals).
- **HR (~1 Hz)** → already aligned; forward-filled to the grid if timestamps drift slightly.
- **IBI (irregular)** → beats are assigned to the second in which they occur; HRV metrics are computed on a **trailing rolling window** (e.g., last 60s of beats) and sampled onto the 1 Hz grid.

### 3.3 Window generation
Two levels of windows:
1. **Feature windows** (used inside `features/`): rolling windows of width `w` (e.g., 30s, 60s, 300s) ending at time `t`, used to compute rolling statistics, HRV metrics, activity levels, etc. → produces a feature vector `x(t)`.
2. **Model (sequence) windows**: a contiguous sequence of `x(t-L+1..t)` (input sequence length `L`, default 600s = 10 min) used as model input; targets are `HR(t+60s)`, `HR(t+300s)`, `HR(t+600s)`.

### 3.4 Missing value handling
- Gaps ≤ 5 s: linear interpolation.
- Gaps 5–60 s: forward-fill with a "stale data" flag feature (1/0) so the model can down-weight it.
- Gaps > 60 s (sensor off, charging, button press artifacts): segment the recording — do **not** build sequence windows that straddle a gap.

### 3.5 Signal cleaning
- **BVP**: band-pass filter 0.5–8 Hz (Butterworth, order 3) to remove baseline drift and high-frequency noise before peak detection.
- **EDA**: low-pass filter ~1 Hz (Butterworth) then split into tonic (very slow, moving-average) and phasic (residual) components.
- **ACC**: remove gravity by high-pass filtering or by subtracting a slow moving average per axis before computing magnitude.
- **TEMP**: light moving-average smoothing (sensor is already slow/clean).
- **IBI**: discard physiologically implausible values (`< 0.3s` i.e. >200bpm, `> 2.0s` i.e. <30bpm), then optionally apply Malik/Karlsson-style ectopic-beat correction.
- **HR**: clip to plausible range (30–220 bpm); short spikes (1 sample, > 40 bpm jump and reverts) are treated as artifacts and interpolated.

---

## 4. Feature Engineering

All features are computed causally (only using data up to time `t`) so they are valid for real-time deployment.

### HR features (window sizes: 30s, 60s, 300s)
- Rolling mean, rolling std
- Trend = slope of linear regression of HR over the window
- Rate of change = `HR(t) - HR(t-Δ)` for `Δ ∈ {5s, 30s, 60s}`
- Deviation from personal resting HR baseline (see §6)

### IBI / HRV features (sliding 60s / 300s windows of beats)
- **RMSSD** = `sqrt(mean((IBI[i+1] - IBI[i])^2))`
- **SDNN** = `std(IBI)`
- **pNN50** = `fraction of |IBI[i+1]-IBI[i]| > 50ms`
- Mean HR from IBI = `60 / mean(IBI)`
- Optionally LF/HF power ratio (frequency-domain HRV) if window long enough (≥120s)

### ACC features (1s aggregation + rolling 10s/60s)
- Magnitude `m(t) = sqrt(x^2+y^2+z^2)` (gravity-removed)
- Activity intensity = rolling mean/std of `m(t)`
- Movement detection = binary flag, `mean(m) > activity_threshold`
- Posture/activity estimation = coarse classifier from mean axis orientation + intensity buckets (rest / light / moderate / vigorous), see §4.1

### BVP features (computed at 64Hz, aggregated to 1s)
- Pulse peak detection (`scipy.signal.find_peaks` on cleaned BVP)
- Pulse rate from peaks (cross-check vs HR.csv)
- Pulse amplitude (peak-to-trough), and amplitude variability (std of amplitudes in window)
- Signal quality index (e.g., peak regularity / SNR proxy)

### TEMP features
- Rolling mean / trend (slope over 5min)
- Deviation from personal baseline temperature (EWMA of TEMP over calibration period)

### EDA features
- Tonic level (slow moving average)
- Phasic activity = count/magnitude of SCRs (skin-conductance responses) above threshold in window

### Time / circadian features
- Hour of day (encoded as `sin`/`cos` of `2π·hour/24`)
- Day of week (`sin`/`cos` of `2π·dow/7`)
- Minutes since wake / sleep proxy if available (future work)

### Circadian baseline (`src/features/circadian.py`)
For each hour-of-day `h ∈ {0..23}`, maintain a **causal expanding mean/std** of HR restricted
to samples from that hour (1-step-shifted so the current sample never leaks into its own
baseline):
```
HR_circadian_mean(t) = mean(HR[s] for s < t, hour(s) == hour(t))
HR_circadian_std(t)  = std(HR[s]  for s < t, hour(s) == hour(t))
HR_circadian_zscore(t) = (HR(t) - HR_circadian_mean(t)) / HR_circadian_std(t)
```
This captures "is HR unusual *for this person at this time of day*" — e.g. HR=85 at 3am is far
more abnormal than HR=85 at 3pm, even if both are within the person's all-day range.
`HR_circadian_zscore` feeds directly into the combined anomaly score (§7.5) and the
Personal Digital Twin's circadian table (§8.1) is the non-causal, full-history version of the
same per-hour mean/std/count, built once from a subject's full recording.

### Physiological state classifier (`src/features/state_classifier.py`)
A causal, rule-based classifier assigns each sample one of seven states —
`sleep, rest, focused_work, walking, exercise, recovery, unknown` — using activity bucket
(§4.1), HR relative to the circadian baseline, time-of-day, and EDA phasic activity:
```
if activity_bucket >= MODERATE:                          -> exercise
elif recent exercise (<=10min ago) and HR > circadian+3:  -> recovery
elif activity_bucket == LIGHT:                            -> walking
elif activity_bucket == REST and night hours and
     HR <= circadian_mean:                                -> sleep
elif activity_bucket == REST and day hours and
     EDA_phasic_activity above its rolling median:        -> focused_work
elif activity_bucket == REST:                             -> rest
else:                                                      -> unknown
```
The numeric `physio_state` is included as a model feature (giving the forecaster explicit
context), while `physio_state_name` is excluded from training (categorical, for
display/explanation only). This state also drives the heart-rate-recovery features below and
the Digital Twin's per-state baselines (§8.1).

### Heart-rate-recovery / fitness features (`src/features/recovery_features.py`)
While `physio_state == exercise`, track a running peak HR for the current bout
(`exercise_peak_hr`). Once activity drops but HR is still elevated (`physio_state ==
recovery`), compute:
```
hr_recovery_bpm(t)            = max(0, exercise_peak_hr - HR(t))
hr_recovery_rate_bpm_per_min  = hr_recovery_bpm(t) / (time_since_exercise_peak_s(t) / 60)
```
A **faster** recovery rate (HR drops quickly after exertion) is a classic marker of better
cardiovascular fitness; a persistent slowdown in this rate over weeks/months is itself a
useful longitudinal health signal (tracked by the online-update pipeline, §8.4). All four
columns default to 0 outside of exercise/recovery so they never introduce NaNs into the
training windows.

### 4.1 Activity bucket heuristic
```
intensity = rolling_mean(|ACC_magnitude - 1g|, 10s)
if intensity < 0.05g:        REST
elif intensity < 0.2g:       LIGHT
elif intensity < 0.5g:       MODERATE
else:                         VIGOROUS
```
Thresholds are config defaults; for personalization they are re-calibrated per subject from
the distribution of `intensity` observed during the calibration period (e.g., 10th/50th/90th percentiles).

---

## 5. Model Design

| Model | Personalization | Latency | Real-time streaming | On-device footprint | Notes |
|---|---|---|---|---|---|
| **LSTM** | Good (fine-tune per user) | Medium | Needs hidden-state carry | Small (~10-50k params) | Strong baseline, well-understood |
| **GRU** | Good | Medium-low | Good — fewer gates than LSTM | Smallest of the RNNs | Slightly cheaper than LSTM, similar accuracy |
| **TCN** (dilated causal conv) | Good (per-user fine-tune of head) | **Low** — fully parallel, fixed receptive field | Good with a sliding buffer (cache last *L* steps) | Small, very fast inference | Best latency/accuracy tradeoff; easy to quantize |
| **Transformer (time-series)** | Good but data-hungry | High | Poor for tight real-time loops (O(L²) attention) | Largest | Best for long-range / multi-signal fusion if data is abundant (15+ days helps here) |
| **Kalman Filter + ML hybrid** | Excellent (state is inherently per-user) | **Lowest** | Ideal — O(1) update per sample | Tiny | Great for the *baseline/resting-HR tracker* and as a fallback during cold-start |
| **XGBoost (tabular benchmark)** | Good (retrain per user is cheap/fast) | Low | Good — single feature vector per step, no sequence state | Small | `src/models/xgb_model.py`. One regressor per target column, fed the *current* engineered feature vector (rolling windows already encode recent history). Don't assume deep learning wins: on short/noisy per-subject recordings, a well-tuned XGBoost often matches or beats the sequence models on MAE — always benchmark it (`--model xgboost` in `run_training.py`) before committing to a deep architecture. |

### Recommendation
**Primary forecaster: a lightweight causal TCN**, because it gives the best combination of
(a) personalization via per-user fine-tuning of the final layers, (b) low, deterministic
latency suitable for a wearable/phone, and (c) easy training on short calibration windows
without the vanishing-gradient/instability issues of RNNs.

**Complementary: a Kalman-filter baseline tracker** runs alongside the TCN to maintain a
slowly-varying estimate of the person's *resting HR* and its variance — this gives (i) an
instant, near-zero-cost fallback during cold start (before enough data exists to train the
TCN), and (ii) the adaptive-threshold statistics used by the anomaly detector (§7).

GRU/LSTM are implemented as comparison baselines (`src/models/lstm_gru.py`); a minimal
Transformer is included (`src/models/transformer.py`) for completeness/benchmarking once
15+ days of data are available, where its long-range attention can help with circadian
patterns. XGBoost (`src/models/xgb_model.py`) is included as a non-deep-learning benchmark —
`run_training.py --model {tcn,lstm,gru,transformer,xgboost}` trains any of the five on the
same feature table and `evaluate_forecast` (MAE/RMSE/MAPE) makes them directly comparable.

### Architecture (TCN, default)
```
Input: x ∈ R^{L x F}   (L=600 timesteps @1Hz = 10 min, F = #features)
  -> Causal Conv1d(F -> 32, kernel=3, dilation=1) + ReLU + residual
  -> Causal Conv1d(32 -> 32, kernel=3, dilation=2) + ReLU + residual
  -> Causal Conv1d(32 -> 32, kernel=3, dilation=4) + ReLU + residual
  -> Causal Conv1d(32 -> 32, kernel=3, dilation=8) + ReLU + residual
  -> Global pooling of last timestep (full receptive field ~ 2*(3-1)*(1+2+4+8)+1 = 91s;
     stack more blocks / larger kernel to cover 600s, see code for `receptive_field`)
  -> Dense(32 -> 3)            # mean prediction for [+1min, +5min, +10min]
  -> Dense(32 -> 3, softplus)  # predicted std (heteroscedastic uncertainty) for the same horizons
```
Trained with a Gaussian negative-log-likelihood loss so the model outputs both a point
forecast and a calibrated uncertainty, which feeds directly into the prediction-interval
anomaly method (§7.3).

---

## 6. Forecasting Task

- **Input sequence length `L`**: 600 s (10 minutes) of 1 Hz multivariate features by default
  (configurable; with 15+ days of data, longer context — e.g. 60 min — can be used for the
  Transformer variant to capture circadian effects).
- **Prediction horizons `H`**: {60s, 300s, 600s} → HR 1/5/10 minutes ahead, predicted jointly
  (multi-task output head) so the model is forced to learn a coherent trajectory rather than
  three independent models.
- **Training targets**: `y = [HR(t+60), HR(t+300), HR(t+600)]`, each paired with a predicted
  std `σ_h` for uncertainty.
- **Sample construction**: sliding window with stride `s` (default 10s) over each contiguous
  (gap-free) segment; windows are dropped if any required target falls beyond the segment end.

### 6.1 Multi-target forecasting (HR + HRV)
With `model.predict_rmssd: true` (default), the target vector is extended with
`RMSSD(t+60), RMSSD(t+300), RMSSD(t+600)` — i.e. `y` has `2 * len(horizons)` columns. Since
every sequence model (TCN/LSTM/GRU/Transformer) already has a generic `n_horizons`-sized
output head, this required **no architecture change**: `n_horizons` is simply set to
`len(target_cols)`. Forecasting HRV alongside HR gives an early-warning signal that often
moves *before* HR does (e.g. RMSSD dropping ahead of a stress-induced HR rise), and is used
by the illness score (§7.6).

---

## 7. Anomaly Detection

Let `ŷ_h(t)` be the model's point forecast for horizon `h` made at time `t`, `σ̂_h(t)` its
predicted std, and `y_h(t)` the realized HR at `t+h`.

### 7.1 Residual / deviation score
```
r_h(t) = y_h(t) - ŷ_h(t)            (raw deviation, bpm)
```

### 7.2 Z-score method (adaptive)
Maintain an exponentially-weighted mean/variance of recent residuals (per horizon, per
activity bucket — see §8):
```
μ_h(t)   = (1-λ) * μ_h(t-1)   + λ * r_h(t)
σ²_h(t)  = (1-λ) * σ²_h(t-1)  + λ * (r_h(t) - μ_h(t))²
z_h(t)   = (r_h(t) - μ_h(t)) / σ_h(t)
```
`λ` (e.g. 0.01–0.05) controls how fast the "normal residual" baseline adapts — slow enough to
not absorb genuine anomalies, fast enough to track drift (fitness change, season).
**Anomaly if `|z_h(t)| > z_thresh` (default 3.0)** for `≥K` consecutive samples (default K=3,
debouncing transient noise).

### 7.3 Prediction-interval method (uses model uncertainty directly)
```
PI_h(t) = [ŷ_h(t) - c·σ̂_h(t), ŷ_h(t) + c·σ̂_h(t)]   (c≈1.96 for ~95% interval)
Anomaly if y_h(t) ∉ PI_h(t)
```
Anomaly score (continuous, for ranking/severity):
```
score_h(t) = |y_h(t) - ŷ_h(t)| / σ̂_h(t)
```
(identical form to a z-score, but `σ̂_h` is the model's *input-conditional* uncertainty rather
than a global residual statistic — captures e.g. "uncertainty is naturally higher during
exercise, so a bigger deviation there is less surprising").

### 7.4 Bayesian / sequential approach
Model the residual stream with a simple state-space model (e.g., a 1-D Kalman filter on
`r_h(t)`); the filter's innovation (prediction error of the residual itself, normalized by
its own predicted variance) is a second-order anomaly score that catches *gradual drift*
(Case 3 in the brief): a slow, sustained increase in `r_h(t)` will push the Kalman state away
from zero even if no single sample crosses the z-score threshold.

For HRV collapse (Case 4): track `RMSSD(t)` with its own EWMA baseline `μ_RMSSD, σ_RMSSD`
(per activity bucket) and flag `z_RMSSD(t) = (RMSSD(t) - μ_RMSSD)/σ_RMSSD < -z_thresh`
(a *drop* below normal, one-sided).

### 7.5 Combined anomaly score
```
AnomalyScore(t) = max( |z_1min(t)|, |z_5min(t)|, |z_10min(t)|,
                       |z_RMSSD(t)|, |KalmanInnovation(t)|, |z_circadian(t)| )
```
`z_circadian(t)` is the circadian-deviation z-score from §4 (`HR_circadian_zscore`) — it adds
a *slow, person-and-time-of-day-aware* check that complements the fast EWMA residual z-scores:
a sustained HR rise that is still "in range" of the global residual statistics but unusual
*for 3am specifically* (Scenario G, silent drift) will surface here even before the
Kalman-innovation term reacts.

Severity levels (configurable): `score < 2` → normal, `2 ≤ score < 3` → watch (log only),
`score ≥ 3` → alert, `score ≥ 5` → urgent alert.

### 7.6 Illness detection score & explainable alerts (`src/anomaly/explain.py`)
Three signals classically co-move during infection/fever — HR rises, HRV (RMSSD) falls, skin
temperature rises. Any one alone is often noise; all three moving together is a much
stronger signal:
```
IllnessScore(t) = mean( clip(z_circadian(t), 0, ∞)/3,        # HR above expected, scaled
                         clip(RMSSD_pct_drop(t), 0, ∞)/0.35,  # % RMSSD drop, scaled
                         clip(TEMP_baseline_deviation(t), 0, ∞)/1.0 )  # °C above baseline
```
scaled so a "typical fever" (HR z≈3, RMSSD drop≈35%, TEMP +1°C) scores ≈1.0.

Every alert is paired with a **human-readable reason string** built from the same
components, e.g.:
```
"+HR 22bpm above expected, RMSSD dropped 35%, Temp elevated +1.2C, Activity: rest"
```
so a clinician/user sees *why* the system flagged the sample, not just a bare score. This is
the `alert_reason` column produced by `inference_pipeline.run_inference`.

---

## 8. Personalization Strategy

| Approach | Role in this system |
|---|---|
| **Global (population) model** | Cold-start prior only — `pretrain_pipeline.run_pretraining` trains once across all 35 Stress-Predict subjects (pooled windows, intersected feature columns); provides a reasonable initialization so a brand-new user isn't starting from random weights. |
| **Personal model** | The end-state for each user — TCN (or LSTM/GRU/Transformer/XGBoost) whose weights (and the Kalman baseline tracker's parameters) are specialized to that individual via `train_pipeline.run_training` or `finetune_pipeline.run_finetune`. |
| **Transfer learning** | `finetune_pipeline.run_finetune` loads the global checkpoint + scaler and continues training on one subject's windows only, at `lr/5` for `epochs/3` by default; `--freeze-backbone` freezes everything except the `mean_head`/`std_head` output layers so only the final mapping is personalized. |
| **Fine-tuning** | Same `finetune_pipeline.run_finetune`, re-run periodically on the most recent N days with a low learning rate, to track slow physiological drift (fitness, season, age). |
| **Online learning** | `online_update_pipeline.run_online_update` (§8.4) — a lighter-weight, more frequent (e.g. nightly) update on just the most recent window, plus incremental scaler adaptation. Between updates, the **Kalman baseline tracker** and the **EWMA residual statistics** (§7.2) update every sample — the fast-adapting layer that keeps anomaly thresholds current without retraining the network. |
| **Personal Digital Twin** | §8.1 — replaces population norms as the anomaly comparison baseline. |

**Recommended pipeline**: Global pretrain (once, `run_pretraining.py --all`) → per-user
transfer + fine-tune at the end of calibration (`run_finetune.py`) → periodic incremental
online update (`run_online_update.py`) + continuous online EWMA/Kalman updates. This gives
cold-start coverage, individual specificity, and adaptation to drift, while keeping the
expensive step (full fine-tune) infrequent.

Activity-bucket conditioning (§4.1) and physiological-state conditioning (§4) are applied
throughout: baselines, residual statistics, and thresholds are tracked **per activity
bucket** (REST/LIGHT/MODERATE/VIGOROUS) so that "normal" is contextual — Case 2 in the brief
(HR 145 while running, expected 140) is not flagged, because both the forecast and the
residual statistics are conditioned on the VIGOROUS bucket.

### 8.1 Personal Digital Twin (`src/personalization/digital_twin.py`)
A small JSON profile, built once from a subject's full feature table
(`scripts/build_digital_twin.py --subject S01`):
```json
{
  "subject_id": "S01",
  "resting_hr": 64.2,
  "sleep_hr": 58.7,
  "walking_hr": 78.4,
  "running_hr": 132.1,
  "avg_rmssd": 41.3,
  "circadian_profile": { "0": {"hr_mean": 59.1, "hr_std": 2.3}, "1": {...}, ... "23": {...} }
}
```
`resting_hr`/`sleep_hr`/`walking_hr`/`running_hr` are the mean HR observed while
`physio_state` was REST/SLEEP/WALKING/EXERCISE respectively (§4); `circadian_profile` is the
full-history per-hour-of-day mean/std (`circadian.circadian_profile_table`).

### 8.2 Digital Twin score (health-drift score)
At inference time, `digital_twin_score(current_hr, hour_of_day, twin, physio_state)` computes:
```
ExpectedHR(t) = circadian_profile[hour(t)].hr_mean   (fallback: state-specific baseline)
DigitalTwinScore(t) = (HR(t) - ExpectedHR(t)) / circadian_profile[hour(t)].hr_std
```
This is **this person's own history**, not a population table, as the comparison baseline —
the same absolute HR can be perfectly normal for one person and a clear deviation for
another. `run_inference(..., twin=twin)` adds a `digital_twin_score` column when a twin
profile is supplied.

### 8.3 Population-to-personal transfer learning
1. `scripts/run_pretraining.py --all --model tcn` → pools windows from every subject
   (intersecting feature columns so all subjects contribute even if a few features are
   subject-specific), fits one global scaler, trains one model → `models/global/tcn/`.
2. `scripts/run_finetune.py --subject S01 --pretrained models/global/tcn [--freeze-backbone]`
   → loads the global weights + scaler, continues training on `S01`'s windows only (low
   LR, few epochs) → `models/S01/finetuned/`, loadable like any personal model.

### 8.4 Online / incremental learning (`src/pipeline/online_update_pipeline.py`)
`scripts/run_online_update.py --subject S01 --model-dir models/S01/tcn`, intended to run
periodically (cron):
- Takes only the most recent `online.recent_fraction` of windows (default 20%).
- Calls `scaler.partial_fit(...)` on that recent data so feature normalization tracks slow
  shifts in the person's baseline (e.g. resting HR trending down as fitness improves).
- Continues training the existing model for `online.epochs` (default 3) at
  `model.lr * online.lr_fraction` (default 0.1x) — small enough to adapt without
  catastrophic forgetting of the longer-history fine-tune.

---

## 9. Simulation Environment

`src/simulation/simulator.py` implements a `PhysiologicalSimulator` that can either (a)
generate a fully synthetic multi-day trace from scratch, or (b) take a real subject's
preprocessed trace and inject scenario perturbations. All scenarios operate on the unified
1 Hz feature table (HR, IBI-derived, ACC magnitude, BVP-derived) so they flow through the
same pipeline as real data.

| Scenario | Synthetic modification |
|---|---|
| **A. Normal day** | Circadian HR baseline `HR_base(t) = 65 + 8*sin(2π(t-6h)/24h)` + activity-driven bumps from a simulated daily activity schedule (sleep/work/exercise blocks) + Gaussian noise (σ≈1.5bpm). IBI generated as `60/HR + noise`; ACC magnitude follows the activity schedule. |
| **B. Sudden HR spike** | Add a short pulse `Δ(t) = A * exp(-((t-t0)/τ)^2)` to HR (e.g., `A=30bpm`, `τ=30s`) **without** a corresponding ACC change — simulates a panic attack / arrhythmia event. IBI shortens proportionally (`IBI = 60/(HR_base+Δ)`); RMSSD typically drops during the spike. |
| **C. Gradual HR drift** | Add a slow ramp `Δ(t) = k*(t-t0)` over tens of minutes to hours with activity held constant — models e.g. dehydration, developing fever, or overtraining onset (Case 3 in the brief). |
| **D. Exercise event** | Raise ACC magnitude/variance to "vigorous" levels for a block, and drive HR toward an exercise target via a first-order lag: `HR(t) = HR(t-1) + (HR_target - HR(t-1))/τ_ex` (τ_ex≈60-120s on the way up, longer on recovery) — produces realistic onset + recovery curves correlated with ACC. |
| **E. Illness / fever** | Shift the circadian HR baseline up by `2-3 bpm per °C` of simulated TEMP elevation, raise resting HR by ~10-20bpm, and reduce HRV (lower RMSSD/SDNN) — TEMP trace is shifted upward by 1-3°C over hours. |
| **F. Sensor noise** | Inject white noise, brief dropout segments (NaNs), and motion-artifact bursts into BVP/ACC (and consequently IBI gaps), without changing underlying physiology — tests robustness/false-positive rate of the pipeline. |
| **G. Silent physiological drift** | HR ramps slowly (e.g. 72→90bpm) over 4-6 hours while ACC/activity is actively held near zero (resting) throughout — unlike Scenario C, there is *no* activity-pattern change to explain the rise. RMSSD falls slightly as the ramp progresses. Designed to stress-test the circadian-deviation (§4) and Kalman-innovation (§7.4) anomaly terms, which are the ones built to catch slow trends that per-step residual z-scores can miss. |

Each scenario returns the modified trace **and** a ground-truth anomaly label vector
(1 during the injected event window, 0 otherwise), enabling the evaluation metrics in §10.

---

## 10. Evaluation

### Forecasting metrics (per horizon h ∈ {1,5,10 min})
- **MAE** = `mean(|y_h - ŷ_h|)`
- **RMSE** = `sqrt(mean((y_h - ŷ_h)^2))`
- **MAPE** = `mean(|y_h - ŷ_h| / y_h) * 100%`
- Baselines for comparison: persistence (`ŷ_h = HR(t)`), moving average.

### Anomaly metrics (on simulated scenarios with ground truth)
- **Precision** = `TP / (TP+FP)`, **Recall** = `TP / (TP+FN)`, **F1**
- **ROC-AUC** over the continuous `AnomalyScore(t)`
- **Time-to-detection**: delay between scenario onset and first alert
- **False-alarm rate**: alerts per hour during Scenario A (normal day)

### What matters most
- For **forecasting**, MAE/RMSE at the **1-minute horizon** matter most for anomaly
  responsiveness (it directly feeds the residual score); MAPE is most useful for comparing
  across subjects with very different resting HR.
- For **anomaly detection**, **Recall on Scenarios B/D/E (acute events)** and **false-alarm
  rate on Scenario A** are the key product metrics — a system that misses a spike or alerts
  constantly during normal life is unusable. ROC-AUC is the best single summary for model
  selection/tuning of `z_thresh`.

---

## 11. Production Folder Structure

```
personalized_hrv_system/
├── DESIGN.md
├── requirements.txt
├── configs/
│   └── config.yaml              # sampling rates, window sizes, horizons, thresholds, paths
├── src/
│   ├── data/
│   │   ├── loader.py             # parse raw E4 CSVs -> per-signal DataFrames
│   │   ├── sync.py               # resample + merge onto common 1Hz grid
│   │   └── cleaning.py           # filtering, artifact removal, gap handling
│   ├── features/
│   │   ├── hr_features.py
│   │   ├── ibi_features.py       # RMSSD, SDNN, pNN50, ...
│   │   ├── acc_features.py       # magnitude, activity buckets
│   │   ├── bvp_features.py       # peaks, amplitude
│   │   ├── temp_features.py
│   │   ├── eda_features.py
│   │   ├── time_features.py      # circadian sin/cos encodings
│   │   ├── circadian.py          # per-hour-of-day HR baseline + circadian z-score
│   │   ├── state_classifier.py   # physiological state (sleep/rest/focused_work/walking/exercise/recovery)
│   │   ├── recovery_features.py  # heart-rate-recovery (HRR) features
│   │   └── build_features.py     # orchestrates all of the above -> feature table
│   ├── models/
│   │   ├── datasets.py           # windowing -> torch Dataset
│   │   ├── tcn.py                # recommended forecaster
│   │   ├── lstm_gru.py           # baselines
│   │   ├── transformer.py        # benchmark for long-context
│   │   ├── xgb_model.py          # XGBoost tabular-feature benchmark
│   │   ├── kalman.py             # baseline-tracker / online layer
│   │   └── train.py              # training loop, NLL loss, checkpoints
│   ├── anomaly/
│   │   ├── scoring.py            # residual, z-score, prediction interval, circadian/RMSSD terms
│   │   ├── adaptive.py           # EWMA baselines, Kalman innovation, activity buckets
│   │   └── explain.py            # illness score + human-readable alert reasons
│   ├── personalization/
│   │   └── digital_twin.py       # personal digital twin profile (resting/sleep/walk/run HR, circadian table)
│   ├── simulation/
│   │   └── simulator.py          # scenarios A-G
│   └── pipeline/
│       ├── train_pipeline.py         # end-to-end: load -> features -> train -> save (tcn/lstm/gru/transformer/xgboost)
│       ├── inference_pipeline.py     # streaming inference + anomaly scoring + explainability
│       ├── pretrain_pipeline.py      # global pretraining across subjects (transfer learning, stage 1)
│       ├── finetune_pipeline.py      # per-subject fine-tune of a pretrained model (stage 2)
│       └── online_update_pipeline.py # incremental online adaptation
└── scripts/
    ├── run_preprocessing.py
    ├── run_training.py            # --model tcn|lstm|gru|transformer|xgboost
    ├── run_inference.py
    ├── run_simulation.py
    ├── build_digital_twin.py
    ├── run_pretraining.py
    ├── run_finetune.py
    └── run_online_update.py
```

---

## 12. Future Improvements

1. **Multi-day calibration**: extend window/feature configs for 15+ day continuous traces;
   add sleep-stage-aware circadian features (the current circadian baseline uses hour-of-day
   only — true sleep-stage detection would sharpen the `sleep` state and its baseline).
2. **On-device deployment**: quantize the TCN (int8), implement a streaming ring-buffer
   inference loop in Swift/Kotlin; Kalman tracker as the always-on fallback.
3. **Federated / privacy-preserving fine-tuning**: keep personal data on-device, share only
   model deltas for the global-prior update.
4. **Learned activity/state classifier**: replace the heuristic activity-bucket and
   physiological-state classifiers (§4.1, §4) with small supervised models (HAR) once labeled
   activity data is available.
5. **Multi-modal fusion**: incorporate SpO2/BP if future hardware provides them, following
   the Temporal-Fusion-Transformer multi-vital approach referenced in the research notes.
6. **Generative augmentation**: train a conditional generative model (e.g., diffusion / RCGAN)
   on calibration data to expand simulation scenario diversity beyond rule-based injections
   (currently Scenarios A-G).
7. **Closed-loop alert tuning**: use user feedback (acknowledged/dismissed alerts) to adapt
   `z_thresh` per user (reduces false-alarm fatigue), and to weight the components of
   `IllnessScore` (§7.6) per user.
8. **Digital twin enrichment**: extend the Digital Twin profile (§8.1) beyond HR/RMSSD —
   personal TEMP baseline, EDA arousal baseline, and recovery-rate trend over time (from
   §8.4's online updates), so the "expected" comparison improves as more history accumulates.
```
