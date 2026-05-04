<div align="center">

# LSTM Time-Series Forecasting — and Why Multi-Step is Hard

**Forecast a synthetic series with a known generative process. Compare LSTM (one-step + free-run) against naive baselines.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![framework](https://img.shields.io/badge/framework-PyTorch-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Build a time series from known parts — sinusoidal seasonality, piecewise-linear trend, AR(1) noise, one regime shift — and forecast the held-out tail four ways: naive, seasonal naive, LSTM with fresh context at every step, and LSTM running free on its own predictions.

<table>
<tr>
<td align="center" width="33%">
<sub>LSTM (one-step)</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">RMSE 0.501</b><br>
<sub>best — beats both baselines</sub>
</td>
<td align="center" width="33%">
<sub>Seasonal naive</sub><br>
<b style="font-size:1.5em; color:#7A7A7A;">RMSE 0.736</b><br>
<sub>strong baseline (period = 36)</sub>
</td>
<td align="center" width="33%">
<sub>LSTM (free-run)</sub><br>
<b style="font-size:1.5em; color:#C04040;">RMSE 2.403</b><br>
<sub>error compounds over 225 steps</sub>
</td>
</tr>
</table>

| Method | RMSE | MAE | Setup |
|---|---:|---:|---|
| Naive (last-value) | 1.942 | 1.611 | predict y[t] = y[t−1] |
| Seasonal naive | 0.736 | 0.599 | predict y[t] = y[t−36] |
| **LSTM (one-step, with fresh context)** | **0.501** | **0.408** | re-feed actual y at every step |
| LSTM (free-running rollout) | 2.403 | 1.972 | feed own predictions back into context |

<sub>**Headline finding:** the same LSTM is *the best* and *the worst* method here, depending on how you ask it to forecast. With actual history at every step it cleanly beats both naive baselines. Asked to roll forward 225 steps on its own predictions, the small per-step errors compound into a forecast that drifts off in completely the wrong direction. This split is the single most underappreciated thing about deep-learning forecasting.</sub>

---

## Dashboard

### 1. The series — and what's inside it

![series](assets/01_series.png)

**Top**: the full 1500-step series, blue = train (85%), red = test (15%). The series clearly *looks* like one signal, but it isn't — it's a sum of four well-defined components, shown below.

**Bottom**: the four generative components plotted separately.

- The **light-blue sinusoid** is the seasonal cycle (period 36, amplitude 1.5).
- The **gray triangle** is the trend — a piecewise-linear ramp that flips slope around step 825.
- The **red step** is the regime shift at step 1080 (level jumps by +1.2).
- The **light-gray squiggle** is AR(1) noise (φ = 0.55, σ = 0.4).

Knowing the components by construction lets us reason about which methods *should* be able to capture which.

### 2. Forecasts on the test segment

![forecasts](assets/02_forecasts.png)

Four lines on top of the actual (gray):

- **Naive (light gray)**: a flat line at the last training value. Misses everything.
- **Seasonal naive (light blue)**: copies the equivalent cycle from one season ago. Recovers the periodic pattern but misses the regime shift.
- **LSTM one-step (blue)**: tracks the actual series within the noise band — the closest curve to the gray.
- **LSTM free-run (red)**: starts plausibly but drifts off after a few cycles, settling into a phase-shifted oscillation that no longer matches reality.

The visual is the lesson: a deep model with fresh data is dominant; the same model on its own predictions is unreliable.

### 3. Forecast accuracy

![metrics](assets/03_metrics.png)

RMSE and MAE both rank the methods identically: LSTM one-step < Seasonal naive < Naive ≈ LSTM free-run. The 4× gap between LSTM-one-step and LSTM-free-run on the same model is the practical takeaway.

### 4. Training dynamics

![training](assets/04_training.png)

Training MSE drops cleanly from ~1.9 to ~0.2 over 12 epochs. The plateau around epoch 10–12 indicates the LSTM has captured the deterministic structure (seasonality + trend), and the residual training error is essentially the AR(1) noise variance — which is *un-forecastable* by construction.

---

## What's actually happening

### The generative process — fully under our control

```
y(t) = season(t) + trend(t) + ar1_noise(t) + regime_shift(t)
```

By construction, the irreducible part of the series is the AR(1) noise — even an oracle can't predict that better than its conditional mean. So the *theoretically optimal* one-step RMSE on this series is roughly the AR(1) standard deviation σ ≈ 0.4. The LSTM at 0.408 MAE is essentially at that floor.

### Why the LSTM beats seasonal naive at one-step

Seasonal naive copies y[t−36]. It captures seasonality perfectly but is blind to the trend slope flip and the regime shift, both of which produce systematic errors. The LSTM, with its 72-step context window (two full cycles), can pick up the trend direction and respond to the level shift after seeing it once. That's where the 0.501 vs 0.736 gap comes from.

### Why the same LSTM fails on free-run

Multi-step rollout = autoregressive feedback. At each step, the LSTM uses its previous prediction as input for the next prediction. Two compounding effects make this break down:

1. **Distribution shift**: the model was trained on inputs sampled from the *true* series distribution. Its own predictions, slightly off in many small ways, slowly leave that distribution and enter territory the model wasn't trained on.
2. **Errors compound multiplicatively**: a +5% error on step 1, propagated through 225 steps, can blow up.

This is why production forecasting systems either re-condition on fresh data frequently, or are explicitly trained on multi-step prediction (e.g., teacher forcing during training, but evaluated on rollout — or sequence-to-sequence architectures that *output* the whole horizon at once).

### Mental model

| You have… | Use |
|---|---|
| **One-step ahead** prediction with fresh data each tick | LSTM (or a small MLP on lagged features — often equally good) |
| **Multi-step horizon** with no new data | Seasonal naive + structural model (Prophet, ETS), then add ML on residuals |
| **Strong known seasonality** | Seasonal naive is hard to beat; treat it as the floor your model must exceed |
| **Need uncertainty bands** | Stick to statistical methods (ARIMA / state-space) — LSTMs don't natively give intervals |

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
```

Total wall-time: under 30 seconds on CPU. The model is intentionally tiny (1-layer, 32-hidden LSTM) — bumping it to 2-layer 64-hidden makes the training curves prettier but doesn't move test RMSE much, because the irreducible noise is the binding constraint.

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py):

```python
DataConfig(
    n_steps=1500,
    season_period=36,
    season_amp=1.5,
    trend_slope=0.002,
    trend_break_at=0.55,        # where slope flips sign
    ar1_phi=0.55, ar1_sigma=0.4,
    regime_shift_at=0.72,
    regime_shift_size=1.2,
    seed=42,
)
```

Crank `ar1_sigma` up to 1.5 to make the noise drown the signal, or set `season_period=240` to test how well a 72-step context window can still pick up a 240-step cycle (it can't — that's a context-window failure mode).

---

## Project layout

```
06-rnn-lstm-timeseries/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← deterministic time-series generator
├── train.py               ← LSTM + baselines + dashboard figures
├── assets/                ← 4 dashboard PNGs
└── results/metrics.json
```

---

## What I learned

- **Always benchmark against seasonal naive.** It is criminally underrated. On any series with strong seasonality, getting beaten by `y[t] = y[t−period]` is a real risk for ML methods, and it's the *first* baseline I should try before reaching for an LSTM.
- **One-step accuracy and multi-step accuracy are different metrics.** Reporting one without the other is misleading. The LSTM here gets +4% better than seasonal naive on one-step *and* 3× worse on a 225-step rollout. Both are true; both matter.
- **The irreducible-noise floor is real.** When training MSE plateaus at ~σ² of the noise process you injected, the model has *learned the deterministic part of the data and nothing more is recoverable*. That's success, not failure.
- **Multi-step rollout is a property of how you *use* the model, not the model itself.** A bigger LSTM doesn't fix free-running drift. The fix is architectural (sequence-to-sequence) or procedural (re-condition on real data when it arrives).

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
