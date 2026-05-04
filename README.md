# 06 — LSTM Time-Series Forecasting

> Status: 🟡 Planned (skeleton only)

## Goal

Forecast a univariate time series with an LSTM. Compare it against a
classical baseline (ARIMA / Prophet) so the deep-learning value-add is
honest, not assumed.

## Topics covered

- Why vanilla RNNs fail on long sequences (vanishing gradients)
- LSTM gates: input, forget, output — what each one actually does
- Sliding-window dataset construction for sequence-to-one forecasting
- Walk-forward validation (no random shuffling allowed in time series)
- Comparing against ARIMA / Prophet baselines
- Multi-step vs single-step forecasting trade-offs

## Dataset

**Self-generated synthetic data — no third-party / copyrighted datasets.**
A custom time-series generator (`generate_data.py`) combines a
deterministic seasonal pattern, a slow trend, an AR(1) noise process,
and occasional regime changes. Because the generative process is fully
known, the LSTM's forecast can be benchmarked against the *theoretical*
optimal forecast given the true structure — a sharper diagnostic than
just "lower RMSE than ARIMA."

## Tech stack

- Python 3.10+
- PyTorch
- statsmodels (for ARIMA), prophet
- pandas, matplotlib

## Run

```bash
pip install -r requirements.txt
python train.py
```

## Results

_To be filled in after implementation._

## What I learned

_To be filled in after implementation._

---
*Part of the [machine-learning portfolio](../README.md).*
