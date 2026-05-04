"""
Train an LSTM on the synthetic series, compare against simple baselines,
render dashboard figures.

Baselines compared:
    - Naive (next value = last value)
    - Seasonal naive (next value = value `season_period` steps ago)
    - LSTM (single-step forecaster trained on a sliding window)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from generate_data import DataConfig, generate

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"
COLOR_LIGHT_BLUE = "#9EB7D6"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ------------------------------------------------------------- LSTM model --
class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])    # (B, 1)


# ---------------------------------------------------------------- helpers --
def make_windows(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, t = [], []
    for i in range(window, len(y)):
        X.append(y[i - window:i])
        t.append(y[i])
    return np.array(X), np.array(t)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ------------------------------------------------------------- training ---
def train_lstm(y_train: np.ndarray, window: int, epochs: int = 12, lr: float = 5e-3,
               seed: int = 42) -> tuple[LSTMForecaster, dict]:
    torch.manual_seed(seed)
    Xtr, ttr = make_windows(y_train, window)
    Xt = torch.from_numpy(Xtr).float().unsqueeze(-1)         # (N, T, 1)
    yt = torch.from_numpy(ttr).float().unsqueeze(-1)         # (N, 1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict = {"train_loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * yb.size(0)
            n += yb.size(0)
        history["train_loss"].append(loss_sum / n)
        print(f"epoch {ep:2d}  train MSE {loss_sum / n:.4f}")
    return model, history


def lstm_predict_one_step(model: LSTMForecaster, full_series: np.ndarray,
                          window: int, start: int) -> np.ndarray:
    """
    Single-step forecasts using the *actual* history at each step (the
    fair evaluation for a one-step LSTM). Predicts y[start], y[start+1],
    ..., y[len(full_series)-1].
    """
    model.eval()
    n_pred = len(full_series) - start
    preds = np.empty(n_pred)
    with torch.no_grad():
        for i in range(n_pred):
            ctx = full_series[start + i - window: start + i]
            x = torch.from_numpy(ctx.copy()).float().view(1, window, 1)
            preds[i] = float(model(x))
    return preds


def lstm_predict_rollout(model: LSTMForecaster, y_history: np.ndarray, window: int,
                         horizon: int) -> np.ndarray:
    """
    Free-running forecast: feed prediction back into context for next step.
    Multi-step error accumulates — this is the harder evaluation.
    """
    model.eval()
    history = list(y_history[-window:])
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(history[-window:], dtype=torch.float32).view(1, window, 1)
            p = float(model(x))
            preds.append(p)
            history.append(p)
    return np.array(preds)


# ---------------------------------------------------------------- figures --
def fig_full_series(y: np.ndarray, split_at: int, components: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    t = np.arange(len(y))

    # Top: full series with train/test split.
    axes[0].plot(t[:split_at], y[:split_at], color=COLOR_BLUE, linewidth=1.0,
                 label="train")
    axes[0].plot(t[split_at:], y[split_at:], color=COLOR_RED, linewidth=1.0,
                 label="test")
    axes[0].axvline(split_at, color=COLOR_GRAY, linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("y(t)")
    axes[0].set_title("Synthetic series — train / test split")
    axes[0].legend(loc="upper left")

    # Bottom: components
    axes[1].plot(t, components["season"], color=COLOR_LIGHT_BLUE, linewidth=0.9, label="season")
    axes[1].plot(t, components["trend"],  color=COLOR_GRAY,       linewidth=1.4, label="trend")
    axes[1].plot(t, components["regime"], color=COLOR_RED,        linewidth=1.4, label="regime shift")
    axes[1].plot(t, components["noise"],  color=COLOR_LIGHT_GRAY, linewidth=0.6, alpha=0.8, label="AR(1) noise")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("contribution")
    axes[1].set_title("Generative components (known by construction)")
    axes[1].legend(loc="upper left", ncol=2)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_forecasts(y_test: np.ndarray, preds: dict[str, np.ndarray],
                  start: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    t_test = np.arange(start, start + len(y_test))
    ax.plot(t_test, y_test, color=COLOR_GRAY, linewidth=2.0, label="actual",
            alpha=0.85)
    palette = [COLOR_LIGHT_GRAY, COLOR_LIGHT_BLUE, COLOR_BLUE, COLOR_RED]
    for (name, p), c in zip(preds.items(), palette):
        ax.plot(t_test, p, color=c, linewidth=1.4, label=name, alpha=0.95)
    ax.set_xlabel("t"); ax.set_ylabel("y(t)")
    ax.set_title("Forecasts on the held-out test segment")
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bar(metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    names = list(metrics.keys())
    rmses = [metrics[k]["rmse"] for k in names]
    maes = [metrics[k]["mae"] for k in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    palette = [COLOR_RED, COLOR_GRAY, COLOR_BLUE]
    for ax, vals, label in zip(axes, [rmses, maes], ["RMSE", "MAE"]):
        bars = ax.bar(names, vals, color=palette,
                      edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
        ax.set_title(label); ax.tick_params(axis="x", labelrotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=COLOR_TEXT)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Forecast accuracy (lower = better)",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_training_curve(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"],
            color=COLOR_BLUE, marker="o", linewidth=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train MSE")
    ax.set_title("LSTM training loss")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    cfg = DataConfig()
    y, components = generate(cfg)

    split_at = int(0.85 * len(y))           # 85% / 15% train / test split
    y_train, y_test = y[:split_at], y[split_at:]
    window = cfg.season_period * 2          # use 2 cycles of context

    # Baselines (multi-step naive, seasonal naive)
    naive = np.full_like(y_test, y_train[-1])
    seasonal = y[split_at - cfg.season_period:split_at - cfg.season_period + len(y_test)]
    if len(seasonal) < len(y_test):
        seasonal = np.concatenate([seasonal, np.full(len(y_test) - len(seasonal), seasonal[-1])])

    # LSTM
    model, history = train_lstm(y_train, window=window, epochs=12)
    lstm_one_step = lstm_predict_one_step(model, y, window=window, start=split_at)
    lstm_rollout = lstm_predict_rollout(model, y_train, window=window, horizon=len(y_test))

    metrics = {
        "Naive": {"rmse": rmse(y_test, naive), "mae": mae(y_test, naive)},
        "Seasonal naive": {"rmse": rmse(y_test, seasonal), "mae": mae(y_test, seasonal)},
        "LSTM (one-step)": {"rmse": rmse(y_test, lstm_one_step), "mae": mae(y_test, lstm_one_step)},
        "LSTM (free-run)": {"rmse": rmse(y_test, lstm_rollout),  "mae": mae(y_test, lstm_rollout)},
    }

    print(f"\nTest metrics:")
    for k, m in metrics.items():
        print(f"  {k:<18} RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "config": cfg.__dict__,
        "metrics": metrics,
        "lstm_train_loss": history["train_loss"],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_full_series(y, split_at, components, assets / "01_series.png")
    fig_forecasts(y_test,
                  {"Naive": naive, "Seasonal naive": seasonal,
                   "LSTM (one-step)": lstm_one_step, "LSTM (free-run)": lstm_rollout},
                  split_at, assets / "02_forecasts.png")
    fig_metrics_bar(metrics, assets / "03_metrics.png")
    fig_training_curve(history, assets / "04_training.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
