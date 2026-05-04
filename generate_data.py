"""
Synthetic univariate time series with a known generative process.

The series is the sum of:
    - a deterministic seasonal cycle (sinusoid)
    - a slow piecewise-linear trend
    - an AR(1) noise process (correlated random walk)
    - a single regime change halfway through (level shift)

Knowing the components lets us check whether the LSTM picks them up
*qualitatively* — not just that it forecasts a low RMSE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    n_steps: int = 1500
    season_period: int = 36
    season_amp: float = 1.5
    trend_slope: float = 0.002
    trend_break_at: float = 0.55         # fraction of n_steps where slope changes sign
    ar1_phi: float = 0.55
    ar1_sigma: float = 0.4
    regime_shift_at: float = 0.72        # where the level shift occurs
    regime_shift_size: float = 1.2
    seed: int = 42


def generate(cfg: DataConfig) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.n_steps)

    season = cfg.season_amp * np.sin(2 * np.pi * t / cfg.season_period)

    break_at = int(cfg.trend_break_at * cfg.n_steps)
    trend = np.empty(cfg.n_steps)
    trend[:break_at] = cfg.trend_slope * np.arange(break_at)
    trend[break_at:] = (cfg.trend_slope * break_at
                        - cfg.trend_slope * 0.6 * np.arange(cfg.n_steps - break_at))

    noise = np.zeros(cfg.n_steps)
    for i in range(1, cfg.n_steps):
        noise[i] = cfg.ar1_phi * noise[i - 1] + rng.normal(0, cfg.ar1_sigma)

    regime = np.zeros(cfg.n_steps)
    shift_at = int(cfg.regime_shift_at * cfg.n_steps)
    regime[shift_at:] = cfg.regime_shift_size

    y = season + trend + noise + regime
    components = {"season": season, "trend": trend,
                  "noise": noise, "regime": regime}
    return y, components


def save(out_dir: Path, y: np.ndarray, components: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(y, name="y").to_csv(out_dir / "y.csv", index=False)
    df = pd.DataFrame(components)
    df.to_csv(out_dir / "components.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic univariate time series.")
    p.add_argument("--n-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(n_steps=args.n_steps, seed=args.seed)
    y, comp = generate(cfg)
    save(args.out_dir, y, comp)
    print(f"Generated {len(y)} steps. Saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
