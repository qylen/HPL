"""Synthetic-data generator for smoke tests; generated patterns are not market evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def generate_synthetic_dataset(output_path: str | Path, rows: int = 20_000, seed: int = 13) -> pd.DataFrame:
    """Create a synthetic feature table with a deliberately planted weak signal."""
    if rows < 500:
        raise ValueError("rows must be at least 500")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    timestamp = pd.date_range(start, periods=rows, freq="s")
    ofi_norm = rng.normal(0, 0.25, rows)
    trade_imbalance = np.tanh(ofi_norm * 1.8 + rng.normal(0, 0.7, rows))
    spread = np.clip(rng.lognormal(mean=0.0, sigma=0.3, size=rows), 0.2, 5.0)
    depth = rng.lognormal(mean=1.0, sigma=0.5, size=rows)
    volatility = np.abs(rng.normal(0.00015, 0.00007, rows))
    noise = rng.normal(0, 0.00035, rows)
    # This planted dependency exists only to verify pipeline mechanics.
    y = 0.00020 * ofi_norm + 0.00008 * trade_imbalance + noise
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "target_return": y,
        "ofi": ofi_norm * depth,
        "ofi_norm": ofi_norm,
        "ofi_norm_5s": pd.Series(ofi_norm).rolling(5, min_periods=1).mean().to_numpy(),
        "trade_imbalance": trade_imbalance,
        "trade_imbalance_5s": pd.Series(trade_imbalance).rolling(5, min_periods=1).mean().to_numpy(),
        "trade_volume_log": np.log1p(rng.lognormal(0, 0.6, rows)),
        "trade_volume_5s_log": np.log1p(rng.lognormal(1, 0.5, rows)),
        "buy_volume_log": np.log1p(rng.lognormal(-0.2, 0.7, rows)),
        "sell_volume_log": np.log1p(rng.lognormal(-0.2, 0.7, rows)),
        "trade_count_log": np.log1p(rng.poisson(3, rows)),
        "spread_bps": spread,
        "depth_total": depth,
        "depth_imbalance": np.tanh(ofi_norm + rng.normal(0, 0.5, rows)),
        "mid_return_1s": np.r_[0.0, y[:-1]],
        "volatility_5s": pd.Series(volatility).rolling(5, min_periods=3).std().fillna(0).to_numpy(),
        "volatility_30s": pd.Series(volatility).rolling(30, min_periods=10).std().fillna(0).to_numpy(),
        "quote_update_count_log": np.log1p(rng.poisson(4, rows)),
        "quote_age_ms": rng.uniform(0, 1000, rows),
        "time_sin": np.sin(2 * np.pi * (np.arange(rows) % 86400) / 86400),
        "time_cos": np.cos(2 * np.pi * (np.arange(rows) % 86400) / 86400),
        "mid": 100_000 * np.exp(np.cumsum(y)),
    })
    # Keep the exact same feature/target schema as real built data.
    wanted = list(dict.fromkeys(["timestamp", "mid", "target_return", "spread_bps", "quote_age_ms", *FEATURE_COLUMNS]))
    frame = frame[wanted]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
