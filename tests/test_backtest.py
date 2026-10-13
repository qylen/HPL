import numpy as np
import pandas as pd

from hpl1.backtest import run_backtest


def test_backtest_charges_costs_and_closes_final_position():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="s", tz="UTC").astype(str),
        "actual_return": [0.001, 0.001, -0.001, 0.001],
        "prediction": [0.01, 0.01, -0.01, 0.01],
        "spread_bps": [1.0] * 4,
    })
    output, report = run_backtest(frame, taker_fee_bps=1, slippage_bps=0, min_edge_bps=0, hac_lags=1)
    assert len(output) == 4
    assert report["total_cost_bps_approx"] > 0
    assert output.iloc[-1]["liquidation_cost"] > 0
    assert np.isfinite(report["total_return_pct"])


def test_no_signal_stays_flat():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="s", tz="UTC").astype(str),
        "actual_return": np.zeros(5),
        "prediction": np.zeros(5),
        "spread_bps": np.ones(5),
    })
    output, report = run_backtest(frame, taker_fee_bps=5, slippage_bps=1)
    assert (output["position"] == 0).all()
    assert report["total_cost_bps_approx"] == 0
