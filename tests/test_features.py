
import numpy as np
import pandas as pd

from hpl1.features import build_feature_frame


def test_order_flow_and_one_second_target_alignment():
    # Quote changes in second 0 create positive OFI; mid rises in second 1.
    quotes = pd.DataFrame([
        {"type": "quote", "event_time_ms": 100, "seq": 1, "bid": 100.0, "bid_qty": 1.0, "ask": 100.2, "ask_qty": 1.0},
        {"type": "quote", "event_time_ms": 500, "seq": 2, "bid": 100.1, "bid_qty": 2.0, "ask": 100.3, "ask_qty": 1.0},
        {"type": "quote", "event_time_ms": 1100, "seq": 3, "bid": 100.2, "bid_qty": 2.0, "ask": 100.4, "ask_qty": 1.0},
        {"type": "quote", "event_time_ms": 2100, "seq": 4, "bid": 100.3, "bid_qty": 2.0, "ask": 100.5, "ask_qty": 1.0},
    ])
    trades = pd.DataFrame([
        {"type": "trade", "event_time_ms": 600, "seq": 5, "qty": 0.5, "buyer_is_maker": False},
        {"type": "trade", "event_time_ms": 700, "seq": 6, "qty": 0.2, "buyer_is_maker": True},
    ])
    result = build_feature_frame(quotes, trades)
    assert len(result) >= 3
    assert result.loc[0, "ofi"] > 0
    assert result.loc[0, "trade_imbalance"] > 0
    expected = np.log(result.loc[1, "mid"] / result.loc[0, "mid"])
    assert np.isclose(result.loc[0, "target_return"], expected)
    assert pd.isna(result.iloc[-1]["target_return"])


def test_built_csv_columns_are_unique(tmp_path):
    from hpl1.features import save_feature_csv
    from hpl1.synthetic import generate_synthetic_dataset

    synthetic = generate_synthetic_dataset(tmp_path / "synthetic.csv", rows=500)
    assert synthetic.columns.is_unique
    quotes = pd.DataFrame([
        {"event_time_ms": 100, "seq": 1, "bid": 100.0, "bid_qty": 1.0, "ask": 100.2, "ask_qty": 1.0},
        {"event_time_ms": 1100, "seq": 2, "bid": 100.1, "bid_qty": 1.0, "ask": 100.3, "ask_qty": 1.0},
    ])
    trades = pd.DataFrame(columns=["event_time_ms", "qty", "buyer_is_maker"])
    from hpl1.features import build_feature_frame
    frame = build_feature_frame(quotes, trades)
    assert frame.columns.is_unique
    output = tmp_path / "features.csv"
    save_feature_csv(frame, output)
    loaded = pd.read_csv(output)
    assert loaded.columns.is_unique


def test_binance_message_normalization():
    from hpl1.collector import _normalize_message

    quote = _normalize_message({"stream": "btcusdt@bookTicker", "data": {
        "e": "bookTicker", "T": 123, "b": "100.0", "B": "2", "a": "100.2", "A": "3"
    }}, seq=7, received_ms=130)
    trade = _normalize_message({"stream": "btcusdt@aggTrade", "data": {
        "e": "aggTrade", "T": 124, "p": "100.1", "q": "0.5", "m": True
    }}, seq=8, received_ms=131)
    assert quote["type"] == "quote"
    assert quote["bid"] == 100.0 and quote["ask_qty"] == 3.0
    assert trade["type"] == "trade"
    assert trade["buyer_is_maker"] is True
