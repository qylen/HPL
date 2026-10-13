"""Ingestion: schema handling, OFI convention, dedup/order/crossed, gaps, spot CSVs."""
import io
import zipfile
from pathlib import Path

import pytest

from hpl1.ingest import ingest_bookticker_zip, ingest_funding_zip, ingest_trades_zip

BOOK_HEADER = ("update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
               "transaction_time,event_time\n")


def _book_zip(rows: list[tuple]) -> bytes:
    """rows: (uid, bid, bidq, ask, askq, ts_ms)"""
    text = BOOK_HEADER + "".join(
        f"{u},{b},{bq},{a},{aq},{t},{t}\n" for u, b, bq, a, aq, t in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", text)
    return buf.getvalue()


def _write(tmp_path, name, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_ofi_convention_ask_drop_is_positive_pressure(tmp_path):
    """Cont best-level OFI: ask price DROPPING (offer lifted) must contribute
    POSITIVELY — the original repo inverted this sign."""
    base = 1700000000000
    rows = [
        (1, 99.9, 1.0, 100.1, 1.0, base),
        (2, 99.9, 1.0, 100.0, 1.0, base + 10),   # ask DROPS 100.1 -> 100.0
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, stats = ingest_bookticker_zip(p)
    assert stats["applied_events"] == 2
    # both events share one second: first event contributes 0 (no predecessor),
    # the ask-drop event contributes +old ask qty = +1.0
    assert frame["ofi"].sum() == pytest.approx(1.0)


def test_ofi_bid_up_positive_ask_up_negative(tmp_path):
    base = 1700000000000
    rows = [
        (1, 99.9, 2.0, 100.1, 2.0, base),
        (2, 100.0, 3.0, 100.1, 2.0, base + 10),   # bid up -> +3
        (3, 100.0, 3.0, 100.2, 4.0, base + 20),   # ask up -> -4
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, stats = ingest_bookticker_zip(p)
    assert stats["applied_events"] == 3
    assert frame["ofi"].sum() == pytest.approx(3.0 - 4.0)


def test_ofi_same_price_quantity_delta(tmp_path):
    base = 1700000000000
    rows = [
        (1, 100.0, 5.0, 100.2, 5.0, base),
        (2, 100.0, 7.0, 100.2, 3.0, base + 10),  # bid +2, ask -(5-3)=+2
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, _ = ingest_bookticker_zip(p)
    assert frame["ofi"].sum() == pytest.approx(2.0 + 2.0)


def test_duplicate_update_ids_counted_and_skipped(tmp_path):
    base = 1700000000000
    rows = [
        (1, 100.0, 1.0, 100.2, 1.0, base),
        (2, 100.0, 2.0, 100.2, 1.0, base + 10),
        (2, 100.0, 2.0, 100.2, 1.0, base + 20),  # duplicate id
        (3, 100.0, 3.0, 100.2, 1.0, base + 30),
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, stats = ingest_bookticker_zip(p)
    assert stats["dup_events"] == 1
    assert stats["applied_events"] == 3
    assert frame["n_updates"].sum() == 3


def test_out_of_order_ids_counted_and_skipped(tmp_path):
    base = 1700000000000
    rows = [
        (10, 100.0, 1.0, 100.2, 1.0, base),
        (9, 100.0, 9.0, 100.2, 1.0, base + 10),    # regression vs applied 10
        (11, 100.0, 2.0, 100.2, 1.0, base + 20),
        (8, 100.0, 8.0, 100.2, 1.0, base + 30),    # regression vs applied 11
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, stats = ingest_bookticker_zip(p)
    assert stats["out_of_order"] == 2
    assert stats["applied_events"] == 2  # uid 10 and uid 11; 9 and 8 are skipped
    # last applied quote is uid 11 (bid_qty 2.0): the 9/8 rows never touched the book
    assert frame.iloc[-1]["bid_qty"] == pytest.approx(2.0)


def test_crossed_quotes_counted_never_applied(tmp_path):
    base = 1700000000000
    rows = [
        (1, 100.0, 1.0, 100.2, 1.0, base),
        (2, 100.5, 1.0, 100.4, 1.0, base + 1010),   # bid >= ask: crossed, own second
        (3, 100.1, 1.0, 100.3, 1.0, base + 2020),
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, stats = ingest_bookticker_zip(p)
    assert stats["crossed_quotes"] == 1
    assert stats["applied_events"] == 2
    # seconds 0 and 2 have applied quotes; the crossed-only second has NO row
    # (its absence is a coverage gap, detected by the quality report)
    assert len(frame) == 2
    assert frame.iloc[-1]["bid"] == pytest.approx(100.1)  # uid 3 applied
    assert frame.iloc[0]["bid"] == pytest.approx(100.0)   # uid 1 applied


def test_per_second_bucketing_and_gap_seconds(tmp_path):
    base = 1700000000000
    rows = [
        (1, 100.0, 1.0, 100.2, 1.0, base),
        (2, 100.0, 1.0, 100.2, 1.0, base + 1500),  # next second
        (3, 100.0, 1.0, 100.2, 1.0, base + 2500),  # second after
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, _ = ingest_bookticker_zip(p)
    assert len(frame) == 3
    assert list(frame["second"]) == [base // 1000, base // 1000 + 1, base // 1000 + 2]
    assert (frame["n_updates"] == 1).all()


def test_empty_book_archive(tmp_path):
    p = _write(tmp_path, "b.zip", _book_zip([]))
    frame, stats = ingest_bookticker_zip(p)
    assert frame.empty
    assert stats["applied_events"] == 0


def test_futures_trades_zip_schema(tmp_path):
    header = ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
              "transact_time,is_buyer_maker\n")
    text = header + "1,100.0,0.5,1,1,1700000000000,false\n" \
                    "2,100.1,0.2,2,2,1700000000050,true\n" \
                    "3,100.1,0.3,3,3,1700000000050,false\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", text)
    p = _write(tmp_path, "t.zip", buf.getvalue())
    frame = ingest_trades_zip(p, market="futures")
    sec = 1700000000
    row = frame[frame["second"] == sec].iloc[0]
    assert row["n_trades"] == 3
    assert row["buy_vol"] == pytest.approx(0.8)   # false = aggressive buy
    assert row["sell_vol"] == pytest.approx(0.2)  # true = aggressive sell
    assert row["vol"] == pytest.approx(1.0)
    assert row["notional"] == pytest.approx(100.0 * 0.5 + 100.1 * 0.5)


def test_spot_trades_headerless_with_trailing_blank_line(tmp_path):
    # spot aggTrades CSVs: no header, 8 columns, trailing empty line
    text = ("1,100.0,0.5,1,1,1700000000000,false,M\n"
            "2,100.1,0.25,2,2,1700000005000,true,M\n"
            "\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", text)
    p = _write(tmp_path, "t.zip", buf.getvalue())
    frame = ingest_trades_zip(p, market="spot")
    assert len(frame) == 2
    row0 = frame.iloc[0]
    assert row0["buy_vol"] == pytest.approx(0.5)
    assert row0["sell_vol"] == pytest.approx(0.0)


def test_funding_zip_schema(tmp_path):
    text = "calc_time,funding_interval_hours,last_funding_rate\n" \
           "1700044800000,8,0.00010\n1700073600000,8,-0.00005\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", text)
    p = _write(tmp_path, "f.zip", buf.getvalue())
    frame = ingest_funding_zip(p)
    assert len(frame) == 2
    assert frame.iloc[0]["funding_rate"] == pytest.approx(0.0001)
    assert frame.iloc[1]["funding_rate"] == pytest.approx(-0.00005)
    assert set(frame.columns) >= {"second", "funding_rate", "funding_interval_hours"}


def test_funding_zip_headerless_variant(tmp_path):
    # actual Binance fundingRate CSVs have no header row
    text = "1700044800000,8,0.00010\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X.csv", text)
    p = _write(tmp_path, "f.zip", buf.getvalue())
    frame = ingest_funding_zip(p)
    assert frame.iloc[0]["funding_rate"] == pytest.approx(0.0001)


def test_corrupted_zip_raises(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(Exception):
        ingest_bookticker_zip(p)


def test_multi_second_book_ofi_carries_across_seconds(tmp_path):
    base = 1700000000000
    rows = [
        (1, 100.0, 1.0, 100.2, 1.0, base),
        (2, 100.0, 2.0, 100.2, 1.0, base + 1000 + 5),  # OFI +1 in the NEXT second
    ]
    p = _write(tmp_path, "b.zip", _book_zip(rows))
    frame, _ = ingest_bookticker_zip(p)
    assert frame.iloc[0]["ofi"] == pytest.approx(0.0)
    assert frame.iloc[1]["ofi"] == pytest.approx(1.0)
