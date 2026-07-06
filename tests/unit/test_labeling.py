"""
tests/unit/test_labeling.py — Triple-barrier labeling correctness.
"""

from __future__ import annotations

import pandas as pd

from app.training.labeling import label_events_batch, label_triple_barrier


def _make_df(prices_high_low_close: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Make df."""
    df = pd.DataFrame(prices_high_low_close, columns=["high", "low", "close"])
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = 1000
    return df.reset_index(drop=True)


def test_buy_top_barrier_hit_first():
    """Test buy top barrier hit first."""

    df = _make_df(
        [
            (100, 100, 100),
            (104, 99, 102),
            (106, 101, 105),
            (107, 103, 106),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="BUY",
        entry=100,
        stop=95,
        target=105,
        horizon_bars=10,
    )
    assert res.label == 1
    assert res.binary == 1
    assert res.exit_bar_idx == 2
    assert res.barrier_hit == "top"
    assert res.holding_bars == 2


def test_buy_bottom_barrier_hit_first():
    """Test buy bottom barrier hit first."""

    df = _make_df(
        [
            (100, 100, 100),
            (102, 94, 96),
            (103, 95, 100),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="BUY",
        entry=100,
        stop=95,
        target=105,
        horizon_bars=10,
    )
    assert res.label == -1
    assert res.binary == 0
    assert res.exit_bar_idx == 1
    assert res.barrier_hit == "bottom"


def test_timeout_vertical_barrier():
    """Test timeout vertical barrier."""

    df = _make_df(
        [
            (100, 100, 100),
            (102, 99, 100),
            (103, 98, 102),
            (104, 100, 101),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="BUY",
        entry=100,
        stop=90,
        target=110,
        horizon_bars=3,
    )
    assert res.label == 0
    assert res.binary == 0
    assert res.barrier_hit == "timeout"


def test_sell_top_barrier_hit_first():
    """Test sell top barrier hit first."""

    df = _make_df(
        [
            (100, 100, 100),
            (102, 94, 96),
            (103, 95, 99),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="SELL",
        entry=100,
        stop=105,
        target=95,
        horizon_bars=10,
    )
    assert res.label == 1
    assert res.binary == 1
    assert res.barrier_hit == "top"


def test_sell_stop_hit_first():
    """Test sell stop hit first."""

    df = _make_df(
        [
            (100, 100, 100),
            (106, 99, 104),
            (107, 95, 102),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="SELL",
        entry=100,
        stop=105,
        target=95,
        horizon_bars=10,
    )
    assert res.label == -1
    assert res.binary == 0
    assert res.barrier_hit == "bottom"


def test_no_future_data():
    """Test no future data."""

    df = _make_df([(100, 100, 100)])
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="BUY",
        entry=100,
        stop=95,
        target=105,
        horizon_bars=10,
    )
    assert res.barrier_hit == "no_data"
    assert res.exit_bar_idx == -1


def test_atr_fallback_when_no_explicit_levels():
    """Test atr fallback when no explicit levels."""

    df = _make_df(
        [
            (100, 100, 100),
            (105, 99, 103),
        ]
    )
    res = label_triple_barrier(
        df,
        bar_idx=0,
        direction="BUY",
        entry=100,
        atr_at_entry=2.0,
        atr_mult_top=2.0,
        atr_mult_bot=1.0,
        horizon_bars=10,
    )
    assert res.label == 1
    assert res.barrier_hit == "top"


def test_batch_labeling_returns_dataframe():
    """Test batch labeling returns dataframe."""
    df = _make_df(
        [
            (100, 100, 100),
            (105, 99, 103),
            (106, 95, 100),
            (107, 96, 102),
        ]
    )
    events = [
        {
            "bar_idx": 0,
            "direction": "BUY",
            "entry": 100,
            "stop": 95,
            "target": 105,
            "atr_at_entry": 2.0,
        },
        {
            "bar_idx": 1,
            "direction": "BUY",
            "entry": 103,
            "stop": 100,
            "target": 110,
            "atr_at_entry": 2.0,
        },
    ]
    out = label_events_batch(df, events, horizon_bars=5)
    assert len(out) == 2
    assert "label" in out.columns
    assert "binary" in out.columns
    assert "barrier_hit" in out.columns
    assert "exit_bar_idx" in out.columns
    assert "holding_bars" in out.columns

    assert out.iloc[0]["label"] == 1
