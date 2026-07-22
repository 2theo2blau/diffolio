from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from diffolio.config import UniverseConfig
from diffolio.data.universe import (
    Universe,
    build_universe,
    load_candidate_roster,
    normalize_ticker,
    screen_candidates,
    subset_universe,
)
from synthetic import make_ohlcv, trading_days


def test_normalize_ticker_maps_share_classes_to_yahoo():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker("bf.b") == "BF-B"
    assert normalize_ticker("AAPL") == "AAPL"
    # Exchange suffixes are longer than one character and must survive intact.
    assert normalize_ticker("005930.KS") == "005930.KS"


def test_roster_specs_resolve_without_network():
    assert load_candidate_roster("list:aapl, msft,brk.b") == ["AAPL", "BRK-B", "MSFT"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roster.txt"
        path.write_text("# comment\nAAPL\nMSFT\n\n")
        assert load_candidate_roster(f"file:{path}") == ["AAPL", "MSFT"]

        csv = Path(tmp) / "roster.csv"
        pd.DataFrame({"Symbol": ["NVDA", "AMD"]}).to_csv(csv, index=False)
        assert load_candidate_roster(f"file:{csv}") == ["AMD", "NVDA"]


def _screen_config(**kwargs) -> UniverseConfig:
    defaults = dict(
        target_size=None,
        min_price=1.0,
        min_avg_dollar_volume=1.0e6,
        max_missing_frac=0.05,
        require_full_history=True,
    )
    defaults.update(kwargs)
    return UniverseConfig(**defaults)


def test_screen_rejects_penny_illiquid_late_and_sparse():
    index = trading_days(periods=200)
    frames = {
        "GOOD": make_ohlcv(index, seed=1, price0=100.0, volume=5.0e6),
        "PENNY": make_ohlcv(index, seed=2, price0=0.4, volume=5.0e8),
        "THIN": make_ohlcv(index, seed=3, price0=100.0, volume=100.0),
        "LATE": make_ohlcv(index[80:], seed=4, price0=100.0, volume=5.0e6),
        "SPARSE": make_ohlcv(index, seed=5, price0=100.0, volume=5.0e6).iloc[::2],
    }
    start, end = str(index[0].date()), str(index[-1].date())

    result = screen_candidates(
        frames, _screen_config(), start=start, end=end, reference_days=len(index)
    )

    status = result.report["status"].to_dict()
    assert status["GOOD"] == "eligible"
    assert status["PENNY"] == "penny_stock"
    assert status["THIN"] == "illiquid"
    assert status["LATE"] == "late_listing"
    assert status["SPARSE"] == "sparse"
    assert result.eligible == ["GOOD"]


def test_include_and_exclude_override_the_screen():
    index = trading_days(periods=120)
    frames = {
        "KEEP": make_ohlcv(index, seed=1, price0=100.0, volume=5.0e6),
        "TINY": make_ohlcv(index, seed=2, price0=100.0, volume=10.0),
        "BANNED": make_ohlcv(index, seed=3, price0=100.0, volume=5.0e6),
    }
    config = _screen_config(include=["TINY"], exclude=["BANNED"])
    result = screen_candidates(
        frames,
        config,
        start=str(index[0].date()),
        end=str(index[-1].date()),
        reference_days=len(index),
    )
    assert sorted(result.eligible) == ["KEEP", "TINY"]
    assert result.report.loc["BANNED", "status"] == "excluded"


def test_build_universe_ranks_by_liquidity_and_cuts_to_target_size():
    index = trading_days(periods=120)
    volumes = {"A": 9.0e6, "B": 8.0e6, "C": 7.0e6, "D": 6.0e6}
    frames = {
        ticker: make_ohlcv(index, seed=i, price0=100.0, volume=v)
        for i, (ticker, v) in enumerate(volumes.items())
    }
    config = _screen_config(target_size=2)
    start, end = str(index[0].date()), str(index[-1].date())
    screen = screen_candidates(frames, config, start=start, end=end, reference_days=len(index))

    universe = build_universe(screen, config, start=start, end=end)

    assert universe.size == 2
    assert set(universe.tickers) == {"A", "B"}
    # Canonical ordering is alphabetical and stable.
    assert universe.tickers == tuple(sorted(universe.tickers))
    assert universe.index("B") == 1


def test_universe_roundtrip_and_subset():
    universe = Universe(
        market="US",
        benchmark="^GSPC",
        tickers=("A", "B", "C"),
        start="2015-01-01",
        end="2020-01-01",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "universe.json"
        universe.save(path)
        restored = Universe.load(path)
    assert restored.tickers == universe.tickers
    assert restored.index_map == {"A": 0, "B": 1, "C": 2}

    reduced = subset_universe(universe, ["C", "A"])
    assert reduced.tickers == ("A", "C")
    assert reduced.criteria["dropped_in_cleaning"] == ["B"]
