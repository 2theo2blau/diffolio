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


def test_screen_rejects_penny_illiquid_late_and_sparse(real_frames):
    # Real stocks in each role, with thresholds set relative to them over
    # mid-2010 .. mid-2011: F (~$6) is a "penny stock" under min_price=10,
    # NVR (~$26M/day) is illiquid under a $100M floor, KMI listed in Feb 2011.
    # (Not AAPL: split-adjusted, it traded under $10 then too.)
    start, end = "2010-06-01", "2011-06-30"
    frames = real_frames(["JPM", "F", "NVR", "KMI", "MSFT", "^GSPC"], start, end)
    benchmark = frames.pop("^GSPC")
    frames["MSFT"] = frames["MSFT"].iloc[::2]  # injected: no real stock is this sparse
    config = _screen_config(min_price=10.0, min_avg_dollar_volume=1.0e8)

    result = screen_candidates(frames, config, start=start, end=end, reference_days=len(benchmark))

    status = result.report["status"].to_dict()
    assert status["JPM"] == "eligible"
    assert status["F"] == "penny_stock"
    assert status["NVR"] == "illiquid"
    assert status["KMI"] == "late_listing"
    assert status["MSFT"] == "sparse"
    assert result.eligible == ["JPM"]


def test_real_screen_statistics_match_the_bars(real_frames):
    start, end = "2018-01-01", "2018-12-31"
    frames = real_frames(["AAPL", "JPM"], start, end)
    result = screen_candidates(frames, _screen_config(), start=start, end=end)
    for ticker, frame in frames.items():
        row = result.report.loc[ticker]
        assert row["rows"] == len(frame) == 251
        assert row["median_dollar_volume"] == (frame["close"] * frame["volume"]).median()
        assert row["status"] == "eligible"


def test_include_and_exclude_override_the_screen(real_frames):
    start, end = "2018-01-01", "2018-12-31"
    frames = real_frames(["AAPL", "NVR", "JPM"], start, end)
    # NVR fails a $1B liquidity floor but is forced in; JPM passes but is banned.
    config = _screen_config(min_avg_dollar_volume=1.0e9, include=["NVR"], exclude=["JPM"])
    result = screen_candidates(frames, config, start=start, end=end, reference_days=251)
    assert sorted(result.eligible) == ["AAPL", "NVR"]
    assert result.report.loc["JPM", "status"] == "excluded"


def test_build_universe_ranks_by_liquidity_and_cuts_to_target_size(real_frames):
    start, end = "2018-01-01", "2018-12-31"
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"]
    frames = real_frames(tickers, start, end)
    config = _screen_config(target_size=2)
    screen = screen_candidates(frames, config, start=start, end=end, reference_days=251)

    universe = build_universe(screen, config, start=start, end=end)

    liquidity = {t: float((f["close"] * f["volume"]).median()) for t, f in frames.items()}
    top_two = sorted(liquidity, key=liquidity.get, reverse=True)[:2]
    assert universe.size == 2
    assert set(universe.tickers) == set(top_two)
    # Canonical ordering is alphabetical and stable.
    assert universe.tickers == tuple(sorted(top_two))
    assert universe.index(universe.tickers[1]) == 1


def test_the_real_universe_is_the_liquid_top_224(real_dataset):
    universe = real_dataset.universe
    assert universe.size == 224
    assert universe.tickers == tuple(sorted(universe.tickers))
    assert "DHR" not in universe  # excluded in configs/us_sp500.yaml
    assert universe.criteria["dropped_in_cleaning"] == ["CHTR"]
    assert universe.criteria["refilled_from_reserve"] == ["WBD"]


def test_universe_roundtrip_and_subset(real_dataset, tmp_path):
    universe = real_dataset.universe
    path = tmp_path / "universe.json"
    universe.save(path)
    restored = Universe.load(path)
    assert restored.tickers == universe.tickers
    assert restored.index_map == {t: i for i, t in enumerate(universe.tickers)}
    assert restored.criteria == dict(universe.criteria)

    keep = universe.tickers[::2]
    reduced = subset_universe(universe, keep[::-1])
    assert reduced.tickers == keep
    assert reduced.criteria["dropped_in_cleaning"] == sorted(universe.tickers[1::2])
