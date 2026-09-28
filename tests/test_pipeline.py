"""End-to-end build tests (sections 1-4), on real cached yfinance bars.

Every build here is served by ``real_pipeline_config``: a ``list:`` roster of
real tickers, the real ``^GSPC`` benchmark and a temporary copy of the real
cache, with the ``offline`` guard failing any network call.  No provider is
mocked.  The one injected edge case is a mid-series gap (the real data has
none); the refill tests use genuine cleaning drops - CHTR's first bar is a
day late, and DHR's broken spin-off adjustment trips a tightened jump check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diffolio.data.download import PriceCache
from diffolio.data.pipeline import build_dataset, load_dataset

from conftest import REAL_TICKERS

LOOKBACK = 10


def _inject_gap(cache_dir: Path, ticker: str, positions) -> None:
    """Delete bars from the *temporary* cache copy to create a gap."""
    path = PriceCache(cache_dir).path_for(ticker)
    frame = pd.read_parquet(path)
    frame.drop(frame.index[list(positions)]).to_parquet(path)


def _benchmark_days(cache_dir: str, start: str, end: str) -> pd.DatetimeIndex:
    return PriceCache(cache_dir).read("^GSPC", start, end).index


def test_end_to_end_build_produces_a_consistent_dataset(real_pipeline_config, tmp_path):
    config = real_pipeline_config()
    # Drop three JPM bars in mid-2018 from the cache copy (which spans 2010-2024).
    days = _benchmark_days(config.data.cache_dir, "2010-01-01", "2024-12-31")
    gap_at = int(days.get_loc(pd.Timestamp("2018-06-01")))
    _inject_gap(Path(config.data.cache_dir), "JPM", range(gap_at, gap_at + 3))

    dataset = build_dataset(config, output_dir=tmp_path / "out")
    panel, splits = dataset.panel, dataset.splits
    calendar = _benchmark_days(config.data.cache_dir, config.data.start, config.data.end)

    assert panel.n_assets == len(REAL_TICKERS)
    assert panel.n_features == 5
    assert panel.tickers == tuple(sorted(REAL_TICKERS))
    assert panel.benchmark == "^GSPC"
    # ref_window - 1 = 19 warm-up rows are consumed by the price reference.
    warmup = config.features.ref_window - 1
    assert panel.n_days == len(calendar) - warmup
    assert panel.calendar[0] == calendar[warmup]
    assert panel.calendar[-1] == calendar[-1]
    assert np.isfinite(panel.features).all()
    assert np.isfinite(panel.returns).all()

    # The asset with missing bars is forward-filled and flagged; no other is.
    jpm = panel.asset_index("JPM")
    assert panel.observed[:, jpm].sum() == panel.n_days - 3
    others = np.delete(panel.observed, jpm, axis=1)
    assert others.all()

    # Returns are the real open-to-open moves of the adjusted opens.
    aapl = panel.asset_index("AAPL")
    raw = PriceCache(config.data.cache_dir).read("AAPL", config.data.start, config.data.end)
    opens = raw["open"].reindex(panel.calendar).to_numpy()
    np.testing.assert_allclose(
        panel.returns[:-1, aapl], opens[1:] / opens[:-1] - 1.0, rtol=1e-4, atol=1e-6
    )

    train_end = splits.train.stop
    assert splits.train.start == 0 and splits.val.start == train_end
    assert splits.test.stop == panel.n_days
    for _, split in splits.items():
        assert split.tau.min() >= split.start + config.lookback - 1
        assert split.tau.max() <= split.stop - 2
    # Steps whose target touches the gap are excluded (1 of 6 assets > 5%).
    gap_day = int(panel.calendar.get_loc(pd.Timestamp("2018-06-01")))
    assert not np.isin(np.arange(gap_day - 1, gap_day + 3), splits.train.tau).any()

    # Standardisation statistics come from the training split only, so the
    # training block is centred while the later splits are not re-centred.
    train_block = panel.features[:train_end]
    price_features = slice(0, 4)
    assert np.allclose(train_block[..., price_features].mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(train_block[..., price_features].std(axis=(0, 1)), 1.0, atol=1e-2)
    assert float(np.abs(panel.features).max()) <= config.features.clip_sigma + 1e-6

    h, g, r = panel.window(int(splits.test.tau[0]), config.lookback)
    assert h.shape == (panel.n_assets, config.lookback, panel.n_features)
    assert g.shape == (config.lookback, panel.n_features)
    assert r.shape == (panel.n_assets,)


def test_artifacts_are_written_and_reloadable(real_pipeline_config, tmp_path):
    out = tmp_path / "out"
    dataset = build_dataset(real_pipeline_config(), output_dir=out)

    for relative in (
        "panel/meta.json",
        "panel/arrays/features.npy",
        "splits.json",
        "universe.json",
        "build_report.json",
        "config.yaml",
        "diagnostics/screen.csv",
    ):
        assert (out / relative).exists(), relative

    reloaded = load_dataset(out)
    np.testing.assert_array_equal(reloaded.panel.features, dataset.panel.features)
    np.testing.assert_array_equal(reloaded.splits.train.tau, dataset.splits.train.tau)
    assert reloaded.universe.tickers == dataset.universe.tickers
    assert reloaded.report["panel"]["N"] == dataset.panel.n_assets


def test_second_build_hits_the_cache_and_config_changes_invalidate_it(
    real_pipeline_config, tmp_path
):
    out = tmp_path / "out"
    config = real_pipeline_config()
    first = build_dataset(config, output_dir=out)

    # A dataset cache hit reloads without re-running the pipeline.
    cached = build_dataset(config, output_dir=out)
    assert cached.report["built_at"] == first.report["built_at"]

    changed = real_pipeline_config()
    changed.data.cache_dir = config.data.cache_dir
    changed.window.lookback = 20
    rebuilt = build_dataset(changed, output_dir=out)
    assert rebuilt.splits.lookback == 20
    assert rebuilt.panel.metadata["fingerprint"] != cached.panel.metadata["fingerprint"]


def test_build_reports_when_the_universe_collapses(real_pipeline_config, tmp_path):
    config = real_pipeline_config()
    config.universe.min_avg_dollar_volume = 1e15
    with pytest.raises(RuntimeError, match="eligibility screen"):
        build_dataset(config, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# Refilling assets that cleaning drops, on genuine drops in the real data.


def _refill_config(real_pipeline_config, tickers, target_size, **data):
    config = real_pipeline_config(tickers, start="2010-01-01", end="2016-12-31")
    config.universe.target_size = target_size
    config.universe.include = ["CHTR"]  # forced in, so cleaning must drop it
    config.diffusion.gamma_max = 2
    for key, value in data.items():
        setattr(config.data, key, value)
    return config


def test_assets_dropped_in_cleaning_are_refilled_from_the_reserve(
    real_pipeline_config, tmp_path
):
    # CHTR's first bar is 2010-01-05, a day after the calendar starts: it passes
    # the screen's listing grace but cleaning drops it for its leading gap.
    config = _refill_config(real_pipeline_config, ["CHTR", "AAPL", "MSFT", "NVR"], 2)
    dataset = build_dataset(config, output_dir=tmp_path / "out")

    # Kept: CHTR (forced) + AAPL (most liquid); MSFT is next in line.
    assert dataset.panel.tickers == ("AAPL", "MSFT")
    assert dataset.universe.tickers == dataset.panel.tickers
    assert dataset.universe.criteria["dropped_in_cleaning"] == ["CHTR"]
    assert dataset.universe.criteria["refilled_from_reserve"] == ["MSFT"]
    assert dataset.report["cleaning"]["dropped"] == {"CHTR": "leading_gap"}


def test_a_replacement_that_is_itself_dropped_is_replaced_again(
    real_pipeline_config, tmp_path
):
    # With the jump check tightened to 1.5x, DHR's bad 2016 spin-off
    # adjustment (+61% overnight) is caught as an anomaly.  DHR out-trades NVR,
    # so it is the first replacement for CHTR - and is dropped in turn.
    config = _refill_config(
        real_pipeline_config, ["CHTR", "AAPL", "DHR", "NVR"], 2, max_daily_jump=1.5
    )
    dataset = build_dataset(config, output_dir=tmp_path / "out")

    assert dataset.panel.tickers == ("AAPL", "NVR")
    assert dataset.universe.criteria["refilled_from_reserve"] == ["DHR", "NVR"]
    assert dataset.report["cleaning"]["dropped"] == {
        "CHTR": "leading_gap",
        "DHR": "anomalous",
    }


def test_an_exhausted_reserve_leaves_a_smaller_universe(real_pipeline_config, tmp_path):
    config = _refill_config(real_pipeline_config, ["CHTR", "AAPL", "MSFT"], 3)
    dataset = build_dataset(config, output_dir=tmp_path / "out")
    assert dataset.panel.tickers == ("AAPL", "MSFT")
