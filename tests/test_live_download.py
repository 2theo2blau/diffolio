"""Live tests against the real provider (yfinance) - strictly opt-in.

Unlike the rest of the suite these tests touch the network and download *real*
market data: the acquisition layer is a critical point of the architecture and
must actually work against the real provider, not only against stubs.  They are
skipped unless explicitly enabled::

    DIFFOLIO_NETWORK_TESTS=1 python -m pytest tests/test_live_download.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from diffolio.config import DiffolioConfig
from diffolio.data.download import OHLCV_COLUMNS, download_benchmark, download_ohlcv
from diffolio.data.pipeline import build_dataset
from diffolio.data.windows import window_dataset

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("DIFFOLIO_NETWORK_TESTS", "") not in {"1", "true", "yes"},
        reason="live provider tests are opt-in: set DIFFOLIO_NETWORK_TESTS=1",
    ),
]

#: Mega-caps with continuous, liquid trading over the whole study window.
TICKERS = ["AAPL", "MSFT", "JPM", "PG"]
BENCHMARK = "^GSPC"
START, END = "2023-01-01", "2023-12-31"
#: 2023 had ~250 US trading days; allow slack for provider rounding.
MIN_DAYS, MAX_DAYS = 200, 260


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """One shared raw-price cache, so the second test reuses the first's bars."""
    return tmp_path_factory.mktemp("live_cache")


def test_real_download_returns_canonical_daily_bars(cache_dir):
    frames = download_ohlcv(TICKERS, START, END, cache_dir=str(cache_dir))
    benchmark = download_benchmark(BENCHMARK, START, END, cache_dir=str(cache_dir))

    assert set(frames) == set(TICKERS)
    for frame in [*frames.values(), benchmark]:
        assert tuple(frame.columns) == OHLCV_COLUMNS
        assert frame.index.name == "date"
        assert frame.index.tz is None
        assert frame.index.is_monotonic_increasing
        assert str(frame.index[0].date()) >= START
        assert str(frame.index[-1].date()) <= END
        assert MIN_DAYS <= len(frame) <= MAX_DAYS
        assert (frame[["open", "high", "low", "close"]] > 0).all().all()
        assert (frame["high"] >= frame["low"]).all()

    # Equities really traded (an index can report zero volume).
    for frame in frames.values():
        assert (frame["volume"] >= 0).all()
        assert float(frame["volume"].median()) > 0.0


def test_end_to_end_build_with_real_data(cache_dir, tmp_path_factory):
    config = DiffolioConfig()
    config.name = "live_smoke"
    config.universe.roster = "list:" + ",".join(TICKERS)
    config.universe.benchmark = BENCHMARK
    config.universe.target_size = None
    config.universe.min_avg_dollar_volume = 0.0
    config.data.start = START
    config.data.end = END
    config.data.cache_dir = str(cache_dir)
    config.features.ref_window = 5
    config.window.lookback = 16

    dataset = build_dataset(config, output_dir=tmp_path_factory.mktemp("live_out"))

    panel = dataset.panel
    assert panel.benchmark == BENCHMARK
    assert panel.tickers == tuple(sorted(TICKERS))
    # Four warm-up rows are trimmed by the 5-day price reference.
    assert MIN_DAYS - 10 <= panel.n_days <= MAX_DAYS
    assert np.isfinite(panel.features).all()
    assert float(panel.observed.mean()) > 0.98  # mega-caps: essentially no gaps
    for name in ("train", "val", "test"):
        assert dataset.splits[name].n_samples > 0

    # Section 5 consumes the real panel like the model side will.
    windows = window_dataset(dataset, "test")
    sample = windows[0]
    assert sample.h.shape == (panel.n_assets, config.lookback, panel.n_features)
    assert float(sample.r_valid.float().mean()) > 0.9
