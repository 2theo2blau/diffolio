"""Shared fixtures: real S&P 500 data from the local build, and an offline guard.

Tests prefer real data wherever they can (see AGENTS.md).  Two local sources
back them, both produced once by
``python scripts/build_dataset.py -c configs/us_sp500.yaml``:

* ``data/processed/us_sp500`` - the built panel, splits and targets
  (``real_dataset``, ``real_targets``, ``real_subpanel``);
* ``data/cache/ohlcv`` - the raw yfinance bars behind it (``real_frames``,
  ``real_cache``).

The fixtures only *read* these; ``real_cache`` copies the files a test needs
into a temporary cache so a test can never modify the real one.  Without a
local build the dependent tests skip with a pointer to the build command.

Every test except the opt-in ``network`` ones runs under :func:`offline`,
which makes any provider download or roster fetch raise, so "offline" is
enforced rather than assumed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET_DIR = REPO_ROOT / "data" / "processed" / "us_sp500"
REAL_CACHE_DIR = REPO_ROOT / "data" / "cache"
_BUILD_HINT = "build it once with `python scripts/build_dataset.py -c configs/us_sp500.yaml`"

#: Real tickers used by the raw-data tests: liquid, continuously listed.
REAL_TICKERS = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT")
REAL_BENCHMARK = "^GSPC"


@pytest.fixture(autouse=True)
def offline(request, monkeypatch):
    """Fail loudly if a non-network test tries to reach a provider."""
    if request.node.get_closest_marker("network"):
        return

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a test tried to reach the network; use the real-data fixtures, which "
            "read the local cache, or mark the test `network`"
        )

    monkeypatch.setattr("diffolio.data.download._download_batch", refuse)
    monkeypatch.setattr("diffolio.data.universe.urllib.request.urlopen", refuse)


@pytest.fixture(scope="session")
def real_dataset():
    """The processed S&P 500 dataset (memory-mapped, read-only)."""
    from diffolio.data.pipeline import load_dataset

    try:
        return load_dataset(REAL_DATASET_DIR, mmap=True)
    except FileNotFoundError:
        pytest.skip(f"no real dataset at {REAL_DATASET_DIR}; {_BUILD_HINT}")


@pytest.fixture(scope="session")
def real_targets(real_dataset):
    """Section-6 targets for the real dataset (cached next to it)."""
    from diffolio.data.targets import PortfolioTargets, build_targets

    if not PortfolioTargets.exists(REAL_DATASET_DIR / "targets"):
        pytest.skip("real dataset has no cached targets; re-run scripts/build_dataset.py")
    return build_targets(real_dataset, mmap=True)


@pytest.fixture(scope="session")
def real_schedule(real_dataset, real_targets):
    """The section-7 schedule, with sigma_x fitted on the real training split."""
    from diffolio.diffusion import fit_schedule

    return fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)


@pytest.fixture(scope="session")
def real_subpanel(real_dataset) -> Callable[..., "MarketPanel"]:  # noqa: F821
    """Factory for a small, writable ``MarketPanel`` cut from the real build.

    ``real_subpanel(n_days, n_assets, start=0)`` takes a contiguous block of
    days and the first ``n_assets`` assets; returns are re-derived from the
    real opens by ``MarketPanel.build``, exactly as the pipeline does.  The
    arrays are copies, so a test may inject an edge case (e.g. a gap) into
    them without touching the real dataset.
    """
    from diffolio.data.panel import MarketPanel

    source = real_dataset.panel

    def make(n_days: int, n_assets: int, start: int = 0) -> MarketPanel:
        if start + n_days > source.n_days or n_assets > source.n_assets:
            raise ValueError("requested block exceeds the real panel")
        days = slice(start, start + n_days)
        assets = slice(0, n_assets)
        return MarketPanel.build(
            calendar=source.calendar[days],
            tickers=source.tickers[assets],
            feature_names=source.feature_names,
            benchmark=source.benchmark,
            features=np.array(source.features[days, assets]),
            index_features=np.array(source.index_features[days]),
            open_prices=np.array(source.open_prices[days, assets]),
            close_prices=np.array(source.close_prices[days, assets]),
            index_open=np.array(source.index_open[days]),
            index_close=np.array(source.index_close[days]),
            observed=np.array(source.observed[days, assets]),
            index_observed=np.array(source.index_observed[days]),
            metadata={"fingerprint": "real-subpanel"},
        )

    return make


def _require_cache(tickers: Sequence[str]) -> None:
    from diffolio.data.download import PriceCache

    missing = [t for t in tickers if not PriceCache(REAL_CACHE_DIR).path_for(t).exists()]
    if missing:
        pytest.skip(f"real price cache lacks {missing}; {_BUILD_HINT}")


@pytest.fixture(scope="session")
def real_frames() -> Callable[..., dict[str, pd.DataFrame]]:
    """Factory: raw cached yfinance bars, ``real_frames(tickers, start, end)``."""
    from diffolio.data.download import PriceCache

    def load(
        tickers: Sequence[str] = REAL_TICKERS,
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> dict[str, pd.DataFrame]:
        _require_cache(tickers)
        cache = PriceCache.__new__(PriceCache)  # read-only: no mkdir, no manifest
        cache.root = REAL_CACHE_DIR / "ohlcv"
        return {t: pd.read_parquet(cache.path_for(t)).loc[start:end].copy() for t in tickers}

    return load


@pytest.fixture
def real_cache(tmp_path) -> Callable[[Sequence[str]], Path]:
    """Factory: a temporary cache dir holding copies of the real bars for
    ``tickers``, with their manifest entries, so a pipeline build is served
    entirely from real cached data."""
    from diffolio.utils import read_json, write_json

    def make(tickers: Sequence[str]) -> Path:
        _require_cache(tickers)
        from diffolio.data.download import PriceCache

        src = PriceCache.__new__(PriceCache)
        src.root = REAL_CACHE_DIR / "ohlcv"
        manifest = read_json(src.root / "manifest.json")
        dst = tmp_path / "cache" / "ohlcv"
        dst.mkdir(parents=True, exist_ok=True)
        for ticker in tickers:
            shutil.copy2(src.path_for(ticker), dst / src.path_for(ticker).name)
        write_json(dst / "manifest.json", {t: manifest[t] for t in tickers})
        return tmp_path / "cache"

    return make


@pytest.fixture
def real_pipeline_config(real_cache):
    """Factory: a config whose build runs end to end on real cached bars.

    ``real_pipeline_config(tickers=REAL_TICKERS, start=..., end=..., lookback=10)``
    returns a ``DiffolioConfig`` with a ``list:`` roster (no roster fetch),
    the real ``^GSPC`` benchmark, and a temporary copy of the real cache, so
    ``build_dataset`` needs neither mocks nor the network.
    """
    from diffolio.config import DiffolioConfig

    def make(
        tickers: Sequence[str] = REAL_TICKERS,
        start: str = "2018-01-01",
        end: str = "2019-12-31",
        lookback: int = 10,
    ) -> DiffolioConfig:
        cache_dir = real_cache([*tickers, REAL_BENCHMARK])
        config = DiffolioConfig()
        config.name = "real-test"
        config.universe.roster = "list:" + ",".join(tickers)
        config.universe.benchmark = REAL_BENCHMARK
        config.universe.target_size = None
        config.data.start = start
        config.data.end = end
        config.data.cache_dir = str(cache_dir)
        config.window.lookback = lookback
        config.diffusion.gamma_max = min(config.diffusion.gamma_max, len(tickers))
        return config

    return make
