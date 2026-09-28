"""Tests for the acquisition layer that do not touch the network.

The provider itself must be stubbed here - these tests exercise how the
download layer reacts to what yfinance returns (layouts, gaps, outages) - but
the bars the stub returns are real cached bars, re-shaped into yfinance's
response layout (title-cased columns, a tz-aware index, ticker-grouped
columns for multi-ticker requests).
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from diffolio.data.download import (
    OHLCV_COLUMNS,
    PriceCache,
    _split_response,
    download_ohlcv,
    normalize_ohlcv,
)

START, END = "2018-01-02", "2018-03-29"


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            return False
    return True


requires_parquet = pytest.mark.skipif(
    not _has_parquet_engine(), reason="no parquet engine installed"
)


def as_provider_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-shape a canonical frame the way yfinance returns it."""
    raw = frame.copy()
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("America/New_York")
    raw.index.name = "Date"
    return raw


@pytest.fixture
def real_bars(real_frames):
    return real_frames(["AAPL", "MSFT"], START, END)


def test_normalize_ohlcv_canonicalises_schema_and_index(real_bars):
    real = real_bars["AAPL"]
    raw = as_provider_frame(real)
    raw = pd.concat([raw.iloc[5:], raw.iloc[:5], raw.iloc[[5]]])  # unsorted + duplicate

    frame = normalize_ohlcv(raw)

    assert tuple(frame.columns) == OHLCV_COLUMNS
    assert frame.index.tz is None
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing
    assert len(frame) == len(real)  # the duplicated row is collapsed
    assert (frame.index.normalize() == frame.index).all()
    pd.testing.assert_frame_equal(frame, real, check_freq=False, check_names=False)


def test_normalize_ohlcv_rejects_an_incomplete_response(real_bars):
    partial = as_provider_frame(real_bars["AAPL"])[["Open", "Close"]]
    with pytest.raises(KeyError, match="high"):
        normalize_ohlcv(partial)


def test_split_response_handles_grouped_and_flat_payloads(real_bars):
    grouped = pd.concat(
        {t: as_provider_frame(f) for t, f in real_bars.items()}, axis=1
    )
    frames = _split_response(grouped, ["AAPL", "MSFT"])
    assert set(frames) == {"AAPL", "MSFT"}
    assert tuple(frames["AAPL"].columns) == OHLCV_COLUMNS
    pd.testing.assert_frame_equal(
        frames["MSFT"], real_bars["MSFT"], check_freq=False, check_names=False
    )

    # yfinance's newer layout puts the price field first: (Price, Ticker).
    swapped = grouped.swaplevel(axis=1).sort_index(axis=1)
    assert set(_split_response(swapped, ["AAPL", "MSFT"])) == {"AAPL", "MSFT"}

    # A ticker the provider silently dropped simply does not come back.
    assert set(_split_response(grouped, ["AAPL", "ZZZZ"])) == {"AAPL"}

    flat = _split_response(as_provider_frame(real_bars["AAPL"]), ["AAPL"])
    assert set(flat) == {"AAPL"}
    assert len(flat["AAPL"]) == len(real_bars["AAPL"])

    assert _split_response(pd.DataFrame(), ["AAPL"]) == {}


@requires_parquet
def test_price_cache_roundtrip_and_coverage(real_bars, tmp_path):
    frame = real_bars["AAPL"]
    index = frame.index

    cache = PriceCache(tmp_path)
    assert not cache.covers("AAPL", START, END)

    cache.write("AAPL", frame, START, END)
    cache.flush()
    assert (tmp_path / "ohlcv" / "manifest.json").exists()

    reopened = PriceCache(tmp_path)
    assert reopened.covers("AAPL", START, END)
    assert reopened.covers("AAPL", str(index[5].date()), str(index[-5].date()))
    # A request extending beyond what was fetched is not covered.
    assert not reopened.covers("AAPL", "2010-01-01", END)
    assert not reopened.covers("AAPL", START, "2030-01-01")

    sliced = reopened.read("AAPL", str(index[10].date()), str(index[20].date()))
    assert len(sliced) == 11
    pd.testing.assert_frame_equal(sliced, frame.iloc[10:21], check_freq=False)


@requires_parquet
def test_download_uses_the_cache_on_the_second_call(real_bars, tmp_path):
    calls: list[list[str]] = []

    def stub_batch(tickers, *args, **kwargs):
        calls.append(list(tickers))
        return {t: real_bars[t] for t in tickers if t in real_bars}

    with mock.patch("diffolio.data.download._download_batch", stub_batch):
        first = download_ohlcv(["AAPL", "MSFT"], START, END, cache_dir=tmp_path)
        second = download_ohlcv(["AAPL", "MSFT"], START, END, cache_dir=tmp_path)
        forced = download_ohlcv(["AAPL", "MSFT"], START, END, cache_dir=tmp_path, force=True)

    assert set(first) == {"AAPL", "MSFT"}
    assert len(first["AAPL"]) == len(real_bars["AAPL"])
    assert calls == [["AAPL", "MSFT"], ["AAPL", "MSFT"]], "only the first and forced calls hit the API"
    # check_freq is off because an inferred index frequency does not survive
    # the Parquet round-trip; the values and dates do, which is what matters.
    pd.testing.assert_frame_equal(first["AAPL"], second["AAPL"], check_freq=False)
    pd.testing.assert_frame_equal(first["MSFT"], forced["MSFT"], check_freq=False)


@requires_parquet
def test_missing_tickers_are_omitted_rather_than_faked(real_bars, tmp_path):
    # FDXF is a real S&P roster entry for which Yahoo returns no 2010-2024 bars.
    with mock.patch(
        "diffolio.data.download._download_batch",
        lambda t, *a, **k: {"AAPL": real_bars["AAPL"]},
    ):
        frames = download_ohlcv(["AAPL", "FDXF"], START, END, cache_dir=tmp_path)

    assert set(frames) == {"AAPL"}
    assert not PriceCache(tmp_path).covers("FDXF", START, END)


def test_the_offline_guard_blocks_uncached_downloads(tmp_path):
    # conftest's autouse guard: an uncached ticker would need the network.
    with pytest.raises(AssertionError, match="reach the network"):
        download_ohlcv(["AAPL"], START, END, cache_dir=tmp_path)
