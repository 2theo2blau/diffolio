"""Tests for the acquisition layer that do not touch the network."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

from diffolio.data.download import (
    OHLCV_COLUMNS,
    PriceCache,
    _split_response,
    download_ohlcv,
    normalize_ohlcv,
)
from synthetic import make_ohlcv, trading_days


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            return False
    return True


def _provider_frame(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """A frame shaped like a yfinance response: title-cased, tz-aware index."""
    frame = make_ohlcv(index, seed=seed)
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    frame.index = frame.index.tz_localize("America/New_York")
    return frame


def test_normalize_ohlcv_canonicalises_schema_and_index():
    index = trading_days(periods=10)
    raw = _provider_frame(index)
    raw = pd.concat([raw.iloc[5:], raw.iloc[:5], raw.iloc[[5]]])  # unsorted + duplicate

    frame = normalize_ohlcv(raw)

    assert tuple(frame.columns) == OHLCV_COLUMNS
    assert frame.index.tz is None
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 10  # the duplicated row is collapsed
    assert (frame.index.normalize() == frame.index).all()


def test_normalize_ohlcv_rejects_an_incomplete_response():
    frame = pd.DataFrame({"Open": [1.0], "Close": [1.0]}, index=trading_days(periods=1))
    try:
        normalize_ohlcv(frame)
    except KeyError as exc:
        assert "high" in str(exc)
    else:
        raise AssertionError("expected missing columns to raise")


def test_split_response_handles_grouped_and_flat_payloads():
    index = trading_days(periods=8)
    grouped = pd.concat(
        {"AAA": _provider_frame(index, 1), "BBB": _provider_frame(index, 2)}, axis=1
    )
    frames = _split_response(grouped, ["AAA", "BBB"])
    assert set(frames) == {"AAA", "BBB"}
    assert tuple(frames["AAA"].columns) == OHLCV_COLUMNS

    # A ticker the provider silently dropped simply does not come back.
    assert set(_split_response(grouped, ["AAA", "ZZZ"])) == {"AAA"}

    flat = _split_response(_provider_frame(index, 3), ["AAA"])
    assert set(flat) == {"AAA"}
    assert len(flat["AAA"]) == 8

    assert _split_response(pd.DataFrame(), ["AAA"]) == {}


def test_price_cache_roundtrip_and_coverage():
    if not _has_parquet_engine():
        print("skipping parquet cache test: no parquet engine installed")
        return

    index = trading_days(start="2015-01-01", periods=60)
    frame = normalize_ohlcv(_provider_frame(index))
    start, end = str(index[0].date()), str(index[-1].date())

    with tempfile.TemporaryDirectory() as tmp:
        cache = PriceCache(tmp)
        assert not cache.covers("AAA", start, end)

        cache.write("AAA", frame, start, end)
        cache.flush()
        assert (Path(tmp) / "ohlcv" / "manifest.json").exists()

        reopened = PriceCache(tmp)
        assert reopened.covers("AAA", start, end)
        assert reopened.covers("AAA", str(index[5].date()), str(index[-5].date()))
        # A request extending beyond what was fetched is not covered.
        assert not reopened.covers("AAA", "2010-01-01", end)
        assert not reopened.covers("AAA", start, "2030-01-01")

        sliced = reopened.read("AAA", str(index[10].date()), str(index[20].date()))
        assert len(sliced) == 11


def test_download_uses_the_cache_on_the_second_call():
    if not _has_parquet_engine():
        print("skipping parquet cache test: no parquet engine installed")
        return

    index = trading_days(start="2015-01-01", periods=40)
    payload = {t: normalize_ohlcv(_provider_frame(index, i)) for i, t in enumerate(["AAA", "BBB"])}
    start, end = str(index[0].date()), str(index[-1].date())
    calls: list[list[str]] = []

    def fake_batch(tickers, *args, **kwargs):
        calls.append(list(tickers))
        return {t: payload[t] for t in tickers if t in payload}

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch("diffolio.data.download._download_batch", fake_batch):
            first = download_ohlcv(["AAA", "BBB"], start, end, cache_dir=tmp)
            second = download_ohlcv(["AAA", "BBB"], start, end, cache_dir=tmp)
            forced = download_ohlcv(["AAA", "BBB"], start, end, cache_dir=tmp, force=True)

    assert set(first) == {"AAA", "BBB"}
    assert len(first["AAA"]) == 40
    assert calls == [["AAA", "BBB"], ["AAA", "BBB"]], "only the first and forced calls hit the API"
    # check_freq is off because an inferred index frequency does not survive
    # the Parquet round-trip; the values and dates do, which is what matters.
    pd.testing.assert_frame_equal(first["AAA"], second["AAA"], check_freq=False)
    pd.testing.assert_frame_equal(first["BBB"], forced["BBB"], check_freq=False)


def test_missing_tickers_are_omitted_rather_than_faked():
    if not _has_parquet_engine():
        print("skipping parquet cache test: no parquet engine installed")
        return

    index = trading_days(start="2015-01-01", periods=20)
    good = normalize_ohlcv(_provider_frame(index))

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch("diffolio.data.download._download_batch", lambda t, *a, **k: {"AAA": good}):
            frames = download_ohlcv(
                ["AAA", "GONE"], str(index[0].date()), str(index[-1].date()), cache_dir=tmp
            )

    assert set(frames) == {"AAA"}
