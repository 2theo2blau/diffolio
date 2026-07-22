"""A failed download must not be cached as a valid empty answer."""

from __future__ import annotations

import tempfile
from unittest import mock

from diffolio.data.download import PriceCache, download_ohlcv, normalize_ohlcv
from test_download import _has_parquet_engine, _provider_frame
from synthetic import trading_days


def test_empty_response_is_not_treated_as_cached():
    if not _has_parquet_engine():
        print("skipping parquet cache test: no parquet engine installed")
        return

    index = trading_days(start="2015-01-01", periods=20)
    start, end = str(index[0].date()), str(index[-1].date())
    good = normalize_ohlcv(_provider_frame(index))
    attempts: list[list[str]] = []

    def flaky_batch(tickers, *args, **kwargs):
        attempts.append(list(tickers))
        return {} if len(attempts) == 1 else {"AAA": good}

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch("diffolio.data.download._download_batch", flaky_batch):
            first = download_ohlcv(["AAA"], start, end, cache_dir=tmp)
            assert first == {}
            assert not PriceCache(tmp).covers("AAA", start, end)

            # The outage is retried rather than served from a poisoned cache.
            second = download_ohlcv(["AAA"], start, end, cache_dir=tmp)

    assert len(attempts) == 2
    assert len(second["AAA"]) == 20
