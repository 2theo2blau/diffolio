"""A failed download must not be cached as a valid empty answer.

The outage has to be simulated (a stubbed provider), but the bars served once
it recovers are real cached bars.
"""

from __future__ import annotations

from unittest import mock

from diffolio.data.download import PriceCache, download_ohlcv
from test_download import END, START, requires_parquet


@requires_parquet
def test_empty_response_is_not_treated_as_cached(real_frames, tmp_path):
    good = real_frames(["AAPL"], START, END)["AAPL"]
    attempts: list[list[str]] = []

    def flaky_batch(tickers, *args, **kwargs):
        attempts.append(list(tickers))
        return {} if len(attempts) == 1 else {"AAPL": good}

    with mock.patch("diffolio.data.download._download_batch", flaky_batch):
        first = download_ohlcv(["AAPL"], START, END, cache_dir=tmp_path)
        assert first == {}
        assert not PriceCache(tmp_path).covers("AAPL", START, END)

        # The outage is retried rather than served from a poisoned cache.
        second = download_ohlcv(["AAPL"], START, END, cache_dir=tmp_path)

    assert len(attempts) == 2
    assert len(second["AAPL"]) == len(good)
