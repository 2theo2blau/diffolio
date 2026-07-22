"""Deterministic synthetic market data so the tests never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(start: str = "2015-01-01", periods: int = 300) -> pd.DatetimeIndex:
    """Business days with no ``freq`` attached, like a real exchange calendar."""
    days = pd.bdate_range(start=start, periods=periods).to_numpy()
    return pd.DatetimeIndex(days, name="date")


def make_ohlcv(
    index: pd.DatetimeIndex,
    seed: int = 0,
    price0: float = 100.0,
    drift: float = 0.0002,
    vol: float = 0.015,
    volume: float = 2.0e6,
) -> pd.DataFrame:
    """A geometric-random-walk OHLCV frame with internally consistent bars."""
    rng = np.random.default_rng(seed)
    n = len(index)
    close = price0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * np.exp(rng.normal(0.0, vol / 3.0, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, vol / 4.0, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, vol / 4.0, n)))
    volumes = rng.lognormal(np.log(volume), 0.25, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volumes},
        index=index,
    )


def make_universe_frames(
    tickers: list[str],
    index: pd.DatetimeIndex,
    price0: float = 100.0,
    volume: float = 2.0e6,
) -> dict[str, pd.DataFrame]:
    return {
        ticker: make_ohlcv(index, seed=i, price0=price0, volume=volume)
        for i, ticker in enumerate(tickers)
    }


def drop_days(frame: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Remove specific rows to simulate missing bars."""
    keep = [i for i in range(len(frame)) if i not in set(positions)]
    return frame.iloc[keep]
