"""Section 2 - raw price data acquisition via yfinance.

Every request is split- and dividend-adjusted (``auto_adjust=True``), which
back-adjusts *open* as well as close.  That matters here because returns in
this pipeline are open-to-open, so an unadjusted open would manufacture a fake
overnight gap on every ex-dividend and split date.

Raw responses are cached to Parquet before any transformation so that re-runs
never re-hit the API.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..utils import get_logger, read_json, write_json

logger = get_logger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_MANIFEST_NAME = "manifest.json"
_SUBDIR = "ohlcv"


class PriceCache:
    """Parquet-backed cache of raw daily bars, one file per ticker."""

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir) / _SUBDIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / _MANIFEST_NAME
        self._manifest: dict[str, dict] = (
            read_json(self._manifest_path) if self._manifest_path.exists() else {}
        )

    def path_for(self, ticker: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
        return self.root / f"{safe}.parquet"

    def covers(self, ticker: str, start: str, end: str) -> bool:
        entry = self._manifest.get(ticker)
        if entry is None or entry.get("empty") or not self.path_for(ticker).exists():
            return False
        return entry["requested_start"] <= start and entry["requested_end"] >= end

    def read(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path_for(ticker))
        return frame.loc[str(start) : str(end)]

    def write(self, ticker: str, frame: pd.DataFrame, start: str, end: str) -> None:
        """Cache a response.

        An empty frame is recorded but never written: a provider outage and a
        genuinely dataless ticker look identical here, and caching the empty
        answer would poison every later run with a silent "no data".
        """
        entry = {
            "requested_start": str(start),
            "requested_end": str(end),
            "rows": int(len(frame)),
            "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        if len(frame):
            frame.to_parquet(self.path_for(ticker))
            entry["first_date"] = str(frame.index[0].date())
            entry["last_date"] = str(frame.index[-1].date())
        else:
            entry["empty"] = True
        self._manifest[ticker] = entry

    def flush(self) -> None:
        write_json(self._manifest_path, self._manifest)


def download_ohlcv(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    force: bool = False,
    batch_size: int = 50,
    max_retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Fetch adjusted daily bars for ``tickers``, using the on-disk cache.

    Returns a mapping ticker -> frame indexed by trading date with columns
    ``[open, high, low, close, volume]``.  Tickers that returned no usable data
    are simply absent from the result (and logged).
    """
    cache = PriceCache(cache_dir)
    frames: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        if not force and cache.covers(ticker, start, end):
            frames[ticker] = cache.read(ticker, start, end)
        else:
            to_fetch.append(ticker)

    if frames:
        logger.info("loaded %d/%d tickers from cache", len(frames), len(tickers))

    for batch in _chunks(to_fetch, batch_size):
        fetched = _download_batch(
            batch, start, end, max_retries=max_retries, cache_dir=cache_dir
        )
        for ticker in batch:
            frame = fetched.get(ticker, _empty_frame())
            cache.write(ticker, frame, start, end)
            if len(frame):
                frames[ticker] = frame
        cache.flush()

    cache.flush()
    missing = [t for t in tickers if t not in frames or not len(frames[t])]
    if missing:
        logger.warning("no data returned for %d tickers: %s", len(missing), _preview(missing))
    return {t: f for t, f in frames.items() if len(f)}


def download_benchmark(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    force: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch the benchmark index over the same range and provider."""
    frames = download_ohlcv(
        [symbol],
        start=start,
        end=end,
        cache_dir=cache_dir,
        force=force,
        batch_size=1,
        max_retries=max_retries,
    )
    if symbol not in frames:
        raise RuntimeError(f"benchmark {symbol} returned no data for {start}..{end}")
    return frames[symbol]


def _download_batch(
    tickers: Sequence[str],
    start: str,
    end: str,
    max_retries: int,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    import yfinance as yf  # imported lazily so tests need no network stack

    if cache_dir is not None:
        # yfinance keeps a timezone SQLite cache under $HOME by default, which
        # fails outright on read-only home directories (containers, CI).
        yf.set_tz_cache_location(str(Path(cache_dir) / "yfinance_tz"))

    # yfinance treats ``end`` as exclusive.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "downloading %d tickers (%s) attempt %d/%d",
                len(tickers),
                _preview(tickers),
                attempt,
                max_retries,
            )
            raw = yf.download(
                tickers=list(tickers),
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            return _split_response(raw, tickers)
        except Exception as exc:  # network/parse failures are transient
            last_error = exc
            wait = 2.0**attempt
            logger.warning("download failed (%s); retrying in %.0fs", exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"failed to download batch {_preview(tickers)}") from last_error


def _split_response(raw: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Turn a (possibly MultiIndex-columned) yfinance response into per-ticker frames."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        level = 0 if set(tickers) & set(raw.columns.get_level_values(0)) else 1
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(level):
                continue
            frame = raw.xs(ticker, axis=1, level=level)
            frame = normalize_ohlcv(frame)
            if len(frame):
                out[ticker] = frame
    else:
        if len(tickers) != 1:
            raise RuntimeError("flat response received for a multi-ticker request")
        frame = normalize_ohlcv(raw)
        if len(frame):
            out[tickers[0]] = frame
    return out


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider frame to the canonical schema and dtypes."""
    frame = frame.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in frame.columns})
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"response is missing columns {missing}; got {list(frame.columns)}")

    frame = frame.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    frame.index = index.normalize()
    frame.index.name = "date"

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    # A bar with no close carries no information; volume-only rows are noise.
    return frame.dropna(subset=["open", "high", "low", "close"], how="all")


def _empty_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="date")
    return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=index, dtype="float64")


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    size = max(int(size), 1)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _preview(items: Sequence[str], limit: int = 5) -> str:
    head = ", ".join(items[:limit])
    return head if len(items) <= limit else f"{head}, ... (+{len(items) - limit})"


def restrict_to_window(
    frames: Mapping[str, pd.DataFrame], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Clip every frame to ``[start, end]`` inclusive."""
    return {t: f.loc[str(start) : str(end)] for t, f in frames.items()}
