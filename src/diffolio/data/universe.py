"""Section 1 - universe selection.

Universe selection is inherently two-phase: the eligibility criteria in the
plan (liquidity, penny-stock exclusion, continuous listing) can only be
evaluated once prices are in hand.  So the pipeline pulls a *candidate roster*
from a reference source, downloads it (section 2), and then screens it down to
the final ordered list of N tickers.  That ordered list is the canonical column
order for every tensor downstream.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from ..config import UniverseConfig
from ..utils import get_logger, read_json, write_json

logger = get_logger(__name__)

#: How many calendar days of slack to allow when checking that an asset was
#: listed for the entire study period (holidays, first-day-of-year effects).
LISTING_GRACE_DAYS = 7

_WIKI_ROSTERS: dict[str, tuple[str, str]] = {
    # name -> (wikipedia url, column holding the ticker)
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    "dow30": ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "Symbol"),
}


@dataclass(frozen=True)
class Universe:
    """An ordered, frozen set of tickers plus the criteria that produced it."""

    market: str
    benchmark: str
    tickers: tuple[str, ...]
    start: str
    end: str
    criteria: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("universe contains duplicate tickers")
        if not self.tickers:
            raise ValueError("universe is empty")

    def __len__(self) -> int:
        return len(self.tickers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tickers)

    def __contains__(self, ticker: object) -> bool:
        return ticker in self.tickers

    @property
    def size(self) -> int:
        """N."""
        return len(self.tickers)

    def index(self, ticker: str) -> int:
        """Canonical asset index of ``ticker``."""
        return self.tickers.index(ticker)

    @property
    def index_map(self) -> dict[str, int]:
        return {ticker: i for i, ticker in enumerate(self.tickers)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "benchmark": self.benchmark,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "criteria": dict(self.criteria),
            "created_at": self.created_at,
            "tickers": list(self.tickers),
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())
        logger.info("wrote universe of %d tickers to %s", self.size, path)

    @classmethod
    def load(cls, path: str | Path) -> "Universe":
        payload = read_json(path)
        return cls(
            market=payload["market"],
            benchmark=payload["benchmark"],
            tickers=tuple(payload["tickers"]),
            start=payload["start"],
            end=payload["end"],
            criteria=payload.get("criteria", {}),
            created_at=payload.get("created_at", ""),
        )


def normalize_ticker(ticker: str) -> str:
    """Map a reference-source symbol onto Yahoo Finance's convention.

    Yahoo writes share classes with a hyphen (``BRK-B``) where index providers
    use a dot (``BRK.B``).  Suffixed foreign listings (``005930.KS``) keep
    their dot, so only a single trailing letter after a dot is rewritten.
    """
    ticker = ticker.strip().upper()
    return re.sub(r"\.([A-Z])$", r"-\1", ticker)


def load_candidate_roster(
    spec: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> list[str]:
    """Resolve a roster spec into a list of candidate tickers.

    Supported specs: ``sp500`` / ``nasdaq100`` / ``dow30`` (scraped from
    Wikipedia and cached), ``file:<path>`` (CSV with a ticker column, or a
    newline-delimited text file) and ``list:AAPL,MSFT``.
    """
    spec = spec.strip()
    if spec.startswith("list:"):
        tickers = [t for t in re.split(r"[,\s]+", spec[len("list:") :]) if t]
    elif spec.startswith("file:"):
        tickers = _roster_from_file(spec[len("file:") :])
    elif spec in _WIKI_ROSTERS:
        tickers = _roster_from_wikipedia(spec, cache_dir=cache_dir, force=force)
    else:
        raise ValueError(
            f"unknown roster spec {spec!r}; expected one of "
            f"{sorted(_WIKI_ROSTERS)} or 'file:<path>' / 'list:A,B'"
        )

    seen: dict[str, None] = {}
    for ticker in tickers:
        seen.setdefault(normalize_ticker(ticker), None)
    resolved = sorted(seen)
    logger.info("roster %s resolved to %d candidates", spec, len(resolved))
    return resolved


def _roster_from_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"roster file {path} does not exist")
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        column = _find_ticker_column(frame)
        return frame[column].dropna().astype(str).tolist()
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("tickers", [])
        return [str(t) for t in payload]
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


def _find_ticker_column(frame: pd.DataFrame) -> str:
    for candidate in ("ticker", "symbol", "Ticker", "Symbol", "TICKER", "SYMBOL"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def _roster_from_wikipedia(
    name: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> list[str]:
    url, column = _WIKI_ROSTERS[name]
    cache_path = Path(cache_dir) / "rosters" / f"{name}.csv" if cache_dir else None

    if cache_path is not None and cache_path.exists() and not force:
        logger.info("using cached roster %s", cache_path)
        return pd.read_csv(cache_path)["ticker"].astype(str).tolist()

    logger.info("fetching roster %s from %s", name, url)
    tables = pd.read_html(url)
    for table in tables:
        if column in table.columns:
            tickers = table[column].dropna().astype(str).tolist()
            break
    else:
        raise RuntimeError(f"no table with a {column!r} column found at {url}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers


@dataclass
class ScreenResult:
    """Outcome of the eligibility screen, kept for auditability."""

    eligible: list[str]
    report: pd.DataFrame

    def rejected(self) -> pd.DataFrame:
        return self.report[self.report["status"] != "eligible"]

    def reason_counts(self) -> dict[str, int]:
        return self.report["status"].value_counts().to_dict()


def screen_candidates(
    frames: Mapping[str, pd.DataFrame],
    config: UniverseConfig,
    start: str,
    end: str,
    reference_days: int | None = None,
) -> ScreenResult:
    """Apply the section-1 eligibility criteria to downloaded candidates.

    ``frames`` maps ticker -> OHLCV frame (already restricted to the study
    window).  ``reference_days`` is the number of trading days the master
    calendar is expected to have; when omitted the busiest candidate is used as
    the reference.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    grace = pd.Timedelta(days=LISTING_GRACE_DAYS)
    excluded = {normalize_ticker(t) for t in config.exclude}
    forced = {normalize_ticker(t) for t in config.include}

    if reference_days is None:
        reference_days = max((len(f) for f in frames.values()), default=0)
    reference_days = max(int(reference_days), 1)

    rows: list[dict[str, Any]] = []
    for ticker in sorted(frames):
        frame = frames[ticker]
        row: dict[str, Any] = {
            "ticker": ticker,
            "rows": len(frame),
            "first_date": frame.index[0] if len(frame) else pd.NaT,
            "last_date": frame.index[-1] if len(frame) else pd.NaT,
            "coverage": len(frame) / reference_days,
            "median_close": float("nan"),
            "min_close": float("nan"),
            "median_dollar_volume": float("nan"),
            "status": "eligible",
        }
        if len(frame):
            close = frame["close"].astype("float64")
            volume = frame["volume"].astype("float64")
            row["median_close"] = float(close.median())
            row["min_close"] = float(close.min())
            row["median_dollar_volume"] = float((close * volume).median())

        status = _screen_one(row, config, start_ts, end_ts, grace)
        if ticker in excluded:
            status = "excluded"
        elif ticker in forced and status != "no_data":
            status = "eligible"
        row["status"] = status
        rows.append(row)

    report = pd.DataFrame(rows).set_index("ticker")
    eligible = report.index[report["status"] == "eligible"].tolist()

    logger.info(
        "screened %d candidates -> %d eligible (%s)",
        len(report),
        len(eligible),
        ", ".join(f"{k}={v}" for k, v in sorted(report["status"].value_counts().items())),
    )
    return ScreenResult(eligible=eligible, report=report)


def _screen_one(
    row: Mapping[str, Any],
    config: UniverseConfig,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    grace: pd.Timedelta,
) -> str:
    if not row["rows"]:
        return "no_data"
    if config.require_full_history:
        if row["first_date"] > start_ts + grace:
            return "late_listing"
        if row["last_date"] < end_ts - grace:
            return "delisted"
    if row["coverage"] < 1.0 - config.max_missing_frac:
        return "sparse"
    if not row["min_close"] > 0:
        return "nonpositive_price"
    if row["median_close"] < config.min_price:
        return "penny_stock"
    if row["median_dollar_volume"] < config.min_avg_dollar_volume:
        return "illiquid"
    return "eligible"


def build_universe(
    screen: ScreenResult,
    config: UniverseConfig,
    start: str,
    end: str,
) -> Universe:
    """Rank the eligible tickers by liquidity and cut to ``target_size``."""
    eligible = list(screen.eligible)
    if not eligible:
        raise RuntimeError(
            "no candidate passed the eligibility screen; loosen universe criteria "
            f"(rejections: {screen.reason_counts()})"
        )

    if config.target_size is not None and len(eligible) > config.target_size:
        liquidity = screen.report.loc[eligible, "median_dollar_volume"]
        forced = [t for t in (normalize_ticker(x) for x in config.include) if t in eligible]
        ranked = [t for t in liquidity.sort_values(ascending=False).index if t not in forced]
        keep = forced + ranked[: max(config.target_size - len(forced), 0)]
        eligible = sorted(keep)
        logger.info("cut universe to target size N=%d by median dollar volume", len(eligible))
    else:
        eligible = sorted(eligible)

    criteria = {
        "roster": config.roster,
        "target_size": config.target_size,
        "min_price": config.min_price,
        "min_avg_dollar_volume": config.min_avg_dollar_volume,
        "max_missing_frac": config.max_missing_frac,
        "require_full_history": config.require_full_history,
        "candidates_screened": int(len(screen.report)),
        "rejections": screen.reason_counts(),
    }
    return Universe(
        market=config.market,
        benchmark=config.benchmark,
        tickers=tuple(eligible),
        start=start,
        end=end,
        criteria=criteria,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


def subset_universe(universe: Universe, keep: Sequence[str]) -> Universe:
    """Re-emit a universe restricted to ``keep``, preserving canonical order."""
    keep_set = set(keep)
    tickers = tuple(t for t in universe.tickers if t in keep_set)
    criteria = dict(universe.criteria)
    criteria["dropped_in_cleaning"] = sorted(set(universe.tickers) - keep_set)
    return Universe(
        market=universe.market,
        benchmark=universe.benchmark,
        tickers=tickers,
        start=universe.start,
        end=universe.end,
        criteria=criteria,
        created_at=universe.created_at,
    )
