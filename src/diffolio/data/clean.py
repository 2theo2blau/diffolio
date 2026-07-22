"""Section 3 (steps 1-3) - calendar construction, alignment and sanity checks.

The output is a set of per-asset frames that all share one master trading
calendar, together with an ``observed`` mask marking which cells are genuine
observations rather than forward fills.  That mask is what stops a
forward-filled price from ever being used as a return target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import DataConfig, UniverseConfig
from ..utils import get_logger

logger = get_logger(__name__)

#: In ``union`` calendar mode, days on which fewer than this fraction of assets
#: traded are treated as non-trading days (half-days, provider glitches).
MIN_ASSET_COVERAGE_PER_DAY = 0.5


@dataclass
class CleaningReport:
    n_input_assets: int = 0
    n_output_assets: int = 0
    n_calendar_days: int = 0
    calendar_start: str = ""
    calendar_end: str = ""
    dropped: dict[str, str] = field(default_factory=dict)
    fill_fraction: dict[str, float] = field(default_factory=dict)
    anomalies: pd.DataFrame | None = None

    def drop_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.dropped.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input_assets": self.n_input_assets,
            "n_output_assets": self.n_output_assets,
            "n_calendar_days": self.n_calendar_days,
            "calendar_start": self.calendar_start,
            "calendar_end": self.calendar_end,
            "drop_counts": self.drop_counts(),
            "dropped": dict(self.dropped),
            "max_fill_fraction": max(self.fill_fraction.values(), default=0.0),
            "mean_fill_fraction": (
                float(np.mean(list(self.fill_fraction.values()))) if self.fill_fraction else 0.0
            ),
        }


@dataclass
class AlignedFrames:
    """Every asset reindexed onto one calendar, plus observation masks."""

    calendar: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]
    observed: dict[str, pd.Series]
    benchmark_frame: pd.DataFrame
    benchmark_observed: pd.Series
    report: CleaningReport

    @property
    def tickers(self) -> list[str]:
        return list(self.frames)


def build_calendar(
    frames: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    mode: str,
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    """Derive the master trading calendar (plan 3.1)."""
    if not frames:
        raise ValueError("cannot build a calendar from zero assets")

    window = (pd.Timestamp(start), pd.Timestamp(end))
    indices = [f.index for f in frames.values() if len(f)]
    union = reduce(lambda a, b: a.union(b), indices)
    union = union[(union >= window[0]) & (union <= window[1])]

    if mode == "union":
        counts = pd.Series(0, index=union, dtype="int64")
        for index in indices:
            counts = counts.add(pd.Series(1, index=index).reindex(union, fill_value=0))
        calendar = union[counts >= MIN_ASSET_COVERAGE_PER_DAY * len(indices)]
    elif mode in {"benchmark", "intersection"}:
        if benchmark is None or not len(benchmark):
            logger.warning("no benchmark frame available; falling back to the union calendar")
            calendar = union
        else:
            bench = benchmark.index[
                (benchmark.index >= window[0]) & (benchmark.index <= window[1])
            ]
            calendar = union.intersection(bench)
        if mode == "intersection":
            calendar = reduce(lambda a, b: a.intersection(b), indices, calendar)
    else:
        raise ValueError(f"unknown calendar mode {mode!r}")

    calendar = pd.DatetimeIndex(sorted(calendar), name="date")
    if len(calendar) < 2:
        raise RuntimeError(f"master calendar has {len(calendar)} day(s); check the date range")
    logger.info(
        "master calendar (%s): %d days, %s .. %s",
        mode,
        len(calendar),
        calendar[0].date(),
        calendar[-1].date(),
    )
    return calendar


def detect_anomalies(
    frame: pd.DataFrame,
    max_daily_jump: float = 10.0,
    max_constant_run: int = 20,
) -> dict[str, Any]:
    """Flag zero/negative prices, unexplained jumps and constant stretches."""
    price_cols = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    prices = frame[price_cols].astype("float64")
    close = prices["close"] if "close" in prices else prices.iloc[:, -1]

    n_nonpositive = int((prices <= 0).to_numpy().sum())
    n_nan = int(prices.isna().to_numpy().sum())

    ratio = (close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    n_jumps = int(((ratio > max_daily_jump) | (ratio < 1.0 / max_daily_jump)).sum())

    longest_run = _longest_constant_run(close.to_numpy())

    inconsistent = 0
    if {"high", "low"} <= set(prices.columns):
        inconsistent = int((prices["high"] < prices["low"]).sum())

    flags = {
        "n_nonpositive": n_nonpositive,
        "n_nan_prices": n_nan,
        "n_jumps": n_jumps,
        "longest_constant_run": longest_run,
        "n_high_below_low": inconsistent,
    }
    flags["is_anomalous"] = bool(
        n_nonpositive or n_jumps or inconsistent or longest_run > max_constant_run
    )
    return flags


def _longest_constant_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    finite = np.isfinite(values)
    same = np.zeros(values.size, dtype=bool)
    same[1:] = (values[1:] == values[:-1]) & finite[1:] & finite[:-1]
    longest = current = 1
    for flag in same[1:]:
        current = current + 1 if flag else 1
        longest = max(longest, current)
    return int(longest)


def align_frames(
    frames: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    data_config: DataConfig,
    universe_config: UniverseConfig,
    start: str,
    end: str,
) -> AlignedFrames:
    """Run the full cleaning pass: anomaly screen -> calendar -> reindex/fill."""
    report = CleaningReport(n_input_assets=len(frames))
    dropped: dict[str, str] = {}

    clipped = {
        ticker: frame.loc[str(start) : str(end)].sort_index()
        for ticker, frame in frames.items()
    }
    for ticker, frame in list(clipped.items()):
        if not len(frame):
            dropped[ticker] = "no_data_in_range"
            clipped.pop(ticker)

    anomaly_rows = {
        ticker: detect_anomalies(frame, data_config.max_daily_jump, data_config.max_constant_run)
        for ticker, frame in clipped.items()
    }
    report.anomalies = pd.DataFrame.from_dict(anomaly_rows, orient="index")
    if data_config.drop_anomalous_assets and len(report.anomalies):
        for ticker in report.anomalies.index[report.anomalies["is_anomalous"]]:
            dropped[ticker] = "anomalous"
            clipped.pop(ticker, None)
    if not clipped:
        raise RuntimeError(f"every asset was dropped during cleaning ({_counts(dropped)})")

    calendar = build_calendar(
        clipped, benchmark, data_config.calendar_mode, start=start, end=end
    )

    # Coverage screen against the master calendar (plan 3.2).
    for ticker, frame in list(clipped.items()):
        observed_days = frame.index.intersection(calendar)
        missing_frac = 1.0 - len(observed_days) / len(calendar)
        if missing_frac > universe_config.max_missing_frac:
            dropped[ticker] = "sparse"
            clipped.pop(ticker)
            continue
        if len(observed_days) and observed_days[0] > calendar[0]:
            # Back-filling a leading gap would inject future information, and a
            # partially listed asset breaks the fixed-N assumption anyway.
            dropped[ticker] = "leading_gap"
            clipped.pop(ticker)

    if not clipped:
        raise RuntimeError(f"every asset was dropped during cleaning ({_counts(dropped)})")

    aligned: dict[str, pd.DataFrame] = {}
    observed: dict[str, pd.Series] = {}
    for ticker, frame in clipped.items():
        filled, mask = _reindex_and_fill(frame, calendar)
        aligned[ticker] = filled
        observed[ticker] = mask
        report.fill_fraction[ticker] = float(1.0 - mask.mean())

    bench_filled, bench_mask = _reindex_and_fill(benchmark, calendar)
    if bench_filled.isna().to_numpy().any():
        raise RuntimeError("benchmark has gaps at the start of the master calendar")

    report.dropped = dropped
    report.n_output_assets = len(aligned)
    report.n_calendar_days = len(calendar)
    report.calendar_start = str(calendar[0].date())
    report.calendar_end = str(calendar[-1].date())
    logger.info(
        "aligned %d/%d assets on %d days (dropped: %s)",
        report.n_output_assets,
        report.n_input_assets,
        len(calendar),
        _counts(dropped) or "none",
    )

    return AlignedFrames(
        calendar=calendar,
        frames=aligned,
        observed=observed,
        benchmark_frame=bench_filled,
        benchmark_observed=bench_mask,
        report=report,
    )


def _reindex_and_fill(
    frame: pd.DataFrame, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.Series]:
    """Reindex onto the calendar, forward-fill prices, zero volume on fills."""
    reindexed = frame.reindex(calendar)
    observed = pd.Series(
        reindexed["close"].notna().to_numpy() & calendar.isin(frame.index),
        index=calendar,
        name="observed",
    )

    price_cols = [c for c in reindexed.columns if c != "volume"]
    reindexed[price_cols] = reindexed[price_cols].ffill()
    if "volume" in reindexed.columns:
        volume = reindexed["volume"].to_numpy(dtype="float64")
        volume[~observed.to_numpy()] = 0.0
        reindexed["volume"] = np.nan_to_num(volume, nan=0.0)
    return reindexed, observed


def _counts(dropped: Mapping[str, str]) -> str:
    counts: dict[str, int] = {}
    for reason in dropped.values():
        counts[reason] = counts.get(reason, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
