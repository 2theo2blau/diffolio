"""Section 3 cleaning tests, on real cached yfinance bars.

The real S&P data is clean, so gaps and anomalies are injected into real
frames where a test needs one; late listing uses a genuine late listing
(KMI, which IPO'd on 2011-02-11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from diffolio.config import DataConfig, UniverseConfig
from diffolio.data.clean import align_frames, build_calendar, detect_anomalies

START, END = "2018-01-01", "2018-12-31"


def _configs(start=START, end=END, **data_kwargs):
    data = DataConfig(start=start, end=end, **data_kwargs)
    universe = UniverseConfig(max_missing_frac=0.05)
    return data, universe


def _drop(frame: pd.DataFrame, positions) -> pd.DataFrame:
    """Remove rows to simulate missing bars (an injected gap)."""
    return frame.drop(frame.index[list(positions)])


def test_calendar_modes_differ_as_documented(real_frames):
    frames = real_frames(["AAPL", "MSFT", "^GSPC"], START, END)
    benchmark = frames.pop("^GSPC")
    n = len(benchmark)
    frames["AAPL"] = _drop(frames["AAPL"], [10])
    frames["MSFT"] = _drop(frames["MSFT"], [20])

    union = build_calendar(frames, benchmark, "union", start=START, end=END)
    bench = build_calendar(frames, benchmark, "benchmark", start=START, end=END)
    intersection = build_calendar(frames, benchmark, "intersection", start=START, end=END)

    assert n == 251  # NYSE trading days in 2018
    assert len(union) == n
    assert bench.equals(pd.DatetimeIndex(benchmark.index, name=bench.name))
    assert len(intersection) == n - 2  # days 10 and 20 each miss one asset


def test_calendar_is_clipped_to_the_requested_range(real_frames):
    frames = real_frames(["AAPL", "^GSPC"], START, END)
    benchmark = frames.pop("^GSPC")
    index = benchmark.index
    calendar = build_calendar(
        frames, benchmark, "benchmark", start=str(index[10].date()), end=str(index[80].date())
    )
    assert calendar[0] == index[10]
    assert calendar[-1] == index[80]


def test_real_bars_pass_the_anomaly_screen(real_frames):
    for ticker, frame in real_frames(["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"]).items():
        flags = detect_anomalies(frame)
        assert flags["is_anomalous"] is False, ticker
        assert flags["n_nonpositive"] == 0 and flags["n_jumps"] == 0


def test_detect_anomalies_flags_jumps_constants_and_bad_prices(real_frames):
    clean = real_frames(["AAPL"], START, END)["AAPL"].iloc[:60]
    assert detect_anomalies(clean)["is_anomalous"] is False

    jumped = clean.copy()
    jumped.iloc[30, jumped.columns.get_loc("close")] *= 50.0
    flags = detect_anomalies(jumped)
    assert flags["n_jumps"] >= 1 and flags["is_anomalous"] is True

    flat = clean.copy()
    flat.iloc[10:40, flat.columns.get_loc("close")] = 42.0
    assert detect_anomalies(flat, max_constant_run=20)["longest_constant_run"] == 30

    negative = clean.copy()
    negative.iloc[5, negative.columns.get_loc("low")] = -1.0
    assert detect_anomalies(negative)["n_nonpositive"] == 1


def test_align_forward_fills_prices_and_marks_them_unobserved(real_frames):
    frames = real_frames(["AAPL", "MSFT", "^GSPC"], START, END)
    benchmark = frames.pop("^GSPC")
    n = len(benchmark)
    last_close = frames["AAPL"]["close"].iloc[29]
    frames["AAPL"] = _drop(frames["AAPL"], [30, 31])
    data, universe = _configs()

    aligned = align_frames(frames, benchmark, data, universe, start=START, end=END)

    assert len(aligned.calendar) == n
    assert set(aligned.frames) == {"AAPL", "MSFT"}
    assert not aligned.frames["AAPL"].isna().to_numpy().any()
    observed = aligned.observed["AAPL"].to_numpy()
    assert observed.sum() == n - 2
    assert not observed[30] and not observed[31]
    assert aligned.frames["AAPL"]["close"].iloc[30] == last_close
    # Volume is zeroed on a non-traded day rather than carried forward.
    assert aligned.frames["AAPL"]["volume"].iloc[30] == 0.0
    assert abs(aligned.report.fill_fraction["AAPL"] - 2 / n) < 1e-9
    assert aligned.observed["MSFT"].all()


def test_align_drops_sparse_and_late_listed_assets(real_frames):
    start, end = "2010-06-01", "2011-06-30"
    frames = real_frames(["AAPL", "MSFT", "KMI", "^GSPC"], start, end)
    benchmark = frames.pop("^GSPC")
    assert frames["KMI"].index[0] == pd.Timestamp("2011-02-11")  # genuinely late
    frames["MSFT"] = _drop(frames["MSFT"], range(0, len(frames["MSFT"]), 4))  # injected
    data, universe = _configs(start, end)

    aligned = align_frames(frames, benchmark, data, universe, start=start, end=end)

    assert aligned.tickers == ["AAPL"]
    assert aligned.report.dropped["MSFT"] == "sparse"
    assert aligned.report.dropped["KMI"] in {"sparse", "leading_gap"}


def test_align_drops_anomalous_assets_when_configured(real_frames):
    frames = real_frames(["AAPL", "MSFT", "^GSPC"], START, END)
    benchmark = frames.pop("^GSPC")
    frames["MSFT"].iloc[60, frames["MSFT"].columns.get_loc("close")] *= 100.0  # injected
    data, universe = _configs()

    aligned = align_frames(frames, benchmark, data, universe, start=START, end=END)
    assert aligned.tickers == ["AAPL"]
    assert aligned.report.dropped["MSFT"] == "anomalous"

    data_keep = DataConfig(start=START, end=END, drop_anomalous_assets=False)
    kept = align_frames(frames, benchmark, data_keep, universe, start=START, end=END)
    assert sorted(kept.tickers) == ["AAPL", "MSFT"]
    assert bool(kept.report.anomalies.loc["MSFT", "is_anomalous"]) is True


def test_benchmark_is_aligned_onto_the_master_calendar(real_frames):
    frames = real_frames(["AAPL", "^GSPC"], START, END)
    benchmark = _drop(frames.pop("^GSPC"), [40])  # injected index gap
    data, universe = _configs(calendar_mode="union")

    aligned = align_frames(frames, benchmark, data, universe, start=START, end=END)

    assert len(aligned.benchmark_frame) == len(aligned.calendar) == 251
    assert not aligned.benchmark_observed.to_numpy()[40]
    assert np.isfinite(aligned.benchmark_frame.to_numpy()).all()
