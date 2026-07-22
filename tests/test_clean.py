from __future__ import annotations

import numpy as np
import pandas as pd

from diffolio.config import DataConfig, UniverseConfig
from diffolio.data.clean import align_frames, build_calendar, detect_anomalies
from synthetic import drop_days, make_ohlcv, make_universe_frames, trading_days


def _configs(**data_kwargs):
    data = DataConfig(start="2015-01-01", end="2030-01-01", **data_kwargs)
    universe = UniverseConfig(max_missing_frac=0.05)
    return data, universe


def test_calendar_modes_differ_as_documented():
    index = trading_days(periods=100)
    frames = make_universe_frames(["A", "B"], index)
    frames["A"] = drop_days(frames["A"], [10])
    frames["B"] = drop_days(frames["B"], [20])
    benchmark = make_ohlcv(index, seed=99)
    kwargs = dict(start="2015-01-01", end="2030-01-01")

    union = build_calendar(frames, benchmark, "union", **kwargs)
    bench = build_calendar(frames, benchmark, "benchmark", **kwargs)
    intersection = build_calendar(frames, benchmark, "intersection", **kwargs)

    assert len(union) == 100
    assert len(bench) == 100
    assert len(intersection) == 98  # days 10 and 20 are missing for one asset each


def test_calendar_is_clipped_to_the_requested_range():
    index = trading_days(periods=100)
    frames = make_universe_frames(["A"], index)
    benchmark = make_ohlcv(index, seed=1)
    calendar = build_calendar(
        frames, benchmark, "benchmark", start=str(index[10].date()), end=str(index[80].date())
    )
    assert calendar[0] == index[10]
    assert calendar[-1] == index[80]


def test_detect_anomalies_flags_jumps_constants_and_bad_prices():
    index = trading_days(periods=60)
    clean = make_ohlcv(index, seed=0)
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


def test_align_forward_fills_prices_and_marks_them_unobserved():
    index = trading_days(periods=100)
    frames = make_universe_frames(["A", "B"], index)
    missing = [30, 31]
    last_close = frames["A"]["close"].iloc[29]
    frames["A"] = drop_days(frames["A"], missing)
    benchmark = make_ohlcv(index, seed=99)
    data, universe = _configs()

    aligned = align_frames(frames, benchmark, data, universe, start="2015-01-01", end="2030-01-01")

    assert len(aligned.calendar) == 100
    assert set(aligned.frames) == {"A", "B"}
    assert not aligned.frames["A"].isna().to_numpy().any()
    observed = aligned.observed["A"].to_numpy()
    assert observed.sum() == 98
    assert not observed[30] and not observed[31]
    assert aligned.frames["A"]["close"].iloc[30] == last_close
    # Volume is zeroed on a non-traded day rather than carried forward.
    assert aligned.frames["A"]["volume"].iloc[30] == 0.0
    assert abs(aligned.report.fill_fraction["A"] - 0.02) < 1e-9


def test_align_drops_sparse_and_late_listed_assets():
    index = trading_days(periods=200)
    frames = make_universe_frames(["FULL", "SPARSE", "LATE"], index)
    frames["SPARSE"] = drop_days(frames["SPARSE"], list(range(0, 200, 4)))
    frames["LATE"] = frames["LATE"].iloc[50:]
    benchmark = make_ohlcv(index, seed=7)
    data, universe = _configs()

    aligned = align_frames(frames, benchmark, data, universe, start="2015-01-01", end="2030-01-01")

    assert aligned.tickers == ["FULL"]
    assert aligned.report.dropped["SPARSE"] == "sparse"
    assert aligned.report.dropped["LATE"] in {"sparse", "leading_gap"}


def test_align_drops_anomalous_assets_when_configured():
    index = trading_days(periods=120)
    frames = make_universe_frames(["OK", "BROKEN"], index)
    frames["BROKEN"].iloc[60, frames["BROKEN"].columns.get_loc("close")] *= 100.0
    benchmark = make_ohlcv(index, seed=3)
    data, universe = _configs()

    aligned = align_frames(frames, benchmark, data, universe, start="2015-01-01", end="2030-01-01")
    assert aligned.tickers == ["OK"]
    assert aligned.report.dropped["BROKEN"] == "anomalous"

    data_keep = DataConfig(start="2015-01-01", end="2030-01-01", drop_anomalous_assets=False)
    kept = align_frames(frames, benchmark, data_keep, universe, start="2015-01-01", end="2030-01-01")
    assert sorted(kept.tickers) == ["BROKEN", "OK"]
    assert bool(kept.report.anomalies.loc["BROKEN", "is_anomalous"]) is True


def test_benchmark_is_aligned_onto_the_master_calendar():
    index = trading_days(periods=80)
    frames = make_universe_frames(["A"], index)
    benchmark = drop_days(make_ohlcv(index, seed=5), [40])
    data, universe = _configs(calendar_mode="union")

    aligned = align_frames(frames, benchmark, data, universe, start="2015-01-01", end="2030-01-01")

    assert len(aligned.benchmark_frame) == len(aligned.calendar) == 80
    assert not aligned.benchmark_observed.to_numpy()[40]
    assert np.isfinite(aligned.benchmark_frame.to_numpy()).all()
