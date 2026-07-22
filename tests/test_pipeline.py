from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

from diffolio.config import DiffolioConfig
from diffolio.data.pipeline import build_dataset, load_dataset
from synthetic import drop_days, make_ohlcv, make_universe_frames, trading_days

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N_DAYS = 400
CALENDAR = trading_days(periods=N_DAYS)


def _config(tmp: str) -> DiffolioConfig:
    config = DiffolioConfig()
    config.name = "synthetic"
    config.universe.roster = "list:" + ",".join(TICKERS)
    config.universe.benchmark = "^TEST"
    config.universe.target_size = None
    config.universe.min_avg_dollar_volume = 0.0
    config.data.start = str(CALENDAR[0].date())
    config.data.end = str(CALENDAR[-1].date())
    config.data.cache_dir = str(Path(tmp) / "cache")
    config.features.ref_window = 5
    config.window.lookback = 10
    return config


def _fake_provider(gap_in: str | None = None):
    """Patch the two download entry points with synthetic frames."""
    index = CALENDAR
    frames = make_universe_frames(TICKERS, index)
    if gap_in:
        frames[gap_in] = drop_days(frames[gap_in], [200, 201, 202])
    benchmark = make_ohlcv(index, seed=123, price0=2000.0)

    def download_ohlcv(tickers, *args, **kwargs):
        return {t: frames[t] for t in tickers if t in frames}

    def download_benchmark(symbol, *args, **kwargs):
        return benchmark

    stack = ExitStack()
    stack.enter_context(mock.patch("diffolio.data.pipeline.download_ohlcv", download_ohlcv))
    stack.enter_context(
        mock.patch("diffolio.data.pipeline.download_benchmark", download_benchmark)
    )
    return stack, index


def test_end_to_end_build_produces_a_consistent_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        stack, index = _fake_provider(gap_in="CCC")
        with stack:
            config = _config(tmp)
            out = Path(tmp) / "out"
            dataset = build_dataset(config, output_dir=out)

        panel, splits = dataset.panel, dataset.splits

        assert panel.n_assets == len(TICKERS)
        assert panel.n_features == 5
        assert panel.tickers == tuple(sorted(TICKERS))
        # Four warm-up rows are consumed by the 5-day price reference.
        assert panel.n_days == N_DAYS - 4
        assert panel.calendar[0] == index[4]
        assert np.isfinite(panel.features).all()
        assert np.isfinite(panel.returns).all()

        # The one asset with missing bars is forward-filled and flagged.
        ccc = panel.asset_index("CCC")
        assert panel.observed[:, ccc].sum() == panel.n_days - 3
        assert panel.observed.sum(axis=0).max() == panel.n_days

        train_end = splits.train.stop
        assert splits.train.start == 0 and splits.val.start == train_end
        assert splits.test.stop == panel.n_days
        for _, split in splits.items():
            assert split.tau.min() >= split.start + config.lookback - 1
            assert split.tau.max() <= split.stop - 2

        # Standardisation statistics come from the training split only, so the
        # training block is centred while the later splits are not re-centred.
        train_block = panel.features[:train_end]
        price_features = slice(0, 4)
        assert np.allclose(train_block[..., price_features].mean(axis=(0, 1)), 0.0, atol=1e-5)
        assert np.allclose(train_block[..., price_features].std(axis=(0, 1)), 1.0, atol=1e-3)
        # Zero volume on a forward-filled bar is a genuine outlier; clip_sigma
        # bounds it rather than letting it dominate the feature's scale.
        assert float(np.abs(panel.features).max()) <= config.features.clip_sigma + 1e-6

        h, g, r = panel.window(int(splits.test.tau[0]), config.lookback)
        assert h.shape == (panel.n_assets, config.lookback, panel.n_features)
        assert g.shape == (config.lookback, panel.n_features)
        assert r.shape == (panel.n_assets,)


def test_artifacts_are_written_and_reloadable():
    with tempfile.TemporaryDirectory() as tmp:
        stack, _ = _fake_provider()
        with stack:
            config = _config(tmp)
            out = Path(tmp) / "out"
            dataset = build_dataset(config, output_dir=out)

        for relative in (
            "panel/meta.json",
            "panel/arrays/features.npy",
            "splits.json",
            "universe.json",
            "build_report.json",
            "config.yaml",
            "diagnostics/screen.csv",
        ):
            assert (out / relative).exists(), relative

        reloaded = load_dataset(out)
        np.testing.assert_array_equal(reloaded.panel.features, dataset.panel.features)
        np.testing.assert_array_equal(reloaded.splits.train.tau, dataset.splits.train.tau)
        assert reloaded.universe.tickers == dataset.universe.tickers
        assert reloaded.report["panel"]["N"] == dataset.panel.n_assets


def test_second_build_hits_the_cache_and_config_changes_invalidate_it():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        stack, _ = _fake_provider()
        with stack:
            config = _config(tmp)
            build_dataset(config, output_dir=out)

        # No provider patched this time: a cache hit must not need the network.
        cached = build_dataset(_config(tmp), output_dir=out)
        assert cached.panel.n_assets == len(TICKERS)

        changed = _config(tmp)
        changed.window.lookback = 20
        stack, _ = _fake_provider()
        with stack:
            rebuilt = build_dataset(changed, output_dir=out)
        assert rebuilt.splits.lookback == 20
        assert rebuilt.panel.metadata["fingerprint"] != cached.panel.metadata["fingerprint"]


def test_build_reports_when_the_universe_collapses():
    with tempfile.TemporaryDirectory() as tmp:
        stack, _ = _fake_provider()
        with stack:
            config = _config(tmp)
            config.universe.min_avg_dollar_volume = 1e15
            try:
                build_dataset(config, output_dir=Path(tmp) / "out")
            except RuntimeError as exc:
                assert "eligibility screen" in str(exc)
            else:
                raise AssertionError("expected an impossible screen to raise")
