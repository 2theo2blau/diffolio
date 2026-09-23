from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffolio.config import DiffolioConfig
from diffolio.data.pipeline import build_dataset, load_dataset
from diffolio.data.splits import make_splits
from diffolio.data.windows import (
    WindowDataset,
    WindowSample,
    stack_windows,
    window_dataset,
)
from synthetic import make_ohlcv, make_universe_frames, trading_days

from test_panel import make_panel

LOOKBACK = 10


def _panel_and_splits(n_days=300, n_assets=4, lookback=LOOKBACK):
    panel = make_panel(n_days=n_days, n_assets=n_assets)
    config = DiffolioConfig()
    config.window.lookback = lookback
    return panel, make_splits(panel, config)


def test_samples_match_the_panel_window_primitive():
    panel, splits = _panel_and_splits()
    ds = WindowDataset(panel, splits.train, LOOKBACK)

    assert len(ds) == splits.train.n_samples
    assert ds.tau.tolist() == splits.train.tau.tolist()

    for i in (0, len(ds) // 2, len(ds) - 1):
        sample = ds[i]
        assert isinstance(sample, WindowSample)
        tau = int(splits.train.tau[i])

        assert sample.h.shape == (panel.n_assets, LOOKBACK, panel.n_features)
        assert sample.g.shape == (LOOKBACK, panel.n_features)
        assert sample.r.shape == (panel.n_assets,)
        assert sample.r_valid.shape == (panel.n_assets,)
        assert sample.h.dtype == torch.float32
        assert sample.r_valid.dtype == torch.bool
        assert sample.tau.dtype == torch.int64
        assert int(sample.tau) == tau

        h_np, g_np, r_np = panel.window(tau, LOOKBACK)
        np.testing.assert_array_equal(sample.h.numpy(), h_np)
        np.testing.assert_array_equal(sample.g.numpy(), g_np)
        np.testing.assert_array_equal(sample.r.numpy(), r_np)
        np.testing.assert_array_equal(sample.r_valid.numpy(), panel.return_valid[tau])

        # Orientation: h's last time row is the decision day, the first one is
        # the oldest day of the window.  This catches a silent transpose.
        np.testing.assert_array_equal(sample.h[:, -1, :].numpy(), panel.features[tau])
        np.testing.assert_array_equal(
            sample.h[:, 0, :].numpy(), panel.features[tau - LOOKBACK + 1]
        )


def test_dataloader_collates_samples_into_batched_tensors():
    panel, splits = _panel_and_splits()
    ds = WindowDataset(panel, splits.train, LOOKBACK)
    loader = DataLoader(ds, batch_size=7, shuffle=False)

    batch = next(iter(loader))

    assert isinstance(batch, WindowSample)
    assert batch.h.shape == (7, panel.n_assets, LOOKBACK, panel.n_features)
    assert batch.g.shape == (7, LOOKBACK, panel.n_features)
    assert batch.r.shape == (7, panel.n_assets)
    assert batch.r_valid.shape == (7, panel.n_assets)
    assert batch.r_valid.dtype == torch.bool
    assert batch.tau.tolist() == splits.train.tau[:7].tolist()

    for row in range(7):
        torch.testing.assert_close(batch.h[row], ds[row].h)
        torch.testing.assert_close(batch.r[row], ds[row].r)
        torch.testing.assert_close(batch.r_valid[row], ds[row].r_valid)


def test_shuffled_loader_with_workers_covers_every_step_once():
    panel, splits = _panel_and_splits()
    ds = WindowDataset(panel, splits.train, LOOKBACK)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

    seen: list[int] = []
    for batch in loader:
        assert batch.h.is_contiguous()
        assert batch.h.shape[1:] == (panel.n_assets, LOOKBACK, panel.n_features)
        seen.extend(batch.tau.tolist())

    assert sorted(seen) == splits.train.tau.tolist()


def test_stack_windows_matches_itemwise_slicing():
    panel, splits = _panel_and_splits()
    ds = WindowDataset(panel, splits.train, LOOKBACK)
    taus = splits.train.tau[::5]

    batch = stack_windows(panel, taus, LOOKBACK)
    assert batch.h.shape == (len(taus), panel.n_assets, LOOKBACK, panel.n_features)
    assert batch.tau.tolist() == taus.tolist()

    positions = np.searchsorted(splits.train.tau, taus)
    for row, position in enumerate(positions):
        sample = ds[int(position)]
        torch.testing.assert_close(batch.h[row], sample.h)
        torch.testing.assert_close(batch.g[row], sample.g)
        torch.testing.assert_close(batch.r[row], sample.r)
        torch.testing.assert_close(batch.r_valid[row], sample.r_valid)
        assert int(batch.tau[row]) == int(sample.tau)

    # The dataset's own stack() gathers every step; a subset keeps order.
    full = ds.stack()
    assert full.tau.tolist() == splits.train.tau.tolist()
    torch.testing.assert_close(full.h, stack_windows(panel, splits.train, LOOKBACK).h)

    subset = ds.stack(indices=[3, 1, 0])
    assert subset.tau.tolist() == splits.train.tau[[3, 1, 0]].tolist()


def test_out_of_range_steps_are_rejected():
    panel, splits = _panel_and_splits()
    n_days = panel.n_days
    # A usable step needs the window [tau-L+1, tau] and the target leg tau+1.
    for bad in ([LOOKBACK - 2], [0], [n_days - 1], [LOOKBACK - 1, n_days - 1]):
        try:
            WindowDataset(panel, bad, LOOKBACK)
        except IndexError:
            pass
        else:
            raise AssertionError(f"tau={bad} should be rejected by WindowDataset")

        try:
            stack_windows(panel, bad, LOOKBACK)
        except IndexError:
            pass
        else:
            raise AssertionError(f"tau={bad} should be rejected by stack_windows")

    # The boundary steps themselves are fine.
    WindowDataset(panel, [LOOKBACK - 1, n_days - 2], LOOKBACK)


def test_a_panel_tensors_view_is_accepted_and_checked():
    panel, splits = _panel_and_splits()
    from_view = WindowDataset(panel.torch(), splits.train, LOOKBACK)
    from_panel = WindowDataset(panel, splits.train, LOOKBACK)
    torch.testing.assert_close(from_view[0].h, from_panel[0].h)
    torch.testing.assert_close(from_view[0].r, from_panel[0].r)

    # A hand-mangled view must fail loudly rather than transpose silently.
    broken = panel.torch()
    broken.returns = broken.returns[:, :2]
    try:
        WindowDataset(broken, splits.train, LOOKBACK)
    except ValueError as exc:
        assert "returns" in str(exc)
    else:
        raise AssertionError("expected a shape mismatch to raise")


def test_an_empty_step_list_yields_an_empty_dataset():
    panel, splits = _panel_and_splits()
    ds = WindowDataset(panel, np.empty(0, dtype=np.int64), LOOKBACK)

    assert len(ds) == 0
    batch = ds.stack()
    assert batch.h.shape == (0, panel.n_assets, LOOKBACK, panel.n_features)
    assert batch.tau.shape == (0,)


# ---------------------------------------------------------------------------
# End-to-end: sections 1-4 build -> section 5 window dataset.


TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N_DAYS = 400
CALENDAR = trading_days(periods=N_DAYS)


def _pipeline_config(tmp: str) -> DiffolioConfig:
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
    config.window.lookback = LOOKBACK
    return config


def _fake_provider():
    index = CALENDAR
    frames = make_universe_frames(TICKERS, index)
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
    return stack


def test_windows_from_a_built_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        with _fake_provider():
            config = _pipeline_config(tmp)
            dataset = build_dataset(config, output_dir=Path(tmp) / "out")

        expected_h = (dataset.n_assets, config.lookback, config.num_features)
        for split in ("train", "val", "test"):
            ds = window_dataset(dataset, split)
            assert len(ds) == dataset.splits[split].n_samples
            assert ds.split_name == split

            sample = ds[0]
            assert sample.h.shape == expected_h
            assert sample.g.shape == (config.lookback, config.num_features)
            assert sample.r.shape == (dataset.n_assets,)
            # Every step's window and target stay inside its own split.
            assert sample.tau >= dataset.splits[split].start + config.lookback - 1
            assert sample.tau <= dataset.splits[split].stop - 2

        # The three splits partition the usable steps without overlap.
        all_tau = np.concatenate(
            [window_dataset(dataset, name).tau for name in ("train", "val", "test")]
        )
        assert len(set(all_tau.tolist())) == len(all_tau)

        # A memory-mapped reload feeds bit-identical samples.
        reloaded = load_dataset(Path(tmp) / "out", mmap=True)
        from_mmap = window_dataset(reloaded, "train")
        from_ram = window_dataset(dataset, "train")
        for i in (0, 3, len(from_ram) - 1):
            torch.testing.assert_close(from_mmap[i].h, from_ram[i].h)
            torch.testing.assert_close(from_mmap[i].r, from_ram[i].r)
            torch.testing.assert_close(from_mmap[i].r_valid, from_ram[i].r_valid)

        # And a real DataLoader round trip on the reloaded panel.
        loader = DataLoader(from_mmap, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        assert batch.h.shape == (32, *expected_h)
        assert batch.r_valid.dtype == torch.bool
