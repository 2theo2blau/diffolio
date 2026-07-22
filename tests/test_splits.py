from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from diffolio.config import DiffolioConfig
from diffolio.data.splits import SplitIndices, compute_split_bounds, make_splits
from test_panel import make_panel


def _config(lookback: int = 10) -> DiffolioConfig:
    config = DiffolioConfig()
    config.window.lookback = lookback
    return config


def test_split_bounds_follow_the_7_1_2_ratio():
    assert compute_split_bounds(1000, 0.7, 0.1) == (700, 800)
    assert compute_split_bounds(1001, 0.7, 0.1) == (700, 800)  # floor, per plan 4.1


def test_splits_cover_the_calendar_without_overlap():
    panel = make_panel(n_days=300, n_assets=4)
    splits = make_splits(panel, _config())

    assert splits.train.start == 0
    assert splits.train.stop == splits.val.start
    assert splits.val.stop == splits.test.start
    assert splits.test.stop == panel.n_days
    assert (splits.train.n_days, splits.val.n_days, splits.test.n_days) == (210, 30, 60)


def test_no_window_or_target_crosses_a_boundary():
    lookback = 10
    panel = make_panel(n_days=300, n_assets=4)
    splits = make_splits(panel, _config(lookback))

    for _, split in splits.items():
        assert split.n_samples > 0
        assert split.tau.min() >= split.start + lookback - 1
        # tau + 1 supplies the second leg of the return target.
        assert split.tau.max() <= split.stop - 2

    all_tau = np.concatenate([s.tau for s in splits])
    assert len(set(all_tau.tolist())) == len(all_tau)


def test_windows_over_forward_filled_data_are_dropped():
    lookback = 10
    panel = make_panel(n_days=300, n_assets=4)
    # Blank out one asset for a fortnight in the middle of the training split.
    panel.observed[100:114, 0] = False
    panel.returns, panel.return_valid = _recompute(panel)

    splits = make_splits(panel, _config(lookback))
    train_tau = set(splits.train.tau.tolist())

    # Steps 99..113 lose asset 0's target (25% of assets, over the 5% cap), and
    # any window holding 3+ filled cells out of L*N = 40 is over the fill cap.
    assert not any(tau in train_tau for tau in range(99, 121))
    assert 98 in train_tau  # windows and targets entirely before the gap survive
    assert 122 in train_tau  # ...and far enough after it


def test_index_gaps_also_invalidate_windows():
    panel = make_panel(n_days=300, n_assets=4)
    panel.index_observed[150:160] = False

    splits = make_splits(panel, _config(10))
    train_tau = set(splits.train.tau.tolist())
    assert 149 in train_tau
    assert 155 not in train_tau


def test_too_short_a_calendar_is_an_explicit_error():
    panel = make_panel(n_days=40, n_assets=2)
    try:
        make_splits(panel, _config(lookback=256))
    except RuntimeError as exc:
        assert "usable decision step" in str(exc)
    else:
        raise AssertionError("expected an error when L exceeds the split length")


def test_splits_roundtrip_to_json():
    panel = make_panel(n_days=300, n_assets=4)
    splits = make_splits(panel, _config())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits.json"
        splits.save(path)
        restored = SplitIndices.load(path)

    assert restored.lookback == splits.lookback
    for name, split in splits.items():
        np.testing.assert_array_equal(restored[name].tau, split.tau)
    assert "train" in restored.describe(panel.calendar)


def _recompute(panel):
    from diffolio.data.panel import compute_open_to_open_returns

    return compute_open_to_open_returns(panel.open_prices, panel.observed)
