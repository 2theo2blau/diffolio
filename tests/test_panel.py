from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from diffolio.data.panel import MarketPanel, compute_open_to_open_returns
from synthetic import trading_days


def make_panel(n_days: int = 40, n_assets: int = 3, n_features: int = 5) -> MarketPanel:
    rng = np.random.default_rng(0)
    calendar = trading_days(periods=n_days)
    observed = np.ones((n_days, n_assets), dtype=bool)
    return MarketPanel.build(
        calendar=calendar,
        tickers=[f"T{i}" for i in range(n_assets)],
        feature_names=["open", "high", "low", "close", "volume"][:n_features],
        benchmark="^IDX",
        features=rng.normal(size=(n_days, n_assets, n_features)),
        index_features=rng.normal(size=(n_days, n_features)),
        open_prices=100.0 + rng.normal(size=(n_days, n_assets)).cumsum(axis=0) * 0.1,
        close_prices=100.0 + rng.normal(size=(n_days, n_assets)).cumsum(axis=0) * 0.1,
        index_open=1000.0 + rng.normal(size=n_days).cumsum() * 0.5,
        index_close=1000.0 + rng.normal(size=n_days).cumsum() * 0.5,
        observed=observed,
        index_observed=np.ones(n_days, dtype=bool),
        metadata={"fingerprint": "test"},
    )


def test_open_to_open_returns_match_the_definition():
    opens = np.array([[10.0, 100.0], [11.0, 90.0], [11.0, 99.0]])
    observed = np.ones_like(opens, dtype=bool)

    returns, valid = compute_open_to_open_returns(opens, observed)

    np.testing.assert_allclose(returns[0], [0.1, -0.1], rtol=1e-6)
    np.testing.assert_allclose(returns[1], [0.0, 0.1], rtol=1e-6)
    # The final day has no t+1 open, so it carries no usable target.
    assert valid[0].all() and valid[1].all()
    assert not valid[2].any()
    assert returns[2].tolist() == [0.0, 0.0]


def test_returns_touching_a_filled_bar_are_invalid_but_zero_filled():
    opens = np.array([[10.0], [11.0], [12.0], [13.0]])
    observed = np.array([[True], [False], [True], [True]])

    returns, valid = compute_open_to_open_returns(opens, observed)

    assert valid[:, 0].tolist() == [False, False, True, False]
    assert np.isfinite(returns).all()
    assert returns[0, 0] == 0.0


def test_panel_validates_shapes():
    panel = make_panel()
    panel.validate()
    assert (panel.n_days, panel.n_assets, panel.n_features) == (40, 3, 5)

    panel.features = panel.features[:, :2]
    try:
        panel.validate()
    except ValueError as exc:
        assert "features" in str(exc)
    else:
        raise AssertionError("expected a shape mismatch to raise")


def test_window_returns_the_documented_shapes():
    panel = make_panel(n_days=40, n_assets=3)
    h, g, r = panel.window(tau=20, lookback=10)

    assert h.shape == (3, 10, 5)
    assert g.shape == (10, 5)
    assert r.shape == (3,)
    # h is the transposed slice X[tau-L+1 : tau+1].
    np.testing.assert_allclose(h[:, -1, :], panel.features[20])
    np.testing.assert_allclose(g[0], panel.index_features[11])

    for bad_tau in (8, panel.n_days - 1):
        try:
            panel.window(tau=bad_tau, lookback=10)
        except IndexError:
            pass
        else:
            raise AssertionError(f"tau={bad_tau} should have no valid window")


def test_save_load_roundtrip_including_mmap():
    panel = make_panel()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "panel"
        panel.save(directory)
        assert MarketPanel.exists(directory)

        for mmap in (False, True):
            restored = MarketPanel.load(directory, mmap=mmap)
            assert restored.tickers == panel.tickers
            assert restored.calendar.equals(panel.calendar)
            assert restored.metadata["fingerprint"] == "test"
            np.testing.assert_allclose(restored.features, panel.features)
            np.testing.assert_array_equal(restored.return_valid, panel.return_valid)
            # A memory-mapped panel still hands out usable torch tensors.
            tensors = restored.torch()
            assert tensors.features.shape == (40, 3, 5)
            assert tensors.features.dtype == torch.float32
            assert tensors.return_valid.dtype == torch.bool


def test_torch_view_shares_memory_when_not_mmapped():
    panel = make_panel()
    tensors = panel.torch()
    assert tensors.features.data_ptr() == panel.features.__array_interface__["data"][0]
    assert tensors.returns.shape == (40, 3)
