from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from diffolio.data.panel import MarketPanel, compute_open_to_open_returns


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


def test_real_returns_are_open_to_open_of_the_adjusted_opens(real_dataset):
    panel = real_dataset.panel
    aapl = panel.asset_index("AAPL")
    opens = np.asarray(panel.open_prices[:, aapl], dtype=np.float64)
    expected = (opens[1:] - opens[:-1]) / opens[:-1]
    # Returns come from the float64 opens; the stored opens are float32, so
    # recomputing from them agrees to float32 rounding (~1e-7 absolute).
    np.testing.assert_allclose(panel.returns[:-1, aapl], expected, rtol=1e-5, atol=1e-6)
    assert panel.return_valid[:-1].all()  # no forward fills in the real build
    assert not panel.return_valid[-1].any()  # the last day has no t+1 open


def test_panel_validates_shapes(real_subpanel):
    panel = real_subpanel(n_days=40, n_assets=3)
    panel.validate()
    assert (panel.n_days, panel.n_assets, panel.n_features) == (40, 3, 5)

    panel.features = panel.features[:, :2]
    try:
        panel.validate()
    except ValueError as exc:
        assert "features" in str(exc)
    else:
        raise AssertionError("expected a shape mismatch to raise")


def test_the_real_panel_is_valid_and_finite(real_dataset):
    panel = real_dataset.panel
    panel.validate()
    assert panel.n_assets == len(real_dataset.universe.tickers)
    assert panel.tickers == tuple(real_dataset.universe.tickers)
    assert (np.asarray(panel.open_prices) > 0).all()


def test_window_returns_the_documented_shapes(real_dataset):
    panel = real_dataset.panel
    n = panel.n_assets
    h, g, r = panel.window(tau=300, lookback=256)

    assert h.shape == (n, 256, 5)
    assert g.shape == (256, 5)
    assert r.shape == (n,)
    # h is the transposed slice X[tau-L+1 : tau+1].
    np.testing.assert_array_equal(h[:, -1, :], panel.features[300])
    np.testing.assert_array_equal(h[:, 0, :], panel.features[300 - 255])
    np.testing.assert_array_equal(g[0], panel.index_features[300 - 255])
    np.testing.assert_array_equal(r, panel.returns[300])

    for bad_tau in (254, panel.n_days - 1):
        try:
            panel.window(tau=bad_tau, lookback=256)
        except IndexError:
            pass
        else:
            raise AssertionError(f"tau={bad_tau} should have no valid window")


def test_save_load_roundtrip_including_mmap(real_subpanel):
    panel = real_subpanel(n_days=40, n_assets=3)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "panel"
        panel.save(directory)
        assert MarketPanel.exists(directory)

        for mmap in (False, True):
            restored = MarketPanel.load(directory, mmap=mmap)
            assert restored.tickers == panel.tickers
            assert restored.calendar.equals(panel.calendar)
            assert restored.metadata["fingerprint"] == "real-subpanel"
            np.testing.assert_array_equal(restored.features, panel.features)
            np.testing.assert_array_equal(restored.returns, panel.returns)
            np.testing.assert_array_equal(restored.return_valid, panel.return_valid)
            # A memory-mapped panel still hands out usable torch tensors.
            tensors = restored.torch()
            assert tensors.features.shape == (40, 3, 5)
            assert tensors.features.dtype == torch.float32
            assert tensors.return_valid.dtype == torch.bool


def test_torch_view_shares_memory_when_not_mmapped(real_subpanel):
    panel = real_subpanel(n_days=40, n_assets=3)
    tensors = panel.torch()
    assert tensors.features.data_ptr() == panel.features.__array_interface__["data"][0]
    assert tensors.returns.shape == (40, 3)
