"""Section 6 tests, on the real S&P 500 build wherever possible.

The full-scale checks run on ``real_dataset`` / ``real_targets`` (N = 224).
Tests that mutate data or write a cache use a small writable block of the real
panel (``real_subpanel``); the only injected edge case is an invalid return,
which the real build (no forward fills) does not contain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diffolio.config import DiffolioConfig
from diffolio.data.pipeline import Dataset, build_dataset, load_dataset
from diffolio.data.splits import make_splits
from diffolio.data.targets import (
    PortfolioDataset,
    PortfolioSample,
    PortfolioTargets,
    build_targets,
    compute_targets,
    portfolio_dataset,
)
from diffolio.data.windows import WindowDataset

LOOKBACK = 10


@pytest.fixture
def small_dataset(real_subpanel):
    """Factory: a writable ``Dataset`` over a block of the real panel."""

    def make(n_days=300, n_assets=12, gamma_max=5, root=None) -> Dataset:
        panel = real_subpanel(n_days=n_days, n_assets=n_assets)
        config = DiffolioConfig()
        config.window.lookback = LOOKBACK
        config.diffusion.gamma_max = gamma_max
        return Dataset(
            panel=panel,
            splits=make_splits(panel, config),
            universe=None,
            config=config,
            report={},
            root=root,
        )

    return make


def _reference_target(r: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Plan 6.2-6.3 written out longhand."""
    magnitude = np.where(valid, np.abs(r), -np.inf)
    chosen = np.argsort(-magnitude, kind="stable")[:k]
    chosen = chosen[valid[chosen]]
    x = np.zeros_like(r, dtype=np.float64)
    x[chosen] = r[chosen] / np.abs(r[chosen]).sum()
    return x


def test_real_targets_cover_every_usable_step(real_dataset, real_targets):
    assert real_targets.risk_sizes == (220, 176, 132, 88, 44)  # plan 6.1, N = 224
    all_tau = np.sort(np.concatenate([s.tau for s in real_dataset.splits]))
    assert real_targets.tau.tolist() == all_tau.tolist()
    assert real_targets.targets.shape == (all_tau.size, 5, real_dataset.n_assets)
    assert real_targets.base.dtype == np.float32
    assert real_targets.targets.dtype == np.float32
    assert real_targets.metadata["fingerprint"] == real_dataset.panel.metadata["fingerprint"]
    assert real_targets.metadata["degenerate_tau"] == []


def test_real_targets_match_the_longhand_definition(real_dataset, real_targets):
    panel = real_dataset.panel
    rows = np.linspace(0, real_targets.tau.size - 1, 25).astype(int)
    for row in rows:
        tau = int(real_targets.tau[row])
        r = np.asarray(panel.returns[tau], dtype=np.float64)
        valid = np.asarray(panel.return_valid[tau])
        np.testing.assert_allclose(real_targets.base[row], r / np.abs(r).sum(), atol=1e-7)
        for gamma, k in enumerate(real_targets.risk_sizes):
            expected = _reference_target(r, valid, k)
            np.testing.assert_allclose(real_targets.targets[row, gamma], expected, atol=1e-7)


def test_every_real_target_has_unit_l1_norm_and_the_right_support(real_targets):
    targets = np.asarray(real_targets.targets)
    np.testing.assert_allclose(np.abs(real_targets.base).sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(targets).sum(-1), 1.0, atol=1e-5)

    support = (targets != 0).sum(-1)  # (S, G)
    assert (support <= np.array(real_targets.risk_sizes)[None, :]).all()
    # Real returns are rarely exactly zero, so each level is (nearly) full.
    assert (support >= np.array(real_targets.risk_sizes)[None, :] - 5).all()
    # Supports are nested: a riskier level holds a subset of a safer one's assets.
    nonzero = targets != 0
    assert (nonzero[:, 1:] <= nonzero[:, :-1]).all()


def test_riskier_levels_concentrate_on_the_largest_moves(real_targets):
    targets = np.asarray(real_targets.targets)
    # The largest single weight grows with gamma at every step.
    peak = np.abs(targets).max(-1)  # (S, G)
    assert (np.diff(peak, axis=1) >= -1e-7).all()


def test_signs_are_preserved_so_shorts_appear(real_dataset, real_targets):
    targets = np.asarray(real_targets.targets)
    held = targets != 0
    r_sign = np.sign(np.asarray(real_dataset.panel.returns)[real_targets.tau])[:, None, :]
    assert (np.sign(targets)[held] == np.broadcast_to(r_sign, held.shape)[held]).all()
    assert (targets < 0).any() and (targets > 0).any()


def test_invalid_returns_never_receive_weight(small_dataset):
    dataset = small_dataset()
    panel = dataset.panel
    tau = int(dataset.splits.train.tau[0])
    # Injected edge case: mark one asset's return as a filled bar carrying a
    # huge stale value, which the top-k selection must ignore.
    panel.return_valid[tau, 3] = False
    panel.returns[tau, 3] = 0.9

    targets = build_targets(dataset)
    row = targets.rows([tau])[0]
    assert (targets.targets[row, :, 3] == 0).all()
    assert targets.base[row, 3] == 0
    np.testing.assert_allclose(np.abs(targets.targets[row]).sum(-1), 1.0, atol=1e-5)


def test_compute_targets_is_batched_and_matches_the_cache(real_dataset, real_targets):
    tau = real_targets.tau[:16]
    r = torch.from_numpy(np.asarray(real_dataset.panel.returns[tau], dtype=np.float64))
    valid = torch.from_numpy(np.asarray(real_dataset.panel.return_valid[tau]))
    base, x0 = compute_targets(r, valid, gamma_max=5)

    assert base.shape == (16, 224) and x0.shape == (16, 5, 224)
    np.testing.assert_allclose(base.numpy(), real_targets.base[:16], atol=1e-7)
    np.testing.assert_allclose(x0.numpy(), real_targets.targets[:16], atol=1e-7)

    # A different gamma_max changes k_gamma = floor(224 / 2) * (2 - gamma) = [224, 112].
    _, x0_two = compute_targets(r, valid, gamma_max=2)
    assert (x0_two[:, 1] != 0).sum(-1).max() <= 112


def test_cache_round_trip_and_invalidation(small_dataset, tmp_path):
    dataset = small_dataset(root=tmp_path)
    first = build_targets(dataset)
    assert PortfolioTargets.exists(tmp_path / "targets")

    again = build_targets(dataset, mmap=True)
    np.testing.assert_array_equal(again.targets, first.targets)
    assert isinstance(again.targets, np.memmap)

    # A different gamma_max must not reuse the stale cache.
    dataset.config.diffusion.gamma_max = 3
    rebuilt = build_targets(dataset)
    assert rebuilt.gamma_max == 3
    assert rebuilt.targets.shape[1] == 3

    # Nor may a dataset with a different fingerprint.
    dataset.panel.metadata["fingerprint"] = "other"
    assert build_targets(dataset).metadata["fingerprint"] == "other"


def test_select_and_missing_steps(real_dataset, real_targets):
    train = real_targets.select(real_dataset.splits.train)
    assert train.tau.tolist() == real_dataset.splits.train.tau.tolist()

    with pytest.raises(KeyError):
        real_targets.rows([0])  # tau = 0 has no complete window, so no target


def test_portfolio_dataset_pairs_windows_with_targets(real_dataset, real_targets):
    ds = portfolio_dataset(real_dataset, "train", real_targets)
    windows = WindowDataset(ds.tensors, real_dataset.splits.train, real_dataset.config.lookback)
    n = real_dataset.n_assets

    assert len(ds) == len(windows)
    for i in (0, len(ds) - 1):
        sample = ds[i]
        assert isinstance(sample, PortfolioSample)
        torch.testing.assert_close(sample.h, windows[i].h)
        assert sample.x0.shape == (5, n)
        row = real_targets.rows([int(sample.tau)])[0]
        np.testing.assert_array_equal(sample.x0.numpy(), real_targets.targets[row])
        np.testing.assert_array_equal(sample.x_base.numpy(), real_targets.base[row])

    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    assert isinstance(batch, PortfolioSample)
    assert batch.x0.shape == (8, 5, n)
    rows = real_targets.rows(batch.tau.numpy())
    np.testing.assert_array_equal(batch.x0.numpy(), real_targets.targets[rows])

    stacked = ds.stack([2, 0])
    assert stacked.tau.tolist() == real_dataset.splits.train.tau[[2, 0]].tolist()
    torch.testing.assert_close(stacked.x0[1], ds[0].x0)


def test_portfolio_dataset_rejects_mismatched_targets(real_dataset, small_dataset):
    other = build_targets(small_dataset(n_assets=10))
    with pytest.raises(ValueError):
        PortfolioDataset(
            real_dataset.panel, real_dataset.splits.train, real_dataset.config.lookback, other
        )


def test_targets_from_a_built_dataset_end_to_end(real_pipeline_config, tmp_path):
    config = real_pipeline_config()
    dataset = build_dataset(config, output_dir=tmp_path / "out")
    assert dataset.root == tmp_path / "out"

    targets = build_targets(dataset)
    assert (tmp_path / "out" / "targets" / "meta.json").exists()

    reloaded = load_dataset(tmp_path / "out", mmap=True)
    cached = build_targets(reloaded)
    np.testing.assert_array_equal(cached.targets, targets.targets)

    for split in ("train", "val", "test"):
        ds = portfolio_dataset(reloaded, split, cached)
        assert len(ds) == reloaded.splits[split].n_samples
        np.testing.assert_allclose(ds.x0.abs().sum(-1).numpy(), 1.0, atol=1e-5)
