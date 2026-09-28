from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
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

from test_panel import make_panel
from test_windows import _fake_provider, _pipeline_config

LOOKBACK = 10


def _dataset(n_days=300, n_assets=12, gamma_max=5, root=None) -> Dataset:
    panel = make_panel(n_days=n_days, n_assets=n_assets)
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


def _reference_target(r: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Plan 6.2-6.3 written out longhand."""
    magnitude = np.where(valid, np.abs(r), -np.inf)
    chosen = np.argsort(-magnitude, kind="stable")[:k]
    chosen = chosen[valid[chosen]]
    x = np.zeros_like(r, dtype=np.float64)
    x[chosen] = r[chosen] / np.abs(r[chosen]).sum()
    return x


def test_targets_match_the_longhand_definition():
    dataset = _dataset()
    targets = build_targets(dataset)
    panel = dataset.panel

    assert targets.risk_sizes == (10, 8, 6, 4, 2)  # floor(12 / 5) = 2
    all_tau = np.sort(np.concatenate([s.tau for s in dataset.splits]))
    assert targets.tau.tolist() == all_tau.tolist()
    assert targets.targets.shape == (all_tau.size, 5, panel.n_assets)
    assert targets.base.dtype == np.float32 and targets.targets.dtype == np.float32

    for row in (0, targets.tau.size // 2, targets.tau.size - 1):
        tau = int(targets.tau[row])
        r = panel.returns[tau].astype(np.float64)
        valid = panel.return_valid[tau]
        np.testing.assert_allclose(targets.base[row], r / np.abs(r).sum(), atol=1e-7)
        for gamma, k in enumerate(targets.risk_sizes):
            expected = _reference_target(r, valid, k)
            np.testing.assert_allclose(targets.targets[row, gamma], expected, atol=1e-7)


def test_every_target_has_unit_l1_norm_and_the_right_support():
    targets = build_targets(_dataset())

    np.testing.assert_allclose(np.abs(targets.base).sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(targets.targets).sum(-1), 1.0, atol=1e-5)
    support = (targets.targets != 0).sum(-1)  # (S, G)
    assert (support <= np.array(targets.risk_sizes)[None, :]).all()
    # Supports are nested: a riskier level holds a subset of a safer one's assets.
    nonzero = targets.targets != 0
    assert (nonzero[:, 1:] <= nonzero[:, :-1]).all()


def test_signs_are_preserved_so_shorts_appear():
    targets = build_targets(_dataset())
    panel = _dataset().panel
    held = targets.targets != 0
    r_sign = np.sign(panel.returns[targets.tau])[:, None, :]
    assert (np.sign(targets.targets)[held] == np.broadcast_to(r_sign, held.shape)[held]).all()
    assert (targets.targets < 0).any()


def test_invalid_returns_never_receive_weight():
    dataset = _dataset()
    panel = dataset.panel
    tau = int(dataset.splits.train.tau[0])
    # Mark one asset's return as a filled bar with a huge (stale) value.
    panel.return_valid[tau, 3] = False
    panel.returns[tau, 3] = 0.9

    targets = build_targets(dataset)
    row = targets.rows([tau])[0]
    assert (targets.targets[row, :, 3] == 0).all()
    assert targets.base[row, 3] == 0
    np.testing.assert_allclose(np.abs(targets.targets[row]).sum(-1), 1.0, atol=1e-5)


def test_compute_targets_is_batched_and_device_agnostic():
    r = torch.tensor([[0.04, -0.01, 0.03, -0.02, 0.0]] * 2)
    valid = torch.ones_like(r, dtype=torch.bool)
    base, x0 = compute_targets(r, valid, gamma_max=2)

    assert base.shape == (2, 5) and x0.shape == (2, 2, 5)
    # k = floor(5 / 2) * (2 - gamma) = [4, 2]
    assert (x0 != 0).sum(-1).tolist() == [[4, 2], [4, 2]]
    torch.testing.assert_close(x0[0, 1], torch.tensor([4 / 7, 0.0, 3 / 7, 0.0, 0.0]))


def test_cache_round_trip_and_invalidation():
    with tempfile.TemporaryDirectory() as tmp:
        dataset = _dataset(root=Path(tmp))
        first = build_targets(dataset)
        assert PortfolioTargets.exists(Path(tmp) / "targets")

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


def test_select_and_missing_steps():
    dataset = _dataset()
    targets = build_targets(dataset)
    train = targets.select(dataset.splits.train)
    assert train.tau.tolist() == dataset.splits.train.tau.tolist()

    try:
        targets.rows([0])  # tau = 0 has no complete window, so it has no target
    except KeyError:
        pass
    else:
        raise AssertionError("a missing decision step should raise")


def test_portfolio_dataset_pairs_windows_with_targets():
    dataset = _dataset()
    targets = build_targets(dataset)
    ds = portfolio_dataset(dataset, "train", targets)
    windows = WindowDataset(dataset.panel, dataset.splits.train, LOOKBACK)

    assert len(ds) == len(windows)
    for i in (0, len(ds) - 1):
        sample = ds[i]
        assert isinstance(sample, PortfolioSample)
        torch.testing.assert_close(sample.h, windows[i].h)
        assert sample.x0.shape == (5, dataset.n_assets)
        row = targets.rows([int(sample.tau)])[0]
        np.testing.assert_array_equal(sample.x0.numpy(), targets.targets[row])
        np.testing.assert_array_equal(sample.x_base.numpy(), targets.base[row])

    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    assert isinstance(batch, PortfolioSample)
    assert batch.x0.shape == (8, 5, dataset.n_assets)
    rows = targets.rows(batch.tau.numpy())
    np.testing.assert_array_equal(batch.x0.numpy(), targets.targets[rows])

    stacked = ds.stack([2, 0])
    assert stacked.tau.tolist() == dataset.splits.train.tau[[2, 0]].tolist()
    torch.testing.assert_close(stacked.x0[1], ds[0].x0)


def test_portfolio_dataset_rejects_mismatched_targets():
    dataset = _dataset()
    other = build_targets(_dataset(n_assets=10))
    try:
        PortfolioDataset(dataset.panel, dataset.splits.train, LOOKBACK, other)
    except ValueError:
        pass
    else:
        raise AssertionError("an N mismatch should raise")


def test_targets_from_a_built_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        with _fake_provider():
            config = _pipeline_config(tmp)
            dataset = build_dataset(config, output_dir=Path(tmp) / "out")
        assert dataset.root == Path(tmp) / "out"

        targets = build_targets(dataset)
        assert (Path(tmp) / "out" / "targets" / "meta.json").exists()

        reloaded = load_dataset(Path(tmp) / "out", mmap=True)
        cached = build_targets(reloaded)
        np.testing.assert_array_equal(cached.targets, targets.targets)

        for split in ("train", "val", "test"):
            ds = portfolio_dataset(reloaded, split, cached)
            assert len(ds) == reloaded.splits[split].n_samples
            np.testing.assert_allclose(ds.x0.abs().sum(-1).numpy(), 1.0, atol=1e-5)
