"""Section 5 - sliding window construction.

Turns the aligned panel into per-decision-step samples ``(h_tau, g_tau, r_tau)``:

    h_tau = X[tau-L+1 : tau+1].transpose((1, 0, 2))    (N, L, F) asset window
    g_tau = G[tau-L+1 : tau+1]                          (L, F)    index window
    r_tau = (open[tau+1] - open[tau]) / open[tau]       (N,)      return target

``r_tau`` is already materialised on the panel: ``compute_open_to_open_returns``
stores the payoff of a position opened at ``tau`` in row ``tau`` of
``panel.returns``.  Which decision steps exist at all is decided by the
section-4 splits - a step is usable only when its whole window *and* the second
leg of its target lie inside one split, and when neither rests on too much
forward-filled data (plan 5.3).  This module therefore consumes ``Split.tau``
directly instead of re-deriving any of that, so the partitions stay
authoritative in one place.

**Storage (plan 5.2, deliberately deviated).**  The plan suggests materialising
every window into a memory-mapped array or LMDB.  But windows are overlapping,
deterministic slices of the panel - each trading day appears in ``L``
consecutive windows - so a materialised copy would be ~``L`` times larger than
the panel itself (hundreds of GB at the paper's scale) while saving nothing.
The panel ``.npy`` files already *are* the memory-mapped store, and slicing a
window out of them costs one ~``N * L * F`` copy (~1 MB at the paper's scale):
negligible next to the model's cross-asset attention.  :class:`WindowDataset`
therefore slices on demand and there is nothing extra to cache or rebuild.

Sample layout - and, with a leading batch axis, the collated batch layout::

    WindowSample.h        (N, L, F) float32   per-asset look-up window
    WindowSample.g        (L, F)    float32   index look-up window
    WindowSample.r        (N,)      float32   open-to-open return target
    WindowSample.r_valid  (N,)      bool       True where r is a genuine return
    WindowSample.tau      ()        int64      decision step on the master calendar

``r_valid`` rides along because a few assets can still carry an invalid
(zero-filled) target at a usable step: the section-4 screen caps them, it does
not eliminate them.  Section 6's top-k selection can use the mask to keep
pseudo-optimal portfolios on genuinely traded assets.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..utils import get_logger
from .panel import MarketPanel, PanelTensors
from .pipeline import Dataset
from .splits import Split

logger = get_logger(__name__)

__all__ = ["WindowDataset", "WindowSample", "stack_windows", "window_dataset"]


class WindowSample(NamedTuple):
    """Tensors for one decision step.

    A collated batch has the same fields with a leading batch axis ``(B, ...)``,
    per the shape conventions at the top of :mod:`diffolio.data.panel`.
    """

    h: torch.Tensor  # (N, L, F) float32 asset look-up window
    g: torch.Tensor  # (L, F) float32 index look-up window
    r: torch.Tensor  # (N,) float32 open-to-open return target
    r_valid: torch.Tensor  # (N,) bool genuine-return mask
    tau: torch.Tensor  # () int64 decision step on the master calendar


class WindowDataset(TorchDataset):
    """The section-5 samples for one split, indexed by position in ``tau``.

    ``panel`` may be a :class:`~diffolio.data.panel.MarketPanel` (a panel loaded
    with ``mmap=True`` is copied into memory once, by ``PanelTensors.torch``)
    or an existing :class:`~diffolio.data.panel.PanelTensors` view.  Tensors
    stay on CPU: ``DataLoader`` workers share them copy-on-write, and the
    training loop moves batches to the device.
    """

    def __init__(
        self,
        panel: MarketPanel | PanelTensors,
        tau: Split | Sequence[int] | np.ndarray,
        lookback: int,
    ) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        tensors = _as_tensors(panel)
        self.split_name = tau.name if isinstance(tau, Split) else None
        if isinstance(tau, Split):
            tau = tau.tau
        tau = np.ascontiguousarray(tau, dtype=np.int64)
        _check_tau(tau, n_days=int(tensors.features.shape[0]), lookback=lookback)

        self.tensors = tensors
        self.tau = tau
        self.lookback = int(lookback)
        self.n_days = int(tensors.features.shape[0])
        self.n_assets = int(tensors.features.shape[1])
        self.n_features = int(tensors.features.shape[2])
        logger.info("built %s", self.describe())

    # -- dataset protocol ----------------------------------------------------
    def __len__(self) -> int:
        return int(self.tau.size)

    def __getitem__(self, index: int) -> WindowSample:
        tau = int(self.tau[index])
        lo = tau - self.lookback + 1
        return WindowSample(
            h=self.tensors.features[lo : tau + 1].permute(1, 0, 2).contiguous(),
            g=self.tensors.index_features[lo : tau + 1],
            r=self.tensors.returns[tau],
            r_valid=self.tensors.return_valid[tau],
            tau=torch.tensor(tau, dtype=torch.int64),
        )

    # -- batched access without a DataLoader ----------------------------------
    def stack(self, indices: Sequence[int] | np.ndarray | None = None) -> WindowSample:
        """All (or a subset of the) samples, gathered in one shot."""
        if indices is None:
            tau = self.tau
        else:
            tau = self.tau[np.asarray(indices, dtype=np.int64)]
        return stack_windows(self.tensors, tau, self.lookback)

    def describe(self) -> str:
        name = self.split_name or "windows"
        return (
            f"WindowDataset({name}: {len(self)} steps, N={self.n_assets}, "
            f"L={self.lookback}, F={self.n_features})"
        )


def stack_windows(
    panel: MarketPanel | PanelTensors,
    tau: Split | Sequence[int] | np.ndarray,
    lookback: int,
) -> WindowSample:
    """Assemble the samples for many decision steps in one vectorised gather.

    Handy at inference time (section 12 walks the test steps one by one) and
    wherever a batch is wanted without a ``DataLoader``.
    """
    tensors = _as_tensors(panel)
    if isinstance(tau, Split):
        tau = tau.tau
    tau = np.ascontiguousarray(tau, dtype=np.int64)
    _check_tau(tau, n_days=int(tensors.features.shape[0]), lookback=lookback)

    # Day indices of every window: (B, L) with the last column == tau.
    offsets = np.arange(-(lookback - 1), 1, dtype=np.int64)
    window_index = torch.as_tensor(tau[:, None] + offsets[None, :], dtype=torch.int64)
    step_index = torch.as_tensor(tau, dtype=torch.int64)

    return WindowSample(
        h=tensors.features[window_index].permute(0, 2, 1, 3).contiguous(),
        g=tensors.index_features[window_index],
        r=tensors.returns[step_index],
        r_valid=tensors.return_valid[step_index],
        tau=step_index,
    )


def window_dataset(dataset: Dataset, split: str = "train") -> WindowDataset:
    """The section-5 samples for one split of a built dataset (sections 1-4)."""
    return WindowDataset(
        dataset.panel,
        dataset.splits[split],
        dataset.config.lookback,
    )


def _as_tensors(panel: MarketPanel | PanelTensors) -> PanelTensors:
    """Normalise to a ``PanelTensors`` view (memory-mapped arrays copy once)."""
    if isinstance(panel, MarketPanel):
        return panel.torch()
    if not isinstance(panel, PanelTensors):
        raise TypeError(f"expected MarketPanel or PanelTensors, got {type(panel).__name__}")
    _check_shapes(panel)
    return panel


def _check_shapes(tensors: PanelTensors) -> None:
    """Guard against silently transposed hand-built views (plan, shape notes)."""
    t, n, f = tensors.features.shape
    expected = {"index_features": (t, f), "returns": (t, n), "return_valid": (t, n)}
    for name, shape in expected.items():
        actual = tuple(getattr(tensors, name).shape)
        if actual != shape:
            raise ValueError(
                f"{name} has shape {actual}, expected {shape} "
                f"(features is {(t, n, f)})"
            )


def _check_tau(tau: np.ndarray, n_days: int, lookback: int) -> None:
    """Usable decision steps need a full window and an existing target leg."""
    if tau.size == 0:
        return
    lo, hi = int(tau.min()), int(tau.max())
    if lo < lookback - 1 or hi > n_days - 2:
        raise IndexError(
            f"decision steps must lie in [{lookback - 1}, {n_days - 2}] for a "
            f"complete look-up window and return target; got [{lo}, {hi}]"
        )
