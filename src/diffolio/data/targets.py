"""Section 6 - risk-dependent pseudo-optimal portfolio synthesis.

For every usable decision step ``tau`` this builds

* the base pseudo-optimal portfolio ``x^(tau) = f(r_tau) = r_tau / sum |r_tau|``
  of Eq. (12).  Section 7 estimates the noise scale ``sigma_x`` over these,
  training steps only;
* the risk-dependent targets ``x_0^(tau, gamma) = f_gamma(r_tau)`` of Eqs.
  (15)-(16), one per risk level ``gamma in 0 .. gamma_max - 1``: the top
  ``k_gamma`` assets by ``|r_tau,n|`` keep their signed, L1-normalised return,
  every other asset gets zero.  These are what the diffusion model learns to
  denoise towards.

The normalisers live in :mod:`diffolio.portfolio` so that the auxiliary loss
(section 10) and risk guidance (section 12) reuse the very same code.

Assets whose return at ``tau`` is not genuine (``r_valid`` False - a
forward-filled leg) are never selected, so no target puts weight on a
zero-filled return.

Everything here is a deterministic function of ``panel.returns``, so the result
is cached next to the dataset under ``targets/`` and keyed by the dataset
fingerprint and ``gamma_max``.  Layout (``S`` usable steps across all splits,
``G = gamma_max``)::

    PortfolioTargets.tau      (S,)       int64    decision steps, sorted
    PortfolioTargets.base     (S, N)     float32  x^(tau)
    PortfolioTargets.targets  (S, G, N)  float32  x_0^(tau, gamma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from ..portfolio import l1_normalize, risk_sizes, top_k_normalize
from ..utils import get_logger, read_json, write_json
from .panel import MarketPanel, PanelTensors
from .pipeline import Dataset
from .splits import Split, SplitIndices
from .windows import WindowDataset

logger = get_logger(__name__)

__all__ = [
    "PortfolioDataset",
    "PortfolioSample",
    "PortfolioTargets",
    "build_targets",
    "compute_targets",
    "portfolio_dataset",
]

_TARGETS_DIR = "targets"
_META_NAME = "meta.json"
_ARRAY_FIELDS = ("tau", "base", "targets")
#: Tolerance on ``sum |x| = 1`` (plan 6.4); targets are stored as float32.
_L1_TOLERANCE = 1e-5


@dataclass
class PortfolioTargets:
    """Cached section-6 portfolios for a set of decision steps."""

    tau: np.ndarray  # (S,) int64, sorted
    base: np.ndarray  # (S, N) float32
    targets: np.ndarray  # (S, G, N) float32
    risk_sizes: tuple[int, ...]  # (G,) k_gamma
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gamma_max(self) -> int:
        return len(self.risk_sizes)

    @property
    def n_assets(self) -> int:
        return int(self.base.shape[1])

    def rows(self, tau: Split | Sequence[int] | np.ndarray) -> np.ndarray:
        """Row positions of the given decision steps; raises if one is missing."""
        if isinstance(tau, Split):
            tau = tau.tau
        tau = np.asarray(tau, dtype=np.int64)
        pos = np.searchsorted(self.tau, tau)
        found = (pos < self.tau.size) & (self.tau[np.minimum(pos, self.tau.size - 1)] == tau)
        if not found.all():
            missing = tau[~found][:5].tolist()
            raise KeyError(f"no portfolio targets for decision steps {missing}")
        return pos

    def select(self, tau: Split | Sequence[int] | np.ndarray) -> "PortfolioTargets":
        """The targets for a subset of decision steps (e.g. one split)."""
        pos = self.rows(tau)
        return PortfolioTargets(
            tau=self.tau[pos],
            base=self.base[pos],
            targets=self.targets[pos],
            risk_sizes=self.risk_sizes,
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        s, n = self.base.shape
        if self.tau.shape != (s,):
            raise ValueError(f"tau has shape {self.tau.shape}, expected {(s,)}")
        if self.targets.shape != (s, self.gamma_max, n):
            raise ValueError(
                f"targets has shape {self.targets.shape}, expected {(s, self.gamma_max, n)}"
            )
        if s and np.any(np.diff(self.tau) <= 0):
            raise ValueError("tau must be strictly increasing")

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in _ARRAY_FIELDS:
            np.save(directory / f"{name}.npy", getattr(self, name))
        write_json(
            directory / _META_NAME,
            {"risk_sizes": list(self.risk_sizes), "metadata": self.metadata},
        )
        logger.info("saved portfolio targets (%d steps) to %s", self.tau.size, directory)

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = False) -> "PortfolioTargets":
        directory = Path(directory)
        meta = read_json(directory / _META_NAME)
        mode = "r" if mmap else None
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode=mode) for name in _ARRAY_FIELDS
        }
        targets = cls(
            risk_sizes=tuple(int(k) for k in meta["risk_sizes"]),
            metadata=meta.get("metadata", {}),
            **arrays,
        )
        targets.validate()
        return targets

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / _META_NAME).exists() and all(
            (directory / f"{name}.npy").exists() for name in _ARRAY_FIELDS
        )


def compute_targets(
    returns: torch.Tensor,
    valid: torch.Tensor,
    gamma_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Base portfolios ``(..., N)`` and risk targets ``(..., G, N)`` for returns.

    ``returns`` / ``valid`` are ``(..., N)``.  Pure and batched, so the training
    loop can also call it directly on ``WindowSample.r`` if it prefers.
    """
    n_assets = returns.shape[-1]
    k = torch.tensor(risk_sizes(n_assets, gamma_max), device=returns.device)
    returns = torch.where(valid, returns, torch.zeros_like(returns))
    base = l1_normalize(returns)
    targets = top_k_normalize(returns.unsqueeze(-2), k, valid=valid.unsqueeze(-2))
    return base, targets


def build_targets(
    dataset: Dataset,
    force: bool = False,
    directory: str | Path | None = None,
    mmap: bool = False,
) -> PortfolioTargets:
    """Build (or reload) the section-6 targets for every usable step of ``dataset``.

    The cache lives in ``directory`` (default ``<dataset.root>/targets``) and is
    reused only when the dataset fingerprint and ``gamma_max`` both match.  A
    dataset without a root is computed in memory and not cached.
    """
    gamma_max = dataset.config.diffusion.gamma_max
    fingerprint = dataset.panel.metadata.get("fingerprint")
    if directory is None and dataset.root is not None:
        directory = Path(dataset.root) / _TARGETS_DIR

    if directory is not None and not force and PortfolioTargets.exists(directory):
        cached = PortfolioTargets.load(directory, mmap=mmap)
        meta = cached.metadata
        if meta.get("fingerprint") == fingerprint and cached.gamma_max == gamma_max:
            logger.info("reusing cached portfolio targets at %s", directory)
            return cached
        logger.info("cached portfolio targets are stale (fingerprint/gamma_max); rebuilding")

    targets = _compute_for_splits(dataset.panel, dataset.splits, gamma_max)
    targets.metadata["fingerprint"] = fingerprint
    if directory is not None:
        targets.save(directory)
    return targets


def _compute_for_splits(
    panel: MarketPanel, splits: SplitIndices, gamma_max: int
) -> PortfolioTargets:
    tau = np.sort(np.concatenate([split.tau for split in splits])).astype(np.int64)
    if np.unique(tau).size != tau.size:
        raise ValueError("splits share decision steps; they must partition the usable steps")

    # float64 so that the float32 copies satisfy sum |x| = 1 to rounding.
    returns = torch.from_numpy(np.asarray(panel.returns[tau], dtype=np.float64))
    valid = torch.from_numpy(np.asarray(panel.return_valid[tau], dtype=bool))
    base, targets = compute_targets(returns, valid, gamma_max)
    sizes = tuple(risk_sizes(panel.n_assets, gamma_max))

    # A step whose genuine returns are all zero has no pseudo-optimal
    # portfolio; it maps to all-zero weights.  Report rather than hide it.
    degenerate = (returns.abs() * valid).sum(dim=-1) == 0
    if degenerate.any():
        logger.warning(
            "%d decision step(s) have no non-zero genuine return; their targets are "
            "all zero: %s",
            int(degenerate.sum()),
            tau[degenerate.numpy()][:10].tolist(),
        )
    _check_unit_l1(base, targets, degenerate, tau)

    logger.info(
        "built portfolio targets: %d steps, N=%d, gamma_max=%d, k_gamma=%s",
        tau.size,
        panel.n_assets,
        gamma_max,
        list(sizes),
    )
    result = PortfolioTargets(
        tau=tau,
        base=base.numpy().astype(np.float32),
        targets=targets.numpy().astype(np.float32),
        risk_sizes=sizes,
        metadata={
            "gamma_max": gamma_max,
            "degenerate_tau": tau[degenerate.numpy()].tolist(),
        },
    )
    result.validate()
    return result


def _check_unit_l1(
    base: torch.Tensor, targets: torch.Tensor, degenerate: torch.Tensor, tau: np.ndarray
) -> None:
    """Plan 6.4: every non-degenerate portfolio has ``sum |x| = 1``."""
    ok = ~degenerate
    base_err = (base.abs().sum(-1) - 1.0).abs()[ok]
    target_err = (targets.abs().sum(-1) - 1.0).abs()[ok]
    worst = max(
        float(base_err.max()) if base_err.numel() else 0.0,
        float(target_err.max()) if target_err.numel() else 0.0,
    )
    if worst > _L1_TOLERANCE:
        raise RuntimeError(f"portfolio targets violate sum |x| = 1 (max error {worst:.3g})")


# ---------------------------------------------------------------------------
# Training-side view: section-5 windows plus their section-6 targets.


class PortfolioSample(NamedTuple):
    """A :class:`~diffolio.data.windows.WindowSample` plus its portfolios.

    Collated batches gain a leading batch axis, as for ``WindowSample``.
    """

    h: torch.Tensor  # (N, L, F) float32
    g: torch.Tensor  # (L, F) float32
    r: torch.Tensor  # (N,) float32
    r_valid: torch.Tensor  # (N,) bool
    tau: torch.Tensor  # () int64
    x_base: torch.Tensor  # (N,) float32 base pseudo-optimal portfolio x^(tau)
    x0: torch.Tensor  # (G, N) float32 risk targets x_0^(tau, gamma), row = gamma


class PortfolioDataset(WindowDataset):
    """Section-5 windows paired with the section-6 targets for the same steps.

    All ``G`` risk targets ride along with each sample; the training loop
    draws ``gamma`` per sample and indexes ``x0[:, gamma]`` itself.
    """

    def __init__(
        self,
        panel: MarketPanel | PanelTensors,
        tau: Split | Sequence[int] | np.ndarray,
        lookback: int,
        targets: PortfolioTargets,
    ) -> None:
        super().__init__(panel, tau, lookback)
        if targets.n_assets != self.n_assets:
            raise ValueError(
                f"targets cover N={targets.n_assets} assets, windows have N={self.n_assets}"
            )
        pos = targets.rows(self.tau)
        self.x_base = torch.from_numpy(np.array(targets.base[pos], dtype=np.float32))
        self.x0 = torch.from_numpy(np.array(targets.targets[pos], dtype=np.float32))
        self.risk_sizes = targets.risk_sizes

    def __getitem__(self, index: int) -> PortfolioSample:
        return PortfolioSample(*super().__getitem__(index), self.x_base[index], self.x0[index])

    def stack(self, indices: Sequence[int] | np.ndarray | None = None) -> PortfolioSample:
        rows = np.arange(len(self)) if indices is None else np.asarray(indices, dtype=np.int64)
        rows_t = torch.as_tensor(rows, dtype=torch.int64)
        return PortfolioSample(*super().stack(rows), self.x_base[rows_t], self.x0[rows_t])


def portfolio_dataset(
    dataset: Dataset,
    split: str = "train",
    targets: PortfolioTargets | None = None,
) -> PortfolioDataset:
    """Windows and targets for one split of a built dataset (sections 5-6)."""
    if targets is None:
        targets = build_targets(dataset)
    return PortfolioDataset(
        dataset.panel, dataset.splits[split], dataset.config.lookback, targets
    )
