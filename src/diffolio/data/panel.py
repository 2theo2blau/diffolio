"""The aligned market panel - the hand-off object between ingestion and models.

Tensor shape conventions used everywhere downstream (see plan, cross-cutting
notes).  ``T`` is the number of trading days on the master calendar, ``N`` the
number of assets, ``F`` the number of price features:

===================  =====================  ==========================================
name                 shape                  meaning
===================  =====================  ==========================================
``features``  (X)    ``(T, N, F)``          normalised per-asset price features
``index_features``   ``(T, F)``             normalised benchmark features
``open_prices``      ``(T, N)``             adjusted opens, raw scale
``returns``   (r)    ``(T, N)``             ``(open[t+1] - open[t]) / open[t]``
``observed``         ``(T, N)`` bool        True where the bar was traded, not filled
===================  =====================  ==========================================

Sliding windows (plan section 5) slice this as ``X[tau-L+1 : tau+1]`` and
transpose to ``(N, L, F)``; batched tensors then carry a leading batch axis:
``h (B, N, L, F)``, ``g (B, L, F)``, ``r (B, N)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..utils import get_logger, read_json, write_json

logger = get_logger(__name__)

_ARRAY_FIELDS = (
    "features",
    "index_features",
    "open_prices",
    "close_prices",
    "index_open",
    "index_close",
    "returns",
    "return_valid",
    "index_returns",
    "index_return_valid",
    "observed",
    "index_observed",
)
_META_NAME = "meta.json"
_ARRAY_DIR = "arrays"


@dataclass
class PanelTensors:
    """Torch view of the panel, ready to be indexed by a ``Dataset``."""

    features: torch.Tensor  # (T, N, F) float32
    index_features: torch.Tensor  # (T, F) float32
    returns: torch.Tensor  # (T, N) float32
    return_valid: torch.Tensor  # (T, N) bool
    observed: torch.Tensor  # (T, N) bool
    open_prices: torch.Tensor  # (T, N) float32

    def to(self, device: str | torch.device) -> "PanelTensors":
        return PanelTensors(**{k: v.to(device) for k, v in vars(self).items()})


@dataclass
class MarketPanel:
    """Gap-free, calendar-aligned market data for a fixed universe."""

    calendar: pd.DatetimeIndex
    tickers: tuple[str, ...]
    feature_names: tuple[str, ...]
    benchmark: str

    features: np.ndarray
    index_features: np.ndarray
    open_prices: np.ndarray
    close_prices: np.ndarray
    index_open: np.ndarray
    index_close: np.ndarray
    returns: np.ndarray
    return_valid: np.ndarray
    index_returns: np.ndarray
    index_return_valid: np.ndarray
    observed: np.ndarray
    index_observed: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)

    # -- shape helpers ------------------------------------------------------
    @property
    def n_days(self) -> int:
        """T."""
        return len(self.calendar)

    @property
    def n_assets(self) -> int:
        """N."""
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        """F."""
        return len(self.feature_names)

    def asset_index(self, ticker: str) -> int:
        return self.tickers.index(ticker)

    def date_index(self, date: str | pd.Timestamp) -> int:
        return int(self.calendar.get_loc(pd.Timestamp(date)))

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        calendar: pd.DatetimeIndex,
        tickers: Sequence[str],
        feature_names: Sequence[str],
        benchmark: str,
        features: np.ndarray,
        index_features: np.ndarray,
        open_prices: np.ndarray,
        close_prices: np.ndarray,
        index_open: np.ndarray,
        index_close: np.ndarray,
        observed: np.ndarray,
        index_observed: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MarketPanel":
        """Assemble a panel and derive the open-to-open return targets."""
        returns, return_valid = compute_open_to_open_returns(open_prices, observed)
        index_returns, index_return_valid = compute_open_to_open_returns(
            index_open[:, None], index_observed[:, None]
        )

        panel = cls(
            calendar=pd.DatetimeIndex(calendar, name="date"),
            tickers=tuple(tickers),
            feature_names=tuple(feature_names),
            benchmark=benchmark,
            features=np.ascontiguousarray(features, dtype="float32"),
            index_features=np.ascontiguousarray(index_features, dtype="float32"),
            open_prices=np.ascontiguousarray(open_prices, dtype="float32"),
            close_prices=np.ascontiguousarray(close_prices, dtype="float32"),
            index_open=np.ascontiguousarray(index_open, dtype="float32"),
            index_close=np.ascontiguousarray(index_close, dtype="float32"),
            returns=returns,
            return_valid=return_valid,
            index_returns=index_returns[:, 0],
            index_return_valid=index_return_valid[:, 0],
            observed=np.ascontiguousarray(observed, dtype=bool),
            index_observed=np.ascontiguousarray(index_observed, dtype=bool),
            metadata=dict(metadata or {}),
        )
        panel.validate()
        return panel

    def validate(self) -> None:
        t, n, f = self.n_days, self.n_assets, self.n_features
        expected = {
            "features": (t, n, f),
            "index_features": (t, f),
            "open_prices": (t, n),
            "close_prices": (t, n),
            "index_open": (t,),
            "index_close": (t,),
            "returns": (t, n),
            "return_valid": (t, n),
            "index_returns": (t,),
            "index_return_valid": (t,),
            "observed": (t, n),
            "index_observed": (t,),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")
        if len(set(self.tickers)) != n:
            raise ValueError("panel contains duplicate tickers")
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates:
            raise ValueError("calendar must be strictly increasing")
        for name in ("features", "index_features", "returns", "open_prices"):
            array = getattr(self, name)
            if not np.isfinite(array).all():
                bad = int((~np.isfinite(array)).sum())
                raise ValueError(f"{name} contains {bad} non-finite entries")
        if (self.open_prices <= 0).any():
            raise ValueError("open_prices contains non-positive values")

    # -- derived views ------------------------------------------------------
    def torch(self, device: str | torch.device = "cpu") -> PanelTensors:
        """Materialise the arrays a model needs as torch tensors."""

        def as_tensor(array: np.ndarray) -> torch.Tensor:
            # np.memmap views are read-only; copy so torch owns writable memory.
            if isinstance(array, np.memmap):
                array = np.array(array)
            return torch.from_numpy(array).to(device)

        return PanelTensors(
            features=as_tensor(self.features),
            index_features=as_tensor(self.index_features),
            returns=as_tensor(self.returns),
            return_valid=as_tensor(self.return_valid),
            observed=as_tensor(self.observed),
            open_prices=as_tensor(self.open_prices),
        )

    def window(self, tau: int, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The section-5 sample at decision step ``tau``: ``(h, g, r)``.

        ``h`` is ``(N, L, F)``, ``g`` is ``(L, F)`` and ``r`` is ``(N,)``.
        """
        if tau < lookback - 1 or tau >= self.n_days - 1:
            raise IndexError(
                f"tau={tau} has no complete look-up window / return target "
                f"(valid range: {lookback - 1}..{self.n_days - 2})"
            )
        lo = tau - lookback + 1
        h = np.transpose(self.features[lo : tau + 1], (1, 0, 2))
        g = self.index_features[lo : tau + 1]
        return h, g, self.returns[tau]

    def describe(self) -> str:
        return (
            f"MarketPanel(T={self.n_days}, N={self.n_assets}, F={self.n_features}, "
            f"{self.calendar[0].date()}..{self.calendar[-1].date()}, "
            f"benchmark={self.benchmark})"
        )

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        (directory / _ARRAY_DIR).mkdir(parents=True, exist_ok=True)
        for name in _ARRAY_FIELDS:
            np.save(directory / _ARRAY_DIR / f"{name}.npy", getattr(self, name))
        write_json(
            directory / _META_NAME,
            {
                "calendar": [str(d.date()) for d in self.calendar],
                "tickers": list(self.tickers),
                "feature_names": list(self.feature_names),
                "benchmark": self.benchmark,
                "shape": {"T": self.n_days, "N": self.n_assets, "F": self.n_features},
                "metadata": self.metadata,
            },
        )
        logger.info("saved %s to %s", self.describe(), directory)

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = False) -> "MarketPanel":
        directory = Path(directory)
        meta = read_json(directory / _META_NAME)
        mode = "r" if mmap else None
        arrays = {
            name: np.load(directory / _ARRAY_DIR / f"{name}.npy", mmap_mode=mode)
            for name in _ARRAY_FIELDS
        }
        panel = cls(
            calendar=pd.DatetimeIndex(pd.to_datetime(meta["calendar"]), name="date"),
            tickers=tuple(meta["tickers"]),
            feature_names=tuple(meta["feature_names"]),
            benchmark=meta["benchmark"],
            metadata=meta.get("metadata", {}),
            **arrays,
        )
        panel.validate()
        return panel

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / _META_NAME).exists()


def compute_open_to_open_returns(
    open_prices: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``r[t] = (open[t+1] - open[t]) / open[t]``, with a validity mask.

    The return at ``t`` is the payoff of a position opened at ``t``'s open and
    closed at ``t+1``'s open, so it is the target for a decision made at ``t``.
    A return is invalid if either leg is a forward-filled (i.e. not actually
    traded) bar, or on the final day, where ``t+1`` does not exist.  Invalid
    entries are zero-filled so no NaN can leak into a tensor.
    """
    prices = np.asarray(open_prices, dtype="float64")
    observed = np.asarray(observed, dtype=bool)
    returns = np.zeros_like(prices, dtype="float64")
    valid = np.zeros(prices.shape, dtype=bool)

    if prices.shape[0] > 1:
        current, nxt = prices[:-1], prices[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = (nxt - current) / current
        good = (
            np.isfinite(step)
            & (current > 0)
            & (nxt > 0)
            & observed[:-1]
            & observed[1:]
        )
        returns[:-1] = np.where(good, step, 0.0)
        valid[:-1] = good

    return returns.astype("float32"), valid
