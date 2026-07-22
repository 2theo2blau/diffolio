"""Section 4 - chronological train/validation/test partition (7:1:2).

A split is described by a day range ``[start, stop)`` on the master calendar
*and* by the set of decision steps ``tau`` that are actually usable inside it.
A step is usable only when

* the whole look-up window ``[tau - L + 1, tau]`` lies inside the split, and
* the return target's second leg ``tau + 1`` also lies inside the split,

so no sample ever straddles a boundary and no label is drawn from the next
period.  Windows sitting on top of too much forward-filled data are dropped as
well (plan 5.3), which is why this lives after cleaning rather than before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..config import DiffolioConfig, SplitConfig
from ..utils import get_logger, read_json, write_json
from .panel import MarketPanel

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """One contiguous period plus its usable decision steps."""

    name: str
    start: int  # first calendar index in the split
    stop: int  # one past the last calendar index
    tau: np.ndarray  # int64, sorted, usable decision steps

    @property
    def n_days(self) -> int:
        return self.stop - self.start

    @property
    def n_samples(self) -> int:
        return int(self.tau.size)

    @property
    def day_slice(self) -> slice:
        return slice(self.start, self.stop)

    def dates(self, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return calendar[self.tau]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": int(self.start),
            "stop": int(self.stop),
            "n_days": int(self.n_days),
            "n_samples": self.n_samples,
            "tau": self.tau.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Split":
        return cls(
            name=payload["name"],
            start=int(payload["start"]),
            stop=int(payload["stop"]),
            tau=np.asarray(payload["tau"], dtype="int64"),
        )


@dataclass
class SplitIndices:
    """The three splits, keyed by name."""

    lookback: int
    train: Split
    val: Split
    test: Split

    def __getitem__(self, name: str) -> Split:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def __iter__(self) -> Iterator[Split]:
        return iter((self.train, self.val, self.test))

    def items(self) -> list[tuple[str, Split]]:
        return [(name, self[name]) for name in SPLIT_NAMES]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            **{name: split.to_dict() for name, split in self.items()},
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndices":
        payload = read_json(path)
        return cls(
            lookback=int(payload["lookback"]),
            **{name: Split.from_dict(payload[name]) for name in SPLIT_NAMES},
        )

    def describe(self, calendar: pd.DatetimeIndex | None = None) -> str:
        lines = []
        for name, split in self.items():
            span = ""
            if calendar is not None and split.n_days:
                first = calendar[split.start].date()
                last = calendar[split.stop - 1].date()
                span = f" {first}..{last}"
            lines.append(
                f"  {name:<5} days[{split.start}:{split.stop}]{span} "
                f"-> {split.n_samples} samples"
            )
        return "\n".join(lines)


def compute_split_bounds(
    n_days: int, train_frac: float, val_frac: float
) -> tuple[int, int]:
    """Return ``(train_end, val_end)`` calendar indices (plan 4.1)."""
    if n_days < 3:
        raise ValueError(f"need at least 3 trading days to split, got {n_days}")
    # The epsilon guards the floor against binary representation error: 0.7 +
    # 0.1 is 0.7999999999999999, which would shave a day off the validation
    # split at every round number of trading days.
    eps = 1e-9
    train_end = int(np.floor(train_frac * n_days + eps))
    val_end = int(np.floor((train_frac + val_frac) * n_days + eps))
    train_end = max(1, min(train_end, n_days - 2))
    val_end = max(train_end + 1, min(val_end, n_days - 1))
    return train_end, val_end


def make_splits(panel: MarketPanel, config: DiffolioConfig) -> SplitIndices:
    """Partition the calendar and enumerate the usable decision steps."""
    lookback = config.lookback
    split_config = config.split
    train_end, val_end = compute_split_bounds(
        panel.n_days, split_config.train_frac, split_config.val_frac
    )
    bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, panel.n_days),
    }

    splits = {
        name: Split(
            name=name,
            start=start,
            stop=stop,
            tau=_usable_steps(panel, start, stop, lookback, split_config),
        )
        for name, (start, stop) in bounds.items()
    }

    empty = [name for name, split in splits.items() if split.n_samples == 0]
    if empty:
        raise RuntimeError(
            f"splits {empty} contain no usable decision step with L={lookback}; "
            f"the calendar has {panel.n_days} days, so each split needs at least "
            f"L + 1 = {lookback + 1} of them"
        )

    indices = SplitIndices(lookback=lookback, **splits)
    logger.info("split calendar (L=%d):\n%s", lookback, indices.describe(panel.calendar))
    return indices


def _usable_steps(
    panel: MarketPanel,
    start: int,
    stop: int,
    lookback: int,
    config: SplitConfig,
) -> np.ndarray:
    """Decision steps whose window and target both sit inside ``[start, stop)``."""
    first = start + lookback - 1
    last = stop - 2  # tau + 1 must remain inside the split
    if last < first:
        return np.empty(0, dtype="int64")

    candidates = np.arange(first, last + 1, dtype="int64")

    target_frac = panel.return_valid[candidates].mean(axis=1)
    keep = target_frac >= config.min_valid_target_frac

    # Fraction of forward-filled cells inside each (N, L) window, via a
    # cumulative sum so the cost is O(T) rather than O(T * L).
    filled = (~panel.observed).sum(axis=1).astype("float64")  # (T,)
    cumulative = np.concatenate([[0.0], np.cumsum(filled)])
    window_filled = cumulative[candidates + 1] - cumulative[candidates - lookback + 1]
    keep &= (window_filled / (lookback * panel.n_assets)) <= config.max_window_fill_frac

    index_filled = (~panel.index_observed).astype("float64")
    index_cumulative = np.concatenate([[0.0], np.cumsum(index_filled)])
    index_window = (
        index_cumulative[candidates + 1] - index_cumulative[candidates - lookback + 1]
    )
    keep &= (index_window / lookback) <= config.max_window_fill_frac

    return candidates[keep]


def split_of(indices: SplitIndices, tau: int) -> str | None:
    """Which split owns decision step ``tau`` (``None`` if it is unusable)."""
    for name, split in indices.items():
        if bool(np.isin(tau, split.tau)):
            return name
    return None
