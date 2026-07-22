"""Section 3 (step 4) - the F price features and their normalisation.

Two stages, deliberately separated by leakage risk:

1. A *causal, per-asset* transform (``FeatureBuilder``).  OHLC are divided by a
   trailing reference price and volume is log-relative to its own trailing
   mean, so every asset ends up on a comparable scale without any statistic
   that spans the train/test boundary.  Because the reference only ever looks
   backwards, this stage is leak-free by construction.
2. An optional *global* z-score (``FeatureStandardizer``) whose mean/std are
   fitted on the training split only and then frozen, per plan 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..utils import get_logger

logger = get_logger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "vwap", "adj_close")


class FeatureBuilder:
    """Turn a cleaned OHLCV frame into an ``(T, F)`` float32 feature block."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.columns: tuple[str, ...] = tuple(config.columns)

    @property
    def num_features(self) -> int:
        """F."""
        return len(self.columns)

    @property
    def warmup(self) -> int:
        """Leading rows that come out undefined and must be trimmed."""
        needs_window = (
            self.config.price_normalization == "rolling_ref"
            or self.config.volume_transform == "log_rel"
        )
        if needs_window:
            return max(self.config.ref_window - 1, 0)
        if self.config.price_normalization == "prev_close":
            return 1
        return 0

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the causal transform; the first ``warmup`` rows are NaN."""
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing feature columns {missing}")

        window = max(self.config.ref_window, 1)
        close = frame["close"].astype("float64")
        reference = self._price_reference(close, window)

        blocks: list[np.ndarray] = []
        for column in self.columns:
            series = frame[column].astype("float64")
            if column == "volume":
                values = self._transform_volume(series, window)
            elif column in PRICE_COLUMNS:
                values = self._transform_price(series, reference)
            else:
                values = series.to_numpy()
            blocks.append(np.asarray(values, dtype="float64"))

        out = np.stack(blocks, axis=1)
        return out.astype("float32", copy=False)

    def _price_reference(self, close: pd.Series, window: int) -> pd.Series | None:
        mode = self.config.price_normalization
        if mode == "rolling_ref":
            return close.rolling(window, min_periods=window).mean()
        if mode == "prev_close":
            return close.shift(1)
        if mode == "none":
            return None
        raise ValueError(f"unknown price_normalization {mode!r}")

    def _transform_price(self, series: pd.Series, reference: pd.Series | None) -> np.ndarray:
        if reference is None:
            return series.to_numpy()
        ratio = series / reference.replace(0.0, np.nan)
        # Centre on zero: a value of 0 means "at the reference price".
        return (ratio - 1.0).to_numpy()

    def _transform_volume(self, series: pd.Series, window: int) -> np.ndarray:
        mode = self.config.volume_transform
        log_volume = np.log1p(series.clip(lower=0.0))
        if mode == "log_rel":
            baseline = log_volume.rolling(window, min_periods=window).mean()
            return (log_volume - baseline).to_numpy()
        if mode == "log":
            return log_volume.to_numpy()
        if mode == "none":
            return series.to_numpy()
        raise ValueError(f"unknown volume_transform {mode!r}")


@dataclass
class FeatureStats:
    """Per-feature mean/std, fitted once on the training split and frozen."""

    mean: np.ndarray
    std: np.ndarray
    clip_sigma: float = 0.0

    @classmethod
    def fit(
        cls, array: np.ndarray, clip_sigma: float = 0.0, eps: float = 1e-8
    ) -> "FeatureStats":
        """Fit over every axis but the last (time, and assets when present)."""
        flat = array.reshape(-1, array.shape[-1]).astype("float64")
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
        return cls(
            mean=np.nan_to_num(mean).astype("float32"),
            std=std.astype("float32"),
            clip_sigma=float(clip_sigma),
        )

    def transform(self, array: np.ndarray) -> np.ndarray:
        out = (array.astype("float32") - self.mean) / self.std
        if self.clip_sigma and self.clip_sigma > 0:
            out = np.clip(out, -self.clip_sigma, self.clip_sigma)
        return out.astype("float32", copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_sigma": self.clip_sigma,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStats":
        return cls(
            mean=np.asarray(payload["mean"], dtype="float32"),
            std=np.asarray(payload["std"], dtype="float32"),
            clip_sigma=float(payload.get("clip_sigma", 0.0)),
        )


def build_feature_tensors(
    frames: dict[str, pd.DataFrame],
    tickers: Sequence[str],
    benchmark_frame: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assemble ``X (T, N, F)`` and ``G (T, F)`` in canonical ticker order.

    Returns the two tensors plus the warm-up length that the caller must trim
    from the head of every time-indexed array.
    """
    builder = FeatureBuilder(config)
    blocks = [builder.transform(frames[ticker]) for ticker in tickers]
    features = np.stack(blocks, axis=1)  # (T, N, F)
    index_features = builder.transform(benchmark_frame)  # (T, F)
    logger.info(
        "built features X%s and G%s (warmup=%d rows)",
        features.shape,
        index_features.shape,
        builder.warmup,
    )
    return features, index_features, builder.warmup
