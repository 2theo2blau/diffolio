"""Central configuration for the Diffolio pipeline.

Every hyperparameter that is shared between components lives here so that the
data pipeline and the model can never drift apart.  In particular ``d`` (the
encoder width) is *derived* as ``L * F`` rather than stored, per the plan's
cross-cutting notes.

Only the ``universe`` / ``data`` / ``features`` / ``window`` / ``split``
sections are consumed by the ingestion components (plan sections 1-4); the
remaining sections are declared up front with the paper's defaults so later
components read from the same object.
"""

import copy
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

__all__ = [
    "UniverseConfig",
    "DataConfig",
    "FeatureConfig",
    "WindowConfig",
    "SplitConfig",
    "DiffusionConfig",
    "ModelConfig",
    "TrainingConfig",
    "DiffolioConfig",
]


CalendarMode = Literal["benchmark", "union", "intersection"]
PriceNormalization = Literal["rolling_ref", "prev_close", "none"]
VolumeTransform = Literal["log_rel", "log", "none"]


@dataclass
class UniverseConfig:
    """Section 1 - which assets make up the investable universe."""

    market: str = "US"
    #: Candidate roster: ``sp500``, ``nasdaq100``, ``file:<path>`` (one ticker
    #: per line / a ``ticker`` column) or ``list:AAPL,MSFT,...``.
    roster: str = "sp500"
    benchmark: str = "^GSPC"
    #: Trading days per year, used by later annualisation (365 for crypto).
    trading_days_per_year: int = 252
    #: Keep at most this many assets, ranked by median dollar volume.  ``None``
    #: keeps every eligible asset.  The paper uses N = 224 for the U.S. market.
    target_size: int | None = 224
    #: Reject an asset unless it has this many assets' worth of history.
    min_price: float = 1.0
    min_avg_dollar_volume: float = 5.0e6
    #: Fraction of the master calendar an asset may be missing before it is
    #: dropped outright (plan 3.2).
    max_missing_frac: float = 0.05
    #: Require the first bar on/before ``data.start`` and the last on/after
    #: ``data.end`` -- i.e. continuous listing across the whole study period.
    require_full_history: bool = True
    exclude: list[str] = field(default_factory=list)
    #: Tickers that are always kept (subject to data availability).
    include: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Sections 2-3 - acquisition and cleaning."""

    start: str = "2010-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data/cache"
    #: How the master trading calendar is derived (plan 3.1).  ``benchmark``
    #: uses the index's own trading days, ``intersection`` keeps only days on
    #: which every asset traded, ``union`` keeps every observed day.
    calendar_mode: CalendarMode = "benchmark"
    #: Sanity checks (plan 3.3).
    max_daily_jump: float = 10.0
    max_constant_run: int = 20
    #: Drop an asset whose longest constant-price stretch exceeds
    #: ``max_constant_run`` or that trips the jump/non-positive-price checks.
    drop_anomalous_assets: bool = True
    download_batch_size: int = 50
    download_max_retries: int = 3

    def __post_init__(self) -> None:
        # YAML parses bare 2010-01-01 into a date; keep dates as ISO strings.
        self.start = str(self.start)
        self.end = str(self.end)


@dataclass
class FeatureConfig:
    """Section 3.4 - the F price features and their per-asset normalisation."""

    columns: list[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    #: ``rolling_ref`` divides OHLC by a trailing mean close (causal, so it
    #: leaks nothing across the split boundary); ``prev_close`` divides by the
    #: previous close; ``none`` leaves raw prices.
    price_normalization: PriceNormalization = "rolling_ref"
    ref_window: int = 20
    #: ``log_rel`` is ``log1p(volume) - trailing mean log1p(volume)``.
    volume_transform: VolumeTransform = "log_rel"
    #: Optional second stage: per-feature z-scoring with statistics fitted on
    #: the training split only (plan 4.3 / validation discipline).
    standardize: bool = True
    #: Clip standardized features to +/- this many sigmas (0 disables).  This
    #: mainly bounds the volume feature on forward-filled bars, where a zero
    #: volume is a genuine but extreme observation.
    clip_sigma: float = 10.0


@dataclass
class WindowConfig:
    """Section 5 - the look-up window shared by data and model."""

    lookback: int = 256  # L


@dataclass
class SplitConfig:
    """Section 4 - chronological 7:1:2 partition."""

    train_frac: float = 0.7
    val_frac: float = 0.1  # test takes the remainder
    #: A decision step is only usable if at least this fraction of assets have
    #: an observed (not forward-filled) open on both tau and tau+1.
    min_valid_target_frac: float = 0.95
    #: ...and if at most this fraction of the (N, L) window cells were filled.
    max_window_fill_frac: float = 0.05


@dataclass
class DiffusionConfig:
    """Section 7 - consumed by later components."""

    num_steps: int = 500  # T
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    gamma_max: int = 5


@dataclass
class ModelConfig:
    """Sections 8-9 - consumed by later components."""

    hidden_dim: int = 128  # d_hid, the LSTM width (the only free width)
    tcn_kernel_size: int = 3
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2])
    time_embedding_activation: Literal["sigmoid", "silu", "gelu"] = "sigmoid"
    mlp_layers: int = 3


@dataclass
class TrainingConfig:
    """Sections 10-11 - consumed by later components."""

    batch_size: int = 128
    learning_rate: float = 1.0e-3
    max_epochs: int = 400
    patience: int = 20
    aux_weight: float = 1.0  # lambda
    seed: int = 0


@dataclass
class DiffolioConfig:
    name: str = "us_sp500"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # -- derived quantities -------------------------------------------------
    @property
    def lookback(self) -> int:
        """L."""
        return self.window.lookback

    @property
    def num_features(self) -> int:
        """F."""
        return len(self.features.columns)

    @property
    def embed_dim(self) -> int:
        """d = L * F (plan section 8: derived, never configured directly)."""
        return self.lookback * self.num_features

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffolioConfig":
        return _from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiffolioConfig":
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return cls.from_dict(payload)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def dataset_fingerprint(self) -> str:
        """Hash of every field that changes the built dataset.

        Used to invalidate the on-disk panel cache; model/training settings are
        deliberately excluded so that re-tuning the network does not force a
        rebuild.
        """
        payload = {
            "universe": dataclasses.asdict(self.universe),
            "data": {
                k: v
                for k, v in dataclasses.asdict(self.data).items()
                if k not in {"cache_dir", "download_batch_size", "download_max_retries"}
            },
            "features": dataclasses.asdict(self.features),
            "window": dataclasses.asdict(self.window),
            "split": dataclasses.asdict(self.split),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        if self.lookback < 2:
            raise ValueError("window.lookback must be >= 2")
        if not self.features.columns:
            raise ValueError("features.columns must not be empty")
        missing = {"open", "close"} - set(self.features.columns)
        if missing:
            raise ValueError(
                f"features.columns must contain {sorted(missing)}: open prices "
                "define the return target and close prices the price reference"
            )
        if not 0.0 < self.split.train_frac < 1.0:
            raise ValueError("split.train_frac must lie in (0, 1)")
        if not 0.0 < self.split.val_frac < 1.0:
            raise ValueError("split.val_frac must lie in (0, 1)")
        if self.split.train_frac + self.split.val_frac >= 1.0:
            raise ValueError("split.train_frac + split.val_frac must leave room for test")
        if self.data.start >= self.data.end:
            raise ValueError("data.start must precede data.end")
        if self.universe.target_size is not None and self.universe.target_size < 2:
            raise ValueError("universe.target_size must be >= 2")


def _from_dict(cls: type, data: Any) -> Any:
    """Recursively build (nested) dataclasses from plain dicts."""
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _from_dict(hint, value)
        elif isinstance(value, (list, tuple)):
            kwargs[key] = list(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def merge_overrides(config: DiffolioConfig, overrides: Sequence[str]) -> DiffolioConfig:
    """Apply ``section.key=value`` CLI overrides to a copy of ``config``."""
    merged = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"malformed override {override!r}, expected section.key=value")
        dotted, raw = override.split("=", 1)
        target: Any = merged
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section {part!r} in {override!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config key {leaf!r} in {override!r}")
        value = yaml.safe_load(raw)
        if isinstance(getattr(target, leaf), str) and not isinstance(value, str):
            value = raw  # e.g. dates, which YAML would otherwise coerce
        setattr(target, leaf, value)
    return merged
