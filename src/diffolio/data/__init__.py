"""Data ingestion for Diffolio (plan sections 1-4).

The pipeline is a straight line::

    universe.load_candidate_roster  ->  download.download_ohlcv
      ->  universe.screen_candidates / build_universe
      ->  clean.align_frames  ->  features.build_feature_tensors
      ->  panel.MarketPanel   ->  splits.make_splits

``pipeline.build_dataset`` runs all of it and caches the result.
"""

from .clean import AlignedFrames, CleaningReport, align_frames, build_calendar, detect_anomalies
from .download import PriceCache, download_benchmark, download_ohlcv
from .features import FeatureBuilder, FeatureStats, build_feature_tensors
from .panel import MarketPanel, PanelTensors, compute_open_to_open_returns
from .pipeline import Dataset, build_dataset, default_output_dir, load_dataset
from .splits import Split, SplitIndices, compute_split_bounds, make_splits
from .universe import (
    ScreenResult,
    Universe,
    build_universe,
    load_candidate_roster,
    normalize_ticker,
    screen_candidates,
)

__all__ = [
    "AlignedFrames",
    "CleaningReport",
    "Dataset",
    "FeatureBuilder",
    "FeatureStats",
    "MarketPanel",
    "PanelTensors",
    "PriceCache",
    "ScreenResult",
    "Split",
    "SplitIndices",
    "Universe",
    "align_frames",
    "build_calendar",
    "build_dataset",
    "build_feature_tensors",
    "build_universe",
    "compute_open_to_open_returns",
    "compute_split_bounds",
    "default_output_dir",
    "detect_anomalies",
    "download_benchmark",
    "download_ohlcv",
    "load_candidate_roster",
    "load_dataset",
    "make_splits",
    "normalize_ticker",
    "screen_candidates",
]
