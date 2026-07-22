"""End-to-end dataset build: sections 1-4 wired together.

    roster -> download -> screen -> universe -> align/clean -> features -> splits

The whole thing is deterministic given the config, so the result is cached on
disk under a fingerprint of the config's data-relevant fields; a second call
with the same config just reloads it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DiffolioConfig
from ..utils import get_logger, read_json, write_json
from .clean import align_frames
from .download import download_benchmark, download_ohlcv
from .features import FeatureStats, build_feature_tensors
from .panel import MarketPanel
from .splits import SplitIndices, compute_split_bounds, make_splits
from .universe import (
    Universe,
    build_universe,
    load_candidate_roster,
    normalize_ticker,
    screen_candidates,
    subset_universe,
)

logger = get_logger(__name__)

_PANEL_DIR = "panel"
_UNIVERSE_FILE = "universe.json"
_SPLITS_FILE = "splits.json"
_REPORT_FILE = "build_report.json"
_CONFIG_FILE = "config.yaml"


@dataclass
class Dataset:
    """Everything the model side needs from the ingestion pipeline."""

    panel: MarketPanel
    splits: SplitIndices
    universe: Universe
    config: DiffolioConfig
    report: dict[str, Any]

    @property
    def n_assets(self) -> int:
        return self.panel.n_assets

    def describe(self) -> str:
        return f"{self.panel.describe()}\n{self.splits.describe(self.panel.calendar)}"


def default_output_dir(config: DiffolioConfig) -> Path:
    return Path("data/processed") / config.name


def build_dataset(
    config: DiffolioConfig,
    output_dir: str | Path | None = None,
    force: bool = False,
    force_download: bool = False,
    mmap: bool = False,
) -> Dataset:
    """Build (or reload) the aligned panel and splits for ``config``."""
    config.validate()
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(config)
    fingerprint = config.dataset_fingerprint()

    cached = _try_load(output_dir, config, fingerprint, mmap=mmap) if not force else None
    if cached is not None:
        logger.info("reusing cached dataset at %s (%s)", output_dir, fingerprint)
        return cached

    start, end = config.data.start, config.data.end
    cache_dir = config.data.cache_dir

    logger.info("building dataset %s [%s .. %s]", config.name, start, end)

    benchmark_frame = download_benchmark(
        config.universe.benchmark,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force=force_download,
        max_retries=config.data.download_max_retries,
    )

    candidates = _candidate_tickers(config, cache_dir, force_download)
    frames = download_ohlcv(
        candidates,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force=force_download,
        batch_size=config.data.download_batch_size,
        max_retries=config.data.download_max_retries,
    )
    if not frames:
        raise RuntimeError("no candidate returned any data; check the date range and tickers")

    screen = screen_candidates(
        frames,
        config.universe,
        start=start,
        end=end,
        reference_days=len(benchmark_frame),
    )
    universe = build_universe(screen, config.universe, start=start, end=end)

    aligned = align_frames(
        {t: frames[t] for t in universe.tickers},
        benchmark=benchmark_frame,
        data_config=config.data,
        universe_config=config.universe,
        start=start,
        end=end,
    )
    universe = subset_universe(universe, aligned.tickers)
    if universe.size < 2:
        raise RuntimeError(f"only {universe.size} asset(s) survived cleaning")

    panel, feature_stats = _assemble_panel(aligned, universe, config, fingerprint)
    splits = make_splits(panel, config)
    panel.metadata["splits"] = {
        name: {"start": s.start, "stop": s.stop, "n_samples": s.n_samples}
        for name, s in splits.items()
    }

    report = {
        "name": config.name,
        "fingerprint": fingerprint,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_candidates": len(candidates),
        "n_downloaded": len(frames),
        "screen": screen.reason_counts(),
        "cleaning": aligned.report.to_dict(),
        "universe": universe.to_dict(),
        "panel": {
            "T": panel.n_days,
            "N": panel.n_assets,
            "F": panel.n_features,
            "d": config.embed_dim,
            "start": str(panel.calendar[0].date()),
            "end": str(panel.calendar[-1].date()),
        },
        # The full tau lists live in splits.json; the report keeps the summary.
        "splits": {
            "lookback": splits.lookback,
            **{
                name: {k: v for k, v in split.to_dict().items() if k != "tau"}
                for name, split in splits.items()
            },
        },
        "feature_stats": {k: v.to_dict() for k, v in feature_stats.items()},
    }

    _persist(output_dir, config, panel, splits, universe, report, screen, aligned)
    logger.info("dataset ready:\n%s", f"{panel.describe()}\n{splits.describe(panel.calendar)}")
    return Dataset(panel=panel, splits=splits, universe=universe, config=config, report=report)


def load_dataset(
    output_dir: str | Path,
    mmap: bool = False,
) -> Dataset:
    """Load a previously built dataset directory."""
    output_dir = Path(output_dir)
    config = DiffolioConfig.from_yaml(output_dir / _CONFIG_FILE)
    dataset = _try_load(output_dir, config, config.dataset_fingerprint(), mmap=mmap)
    if dataset is None:
        raise FileNotFoundError(f"no complete dataset found at {output_dir}")
    return dataset


def _try_load(
    output_dir: Path,
    config: DiffolioConfig,
    fingerprint: str,
    mmap: bool,
) -> Dataset | None:
    panel_dir = output_dir / _PANEL_DIR
    required = [panel_dir, output_dir / _SPLITS_FILE, output_dir / _UNIVERSE_FILE]
    if not MarketPanel.exists(panel_dir) or not all(p.exists() for p in required):
        return None

    panel = MarketPanel.load(panel_dir, mmap=mmap)
    if panel.metadata.get("fingerprint") != fingerprint:
        logger.info(
            "cached dataset fingerprint %s != %s; rebuilding",
            panel.metadata.get("fingerprint"),
            fingerprint,
        )
        return None

    report_path = output_dir / _REPORT_FILE
    return Dataset(
        panel=panel,
        splits=SplitIndices.load(output_dir / _SPLITS_FILE),
        universe=Universe.load(output_dir / _UNIVERSE_FILE),
        config=config,
        report=read_json(report_path) if report_path.exists() else {},
    )


def _candidate_tickers(config: DiffolioConfig, cache_dir: str, force: bool) -> list[str]:
    roster = load_candidate_roster(config.universe.roster, cache_dir=cache_dir, force=force)
    forced = [normalize_ticker(t) for t in config.universe.include]
    excluded = {normalize_ticker(t) for t in config.universe.exclude}
    candidates = sorted({*roster, *forced} - excluded)
    logger.info("universe candidates: %d tickers", len(candidates))
    return candidates


def _assemble_panel(
    aligned,
    universe: Universe,
    config: DiffolioConfig,
    fingerprint: str,
) -> tuple[MarketPanel, dict[str, FeatureStats]]:
    tickers = list(universe.tickers)
    features, index_features, warmup = build_feature_tensors(
        aligned.frames, tickers, aligned.benchmark_frame, config.features
    )

    calendar = aligned.calendar
    open_prices = np.stack(
        [aligned.frames[t]["open"].to_numpy(dtype="float64") for t in tickers], axis=1
    )
    close_prices = np.stack(
        [aligned.frames[t]["close"].to_numpy(dtype="float64") for t in tickers], axis=1
    )
    observed = np.stack(
        [aligned.observed[t].to_numpy(dtype=bool) for t in tickers], axis=1
    )
    index_open = aligned.benchmark_frame["open"].to_numpy(dtype="float64")
    index_close = aligned.benchmark_frame["close"].to_numpy(dtype="float64")
    index_observed = aligned.benchmark_observed.to_numpy(dtype=bool)

    if warmup:
        logger.info("trimming %d warm-up rows consumed by the feature reference", warmup)
        sl = slice(warmup, None)
        calendar = calendar[sl]
        features = features[sl]
        index_features = index_features[sl]
        open_prices = open_prices[sl]
        close_prices = close_prices[sl]
        observed = observed[sl]
        index_open = index_open[sl]
        index_close = index_close[sl]
        index_observed = index_observed[sl]

    _assert_finite(features, "asset features")
    _assert_finite(index_features, "index features")

    feature_stats: dict[str, FeatureStats] = {}
    if config.features.standardize:
        train_end, _ = compute_split_bounds(
            len(calendar), config.split.train_frac, config.split.val_frac
        )
        asset_stats = FeatureStats.fit(features[:train_end], config.features.clip_sigma)
        index_stats = FeatureStats.fit(index_features[:train_end], config.features.clip_sigma)
        features = asset_stats.transform(features)
        index_features = index_stats.transform(index_features)
        feature_stats = {"assets": asset_stats, "index": index_stats}
        logger.info(
            "standardised features with statistics from the first %d training days", train_end
        )

    metadata = {
        "fingerprint": fingerprint,
        "config_name": config.name,
        "lookback": config.lookback,
        "embed_dim": config.embed_dim,
        "warmup_rows_trimmed": warmup,
        "feature_stats": {k: v.to_dict() for k, v in feature_stats.items()},
        "cleaning": aligned.report.to_dict(),
        "trading_days_per_year": config.universe.trading_days_per_year,
    }

    panel = MarketPanel.build(
        calendar=calendar,
        tickers=tickers,
        feature_names=config.features.columns,
        benchmark=universe.benchmark,
        features=features,
        index_features=index_features,
        open_prices=open_prices,
        close_prices=close_prices,
        index_open=index_open,
        index_close=index_close,
        observed=observed,
        index_observed=index_observed,
        metadata=metadata,
    )
    return panel, feature_stats


def _assert_finite(array: np.ndarray, what: str) -> None:
    if np.isfinite(array).all():
        return
    bad = ~np.isfinite(array)
    raise RuntimeError(
        f"{what} contain {int(bad.sum())} non-finite values after the warm-up trim "
        f"(first offending row: {int(np.argmax(bad.any(axis=tuple(range(1, bad.ndim)))))})"
    )


def _persist(
    output_dir: Path,
    config: DiffolioConfig,
    panel: MarketPanel,
    splits: SplitIndices,
    universe: Universe,
    report: dict[str, Any],
    screen,
    aligned,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.save(output_dir / _PANEL_DIR)
    splits.save(output_dir / _SPLITS_FILE)
    universe.save(output_dir / _UNIVERSE_FILE)
    write_json(output_dir / _REPORT_FILE, report)
    config.to_yaml(output_dir / _CONFIG_FILE)

    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    screen.report.to_csv(diagnostics / "screen.csv")
    if aligned.report.anomalies is not None:
        aligned.report.anomalies.to_csv(diagnostics / "anomalies.csv")
    pd.Series(aligned.report.dropped, name="reason").rename_axis("ticker").to_csv(
        diagnostics / "dropped.csv"
    )
