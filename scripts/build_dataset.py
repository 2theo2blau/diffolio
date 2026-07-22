#!/usr/bin/env python
"""Build the Diffolio dataset (plan sections 1-4) from a config file.

Examples::

    python scripts/build_dataset.py --config configs/us_sp500.yaml
    python scripts/build_dataset.py -c configs/us_sp500.yaml \
        --set data.end=2020-12-31 --set universe.target_size=50 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from diffolio.config import DiffolioConfig, merge_overrides  # noqa: E402
from diffolio.data.pipeline import build_dataset, default_output_dir  # noqa: E402
from diffolio.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="path to a YAML config")
    parser.add_argument("-o", "--output-dir", default=None, help="where to write the dataset")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="override a config value (repeatable)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument(
        "--force-download", action="store_true", help="ignore the raw price cache too"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = merge_overrides(DiffolioConfig.from_yaml(args.config), args.overrides)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)

    dataset = build_dataset(
        config,
        output_dir=output_dir,
        force=args.force,
        force_download=args.force_download,
    )

    print()
    print(dataset.describe())
    print(f"\nwritten to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
