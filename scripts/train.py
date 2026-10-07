#!/usr/bin/env python
"""Train Diffolio (plan section 11) on a built dataset.

The dataset must have been built with ``scripts/build_dataset.py`` from the
same config's data sections; it is reloaded from ``data/processed/<name>``.

Examples::

    python scripts/train.py -c configs/us_sp500.yaml
    python scripts/train.py -c configs/us_sp500.yaml --set training.learning_rate=5e-4
    python scripts/train.py -c configs/us_sp500.yaml --resume
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
from diffolio.data.targets import build_targets  # noqa: E402
from diffolio.training import Trainer  # noqa: E402
from diffolio.training.trainer import LAST_CHECKPOINT  # noqa: E402
from diffolio.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="path to a YAML config")
    parser.add_argument("-d", "--dataset-dir", default=None, help="the built dataset (default data/processed/<name>)")
    parser.add_argument("-o", "--output-dir", default=None, help="run directory (default runs/<name>)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="override a config value (repeatable)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    parser.add_argument("--resume", action="store_true", help=f"continue from <output-dir>/{LAST_CHECKPOINT}")
    parser.add_argument("--overwrite", action="store_true", help="start afresh over an existing run")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = merge_overrides(DiffolioConfig.from_yaml(args.config), args.overrides)
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / config.name
    last = output_dir / LAST_CHECKPOINT
    if last.exists() and not (args.resume or args.overwrite):
        print(f"{last} exists; pass --resume to continue it or --overwrite to start afresh", file=sys.stderr)
        return 1
    if args.resume and not last.exists():
        print(f"nothing to resume: {last} does not exist", file=sys.stderr)
        return 1

    # Reuses the cached build when the config's data sections match it.
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_output_dir(config)
    dataset = build_dataset(config, output_dir=dataset_dir)
    trainer = Trainer.from_dataset(
        config, dataset, build_targets(dataset), output_dir=output_dir, device=args.device
    )
    if args.resume:
        trainer.resume(last)
    trainer.fit()

    best = trainer.stopper
    print(f"\nbest validation {config.training.monitor} {best.best:.6g} at epoch {best.best_epoch}")
    print(f"checkpoints and history in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
