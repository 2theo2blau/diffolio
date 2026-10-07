#!/usr/bin/env python
"""Sample portfolios with a trained Diffolio model (plan section 12).

Runs Algorithm 2 over every decision step of a split, at every risk level,
and writes the weights to ``<run>/samples/<split>`` (``<split>_nrg`` with
guidance off).  Sampling settings come from the checkpoint's config, then
the ``sampling`` section of ``--config`` if given, then ``--set``.

Examples::

    python scripts/sample.py -r runs/us_sp500
    python scripts/sample.py -r runs/us_sp500 --set sampling.guidance=false
    python scripts/sample.py -r runs/us_sp500 --split val --set sampling.num_samples=10
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from diffolio.config import DiffolioConfig, merge_overrides  # noqa: E402
from diffolio.data.pipeline import load_dataset  # noqa: E402
from diffolio.sampling import sample_split, samples_dirname  # noqa: E402
from diffolio.training import load_trained, resolve_device  # noqa: E402
from diffolio.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--run", required=True, help="training run directory (holds best.pt)")
    parser.add_argument("--checkpoint", default="best.pt", help="checkpoint file inside the run")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("-c", "--config", default=None, help="take the sampling section from this YAML")
    parser.add_argument("-d", "--dataset-dir", default=None, help="the built dataset (default: the one recorded in the checkpoint)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="sampling.key=value",
        help="override a sampling setting (repeatable)",
    )
    parser.add_argument("-o", "--output-dir", default=None, help="default <run>/samples/<split>[_nrg]")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    if any(not o.startswith("sampling.") for o in args.overrides):
        print("--set only changes sampling.* here; the model is fixed by the checkpoint", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    run = Path(args.run)
    trained = load_trained(run / args.checkpoint, device=device)
    config = copy.deepcopy(trained.config)
    if args.config:
        config.sampling = DiffolioConfig.from_yaml(args.config).sampling
    config = merge_overrides(config, args.overrides)
    config.validate()

    dataset_dir = args.dataset_dir or trained.checkpoint["metadata"].get("dataset_root")
    dataset = load_dataset(dataset_dir)
    samples = sample_split(trained, dataset, args.split, config.sampling, device=device)

    output_dir = Path(args.output_dir) if args.output_dir else run / "samples" / samples_dirname(args.split, config.sampling)
    samples.save(output_dir)

    # Ex-ante proxy risk of the final weights per level: it should rise with gamma.
    vol = np.sqrt(np.asarray(samples.risk, dtype=np.float64) * config.universe.trading_days_per_year)
    print(f"\n{args.split}: {samples.tau.size} steps x {config.sampling.num_samples} samples, "
          f"guidance {'on' if config.sampling.guidance else 'off'}")
    for gamma in range(vol.shape[1]):
        print(f"  gamma={gamma}  k={samples.metadata['risk_sizes'][gamma]:3d}  "
              f"annualised proxy vol {vol[:, gamma].mean():.4f}")
    print(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
