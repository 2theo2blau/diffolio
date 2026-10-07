"""Checkpoints written by the training loop, and rebuilding a model from one.

Two files per run (plan 11.4):

* ``best.pt`` - the weights at the best validation epoch, with everything
  section 12 needs to rebuild them without the training run: the model (the
  head's ``sigma_x`` buffer included), the objective (``W_p``), the
  diffusion schedule, the config, the asset order and the dataset
  fingerprint, plus the seed and the history up to that epoch (plan 11.6).
* ``last.pt`` - the latest epoch, plus the optimiser, LR-scheduler,
  early-stopping and generator states.  Resuming restores the data order
  and noise draws.  On CPU it is bit-identical to an uninterrupted run; on
  GPU, non-deterministic cuDNN/attention kernels make it no more
  reproducible than an uninterrupted run.  A resumed cosine LR schedule keeps
  its original ``T_max``, whatever ``max_epochs`` is now.

Both are plain dicts of tensors and Python primitives, so they load with
``torch.load(weights_only=True)``, and both are written atomically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple

import torch

from ..config import DiffolioConfig
from ..diffusion import DiffusionSchedule
from ..loss import DiffolioLoss
from ..model import DiffolioModel

__all__ = ["CHECKPOINT_FORMAT", "TrainedModel", "load_checkpoint", "load_trained", "save_checkpoint"]

#: Bumped whenever the checkpoint layout changes incompatibly.
CHECKPOINT_FORMAT = 1


class TrainedModel(NamedTuple):
    config: DiffolioConfig
    model: DiffolioModel
    schedule: DiffusionSchedule
    loss: DiffolioLoss
    checkpoint: dict[str, Any]


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """``torch.save`` to a temporary file, then rename over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, **payload}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} has checkpoint format {checkpoint.get('format')}, "
            f"expected {CHECKPOINT_FORMAT}"
        )
    return checkpoint


def load_trained(path: str | Path, device: str | torch.device = "cpu") -> TrainedModel:
    """Rebuild the trained model, schedule and objective from a checkpoint."""
    checkpoint = load_checkpoint(path, map_location=device)
    config = DiffolioConfig.from_dict(checkpoint["config"])
    n_assets, sigma_x = int(checkpoint["n_assets"]), float(checkpoint["sigma_x"])

    model = DiffolioModel.from_config(config, n_assets, sigma_x)
    model.load_state_dict(checkpoint["model"])
    schedule = DiffusionSchedule.from_config(config.diffusion, sigma_x)
    schedule.load_state_dict(checkpoint["schedule"])
    loss = DiffolioLoss.from_config(config, n_assets)
    loss.load_state_dict(checkpoint["loss"])
    for module in (model, schedule, loss):
        module.to(device).eval()
    return TrainedModel(config, model, schedule, loss, checkpoint)
