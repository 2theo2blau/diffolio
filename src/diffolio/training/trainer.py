"""Section 11 - the training loop (Algorithm 1).

Each step draws, for every sample in a batch of training steps ``tau``, a risk
level ``gamma ~ U{0..G-1}``, a timestep ``t ~ U{1..T}`` and noise
``eps ~ N(0, I)``, forms ``x_t = sqrt(abar_t) x_0^(tau, gamma) + sqrt(1 -
abar_t) sigma_x eps`` (plan 11.1) and takes one Adam step on ``L''`` (section
10).  Validation runs the same loss without gradients after every epoch.  The
best epoch is checkpointed, and training stops after ``training.patience``
epochs without improvement or at ``training.max_epochs`` (plan 11.4-11.5).

Choices the paper leaves open, recorded here:

* **Data path.**  The panel (~17 MB at N=224) and the targets are moved to
  the device once and every batch is gathered there by
  :func:`~diffolio.data.windows.stack_windows`, so there is no DataLoader.
* **Epochs.**  An epoch is one shuffled pass over the training steps, so each
  ``tau`` is drawn uniformly, as in plan 11.1, and each is seen once per
  epoch.  The last batch is partial (2372 = 18 x 128 + 68 on the real build).
* **Random streams.**  Three explicit generators, all derived from
  ``training.seed``:
  * a CPU generator shuffles;
  * a device generator draws ``gamma``, ``t`` and ``eps``;
  * validation re-seeds its own generator on every call.
  Validation therefore never moves the training stream, and every epoch is
  scored on the same draws.
* **Validation.**  Every validation step is scored at every risk level with
  ``training.val_repeats`` fixed ``(t, eps)`` draws each.  The encoder runs
  once per window.  A single random draw per sample over 120 validation days
  would be too noisy for early stopping to mean anything.
* **Baseline.**  Epoch 0 is a validation pass of the untrained model.  It
  seeds early stopping, so ``best.pt`` always exists.
* **Precision.**  fp32 by default; see ``TrainingConfig.precision``.  Under
  autocast the losses are still computed in fp32.
"""

from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from ..config import DiffolioConfig
from ..data.panel import PanelTensors
from ..data.pipeline import Dataset
from ..data.targets import PortfolioTargets, build_targets
from ..data.windows import stack_windows
from ..diffusion import DiffusionSchedule, fit_schedule
from ..loss import DiffolioLoss, LossOutput
from ..model import DiffolioModel
from ..utils import get_logger, write_json
from .checkpoint import load_checkpoint, save_checkpoint

logger = get_logger(__name__)

__all__ = [
    "EarlyStopping",
    "LossMetrics",
    "TrainBatch",
    "Trainer",
    "resolve_device",
    "seed_everything",
]

BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
HISTORY_FILE = "history.json"
CONFIG_FILE = "config.yaml"

#: Offsets of the per-stream seeds from ``training.seed``.
_TRAIN_STREAM = 1
_VAL_STREAM = 2

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices) - plan 11.6."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TrainBatch(NamedTuple):
    """One Algorithm-1 batch, everything on the device."""

    h: torch.Tensor  # (B, N, L, F)
    g: torch.Tensor  # (B, L, F)
    r: torch.Tensor  # (B, N) realised returns at tau
    r_valid: torch.Tensor  # (B, N) bool
    x0: torch.Tensor  # (B, N) x_0^(tau, gamma) at each sample's gamma
    x_t: torch.Tensor  # (B, N) noised x0
    t: torch.Tensor  # (B,) int64 in 1..T
    gamma: torch.Tensor  # (B,) int64 in 0..G-1


class LossMetrics(NamedTuple):
    total: float  # L''
    denoise: float
    aux_return: float

    def get(self, name: str) -> float:
        return getattr(self, name)


class _LossSums:
    """Sample-weighted running means of the three loss terms."""

    def __init__(self) -> None:
        self.n = 0
        self.sums: list[torch.Tensor] | None = None

    def add(self, out: LossOutput, n: int) -> None:
        terms = [out.total.detach() * n, out.denoise.detach() * n, out.aux_return.detach() * n]
        self.sums = terms if self.sums is None else [a + b for a, b in zip(self.sums, terms)]
        self.n += n

    def mean(self) -> LossMetrics:
        if self.sums is None:
            raise RuntimeError("no batches were accumulated")
        return LossMetrics(*(float(s) / self.n for s in self.sums))


class EarlyStopping:
    """Tracks the best validation value; stops after ``patience`` epochs
    without a strict improvement (plan 11.5).  A NaN never improves."""

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = int(patience)
        self.best = math.inf
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record an epoch's value; ``True`` if it is a new best."""
        if value < self.best:
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = float(state["best"])
        self.best_epoch = state["best_epoch"]
        self.bad_epochs = int(state["bad_epochs"])


class Trainer:
    """Trains a :class:`DiffolioModel` and its :class:`DiffolioLoss` jointly.

    ``train_tau`` / ``val_tau`` are decision steps of ``tensors``; ``targets``
    must cover them.  ``schedule`` carries ``sigma_x``, which must have been
    fitted on the training split (:meth:`from_dataset` does this).  With an
    ``output_dir``, checkpoints and the history are written there.
    """

    def __init__(
        self,
        config: DiffolioConfig,
        tensors: PanelTensors,
        targets: PortfolioTargets,
        train_tau: Sequence[int] | np.ndarray,
        val_tau: Sequence[int] | np.ndarray,
        schedule: DiffusionSchedule,
        output_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        metadata: dict[str, Any] | None = None,
    ):
        config.validate()
        tc = config.training
        self.config = config
        self.device = resolve_device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.metadata = dict(metadata or {})

        self.train_tau = np.ascontiguousarray(train_tau, dtype=np.int64)
        self.val_tau = np.ascontiguousarray(val_tau, dtype=np.int64)
        if not self.train_tau.size or not self.val_tau.size:
            raise ValueError("training needs at least one training and one validation step")
        n_assets = int(tensors.features.shape[1])
        if targets.n_assets != n_assets:
            raise ValueError(f"targets cover N={targets.n_assets}, the panel has N={n_assets}")
        if targets.gamma_max != config.diffusion.gamma_max:
            raise ValueError(
                f"targets have gamma_max={targets.gamma_max}, the config "
                f"{config.diffusion.gamma_max}"
            )
        if schedule.num_steps != config.diffusion.num_steps:
            raise ValueError(
                f"schedule has T={schedule.num_steps}, the config {config.diffusion.num_steps}"
            )

        self.tensors = tensors.to(self.device)
        self.train_x0 = self._targets_for(targets, self.train_tau)
        self.val_x0 = self._targets_for(targets, self.val_tau)
        self.schedule = schedule.to(self.device)
        self.sigma_x = float(schedule.sigma_x)

        seed_everything(tc.seed)  # before init, so the initial weights follow the seed
        self.model = DiffolioModel.from_config(config, n_assets, self.sigma_x).to(self.device)
        self.loss = DiffolioLoss.from_config(config, n_assets).to(self.device)
        # W_p belongs to the objective; it trains with the model (section 10).
        self.parameters = [*self.model.parameters(), *self.loss.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=tc.learning_rate)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tc.max_epochs)
            if tc.lr_schedule == "cosine"
            else None
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=tc.precision == "fp16")
        self.stopper = EarlyStopping(tc.patience)

        self.shuffle_generator = torch.Generator().manual_seed(tc.seed)
        self.train_generator = torch.Generator(self.device).manual_seed(tc.seed + _TRAIN_STREAM)
        self.epoch = 0
        self.history: list[dict[str, Any]] = []

        if self.output_dir is not None:
            config.to_yaml(self.output_dir / CONFIG_FILE)
        logger.info(
            "trainer: %d train / %d val steps, N=%d, %.2fM parameters (W_p included), "
            "device=%s, precision=%s",
            self.train_tau.size,
            self.val_tau.size,
            n_assets,
            sum(p.numel() for p in self.parameters) / 1e6,
            self.device,
            tc.precision,
        )

    @classmethod
    def from_dataset(
        cls,
        config: DiffolioConfig,
        dataset: Dataset,
        targets: PortfolioTargets | None = None,
        output_dir: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> "Trainer":
        """Train on ``dataset``'s training split, validate on its validation
        split, with ``sigma_x`` fitted on the training split."""
        fingerprint = dataset.panel.metadata.get("fingerprint")
        if config.dataset_fingerprint() != fingerprint:
            raise ValueError(
                "the config's data sections do not match the dataset it is trained on "
                f"({config.dataset_fingerprint()} vs {fingerprint})"
            )
        targets = targets if targets is not None else build_targets(dataset)
        schedule = fit_schedule(config.diffusion, targets, dataset.splits.train)
        metadata = {
            "tickers": list(dataset.panel.tickers),
            "fingerprint": fingerprint,
            "dataset_root": str(dataset.root) if dataset.root is not None else None,
        }
        return cls(
            config,
            dataset.panel.torch(),
            targets,
            dataset.splits.train.tau,
            dataset.splits.val.tau,
            schedule,
            output_dir=output_dir,
            device=device,
            metadata=metadata,
        )

    # -- one step -------------------------------------------------------------
    def make_batch(self, positions: torch.Tensor) -> TrainBatch:
        """Algorithm 1's draws for the training steps at ``positions``
        (indices into ``train_tau``), from the training generator."""
        positions = torch.as_tensor(positions, dtype=torch.int64)
        window = stack_windows(self.tensors, self.train_tau[positions.numpy()], self.config.lookback)
        b, gen = positions.numel(), self.train_generator
        gamma = torch.randint(
            0, self.config.diffusion.gamma_max, (b,), generator=gen, device=self.device
        )
        t = self.schedule.sample_timesteps(b, generator=gen)
        x0 = self.train_x0[positions.to(self.device), gamma]
        x_t = self.schedule.q_sample(x0, t, generator=gen)
        return TrainBatch(window.h, window.g, window.r, window.r_valid, x0, x_t, t, gamma)

    def train_step(self, batch: TrainBatch) -> LossOutput:
        """``zero_grad -> backward -> step`` on ``L''`` (plan 11.3)."""
        self.model.train()
        self.loss.train()
        with self._autocast():
            out = self.model(batch.x_t, batch.t, batch.h, batch.g, batch.gamma)
        losses = self.loss(
            out.x_hat.float(), batch.x0, out.encoding.z_merged.float(), batch.r, batch.r_valid
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(losses.total).backward()
        if self.config.training.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters, self.config.training.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return losses

    def train_epoch(self) -> LossMetrics:
        """One shuffled pass over the training steps."""
        order = torch.randperm(self.train_tau.size, generator=self.shuffle_generator)
        sums = _LossSums()
        for start in range(0, order.numel(), self.config.training.batch_size):
            positions = order[start : start + self.config.training.batch_size]
            sums.add(self.train_step(self.make_batch(positions)), positions.numel())
        metrics = sums.mean()
        if not all(math.isfinite(v) for v in metrics):
            raise FloatingPointError(f"training loss is not finite at epoch {self.epoch + 1}: {metrics}")
        return metrics

    @torch.no_grad()
    def evaluate(self) -> LossMetrics:
        """``L''`` on the validation steps, at every risk level, with
        ``val_repeats`` fixed ``(t, eps)`` draws each (plan 11.4)."""
        self.model.eval()
        self.loss.eval()
        tc, diffusion = self.config.training, self.config.diffusion
        gen = torch.Generator(self.device).manual_seed(tc.seed + _VAL_STREAM)
        k, g_max = tc.val_repeats, diffusion.gamma_max
        sums = _LossSums()
        for start in range(0, self.val_tau.size, tc.batch_size):
            stop = min(start + tc.batch_size, self.val_tau.size)
            window = stack_windows(self.tensors, self.val_tau[start:stop], self.config.lookback)
            b = stop - start
            gamma = torch.arange(g_max, device=self.device).expand(b, k, g_max)
            t = torch.randint(
                1, diffusion.num_steps + 1, (b, k, g_max), generator=gen, device=self.device
            )
            x0 = self.val_x0[start:stop, None].expand(b, k, g_max, -1)  # (b, K, G, N)
            x_t = self.schedule.q_sample(x0, t, generator=gen)
            with self._autocast():
                encoding = self.model.encode(window.h, window.g)
                x_hat = self.model.denoise(x_t, t, gamma, encoding.z_merged)
            losses = self.loss(
                x_hat.float(), x0, encoding.z_merged.float(), window.r, window.r_valid
            )
            sums.add(losses, b)
        return sums.mean()

    # -- the loop ---------------------------------------------------------------
    def fit(self, max_epochs: int | None = None) -> list[dict[str, Any]]:
        """Train until early stopping or ``max_epochs`` (default
        ``training.max_epochs``); returns the per-epoch history.  Continues
        from ``self.epoch`` after :meth:`resume`."""
        tc = self.config.training
        max_epochs = tc.max_epochs if max_epochs is None else int(max_epochs)
        if self.epoch == 0 and not self.history:
            start = time.perf_counter()
            self._end_epoch(None, self.evaluate(), lr=None, seconds=time.perf_counter() - start)

        while self.epoch < max_epochs and not self.stopper.should_stop:
            start = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train = self.train_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            val = self.evaluate()
            self.epoch += 1
            self._end_epoch(train, val, lr=lr, seconds=time.perf_counter() - start)

        if self.stopper.should_stop:
            logger.info(
                "early stop at epoch %d: no improvement in %s for %d epochs (best %.6g at epoch %s)",
                self.epoch,
                tc.monitor,
                tc.patience,
                self.stopper.best,
                self.stopper.best_epoch,
            )
        return self.history

    def _end_epoch(
        self, train: LossMetrics | None, val: LossMetrics, lr: float | None, seconds: float
    ) -> None:
        improved = self.stopper.update(val.get(self.config.training.monitor), self.epoch)
        record: dict[str, Any] = {"epoch": self.epoch, "lr": lr}
        for prefix, metrics in (("train", train), ("val", val)):
            for name in LossMetrics._fields:
                record[f"{prefix}_{name}"] = None if metrics is None else metrics.get(name)
        record.update(seconds=seconds, best=improved)
        self.history.append(record)

        if self.output_dir is not None:
            if improved:
                save_checkpoint(self.output_dir / BEST_CHECKPOINT, self._payload(full=False))
            save_checkpoint(self.output_dir / LAST_CHECKPOINT, self._payload(full=True))
            write_json(self.output_dir / HISTORY_FILE, self.history)

        train_text = (
            "train -" if train is None else
            f"train L''={train.total:.6g} (mse {train.denoise:.4g}, aux {train.aux_return:+.4g})"
        )
        logger.info(
            "epoch %d/%d  %s  val L''=%.6g (mse %.4g, aux %+.4g)  %.1fs%s",
            self.epoch,
            self.config.training.max_epochs,
            train_text,
            val.total,
            val.denoise,
            val.aux_return,
            seconds,
            "  *best" if improved else "",
        )

    # -- checkpoints -----------------------------------------------------------
    def _payload(self, full: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "loss": self.loss.state_dict(),
            "schedule": self.schedule.state_dict(),
            "config": self.config.to_dict(),
            "n_assets": self.model.n_assets,
            "sigma_x": self.sigma_x,
            "seed": self.config.training.seed,
            "monitor": self.config.training.monitor,
            "best_value": self.stopper.best,
            "best_epoch": self.stopper.best_epoch,
            "history": list(self.history),
            "metadata": self.metadata,
        }
        if full:
            payload.update(
                optimizer=self.optimizer.state_dict(),
                lr_scheduler=None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                stopper=self.stopper.state_dict(),
                generators={
                    "shuffle": self.shuffle_generator.get_state(),
                    "train": self.train_generator.get_state(),
                },
            )
        return payload

    def resume(self, path: str | Path) -> None:
        """Restore a ``last.pt`` so :meth:`fit` continues where it stopped."""
        checkpoint = load_checkpoint(path, map_location=self.device)
        if "optimizer" not in checkpoint:
            raise ValueError(f"{path} is a weights-only checkpoint; resume from {LAST_CHECKPOINT}")
        if int(checkpoint["n_assets"]) != self.model.n_assets:
            raise ValueError(f"{path} was trained on N={checkpoint['n_assets']} assets")
        if not math.isclose(float(checkpoint["sigma_x"]), self.sigma_x, rel_tol=1e-6):
            raise ValueError(
                f"{path} was trained with sigma_x={checkpoint['sigma_x']}, this run has "
                f"{self.sigma_x}"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.loss.load_state_dict(checkpoint["loss"])
        self.schedule.load_state_dict(checkpoint["schedule"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None and checkpoint["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.stopper.load_state_dict(checkpoint["stopper"])
        # Generator states must be CPU byte tensors, whatever the map_location.
        self.shuffle_generator.set_state(checkpoint["generators"]["shuffle"].cpu())
        self.train_generator.set_state(checkpoint["generators"]["train"].cpu())
        self.epoch = int(checkpoint["epoch"])
        self.history = list(checkpoint["history"])
        logger.info("resumed from %s at epoch %d", path, self.epoch)

    # -- internals -------------------------------------------------------------
    def _targets_for(self, targets: PortfolioTargets, tau: np.ndarray) -> torch.Tensor:
        """``(S, G, N)`` risk targets aligned with ``tau``, on the device."""
        rows = targets.rows(tau)
        return torch.from_numpy(np.array(targets.targets[rows], dtype=np.float32)).to(self.device)

    def _autocast(self):
        dtype = _AUTOCAST_DTYPES.get(self.config.training.precision)
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=dtype)
