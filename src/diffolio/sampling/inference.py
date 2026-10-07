"""Section 12 at the split level: sample every decision step of a split.

:func:`sample_split` runs Algorithm 2 over a split of the built dataset with a
trained model and returns a :class:`SampleSet`.  The sample set is stored
under ``runs/<name>/samples/<split>`` (``<split>_nrg`` without guidance)::

    tau      (S,)           int64    decision steps
    weights  (S, G, K, N)   float32  w_hat = f_gamma(x_0); K samples per (tau, gamma)
    risk     (S, G, K)      float32  rho'(w_hat) = w_hat^T Sigma_hat_tau w_hat
    meta.json                        settings, seed, checkpoint, tickers, guidance trace

Nothing is averaged away: section 13's backtests, the foresight variant
DF-gamma* and the risk-conformity diagnostics all need every sample at every
risk level.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import SamplingConfig
from ..data.pipeline import Dataset
from ..data.windows import stack_windows
from ..training.checkpoint import TrainedModel
from ..training.trainer import resolve_device
from ..utils import get_logger, read_json, write_json
from .covariance import CovarianceModel
from .sampler import RiskGuidedSampler, proxy_risk

logger = get_logger(__name__)

__all__ = ["SampleSet", "check_compatible", "sample_split", "samples_dirname"]

_ARRAYS = ("tau", "weights", "risk")


@dataclass
class SampleSet:
    tau: np.ndarray  # (S,) int64
    weights: np.ndarray  # (S, G, K, N) float32
    risk: np.ndarray  # (S, G, K) float32
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in _ARRAYS:
            np.save(directory / f"{name}.npy", getattr(self, name))
        write_json(directory / "meta.json", self.metadata)
        logger.info("saved %d x %s samples to %s", self.tau.size, self.weights.shape[1:3], directory)

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = False) -> "SampleSet":
        directory = Path(directory)
        mode = "r" if mmap else None
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode=mode) for name in _ARRAYS}
        return cls(metadata=read_json(directory / "meta.json"), **arrays)


def samples_dirname(split: str, settings: SamplingConfig) -> str:
    return split if settings.guidance else f"{split}_nrg"


def check_compatible(trained: TrainedModel, dataset: Dataset) -> None:
    """The checkpoint must have been trained on this dataset, assets in the
    same order: a reordered universe would fail silently otherwise."""
    meta = trained.checkpoint.get("metadata", {})
    fingerprint = dataset.panel.metadata.get("fingerprint")
    if meta.get("fingerprint") != fingerprint:
        raise ValueError(
            f"checkpoint was trained on dataset {meta.get('fingerprint')}, not {fingerprint}"
        )
    if list(meta.get("tickers", [])) != list(dataset.panel.tickers):
        raise ValueError("checkpoint and dataset list different assets (or a different order)")
    if trained.model.n_assets != dataset.n_assets:
        raise ValueError(f"model has N={trained.model.n_assets}, dataset N={dataset.n_assets}")


def sample_split(
    trained: TrainedModel,
    dataset: Dataset,
    split: str = "test",
    settings: SamplingConfig | None = None,
    device: str | torch.device = "auto",
    tau: Sequence[int] | np.ndarray | None = None,
    trace: bool = True,
) -> SampleSet:
    """Algorithm 2 for every decision step of ``split`` (or the given ``tau``,
    which must belong to it) at every risk level."""
    check_compatible(trained, dataset)
    settings = settings if settings is not None else trained.config.sampling
    config = trained.config
    device = resolve_device(device)
    split_tau = dataset.splits[split].tau
    tau = split_tau if tau is None else np.asarray(tau, dtype=np.int64)
    if not np.isin(tau, split_tau).all():
        raise ValueError(f"some decision steps are not in the {split!r} split")

    tensors = dataset.panel.torch().to(device)
    covariances = CovarianceModel(
        tensors.returns,
        tensors.return_valid,
        settings,
        config.lookback,
        train_tau=dataset.splits.train.tau,
    )
    model, schedule = trained.model.to(device), trained.schedule.to(device)
    sampler = RiskGuidedSampler(
        model,
        schedule,
        config.diffusion.gamma_max,
        guidance=settings.guidance,
        guidance_scale=settings.guidance_scale,
    )
    generator = torch.Generator(device).manual_seed(settings.seed)

    g_max, k, n = config.diffusion.gamma_max, settings.num_samples, model.n_assets
    weights = np.empty((tau.size, g_max, k, n), dtype=np.float32)
    risk = np.empty((tau.size, g_max, k), dtype=np.float32)
    guidance_norm = torch.zeros(schedule.num_steps, g_max, device=device)
    noise_norm = None
    for start in range(0, tau.size, settings.batch_size):
        chunk = tau[start : start + settings.batch_size]
        window = stack_windows(tensors, chunk, config.lookback)
        cov = covariances(chunk)
        out = sampler.sample(window.h, window.g, cov, k, generator=generator, trace=trace)
        stop = start + chunk.size
        weights[start:stop] = out.weights.cpu().numpy()
        risk[start:stop] = proxy_risk(out.weights, cov[:, None, None]).cpu().numpy()
        if trace:
            guidance_norm += out.guidance_norm * chunk.size
            noise_norm = out.noise_norm
        logger.info("sampled %d / %d decision steps", stop, tau.size)

    metadata: dict[str, Any] = {
        "split": split,
        "settings": dataclasses.asdict(settings),
        "checkpoint_epoch": trained.checkpoint.get("epoch"),
        "fingerprint": dataset.panel.metadata.get("fingerprint"),
        "tickers": list(dataset.panel.tickers),
        "num_steps": schedule.num_steps,
        "sigma_x": float(schedule.sigma_x),
        "risk_sizes": sampler.risk_sizes,
    }
    if trace:
        # Per reverse step (index 0 is t = T): the mean guidance shift per
        # risk level, next to the size of the injected noise.
        metadata["trace"] = {
            "guidance_norm": (guidance_norm / tau.size).cpu().tolist(),
            "noise_norm": noise_norm.cpu().tolist(),
        }
    return SampleSet(tau=tau, weights=weights, risk=risk, metadata=metadata)
