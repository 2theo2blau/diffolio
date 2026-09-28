"""Section 7 - diffusion schedule and data-scaled noise statistics.

Diffolio's one change to the standard DDPM is the noise scale: noise enters as
``sigma_x * eps`` with ``eps ~ N(0, I)``, where ``sigma_x`` is the spread of the
pseudo-optimal portfolios, instead of unit variance.  Everything else is the
textbook schedule:

    forward     q(x_t | x_0)       = N( sqrt(abar_t) x_0,  (1 - abar_t) sigma_x^2 I )
    prior       p(x_T)             = N( 0, sigma_x^2 I )
    posterior   q(x_{t-1} | x_t, x_0) = N( c0_t x_0 + ct_t x_t,  vt_t sigma_x^2 I )

    c0_t = sqrt(abar_{t-1}) beta_t / (1 - abar_t)
    ct_t = sqrt(alpha_t) (1 - abar_{t-1}) / (1 - abar_t)
    vt_t = (1 - abar_{t-1}) beta_t / (1 - abar_t)

(Eq. 5 of the paper; Algorithm 2 plugs the network's ``x_hat`` in for ``x_0``
and subtracts the risk-guidance term, which is section 12's business.)

**Indexing.**  Timesteps are 1-based, ``t in 1..T``, as in the paper.  Every
per-step buffer has length ``T + 1`` and index 0 holds the ``t = 0`` boundary
(``abar_0 = 1``), so ``abar[t - 1]`` is always valid and no off-by-one shift is
needed at the call sites.  Index 0 of ``beta`` / the posterior buffers is
unused padding.

``sigma_x`` (plan 7.2) is estimated by :func:`estimate_sigma_x` on the
*training* split only and then frozen into the schedule.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

from .config import DiffusionConfig

if TYPE_CHECKING:
    from .data.splits import Split
    from .data.targets import PortfolioTargets
from .utils import get_logger

logger = get_logger(__name__)

__all__ = ["DiffusionSchedule", "estimate_sigma_x", "fit_schedule", "make_betas"]

#: Nichol & Dhariwal's cap on cosine betas, avoiding a singular final step.
_MAX_COSINE_BETA = 0.999
#: Warn when x_T still carries more than this fraction of x_0 (sqrt(abar_T)):
#: the N(0, sigma_x^2) prior then no longer matches where training ends.
_PRIOR_SIGNAL_WARNING = 0.1

SigmaSource = Literal["base", "risk_targets"]


def make_betas(
    num_steps: int,
    schedule: Literal["linear", "cosine"] = "linear",
    beta_start: float = 1.0e-4,
    beta_end: float = 0.02,
    cosine_offset: float = 0.008,
) -> np.ndarray:
    """``beta_1 .. beta_T`` as a float64 array of length ``T`` (plan 7.1)."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if schedule == "linear":
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError(
                f"linear schedule needs 0 < beta_start <= beta_end < 1, "
                f"got {beta_start}, {beta_end}"
            )
        return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    if schedule == "cosine":
        # abar(t) = cos^2(((t/T + s) / (1 + s)) * pi/2), normalised to abar(0) = 1.
        steps = np.arange(num_steps + 1, dtype=np.float64) / num_steps
        f = np.cos((steps + cosine_offset) / (1.0 + cosine_offset) * math.pi / 2) ** 2
        abar = f / f[0]
        return np.clip(1.0 - abar[1:] / abar[:-1], 0.0, _MAX_COSINE_BETA)
    raise ValueError(f"unknown beta schedule {schedule!r}")


def estimate_sigma_x(
    base: np.ndarray | torch.Tensor,
    risk_targets: np.ndarray | torch.Tensor | None = None,
    source: SigmaSource = "base",
) -> float:
    """``sigma_x = sqrt(E[(x - mu_x)^2])`` with ``mu_x = 0`` (plan 7.2).

    ``base`` is ``(S, N)``, the Eq. (12) portfolios of the *training* steps;
    the paper's definition (section 3.1) averages over ``tau`` and ``n`` only.
    ``source="risk_targets"`` instead pools the ``(S, G, N)`` risk targets, the
    actual denoising targets - a deviation from the paper, and logged as one.
    The caller is responsible for passing training steps only.
    """
    if source == "base":
        values = base
    elif source == "risk_targets":
        if risk_targets is None:
            raise ValueError("source='risk_targets' needs the risk targets")
        values = risk_targets
        logger.warning(
            "sigma_x pooled over the risk-dependent targets x^(tau, gamma): this "
            "deviates from the paper, which defines it over the base portfolios"
        )
    else:
        raise ValueError(f"unknown sigma_x source {source!r}")

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot estimate sigma_x from an empty set of portfolios")
    sigma = float(np.sqrt(np.mean(values**2)))
    if not sigma > 0.0:
        raise ValueError("sigma_x is zero: every portfolio is all-zero")
    logger.info(
        "sigma_x = %.6g over %d portfolios (source=%s, empirical mean %.3g, taken as 0)",
        sigma,
        values.shape[0],
        source,
        float(values.mean()),
    )
    return sigma


class DiffusionSchedule(nn.Module):
    """Precomputed DDPM coefficients plus the frozen ``sigma_x``.

    An ``nn.Module`` with buffers only, so ``.to(device)`` moves it and it is
    saved in a model's ``state_dict`` - ``sigma_x`` included, which keeps the
    training-set estimate attached to the checkpoint it was trained with.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_coef_x0: torch.Tensor
    posterior_coef_xt: torch.Tensor
    posterior_variance_unit: torch.Tensor
    sigma_x: torch.Tensor

    def __init__(self, betas: np.ndarray | torch.Tensor, sigma_x: float) -> None:
        super().__init__()
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1 or betas.size < 1:
            raise ValueError("betas must be a non-empty 1-D array")
        if not ((betas > 0) & (betas < 1)).all():
            raise ValueError("every beta must lie in (0, 1)")
        if not sigma_x > 0:
            raise ValueError(f"sigma_x must be positive, got {sigma_x}")

        # Pad index 0 as the t = 0 boundary: beta_0 = 0, so abar_0 = 1.
        beta = np.concatenate([[0.0], betas])
        alpha = 1.0 - beta
        abar = np.cumprod(alpha)
        abar_prev = np.concatenate([[1.0], abar[:-1]])
        one_minus = 1.0 - abar

        coef_x0 = np.zeros_like(beta)
        coef_xt = np.zeros_like(beta)
        var_unit = np.zeros_like(beta)
        t = slice(1, None)
        coef_x0[t] = np.sqrt(abar_prev[t]) * beta[t] / one_minus[t]
        coef_xt[t] = np.sqrt(alpha[t]) * (1.0 - abar_prev[t]) / one_minus[t]
        var_unit[t] = (1.0 - abar_prev[t]) * beta[t] / one_minus[t]

        def buffer(name: str, value: np.ndarray | float) -> None:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

        buffer("betas", beta)
        buffer("alphas", alpha)
        buffer("alpha_bars", abar)
        buffer("sqrt_alpha_bars", np.sqrt(abar))
        buffer("sqrt_one_minus_alpha_bars", np.sqrt(one_minus))
        buffer("posterior_coef_x0", coef_x0)
        buffer("posterior_coef_xt", coef_xt)
        buffer("posterior_variance_unit", var_unit)
        buffer("sigma_x", float(sigma_x))

        residual = math.sqrt(abar[-1])
        if residual > _PRIOR_SIGNAL_WARNING:
            logger.warning(
                "x_T keeps sqrt(abar_T) = %.3f of x_0 (T=%d): the N(0, sigma_x^2) "
                "prior is a poor match for the end of the forward process; "
                "consider a larger T, a larger beta_end or the cosine schedule",
                residual,
                self.num_steps,
            )

    @classmethod
    def from_config(cls, config: DiffusionConfig, sigma_x: float) -> "DiffusionSchedule":
        betas = make_betas(
            config.num_steps,
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.cosine_offset,
        )
        return cls(betas, sigma_x)

    @property
    def num_steps(self) -> int:
        """T."""
        return int(self.betas.shape[0]) - 1

    # -- sampling helpers -----------------------------------------------------
    def sample_timesteps(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """``t ~ Uniform{1, ..., T}``, shape ``(B,)`` int64 (plan 11.1)."""
        return torch.randint(
            1,
            self.num_steps + 1,
            (batch_size,),
            generator=generator,
            device=self.betas.device,
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) sigma_x eps`` (plan 7.3).

        ``x0`` is ``(B, N)`` (any trailing shape works) and ``t`` is ``(B,)``
        in ``1..T``; ``noise`` is the standard-normal ``eps``.
        """
        t = self._check_t(t, x0)
        if noise is None:
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        return (
            self._gather(self.sqrt_alpha_bars, t, x0) * x0
            + self._gather(self.sqrt_one_minus_alpha_bars, t, x0) * self.sigma_x * noise
        )

    def prior_sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """``x_T ~ N(0, sigma_x^2 I)``."""
        noise = torch.randn(shape, generator=generator, device=self.betas.device, dtype=dtype)
        return self.sigma_x.to(dtype) * noise

    # -- posterior q(x_{t-1} | x_t, x_0) (plan 7.4) ---------------------------
    def posterior_mean(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``c0_t x_0 + ct_t x_t``; pass the network's ``x_hat`` as ``x0``."""
        t = self._check_t(t, x_t)
        return (
            self._gather(self.posterior_coef_x0, t, x_t) * x0
            + self._gather(self.posterior_coef_xt, t, x_t) * x_t
        )

    def posterior_variance(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """``vt_t sigma_x^2``, broadcastable against ``like``; zero at ``t = 1``."""
        t = self._check_t(t, like)
        return self._gather(self.posterior_variance_unit, t, like) * self.sigma_x**2

    def posterior_std(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """``sqrt(vt_t) sigma_x`` - the noise scale of Algorithm 2's update."""
        return self.posterior_variance(t, like).sqrt()

    def describe(self) -> str:
        return (
            f"DiffusionSchedule(T={self.num_steps}, beta={float(self.betas[1]):.3g}.."
            f"{float(self.betas[-1]):.3g}, sigma_x={float(self.sigma_x):.4g}, "
            f"sqrt(abar_T)={float(self.sqrt_alpha_bars[-1]):.3g})"
        )

    # -- internals ------------------------------------------------------------
    def _check_t(self, t: torch.Tensor | int, like: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=like.device, dtype=torch.int64)
        if t.numel() and (int(t.min()) < 1 or int(t.max()) > self.num_steps):
            raise IndexError(
                f"timesteps must lie in 1..{self.num_steps}, got "
                f"{int(t.min())}..{int(t.max())}"
            )
        return t

    @staticmethod
    def _gather(buffer: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """``buffer[t]`` reshaped to broadcast over ``like``'s trailing axes."""
        values = buffer.to(like.device)[t].to(like.dtype)
        return values.reshape(values.shape + (1,) * (like.ndim - values.ndim))


def fit_schedule(
    config: DiffusionConfig,
    targets: "PortfolioTargets",
    train: "Split",
) -> DiffusionSchedule:
    """Estimate ``sigma_x`` on the training split and build the schedule.

    ``train`` must be the training split: validation discipline (plan 4.3)
    forbids fitting ``sigma_x`` on anything the model is evaluated on.
    """
    if train.name != "train":
        raise ValueError(f"sigma_x must be fitted on the training split, got {train.name!r}")
    fitted = targets.select(train)
    sigma_x = estimate_sigma_x(fitted.base, fitted.targets, config.sigma_x_source)
    schedule = DiffusionSchedule.from_config(config, sigma_x)
    logger.info("built %s", schedule.describe())
    return schedule
