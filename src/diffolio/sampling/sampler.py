"""Section 12 - risk-guided sampling (Algorithm 2).

For each decision step, risk level ``gamma`` and sample::

    x_T ~ N(0, sigma_x^2 I)
    for t = T .. 1:
        w_t   = f(x_t)                                   L1 normaliser, Eq. 12
        rho'  = w_t^T Sigma_hat w_t                      Eq. 19
        x_hat = F_gamma(x_t, t, h, g, gamma)
        mu'   = c0_t x_hat + ct_t x_t - zeta^(gamma) grad_{x_t} rho'     Eq. 20
        x_{t-1} = mu' + sqrt(vt_t) sigma_x z,  z ~ N(0, I), z = 0 at t = 1
    w_hat = f_gamma(x_0)                                 Eq. 15, top-k_gamma

with ``zeta^(gamma) = 1 - gamma / (gamma_max - 1)``, so ``gamma = 0`` is
pulled hardest towards low variance and ``gamma = gamma_max - 1`` not at all.

Where the paper contradicts itself, the forward process wins:

* **Mean.**  Eq. 20 and Algorithm 2 write the ``x_t`` coefficient as
  ``sqrt(abar_t) (1 - abar_{t-1}) / (1 - abar_t)``.  Eq. 5, the actual
  posterior of the forward process, has ``sqrt(alpha_t)``.  At large ``t``
  ``sqrt(abar_t)`` would shrink ``x_t`` towards zero, so Eq. 5 is used
  (:meth:`DiffusionSchedule.posterior_mean`).
* **Noise.**  Algorithm 2 draws ``z ~ N(0, sigma_x^2 I)`` and multiplies by
  ``sigma_x`` again.  ``sigma_x`` is applied once, matching the stated
  reverse covariance (plan 12.3 note).

Other choices:

* The gradient goes through ``f`` by autograd, with the denominator attached.
  ``f`` is scale-invariant, so the gradient is orthogonal to ``x_t``.
* The final ``f_gamma`` ranks the sampled ``|x_0|`` only.  The realised-return
  validity mask of training is future information at inference.
* ``sampling.guidance_scale`` multiplies ``zeta`` (1.0 is the paper), and
  ``sampling.guidance = false`` sets it to zero (the DF-nRG ablation).
* Noise is drawn in the same order whether or not guidance is on, so a guided
  and an unguided run with the same seed share every ``x_T`` and ``z``.  At
  ``gamma = gamma_max - 1`` (``zeta = 0``) they are identical.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch

from ..diffusion import DiffusionSchedule
from ..model import DiffolioModel
from ..portfolio import l1_normalize, risk_sizes, top_k_normalize

__all__ = ["RiskGuidedSampler", "SamplerOutput", "proxy_risk", "risk_gradient", "zeta"]


class SamplerOutput(NamedTuple):
    weights: torch.Tensor  # (B, G, S, N) f_gamma(x_0), sum |w| = 1, k_gamma non-zero
    x0: torch.Tensor  # (B, G, S, N) the sampled x_0 before f_gamma
    #: (T, G) mean ||zeta grad rho'|| per step (index 0 is t = T), or None.
    guidance_norm: torch.Tensor | None
    #: (T,) ||sqrt(vt_t) sigma_x z|| expected per step, sqrt(N) * std (index 0 is t = T).
    noise_norm: torch.Tensor | None


def zeta(gamma: torch.Tensor | int, gamma_max: int) -> torch.Tensor:
    """``zeta^(gamma) = 1 - gamma / (gamma_max - 1)``."""
    return 1.0 - torch.as_tensor(gamma, dtype=torch.float32) / (gamma_max - 1)


def proxy_risk(weights: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """``w^T Sigma w`` (Eq. 19); ``cov`` broadcasts as ``(..., N, N)``."""
    return (weights * (cov @ weights.unsqueeze(-1)).squeeze(-1)).sum(-1)


def risk_gradient(x: torch.Tensor, cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``rho'(f(x))`` and its gradient with respect to ``x`` (plan 12.3.3)."""
    with torch.enable_grad():
        leaf = x.detach().requires_grad_(True)
        risk = proxy_risk(l1_normalize(leaf), cov)
        (grad,) = torch.autograd.grad(risk.sum(), leaf)
    return risk.detach(), grad


class RiskGuidedSampler:
    """Algorithm 2 for batches of decision steps, all risk levels at once."""

    def __init__(
        self,
        model: DiffolioModel,
        schedule: DiffusionSchedule,
        gamma_max: int,
        guidance: bool = True,
        guidance_scale: float = 1.0,
    ):
        self.model = model
        self.schedule = schedule
        self.gamma_max = int(gamma_max)
        self.guidance = bool(guidance)
        self.guidance_scale = float(guidance_scale)
        self.risk_sizes = risk_sizes(model.n_assets, self.gamma_max)

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        cov: torch.Tensor,
        num_samples: int,
        gammas: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
        trace: bool = False,
    ) -> SamplerOutput:
        """Sample ``num_samples`` portfolios per decision step and risk level.

        ``h (B, N, L, F)`` / ``g (B, L, F)`` are the look-up windows and
        ``cov (B, N, N)`` the covariance at each step.  ``gammas`` defaults to
        every level.  With ``trace``, the per-step guidance and noise norms
        are returned too.
        """
        self.model.eval()
        gammas = list(range(self.gamma_max)) if gammas is None else [int(x) for x in gammas]
        device = h.device
        b, n, s, levels = h.shape[0], self.model.n_assets, int(num_samples), len(gammas)
        if cov.shape != (b, n, n):
            raise ValueError(f"cov must be (B, N, N) = {(b, n, n)}, got {tuple(cov.shape)}")

        z_merged = self.model.encode(h, g).z_merged
        gamma = torch.tensor(gammas, device=device).view(1, levels, 1).expand(b, levels, s)
        step = zeta(torch.tensor(gammas), self.gamma_max).to(device) * self.guidance_scale
        step = step.view(1, levels, 1, 1) if self.guidance else torch.zeros(1, levels, 1, 1, device=device)
        guided = bool((step > 0).any())
        cov = cov.view(b, 1, 1, n, n)

        x = self.schedule.prior_sample((b, levels, s, n), generator=generator)
        steps = self.schedule.num_steps
        guidance_norm = torch.zeros(steps, levels, device=device) if trace else None
        noise_norm = torch.zeros(steps, device=device) if trace else None
        for i, t in enumerate(range(steps, 0, -1)):
            x_hat = self.model.denoise(x, t, gamma, z_merged)
            mean = self.schedule.posterior_mean(x_hat, x, torch.tensor(t, device=device))
            if guided:
                _, grad = risk_gradient(x, cov)
                correction = step * grad
                mean = mean - correction
                if trace:
                    guidance_norm[i] = correction.norm(dim=-1).mean(dim=(0, 2))
            if t > 1:
                noise = torch.randn(x.shape, generator=generator, device=device)
                std = self.schedule.posterior_std(torch.tensor(t, device=device), x)
                x = mean + std * noise
                if trace:
                    noise_norm[i] = std.reshape(()) * n**0.5
            else:
                x = mean

        k = torch.tensor([self.risk_sizes[level] for level in gammas], device=device).view(1, levels, 1)
        return SamplerOutput(top_k_normalize(x, k), x, guidance_norm, noise_norm)
