"""Section 10 - the joint training objective (Eq. 18).

    L'' = L_denoise - lambda * L_aux

* ``L_denoise`` (10.1) is the batch mean of ``||x_hat - x_0||_2^2``, the
  squared error summed over assets.
* ``L_aux`` (10.2, Eq. 14) is the next-step return of a portfolio read
  straight off the market summary, ``<f(W_p z_merged), r_tau>``.  It is a
  return to *maximise*, so it enters ``L''`` with a minus sign.  It trains the
  encoder to put return-relevant information into ``z_merged``.  ``f`` is the
  L1 normaliser of Eq. 12 (:func:`diffolio.portfolio.l1_normalize`), kept
  differentiable.
* ``lambda`` is ``training.aux_weight`` (1.0 in the paper).

``W_p (N x d)`` is a learnable parameter of the objective, not of the
denoising model: only the loss uses it, and inference never does.
:class:`DiffolioLoss` is therefore an ``nn.Module``.  The training loop must
hand its parameters to the optimiser alongside the model's and save its
``state_dict`` in checkpoints.

Choices the paper leaves open, recorded here:

* ``W_p`` has no bias, as written.
* Assets whose return is invalid at ``tau`` (``r_valid`` False, only after a
  forward-filled bar) get no auxiliary weight.  Their scores are zeroed
  before ``f``, as the section-6 targets never hold them either.  The real
  S&P build has no such returns.
* The optional clip of 10.4 (``training.aux_grad_clip``) bounds the norm of
  the gradient the auxiliary term sends back into ``z_merged``, and so into
  the encoder.  The denoising gradient is untouched.  The clip applies to the
  whole batch's gradient tensor.  ``W_p``'s own gradient is not clipped here;
  that is the training loop's ``clip_grad_norm_`` if wanted.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..config import DiffolioConfig
from ..portfolio import l1_normalize

__all__ = [
    "AuxiliaryProjection",
    "DiffolioLoss",
    "LossOutput",
    "auxiliary_return",
    "denoising_loss",
]


class LossOutput(NamedTuple):
    total: torch.Tensor  # () L'' = denoise - lambda * aux_return, minimised
    denoise: torch.Tensor  # () L_denoise
    aux_return: torch.Tensor  # () L_aux, the batch-mean auxiliary return (maximised)
    aux_weights: torch.Tensor  # (B, N) the auxiliary portfolio f(W_p z_merged)


def denoising_loss(x_hat: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """``mean ||x_hat - x_0||_2^2`` (10.1), summed over assets (last axis) and
    averaged over every leading axis (batch, and risk level if present)."""
    if x_hat.shape != x0.shape:
        raise ValueError(f"x_hat {tuple(x_hat.shape)} and x0 {tuple(x0.shape)} differ")
    return (x_hat - x0).pow(2).sum(-1).mean()


def auxiliary_return(
    weights: torch.Tensor, r: torch.Tensor, r_valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-sample return ``<weights, r>``, shape ``(B,)``.  Invalid returns
    contribute nothing."""
    if r_valid is not None:
        r = torch.where(r_valid, r, torch.zeros_like(r))
    return (weights * r.to(weights.dtype)).sum(-1)


class AuxiliaryProjection(nn.Module):
    """``z_merged -> f(W_p z_merged)``, an L1-normalised portfolio (10.2)."""

    def __init__(self, embed_dim: int, n_assets: int):
        super().__init__()
        self.w_p = nn.Linear(embed_dim, n_assets, bias=False)

    def forward(self, z_merged: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.w_p(z_merged)
        if valid is not None:
            scores = torch.where(valid, scores, torch.zeros_like(scores))
        return l1_normalize(scores)


class DiffolioLoss(nn.Module):
    """``L'' = L_denoise - lambda * L_aux`` (Eq. 18)."""

    def __init__(
        self,
        n_assets: int,
        embed_dim: int,
        aux_weight: float = 1.0,
        aux_grad_clip: float | None = None,
    ):
        super().__init__()
        if aux_weight < 0:
            raise ValueError(f"aux_weight must be >= 0, got {aux_weight}")
        if aux_grad_clip is not None and aux_grad_clip <= 0:
            raise ValueError(f"aux_grad_clip must be positive or None, got {aux_grad_clip}")
        self.n_assets = int(n_assets)
        self.aux_weight = float(aux_weight)
        self.aux_grad_clip = aux_grad_clip
        self.projection = AuxiliaryProjection(embed_dim, n_assets)

    @classmethod
    def from_config(cls, config: DiffolioConfig, n_assets: int) -> "DiffolioLoss":
        return cls(
            n_assets=n_assets,
            embed_dim=config.embed_dim,
            aux_weight=config.training.aux_weight,
            aux_grad_clip=config.training.aux_grad_clip,
        )

    def forward(
        self,
        x_hat: torch.Tensor,
        x0: torch.Tensor,
        z_merged: torch.Tensor,
        r: torch.Tensor,
        r_valid: torch.Tensor | None = None,
    ) -> LossOutput:
        """``x_hat``/``x0`` are ``(B, ..., N)``; ``z_merged`` is ``(B, d)``;
        ``r``/``r_valid`` are the realised returns ``(B, N)`` at each ``tau``."""
        if r.shape != (z_merged.shape[0], self.n_assets):
            raise ValueError(
                f"r must be (B, N) = ({z_merged.shape[0]}, {self.n_assets}), got {tuple(r.shape)}"
            )
        denoise = denoising_loss(x_hat, x0)

        z_aux = z_merged
        if self.aux_grad_clip is not None and z_merged.requires_grad:
            z_aux = z_merged.clone()  # hook only the auxiliary branch's gradient
            z_aux.register_hook(self._clip_gradient)
        weights = self.projection(z_aux, r_valid)
        aux_return = auxiliary_return(weights, r, r_valid).mean()

        total = denoise - self.aux_weight * aux_return
        return LossOutput(total, denoise, aux_return, weights)

    def _clip_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        norm = grad.norm()
        return grad * (self.aux_grad_clip / norm).clamp(max=1.0) if norm > 0 else grad
