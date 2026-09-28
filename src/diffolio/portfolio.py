"""Portfolio normalisers shared by targets, losses and inference.

Two maps turn a raw score vector into portfolio weights with ``sum |w| = 1``:

* :func:`l1_normalize` is ``f`` of Eq. (12), ``f(x) = x / sum_n |x_n|``.  It is
  differentiable end to end: the auxiliary objective (section 10) and the
  risk-guidance gradient (section 12) both backpropagate through it, so the
  denominator is never detached.
* :func:`top_k_normalize` is the risk-dependent ``f_gamma`` of Eq. (15): keep
  the ``k_gamma`` entries of largest magnitude, zero the rest, and L1-normalise
  the survivors.  Section 6 applies it to the realised returns ``r_tau`` to
  build the denoising targets; section 12 applies it to the sampled ``x_0``.

``k_gamma`` itself (Eq. 16) comes from :func:`risk_sizes`.  Signs always pass
through unchanged, so negative entries become short positions.

Shapes follow the package conventions: the asset axis is the last one, and any
leading axes (batch, risk level) broadcast.
"""

from __future__ import annotations

import torch

__all__ = ["risk_sizes", "l1_normalize", "top_k_mask", "top_k_normalize"]

#: Denominator floor for the L1 normaliser.  Only a vector whose entries are
#: all (numerically) zero ever reaches it; it then maps to zero, not NaN.
EPS = 1e-12


def risk_sizes(n_assets: int, gamma_max: int) -> list[int]:
    """``k_gamma = floor(N / gamma_max) * (gamma_max - gamma)`` (Eq. 16).

    Index ``gamma`` of the result is the number of assets held at risk level
    ``gamma`` in ``0 .. gamma_max - 1``.  The floor applies to ``N /
    gamma_max`` before the multiplication, so ``gamma = 0`` (most diversified)
    holds ``k_0 <= N`` assets and ``gamma = gamma_max - 1`` (most
    concentrated) holds ``floor(N / gamma_max)``.
    """
    if gamma_max < 2:
        raise ValueError(f"gamma_max must be >= 2, got {gamma_max}")
    block = n_assets // gamma_max
    if block < 1:
        raise ValueError(
            f"N={n_assets} assets cannot be split into gamma_max={gamma_max} risk "
            f"levels: floor(N / gamma_max) must be >= 1"
        )
    return [block * (gamma_max - gamma) for gamma in range(gamma_max)]


def l1_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """``f(x) = x / sum |x|`` along ``dim`` (Eq. 12), differentiably."""
    return x / x.abs().sum(dim=dim, keepdim=True).clamp_min(eps)


def top_k_mask(
    scores: torch.Tensor,
    k: int | torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean mask of the ``k`` largest-magnitude entries along the last axis.

    ``k`` may be an int or a tensor broadcastable against ``scores.shape[:-1]``
    (e.g. one ``k_gamma`` per sample).  Ties break towards the lower asset
    index, deterministically.  Entries where ``valid`` is False are never
    selected, so fewer than ``k`` survive when fewer than ``k`` are valid.
    """
    magnitude = scores.abs()
    if valid is not None:
        magnitude = magnitude.masked_fill(~valid, float("-inf"))
    order = torch.argsort(magnitude, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    k = torch.as_tensor(k, device=scores.device)
    keep = ranks < k.unsqueeze(-1)
    if valid is not None:
        keep = keep & valid
    return keep


def top_k_normalize(
    x: torch.Tensor,
    k: int | torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """``f_gamma`` (Eq. 15): keep the top-``k`` entries by ``|x|``, L1-normalise.

    See :func:`top_k_mask` for how ``k`` broadcasts and how ``valid`` is used.
    """
    return l1_normalize(torch.where(top_k_mask(x, k, valid), x, torch.zeros_like(x)), eps=eps)
