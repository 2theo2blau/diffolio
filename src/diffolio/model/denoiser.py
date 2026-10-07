"""Section 9 - the risk-aware denoising head ``F_gamma``.

Given the market summary ``z_merged`` from the encoder, predicts the clean
portfolio from a noisy one::

    phi(t)   sinusoidal embedding of t, width d
    omega_t  = act(W_omega phi(t))                         (9.1)
    v_gamma  = risk lookup table, gamma_max x d            (9.2)
    d_t      = (W x_t) * (omega_t + v_gamma)               (9.3, Eq. 17)
    x_hat    = MLP([z_merged || d_t])     R^{2d} -> R^N    (9.4-9.5)

Choices the paper leaves open, recorded here and in ``ModelConfig``:

* ``act`` is ``model.time_embedding_activation`` (sigmoid by default, which
  keeps the gate ``omega_t + v_gamma`` bounded).
* The MLP has ``model.mlp_layers`` linear layers: hidden layers of width ``d``
  joined by ``model.mlp_activation``, then a final ``d -> N`` layer with no
  output activation, since the targets are signed weights.
* ``W`` has no bias, as in Eq. 17.
* ``model.scale_by_sigma_x`` (off by default) makes the head work in units of
  ``sigma_x``: it reads ``x_t / sigma_x`` and returns ``sigma_x * MLP(...)``.
  Both factors could be absorbed into ``W`` and the last layer, so the
  functions the head can represent are unchanged; only the conditioning at
  initialisation differs.  Real weights and noise are O(sigma_x) ~ 6e-3,
  so unscaled, ``W x_t`` starts ~100x smaller than ``z_merged``.  Even so, a
  400-step probe on the real S&P build fitted *worse* with the scaling
  (validation MSE 0.89 vs 0.70 of the zero predictor; 0.74 vs 0.41 for
  t <= 50), so the paper's unscaled form is the default.

Unlike the encoder, the head is tied to ``N`` through ``W`` and the last
layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["DenoisingHead", "sinusoidal_embedding"]

_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Transformer-style position embedding of ``t``, shape ``t.shape + (dim,)``.

    The first half holds ``sin(t * f_i)`` and the second ``cos(t * f_i)`` with
    ``f_i = max_period ** (-i / half)``; an odd ``dim`` gets one zero column.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.to(torch.float32).unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}, expected one of {sorted(_ACTIVATIONS)}")


class DenoisingHead(nn.Module):
    """``(x_t, t, gamma, z_merged) -> x_hat`` (plan 9.1-9.5)."""

    def __init__(
        self,
        n_assets: int,
        embed_dim: int,
        gamma_max: int,
        sigma_x: float = 1.0,
        time_activation: str = "sigmoid",
        mlp_activation: str = "silu",
        mlp_layers: int = 3,
        scale_by_sigma_x: bool = False,
    ):
        super().__init__()
        if mlp_layers < 2:
            raise ValueError(f"mlp_layers must be >= 2, got {mlp_layers}")
        if sigma_x <= 0:
            raise ValueError(f"sigma_x must be positive, got {sigma_x}")
        self.n_assets = int(n_assets)
        self.embed_dim = int(embed_dim)
        self.gamma_max = int(gamma_max)
        self.scale_by_sigma_x = bool(scale_by_sigma_x)
        # Saved with the weights so a checkpoint carries the scale it was trained at.
        self.register_buffer("sigma_x", torch.tensor(float(sigma_x)))

        d = self.embed_dim
        self.w_omega = nn.Linear(d, d)
        self.time_activation = _activation(time_activation)
        self.risk_embedding = nn.Embedding(self.gamma_max, d)
        self.w_x = nn.Linear(self.n_assets, d, bias=False)

        layers: list[nn.Module] = [nn.Linear(2 * d, d)]
        for _ in range(mlp_layers - 2):
            layers += [_activation(mlp_activation), nn.Linear(d, d)]
        layers += [_activation(mlp_activation), nn.Linear(d, self.n_assets)]
        self.mlp = nn.Sequential(*layers)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """``omega_t = act(W_omega phi(t))`` (9.1)."""
        phi = sinusoidal_embedding(t, self.embed_dim).to(self.w_omega.weight.dtype)
        return self.time_activation(self.w_omega(phi))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        gamma: torch.Tensor | int,
        z_merged: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``x_hat``.

        ``x_t`` is ``(B, ..., N)``; ``t`` (in ``1..T``) and ``gamma`` (in
        ``0..gamma_max - 1``) broadcast against ``x_t.shape[:-1]``, and
        ``z_merged (B, d)`` is shared across any axes between ``B`` and ``N``
        (e.g. every risk level of one sample).
        """
        lead = x_t.shape[:-1]
        if x_t.shape[-1] != self.n_assets:
            raise ValueError(f"x_t must end in N={self.n_assets}, got {tuple(x_t.shape)}")
        if z_merged.shape != (lead[0], self.embed_dim):
            raise ValueError(
                f"z_merged must be (B, d) = ({lead[0]}, {self.embed_dim}), "
                f"got {tuple(z_merged.shape)}"
            )
        t = torch.as_tensor(t, device=x_t.device).expand(lead)
        gamma = torch.as_tensor(gamma, device=x_t.device, dtype=torch.long).expand(lead)
        if t.numel() and int(t.min()) < 1:
            raise IndexError(f"timesteps must be >= 1, got {int(t.min())}")
        if gamma.numel() and (int(gamma.min()) < 0 or int(gamma.max()) >= self.gamma_max):
            raise IndexError(
                f"gamma must lie in 0..{self.gamma_max - 1}, got "
                f"{int(gamma.min())}..{int(gamma.max())}"
            )

        scale = self.sigma_x.to(x_t.dtype) if self.scale_by_sigma_x else None
        x_in = x_t / scale if scale is not None else x_t
        gate = self.time_embedding(t) + self.risk_embedding(gamma)  # (..., d)
        d_t = self.w_x(x_in) * gate  # Eq. 17
        z = z_merged.reshape(lead[0], *([1] * (len(lead) - 1)), self.embed_dim).expand(*lead, -1)
        out = self.mlp(torch.cat([z, d_t], dim=-1))
        return out * scale if scale is not None else out
