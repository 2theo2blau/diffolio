"""Section 9.6 - the encoder and denoising head wired into one module.

:class:`DiffolioModel` owns both halves so one optimiser trains them jointly.
Its ``forward`` is the training-time call ``(x_t, t, h, g, gamma) -> x_hat``.
Sampling (section 12) runs the head ``T`` times on the same window, so
:meth:`DiffolioModel.encode` and :meth:`DiffolioModel.denoise` are exposed
separately and the encoder runs once per window, not once per step.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..config import DiffolioConfig
from .denoiser import DenoisingHead
from .encoder import EncoderOutput, MarketEncoder

__all__ = ["DiffolioModel", "DiffolioOutput"]


class DiffolioOutput(NamedTuple):
    x_hat: torch.Tensor  # (B, ..., N) predicted clean portfolio
    encoding: EncoderOutput  # z_merged (B, d) feeds the auxiliary loss (section 10)


class DiffolioModel(nn.Module):
    def __init__(self, encoder: MarketEncoder, head: DenoisingHead):
        super().__init__()
        if encoder.embed_dim != head.embed_dim:
            raise ValueError(
                f"encoder d={encoder.embed_dim} does not match head d={head.embed_dim}"
            )
        self.encoder = encoder
        self.head = head

    @classmethod
    def from_config(
        cls, config: DiffolioConfig, n_assets: int, sigma_x: float
    ) -> "DiffolioModel":
        """Build from the config; ``N`` comes from the dataset and ``sigma_x``
        from the fitted schedule (``DiffusionSchedule.sigma_x``)."""
        m = config.model
        head = DenoisingHead(
            n_assets=n_assets,
            embed_dim=config.embed_dim,
            gamma_max=config.diffusion.gamma_max,
            sigma_x=float(sigma_x),
            time_activation=m.time_embedding_activation,
            mlp_activation=m.mlp_activation,
            mlp_layers=m.mlp_layers,
            scale_by_sigma_x=m.scale_by_sigma_x,
        )
        return cls(MarketEncoder.from_config(config), head)

    @property
    def n_assets(self) -> int:
        return self.head.n_assets

    def encode(self, h: torch.Tensor, g: torch.Tensor) -> EncoderOutput:
        if h.shape[1] != self.n_assets:
            raise ValueError(f"h has N={h.shape[1]} assets, the model was built for {self.n_assets}")
        return self.encoder(h, g)

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        gamma: torch.Tensor | int,
        z_merged: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(x_t, t, gamma, z_merged)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        h: torch.Tensor,
        g: torch.Tensor,
        gamma: torch.Tensor | int,
    ) -> DiffolioOutput:
        encoding = self.encode(h, g)
        return DiffolioOutput(self.denoise(x_t, t, gamma, encoding.z_merged), encoding)
