"""Section 8 - the market dynamics encoder.

Maps a batch of look-up windows to one market summary per sample::

    h  (B, N, L, F)  per-asset windows      -\\
                                              >-  z_merged (B, d),  d = L * F
    g  (B, L, F)     index window           -/

in six stages (plan 8.1-8.6):

1. per-asset temporal encoding: three dilated TCNs ``T_q, T_k, T_v`` run along
   ``L``, with weights shared across assets, giving ``Q, K, V (B, N, L, F)``;
2. cross-asset self-attention on the flattened ``(B, N, d)`` views,
   ``softmax(Q K^T / sqrt(d)) V``;
3. ``h' = LN(h + h_tilde)``, then ``h_final = LN(h' + T_h(h'))``, LayerNorm
   over each asset's ``(L, F)`` block;
4. the index pipeline: LSTM over ``g``, attention ``psi`` over time,
   projection to ``g_final (B, d)``;
5. fusion ``z_n = W_h h_final_n + g_final``;
6. attention pooling over assets, ``z_merged = sum_n alpha_n z_n``.

Nothing depends on ``N``: the TCNs are shared across assets and the attention
and pooling are permutation-equivariant/invariant, so the same weights accept
any number of assets.  ``d`` is derived from ``L`` and ``F`` and never
configured on its own (plan 8, cross-cutting notes).

Choices the paper leaves open, recorded here:

* TCN layers are separated by ReLU with no activation after the last layer, so
  ``Q, K, V`` and the residual branch are not sign-restricted.
* Convolutions use symmetric "same" padding.  Every value in the window is
  already known at the decision step ``tau``, so looking both ways along
  ``L`` leaks nothing.
* ``psi`` and the asset scorer ``s`` are two-layer tanh MLPs of width
  ``d_hid``.
* ``W_h`` has no bias, since ``g_final`` (a biased projection) is added to it.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DiffolioConfig

__all__ = ["DilatedTCN", "EncoderOutput", "IndexEncoder", "MarketEncoder"]


class EncoderOutput(NamedTuple):
    z_merged: torch.Tensor  # (B, d) market summary
    asset_attention: torch.Tensor  # (B, N) pooling weights alpha, rows sum to 1
    time_attention: torch.Tensor  # (B, L) index attention over the window, rows sum to 1


class DilatedTCN(nn.Module):
    """A stack of dilated 1-D convolutions along ``L``, ``F -> F`` channels.

    Input and output are ``(B, N, L, F)``; every asset is convolved
    independently with the same weights.  Padding preserves ``L``.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Sequence[int] = (1, 2)):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve L, got {kernel_size}")
        if not dilations:
            raise ValueError("a TCN needs at least one dilation")
        self.layers = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
            )
            for dilation in dilations
        )
        #: Steps on each side of t that can influence the output at t.
        self.receptive_radius = sum(d * (kernel_size - 1) // 2 for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, length, f = x.shape
        y = x.reshape(b * n, length, f).transpose(1, 2)  # (B*N, F, L)
        for i, conv in enumerate(self.layers):
            y = conv(y)
            if i < len(self.layers) - 1:
                y = F.relu(y)
        return y.transpose(1, 2).reshape(b, n, length, f)


class IndexEncoder(nn.Module):
    """Plan 8.4: LSTM over the index window, attention over time, project to d."""

    def __init__(self, num_features: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.project = nn.Linear(hidden_dim, embed_dim)

    def forward(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u, _ = self.lstm(g)  # (B, L, d_hid)
        weights = torch.softmax(self.score(u).squeeze(-1), dim=-1)  # (B, L)
        g_prime = torch.einsum("bl,blh->bh", weights, u)
        return self.project(g_prime), weights


class MarketEncoder(nn.Module):
    """``(h, g) -> z_merged`` (plan section 8)."""

    def __init__(
        self,
        lookback: int,
        num_features: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.num_features = int(num_features)
        self.embed_dim = self.lookback * self.num_features  # d = L * F

        def tcn() -> DilatedTCN:
            return DilatedTCN(num_features, kernel_size, dilations)

        self.tcn_q, self.tcn_k, self.tcn_v, self.tcn_h = tcn(), tcn(), tcn(), tcn()
        self.norm_attn = nn.LayerNorm([self.lookback, self.num_features])
        self.norm_out = nn.LayerNorm([self.lookback, self.num_features])
        self.index = IndexEncoder(num_features, hidden_dim, self.embed_dim)
        self.w_h = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.asset_score = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    @classmethod
    def from_config(cls, config: DiffolioConfig) -> "MarketEncoder":
        return cls(
            lookback=config.lookback,
            num_features=config.num_features,
            hidden_dim=config.model.hidden_dim,
            kernel_size=config.model.tcn_kernel_size,
            dilations=config.model.tcn_dilations,
        )

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> EncoderOutput:
        self._check_shapes(h, g)
        b, n = h.shape[:2]
        d = self.embed_dim

        # 8.1-8.2: shared per-asset TCNs, then attention across assets.
        q = self.tcn_q(h).reshape(b, n, d)
        k = self.tcn_k(h).reshape(b, n, d)
        v = self.tcn_v(h).reshape(b, n, d)
        h_tilde = F.scaled_dot_product_attention(q, k, v)  # scale 1/sqrt(d)

        # 8.3: residual + LayerNorm, second TCN block, residual + LayerNorm.
        h_prime = self.norm_attn(h + h_tilde.reshape_as(h))
        h_final = self.norm_out(h_prime + self.tcn_h(h_prime))

        # 8.4-8.5: index summary, fused into every asset.
        g_final, time_attention = self.index(g)  # (B, d), (B, L)
        z = self.w_h(h_final.reshape(b, n, d)) + g_final.unsqueeze(1)  # (B, N, d)

        # 8.6: attention pooling across assets.
        alpha = torch.softmax(self.asset_score(z).squeeze(-1), dim=-1)  # (B, N)
        z_merged = torch.einsum("bn,bnd->bd", alpha, z)
        return EncoderOutput(z_merged, alpha, time_attention)

    def _check_shapes(self, h: torch.Tensor, g: torch.Tensor) -> None:
        """Catch silent transposes early (plan, shape conventions)."""
        expected = (self.lookback, self.num_features)
        if h.ndim != 4 or tuple(h.shape[2:]) != expected:
            raise ValueError(f"h must be (B, N, L, F) with (L, F) = {expected}, got {tuple(h.shape)}")
        if g.ndim != 3 or tuple(g.shape[1:]) != expected or g.shape[0] != h.shape[0]:
            raise ValueError(
                f"g must be (B, L, F) = ({h.shape[0]}, {expected[0]}, {expected[1]}), "
                f"got {tuple(g.shape)}"
            )
