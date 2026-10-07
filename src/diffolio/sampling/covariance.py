"""Section 12.1 - the covariance ``Sigma_hat`` behind the proxy risk.

Risk guidance scores a portfolio by ``rho'(w) = w^T Sigma_hat w`` (Eq. 19),
with ``Sigma_hat`` "estimated from historical returns".  Two non-leaky
readings are offered (``sampling.covariance``):

* ``rolling`` (default) - at each decision step ``tau``, the covariance of the
  returns realised inside its look-up window.  The window holds the opens of
  days ``tau - L + 1 .. tau``, which define ``L - 1`` open-to-open returns,
  rows ``tau - L + 1 .. tau - 1`` of ``panel.returns``.  Row ``tau`` is the
  return being predicted (open ``tau`` to open ``tau + 1``), so it is never
  included: everything used is known at the open of ``tau``.  A different
  length can be set with ``sampling.covariance_window``; it always ends at
  row ``tau - 1``.
* ``train`` - one matrix from the realised returns of the training steps.

Raw daily returns are used, not the standardised features.  With ``N = 224``
assets and 255 returns the sample covariance is barely full rank, so
Ledoit-Wolf shrinkage towards a scaled identity is on by default.  It follows
scikit-learn's ``ledoit_wolf`` (centred data, biased ``1/n`` covariance) and
is computed in float64.

Invalid returns (a forward-filled leg) are zero-filled with a warning.  On the
real build only the panel's final row is invalid (it has no next open), so no
usable window contains one.

Estimates are computed on the fly rather than cached (plan 12.1 suggests a
disk cache): one batch of ``(B, N, N)`` matrices costs milliseconds on the
GPU.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..config import SamplingConfig
from ..utils import get_logger

logger = get_logger(__name__)

__all__ = ["CovarianceModel", "covariance", "ledoit_wolf"]


def ledoit_wolf(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ledoit-Wolf shrunk covariance of ``x (..., n, p)`` (n observations of
    p variables), and the shrinkage intensity ``(...)``."""
    n, p = x.shape[-2], x.shape[-1]
    x = x - x.mean(dim=-2, keepdim=True)
    x2 = x.pow(2)
    emp_cov = x.transpose(-1, -2) @ x / n
    trace = x2.sum(dim=-2) / n  # (..., p) diagonal of emp_cov
    mu = trace.sum(dim=-1) / p
    beta_ = (x2.transpose(-1, -2) @ x2).sum(dim=(-1, -2))
    delta_ = (x.transpose(-1, -2) @ x).pow(2).sum(dim=(-1, -2)) / n**2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace.sum(dim=-1) + p * mu.pow(2)) / p
    beta = torch.minimum(beta, delta)
    shrinkage = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)
    eye = torch.eye(p, dtype=x.dtype, device=x.device)
    shrunk = (1 - shrinkage)[..., None, None] * emp_cov + (shrinkage * mu)[..., None, None] * eye
    return shrunk, shrinkage


def covariance(
    x: torch.Tensor, shrinkage: str = "ledoit_wolf", ridge: float = 0.0
) -> torch.Tensor:
    """Covariance of ``x (..., n, p)`` in float64, shrunk and/or ridged."""
    x = x.to(torch.float64)
    if shrinkage == "ledoit_wolf":
        cov, _ = ledoit_wolf(x)
    elif shrinkage == "none":
        centred = x - x.mean(dim=-2, keepdim=True)
        cov = centred.transpose(-1, -2) @ centred / x.shape[-2]
    else:
        raise ValueError(f"unknown shrinkage {shrinkage!r}")
    if ridge:
        cov = cov + ridge * torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    return cov


class CovarianceModel:
    """``Sigma_hat_tau`` for any batch of decision steps.

    ``returns`` / ``valid`` are the panel's ``(T, N)`` open-to-open returns
    and their mask (``PanelTensors.returns`` / ``return_valid``).  ``train_tau``
    is needed only for ``covariance="train"``.  Matrices come back as float32
    ``(B, N, N)`` on the device of ``returns``.
    """

    def __init__(
        self,
        returns: torch.Tensor,
        valid: torch.Tensor,
        settings: SamplingConfig,
        lookback: int,
        train_tau: Sequence[int] | np.ndarray | None = None,
    ):
        self.settings = settings
        self.window = (
            int(settings.covariance_window)
            if settings.covariance_window is not None
            else int(lookback) - 1
        )
        self.returns = torch.where(valid, returns, torch.zeros_like(returns))
        self._invalid = ~valid
        self._fixed: torch.Tensor | None = None
        if settings.covariance == "train":
            if train_tau is None:
                raise ValueError("covariance='train' needs the training steps")
            rows = torch.as_tensor(np.asarray(train_tau, dtype=np.int64), device=returns.device)
            self._warn_invalid(rows)
            self._fixed = covariance(self.returns[rows], settings.shrinkage, settings.ridge).float()
        elif settings.covariance != "rolling":
            raise ValueError(f"unknown covariance {settings.covariance!r}")

    def __call__(self, tau: Sequence[int] | np.ndarray) -> torch.Tensor:
        tau = np.asarray(tau, dtype=np.int64)
        if self._fixed is not None:
            return self._fixed.expand(tau.size, -1, -1)
        if tau.size and int(tau.min()) < self.window:
            raise IndexError(
                f"decision step {int(tau.min())} has fewer than {self.window} past returns"
            )
        offsets = np.arange(-self.window, 0, dtype=np.int64)  # rows tau - W .. tau - 1
        rows = torch.as_tensor(tau[:, None] + offsets[None, :], device=self.returns.device)
        self._warn_invalid(rows)
        cov = covariance(self.returns[rows], self.settings.shrinkage, self.settings.ridge)
        return cov.float()

    def _warn_invalid(self, rows: torch.Tensor) -> None:
        bad = int(self._invalid[rows].sum())
        if bad:
            logger.warning("%d invalid return(s) in the covariance window were zero-filled", bad)
