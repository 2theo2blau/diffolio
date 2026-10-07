"""Training objectives (plan section 10)."""

from .objective import (
    AuxiliaryProjection,
    DiffolioLoss,
    LossOutput,
    auxiliary_return,
    denoising_loss,
)

__all__ = [
    "AuxiliaryProjection",
    "DiffolioLoss",
    "LossOutput",
    "auxiliary_return",
    "denoising_loss",
]
