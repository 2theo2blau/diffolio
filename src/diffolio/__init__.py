"""Diffolio - risk-dependent diffusion portfolio generation.

Currently implemented: plan sections 1-12 - the data pipeline (universe,
acquisition, cleaning, splits, windows, pseudo-optimal targets), the
diffusion schedule, the encoder and denoising network, the joint objective,
the training loop and risk-guided sampling.
"""

from .config import DiffolioConfig
from .utils import setup_logging

__all__ = ["DiffolioConfig", "setup_logging", "__version__"]

__version__ = "0.1.0"
