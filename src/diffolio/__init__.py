"""Diffolio - risk-dependent diffusion portfolio generation.

Currently implemented: plan sections 1-4 (universe selection, price
acquisition, cleaning/alignment and the chronological split).
"""

from .config import DiffolioConfig
from .utils import setup_logging

__all__ = ["DiffolioConfig", "setup_logging", "__version__"]

__version__ = "0.1.0"
