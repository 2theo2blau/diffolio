"""Diffolio - risk-dependent diffusion portfolio generation.

Currently implemented: plan sections 1-5 (universe selection, price
acquisition, cleaning/alignment, the chronological split, and sliding
window construction).
"""

from .config import DiffolioConfig
from .utils import setup_logging

__all__ = ["DiffolioConfig", "setup_logging", "__version__"]

__version__ = "0.1.0"
