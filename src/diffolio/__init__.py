"""Diffolio - risk-dependent diffusion portfolio generation.

Currently implemented: plan sections 1-6 (universe selection, price
acquisition, cleaning/alignment, the chronological split, sliding window
construction, and risk-dependent pseudo-optimal portfolio synthesis).
"""

from .config import DiffolioConfig
from .utils import setup_logging

__all__ = ["DiffolioConfig", "setup_logging", "__version__"]

__version__ = "0.1.0"
