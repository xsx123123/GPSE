"""gpse — Genomic Prediction with Stacking Ensemble for horticultural crops."""

from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("gpse")
except Exception:
    __version__ = "unknown"
