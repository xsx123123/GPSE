"""gpse — Genomic Prediction with Stacking Ensemble for horticultural crops."""

try:
    from importlib.metadata import version as _get_version
except ImportError:
    from importlib_metadata import version as _get_version  # type: ignore[no-redef]

try:
    __version__ = _get_version("gpse")
except Exception:
    __version__ = "unknown"
