"""gpse.core — Core prediction pipeline.

Uses lazy imports to avoid loading heavy ML dependencies at package
import time. ``GenomicPredictorV2`` is loaded on first access.
"""

__all__ = [
    "GenomicPredictorV2",
]


def __getattr__(name: str):
    if name == "GenomicPredictorV2":
        from .prediction_v2 import GenomicPredictorV2

        return GenomicPredictorV2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
