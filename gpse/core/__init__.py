"""gpse.core — Core prediction pipeline.

Uses lazy imports to avoid loading heavy ML dependencies at package
import time. ``GenomicPredictorV2`` and ``GenomicClassifier`` are loaded
on first access.
"""

__all__ = [
    "GenomicPredictorV2",
    "GenomicClassifier",
]


def __getattr__(name: str):
    if name == "GenomicPredictorV2":
        from .prediction_v2 import GenomicPredictorV2

        return GenomicPredictorV2

    if name == "GenomicClassifier":
        from .genomic_classification import GenomicClassifier

        return GenomicClassifier

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
