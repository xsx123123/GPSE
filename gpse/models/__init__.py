"""gpse.models — Model optimizers for regression and classification.

Uses lazy imports to avoid loading optuna and model libraries at package
import time.
"""

__all__ = [
    "ModelOptimizer",
    "ClassificationModelOptimizer",
]


def __getattr__(name: str):
    if name == "ModelOptimizer":
        from .model_optimizers import ModelOptimizer

        return ModelOptimizer

    if name == "ClassificationModelOptimizer":
        from .classification_models import ClassificationModelOptimizer

        return ClassificationModelOptimizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
