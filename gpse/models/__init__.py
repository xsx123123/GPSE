"""gpse.models — Model optimizers for regression and classification.

Uses lazy imports to avoid loading optuna and model libraries at package
import time.
"""

__all__ = [
    "RegressionModelOptimizer",
    "ModelOptimizer",
    "ClassificationModelOptimizer",
]


def __getattr__(name: str):
    if name == "RegressionModelOptimizer":
        from .regression_model_optimizer import RegressionModelOptimizer

        return RegressionModelOptimizer

    if name == "ModelOptimizer":
        from .regression_model_optimizer import ModelOptimizer

        return ModelOptimizer

    if name == "ClassificationModelOptimizer":
        from .classification_model_optimizer import ClassificationModelOptimizer

        return ClassificationModelOptimizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
