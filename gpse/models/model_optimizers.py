"""Backward-compatible imports for regression model optimizers.

Use ``gpse.models.regression_model_optimizer`` for new code.
"""

from gpse.models.regression_model_optimizer import (
    ModelOptimizer,
    RegressionModelOptimizer,
)

__all__ = [
    "RegressionModelOptimizer",
    "ModelOptimizer",
]
