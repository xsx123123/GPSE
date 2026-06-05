"""Backward-compatible shim for classification model optimization.

Use ``gpse.models.classification_model_optimizer`` for new code.
"""

from gpse.models.classification_model_optimizer import ClassificationModelOptimizer

__all__ = ["ClassificationModelOptimizer"]
