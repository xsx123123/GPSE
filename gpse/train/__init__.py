"""Training-stage helpers for model ranking and ensembling."""

__all__ = [
    "GenomicPredictorV2",
    "StackingEnsemble",
    "TOPSISEvaluator",
]


def __getattr__(name: str):
    if name == "GenomicPredictorV2":
        from .predictor import GenomicPredictorV2

        return GenomicPredictorV2

    if name == "StackingEnsemble":
        from .stacking import StackingEnsemble

        return StackingEnsemble

    if name == "TOPSISEvaluator":
        from .topsis import TOPSISEvaluator

        return TOPSISEvaluator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
