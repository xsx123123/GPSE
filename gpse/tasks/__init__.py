"""Task-specific runtime helpers."""

__all__ = ["GenomicClassifier"]


def __getattr__(name: str):
    if name == "GenomicClassifier":
        from .classification import GenomicClassifier

        return GenomicClassifier

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
