"""gpse.utils — Shared utilities for data processing, logging, and evaluation.

Uses lazy imports to avoid loading heavy ML dependencies (pandas, numpy,
scipy, sklearn) until they are actually needed. This keeps ``gpse --help``
and other lightweight CLI commands fast.
"""

__all__ = [
    "logger_init",
    "logger_generator",
    "setup_subprocess_logging",
    "collect_subprocess_logs",
    "calculate_metrics",
    "create_comparison_table",
    "call_topsis_evaluator",
]


def __getattr__(name: str):
    """Lazy-load heavy submodules only when their exports are accessed."""
    if name in ("logger_init", "logger_generator", "setup_subprocess_logging", "collect_subprocess_logs"):
        from .log_utils import logger_init, logger_generator, setup_subprocess_logging, collect_subprocess_logs

        return {
            "logger_init": logger_init,
            "logger_generator": logger_generator,
            "setup_subprocess_logging": setup_subprocess_logging,
            "collect_subprocess_logs": collect_subprocess_logs,
        }[name]

    if name in ("calculate_metrics", "create_comparison_table", "call_topsis_evaluator"):
        from .genomic_utils import (
            calculate_metrics,
            create_comparison_table,
            call_topsis_evaluator,
        )

        return {
            "calculate_metrics": calculate_metrics,
            "create_comparison_table": create_comparison_table,
            "call_topsis_evaluator": call_topsis_evaluator,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
