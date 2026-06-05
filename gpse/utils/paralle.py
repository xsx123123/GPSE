#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for validating GPSE parallel execution settings."""

from __future__ import annotations

import os
from typing import Any


def _log(logger: Any | None, level: str, message: str) -> None:
    """Log through a loguru/logging-compatible object when available."""
    if logger is None:
        return
    log_func = getattr(logger, level, None)
    if callable(log_func):
        log_func(message)


def get_available_cpu_cores(logger: Any | None = None) -> int:
    """Return CPU cores available to this process.

    On Linux, ``sched_getaffinity`` respects CPU affinity and common scheduler
    constraints. If unavailable, fall back to ``os.cpu_count``.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            available = len(os.sched_getaffinity(0))
            if available > 0:
                _log(logger, "debug", f"Detected {available} available CPU core(s) via sched_getaffinity")
                return available
        except OSError as exc:
            _log(logger, "warning", f"Failed to inspect CPU affinity: {exc}; falling back to os.cpu_count()")

    available = os.cpu_count() or 1
    _log(logger, "debug", f"Detected {available} CPU core(s) via os.cpu_count")
    return available


def validate_parallelism(
    n_jobs: int,
    max_workers: int,
    *,
    logger: Any | None = None,
    available_cores: int | None = None,
) -> tuple[int, int]:
    """Validate GPSE process/thread parallelism.

    Parameters
    ----------
    n_jobs
        Number of threads used inside each training worker.
    max_workers
        Number of parallel training worker processes.
    logger
        Optional logger initialized by ``gpse.utils.log_utils``.
    available_cores
        Optional explicit core count, mainly useful for tests.

    Returns
    -------
    tuple[int, int]
        The validated ``(n_jobs, max_workers)`` pair.

    Raises
    ------
    ValueError
        If either value is not positive, or if ``n_jobs * max_workers`` exceeds
        the cores available to the current process.
    """
    if n_jobs < 1:
        message = f"--n_jobs must be >= 1, got {n_jobs}"
        _log(logger, "error", message)
        raise ValueError(message)
    if max_workers < 1:
        message = f"--max_workers must be >= 1, got {max_workers}"
        _log(logger, "error", message)
        raise ValueError(message)

    available = available_cores if available_cores is not None else get_available_cpu_cores(logger)
    if available < 1:
        message = f"Available CPU core count must be >= 1, got {available}"
        _log(logger, "error", message)
        raise ValueError(message)

    requested = n_jobs * max_workers
    _log(
        logger,
        "info",
        (
            "Parallelism check: "
            f"n_jobs={n_jobs}, max_workers={max_workers}, "
            f"requested_threads={requested}, available_cores={available}"
        ),
    )

    if requested > available:
        message = (
            f"Requested parallelism n_jobs * max_workers = {requested}, "
            f"but only {available} CPU core(s) are available. "
            "Please reduce --n_jobs or --max_workers."
        )
        _log(logger, "error", message)
        raise ValueError(message)

    _log(logger, "info", "Parallelism check passed")
    return n_jobs, max_workers
