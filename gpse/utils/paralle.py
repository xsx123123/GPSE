#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for validating and running GPSE parallel execution."""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Generator


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
    repeat_workers: int = 1,
    *,
    logger: Any | None = None,
    available_cores: int | None = None,
) -> tuple[int, int, int]:
    """Validate GPSE process/thread parallelism.

    Parameters
    ----------
    n_jobs
        Number of threads used inside each training worker.
    max_workers
        Number of models trained in parallel.
    repeat_workers
        Number of repeats trained in parallel within each model.
    logger
        Optional logger initialized by ``gpse.utils.log_utils``.
    available_cores
        Optional explicit core count, mainly useful for tests.

    Returns
    -------
        tuple[int, int, int]
        The validated ``(n_jobs, max_workers, repeat_workers)`` tuple.

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
    if repeat_workers < 1:
        message = f"--repeat_workers must be >= 1, got {repeat_workers}"
        _log(logger, "error", message)
        raise ValueError(message)

    available = available_cores if available_cores is not None else get_available_cpu_cores(logger)
    if available < 1:
        message = f"Available CPU core count must be >= 1, got {available}"
        _log(logger, "error", message)
        raise ValueError(message)

    requested = n_jobs * max_workers * repeat_workers
    if requested > available:
        message = (
            "Requested parallelism n_jobs * max_workers * repeat_workers = "
            f"{requested}, but only {available} CPU core(s) are available. "
            "Please reduce --n_jobs, --max_workers, or --repeat_workers."
        )
        _log(logger, "error", message)
        raise ValueError(message)

    _log(
        logger,
        "info",
        (
            f"Parallelism check passed: n_jobs={n_jobs}, model_workers={max_workers}, "
            f"repeat_workers={repeat_workers}, cores={available}"
        ),
    )
    return n_jobs, max_workers, repeat_workers


def derive_parallelism_from_threads(
    threads: int,
    n_models: int,
    n_repeats: int,
    n_jobs: int = 1,
    max_workers: int = 1,
    repeat_workers: int = 1,
) -> tuple[int, int, int]:
    """Derive ``(n_jobs, max_workers, repeat_workers)`` from a total thread budget.

    The strategy keeps ``n_jobs`` unchanged (defaults to 1) and prefers to spend
    the budget on model-level parallelism first, then repeat-level parallelism.
    Values that are still at their argparse defaults (``1``) are treated as
    "not explicitly set" and are overridden; non-default values are preserved.

    Parameters
    ----------
    threads
        Target total number of concurrent training units.
    n_models
        Number of models that will be trained.
    n_repeats
        Number of repeats configured for each model.
    n_jobs
        Current ``--n_jobs`` value.
    max_workers
        Current ``--max_workers`` value.
    repeat_workers
        Current ``--repeat_workers`` value.

    Returns
    -------
    tuple[int, int, int]
        The derived ``(n_jobs, max_workers, repeat_workers)`` tuple.

    Raises
    ------
    ValueError
        If any input is not positive.
    """
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1, got {n_models}")
    if n_repeats < 1:
        raise ValueError(f"--n_repeats must be >= 1, got {n_repeats}")

    derived_n_jobs = n_jobs
    derived_max_workers = max_workers
    derived_repeat_workers = repeat_workers

    if max_workers == 1:
        derived_max_workers = min(n_models, threads)

    # Budget left for repeats inside each model, accounting for n_jobs.
    per_model_budget = threads // derived_max_workers
    if repeat_workers == 1:
        derived_repeat_workers = max(1, min(n_repeats, per_model_budget // derived_n_jobs))

    return derived_n_jobs, derived_max_workers, derived_repeat_workers


@contextlib.contextmanager
def graceful_process_pool(
    max_workers: int,
    *,
    initializer=None,
    initargs=(),
    mp_context=None,
    logger=None,
) -> Generator[ProcessPoolExecutor, None, None]:
    """ProcessPoolExecutor context manager that shuts down promptly on Ctrl+C.

    When the user presses Ctrl+C during a training run, the default
    ``ProcessPoolExecutor`` context manager waits for all worker processes to
    finish their current task before returning. This helper intercepts the
    resulting ``KeyboardInterrupt``, cancels pending futures, and terminates
    any worker processes that are still alive so the CLI exits quickly.

    Parameters
    ----------
    max_workers
        Maximum number of worker processes.
    initializer
        Optional callable run in each worker on start.
    initargs
        Arguments passed to ``initializer``.
    mp_context
        Optional multiprocessing context (e.g. ``mp.get_context("spawn")``).
        When ``None`` the interpreter default is used.
    logger
        Optional loguru/logging-compatible logger for shutdown messages.

    Yields
    ------
    ProcessPoolExecutor
        The underlying executor; submit work and iterate ``as_completed`` as
        usual inside the ``with`` block.
    """
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=initializer,
        initargs=initargs,
    )
    try:
        yield executor
    except KeyboardInterrupt:
        _log(
            logger,
            "warning",
            "Interrupt received (Ctrl+C); shutting down worker processes...",
        )
        executor.shutdown(wait=False, cancel_futures=True)
        _terminate_executor_workers(executor)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        _terminate_executor_workers(executor)


def _terminate_executor_workers(executor: ProcessPoolExecutor) -> None:
    """Best-effort termination of any worker processes still alive."""
    try:
        workers = list(getattr(executor, "_processes", {}).values())
    except Exception:
        return

    for worker in workers:
        try:
            if worker.is_alive():
                worker.terminate()
        except Exception:
            pass

    for worker in workers:
        try:
            worker.join(timeout=0.5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=0.5)
        except Exception:
            pass
