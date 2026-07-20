#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for GPSE parallelism helpers."""

import sys
import os
import time
from concurrent.futures import as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from gpse.utils.paralle import (
    derive_parallelism_from_threads,
    get_available_cpu_cores,
    graceful_process_pool,
    validate_parallelism,
)


class TestDeriveParallelismFromThreads:
    """Unit tests for ``derive_parallelism_from_threads``."""

    def test_basic_regression_budget(self):
        # 14 regression models, 100 threads, 7 repeats -> 14 * 7 = 98
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=100, n_models=14, n_repeats=7
        )
        assert n_jobs == 1
        assert max_workers == 14
        assert repeat_workers == 7

    def test_budget_smaller_than_model_count(self):
        # Only 10 threads for 14 models -> use 10 models, 1 repeat each
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=10, n_models=14, n_repeats=100
        )
        assert n_jobs == 1
        assert max_workers == 10
        assert repeat_workers == 1

    def test_single_repeat_limits_parallelism(self):
        # With n_repeats=1, repeat-level parallelism cannot exceed 1;
        # the leftover budget is recycled into n_jobs (100 // 14 = 7)
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=100, n_models=14, n_repeats=1
        )
        assert n_jobs == 7
        assert max_workers == 14
        assert repeat_workers == 1

    def test_leftover_budget_recycled_into_n_jobs(self):
        # threads=80, 15 models, 2 repeats -> 15 * 2 = 30 workers, n_jobs=2
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=80, n_models=15, n_repeats=2
        )
        assert n_jobs == 2
        assert max_workers == 15
        assert repeat_workers == 2

    def test_explicit_max_workers_preserved(self):
        # User explicitly wants only 7 models in parallel
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=100, n_models=14, n_repeats=100, max_workers=7
        )
        assert n_jobs == 1
        assert max_workers == 7
        assert repeat_workers == 14  # 100 // 7 = 14, capped by n_repeats

    def test_explicit_repeat_workers_preserved(self):
        # User explicitly wants 5 repeat workers per model
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=100, n_models=14, n_repeats=100, repeat_workers=5
        )
        assert n_jobs == 1
        assert max_workers == 14  # 100 // 5 = 20, capped by n_models
        assert repeat_workers == 5

    def test_n_jobs_accounted_for_in_budget(self):
        # If user already set n_jobs=7, repeats per model should drop
        n_jobs, max_workers, repeat_workers = derive_parallelism_from_threads(
            threads=100, n_models=14, n_repeats=100, n_jobs=7
        )
        assert n_jobs == 7
        assert max_workers == 14
        assert repeat_workers == 1  # 100 // 14 // 7 = 1

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            derive_parallelism_from_threads(threads=0, n_models=14, n_repeats=1)
        with pytest.raises(ValueError):
            derive_parallelism_from_threads(threads=100, n_models=0, n_repeats=1)
        with pytest.raises(ValueError):
            derive_parallelism_from_threads(threads=100, n_models=14, n_repeats=0)


class TestValidateParallelism:
    """Unit tests for ``validate_parallelism``."""

    def test_valid_combination(self):
        result = validate_parallelism(
            n_jobs=1, max_workers=4, repeat_workers=2, available_cores=8
        )
        assert result == (1, 4, 2)

    def test_exceeds_available_cores(self):
        with pytest.raises(ValueError):
            validate_parallelism(
                n_jobs=2, max_workers=4, repeat_workers=2, available_cores=8
            )

    def test_negative_value(self):
        with pytest.raises(ValueError):
            validate_parallelism(n_jobs=-1, max_workers=1, repeat_workers=1)


class TestGetAvailableCpuCores:
    """Smoke tests for ``get_available_cpu_cores``."""

    def test_returns_positive_integer(self):
        cores = get_available_cpu_cores()
        assert isinstance(cores, int)
        assert cores >= 1


class TestGracefulProcessPool:
    """Tests for the Ctrl+C-friendly process pool wrapper."""

    def test_normal_execution_completes(self):
        results = []
        with graceful_process_pool(max_workers=2) as executor:
            futures = [executor.submit(pow, i, 2) for i in range(5)]
            for future in as_completed(futures):
                results.append(future.result())
        assert sorted(results) == [0, 1, 4, 9, 16]

    def test_keyboard_interrupt_shuts_down_workers(self):
        def _slow_task():
            time.sleep(5)
            return "done"

        with pytest.raises(KeyboardInterrupt):
            with graceful_process_pool(max_workers=1) as executor:
                future = executor.submit(_slow_task)
                time.sleep(0.3)  # give the worker time to start
                raise KeyboardInterrupt

        # The future should not still be actively running after the interrupt.
        assert not future.running()
