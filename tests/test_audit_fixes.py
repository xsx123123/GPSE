#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08-05 audit fixes (P0/P1/P2)."""

import inspect
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import yaml

from gpse.batch.adaptive_scheduler import compute_concurrency, _upsert_flag
from gpse.utils.parallel import validate_parallelism
from gpse.utils import paralle  # backward-compat shim must keep working

MODELS_YAML = Path(__file__).resolve().parents[1] / "gpse" / "config" / "models.yaml"

GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# P0-1: svm_clf search space must only contain sklearn.svm.SVC parameters
# --------------------------------------------------------------------------- #

def _load_model_entries():
    with open(MODELS_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {entry["name"]: entry for entry in data["models"]}


def test_svm_clf_search_space_is_valid_for_svc():
    from sklearn.svm import SVC

    valid = set(inspect.signature(SVC.__init__).parameters) - {"self"}
    entry = _load_model_entries()["svm_clf"]
    assert entry["import_path"] == "sklearn.svm.SVC"
    names = {param["name"] for param in entry["search_space"]}
    invalid = names - valid
    assert not invalid, f"svm_clf search space has non-SVC params: {invalid}"


# --------------------------------------------------------------------------- #
# P0-3 / P0-6: optimization module no longer caps trials or seeds global RNG
# --------------------------------------------------------------------------- #

def test_optimization_has_no_hard_trial_cap():
    source = (
        Path(__file__).resolve().parents[1] / "gpse" / "train" / "_optimization.py"
    ).read_text(encoding="utf-8")
    assert "min(self.n_trials, 100)" not in source


def test_optimization_does_not_seed_global_numpy_rng():
    source = (
        Path(__file__).resolve().parents[1] / "gpse" / "train" / "_optimization.py"
    ).read_text(encoding="utf-8")
    assert "np.random.seed(" not in source


# --------------------------------------------------------------------------- #
# P0-5: threadpoolctl must be declared as a dependency
# --------------------------------------------------------------------------- #

def test_threadpoolctl_declared_in_pyproject():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "threadpoolctl" in text


# --------------------------------------------------------------------------- #
# P1-7: paralle.py shim re-exports parallel.py
# --------------------------------------------------------------------------- #

def test_paralle_shim_reexports_parallel():
    assert paralle.validate_parallelism is validate_parallelism
    assert hasattr(paralle, "derive_parallelism_from_threads")
    assert hasattr(paralle, "graceful_process_pool")


# --------------------------------------------------------------------------- #
# P1-8: _find_result_file respects the depth limit
# --------------------------------------------------------------------------- #

def test_find_result_file_depth_limit(tmp_path):
    from gpse.predict.core import _find_result_file

    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "model.pkl"
    deep.parent.mkdir(parents=True)
    deep.write_text("x")

    # decoy sitting 5 levels above the model -> must NOT be found
    decoy = tmp_path / "label_encoder.pkl"
    decoy.write_text("x")
    assert _find_result_file(deep, "label_encoder.pkl") is None

    # file within max_up=3 levels -> must be found
    close = tmp_path / "a" / "b" / "c" / "label_encoder.pkl"
    close.write_text("x")
    found = _find_result_file(deep, "label_encoder.pkl")
    assert found == close


# --------------------------------------------------------------------------- #
# Adaptive scheduler: concurrency budget math
# --------------------------------------------------------------------------- #

def test_compute_concurrency_cpu_bound():
    n = compute_concurrency(
        max_threads=90,
        threads_per_trait=8,
        min_free_mem_gb=0,
        mem_per_trait_gb=1,
        max_concurrent_traits=None,
        available_bytes=10**12,
    )
    assert n == 90 // 8


def test_compute_concurrency_memory_bound():
    n = compute_concurrency(
        max_threads=90,
        threads_per_trait=8,
        min_free_mem_gb=64,
        mem_per_trait_gb=32,
        max_concurrent_traits=None,
        available_bytes=100 * GIB,
    )
    assert n == (100 - 64) // 32 == 1


def test_compute_concurrency_respects_hard_cap_and_zero_mem():
    assert compute_concurrency(90, 8, 0, 1, 2, 10**12) == 2
    assert (
        compute_concurrency(90, 8, 64, 32, None, 10 * GIB) == 0
    )


def test_upsert_flag_replaces_and_appends():
    cmd = ["python", "-m", "gpse", "train", "--threads", "4"]
    out = _upsert_flag(cmd, "--threads", "8")
    assert out[out.index("--threads") + 1] == "8"
    out2 = _upsert_flag(out, "--n_jobs", "1")
    assert out2[-2:] == ["--n_jobs", "1"]
