"""Regression tests for SafeKernelRidge and empty-result guards.

SafeKernelRidge works around intermittent segfaults in the multithreaded
LAPACK ``dposv`` of the OpenBLAS build bundled with some scipy wheels by
pinning the internal Cholesky solve to a single thread.
"""

import io

import joblib
import numpy as np
import scipy.linalg
from sklearn.kernel_ridge import KernelRidge

from gpse.models.safe_kernel_ridge import SafeKernelRidge
from gpse.utils.genomic_utils import create_comparison_table


def _toy_data(seed=0, n=120, p=50):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.normal(size=n)


def test_matches_stock_kernel_ridge_linear():
    X, y = _toy_data()
    stock = KernelRidge(alpha=1.0, kernel="linear").fit(X, y)
    safe = SafeKernelRidge(alpha=1.0, kernel="linear").fit(X, y)
    assert np.allclose(stock.dual_coef_, safe.dual_coef_)
    assert np.allclose(stock.predict(X), safe.predict(X))


def test_matches_stock_kernel_ridge_rbf():
    X, y = _toy_data()
    stock = KernelRidge(alpha=0.5, kernel="rbf", gamma=0.1).fit(X, y)
    safe = SafeKernelRidge(alpha=0.5, kernel="rbf", gamma=0.1).fit(X, y)
    assert np.allclose(stock.predict(X), safe.predict(X))


def test_scipy_solve_restored_after_fit():
    X, y = _toy_data()
    original = scipy.linalg.solve
    SafeKernelRidge(alpha=1.0, kernel="linear").fit(X, y)
    assert scipy.linalg.solve is original


def test_joblib_roundtrip():
    X, y = _toy_data()
    model = SafeKernelRidge(alpha=1.0, kernel="linear").fit(X, y)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)
    assert np.allclose(loaded.predict(X), model.predict(X))


def test_create_comparison_table_handles_empty_results(tmp_path):
    # All-models-failed runs must warn instead of raising StopIteration.
    create_comparison_table({}, tmp_path)
