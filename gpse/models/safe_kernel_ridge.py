"""Segfault-safe KernelRidge variant.

Works around intermittent segfaults observed inside ``scipy.linalg.solve``
(``assume_a="pos"``) with the OpenBLAS build bundled in some scipy wheels:
the multithreaded LAPACK ``dposv`` path occasionally dereferences a null
pointer in its parallel kernels (``dtrsm_*copy``/``dgemv``), killing the
whole process. The crash is timing-sensitive and only ever manifests with
more than one BLAS thread.

``SafeKernelRidge`` keeps the expensive kernel-matrix computation fully
multithreaded but pins the tiny ``n_samples x n_samples`` Cholesky solve to
a single thread, which is both safe and effectively free at this size.
"""

from __future__ import annotations

from scipy import linalg
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits


class SafeKernelRidge(KernelRidge):
    """KernelRidge whose internal Cholesky solve runs single-threaded."""

    def fit(self, X, y=None, sample_weight=None):
        original_solve = linalg.solve

        def single_threaded_solve(*args, **kwargs):
            with threadpool_limits(limits=1):
                return original_solve(*args, **kwargs)

        linalg.solve = single_threaded_solve
        try:
            return super().fit(X, y=y, sample_weight=sample_weight)
        finally:
            linalg.solve = original_solve
