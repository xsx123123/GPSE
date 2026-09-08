# gpse.utils.paralle has been renamed to gpse.utils.parallel (typo fix).
#
# Backward-compatibility shim: re-export public names so any lingering
# ``from gpse.utils.paralle import X`` callers keep working until they update
# their import statements. Prefer ``from gpse.utils.parallel import ...`` in
# new code.

from gpse.utils.parallel import (
    derive_parallelism_from_threads,
    get_available_cpu_cores,
    graceful_process_pool,
    validate_parallelism,
)

__all__ = [
    "derive_parallelism_from_threads",
    "get_available_cpu_cores",
    "graceful_process_pool",
    "validate_parallelism",
]
