#!/usr/bin/env bash
# Memory-aware adaptive batch launcher for GPSE low-dim core traits.
#
# Usage (with env overrides):
#   MAX_THREADS=90 MIN_FREE_MEM_GB=64 MEM_PER_TRAIT_GB=32 THREADS_PER_TRAIT=8 \
#       bash batch/launch_lowdim_adaptive.sh [--dry-run]
#
# Recommended conservative defaults (see below):
#   MAX_THREADS=90
#   MIN_FREE_MEM_GB=64
#   MEM_PER_TRAIT_GB=32
#   THREADS_PER_TRAIT=8
#   MIN_THREADS_PER_TRAIT=2
#   MAX_CONCURRENT_TRAITS=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MAX_THREADS:=90}"
: "${MIN_FREE_MEM_GB:=64}"
: "${MEM_PER_TRAIT_GB:=32}"
: "${THREADS_PER_TRAIT:=8}"
: "${MIN_THREADS_PER_TRAIT:=2}"
: "${MAX_CONCURRENT_TRAITS:=2}"
: "${MAX_MEM_RETRIES:=2}"
: "${CONFIG:=${SCRIPT_DIR}/lowdim_core_adaptive.yaml}"
: "${RESULTS_ROOT:=${SCRIPT_DIR}/results_lowdim_adaptive}"

mkdir -p "${RESULTS_ROOT}"

echo "[launch_lowdim_adaptive] REPO_ROOT=${REPO_ROOT}"
echo "[launch_lowdim_adaptive] CONFIG=${CONFIG}"
echo "[launch_lowdim_adaptive] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[launch_lowdim_adaptive] MAX_THREADS=${MAX_THREADS}"
echo "[launch_lowdim_adaptive] MIN_FREE_MEM_GB=${MIN_FREE_MEM_GB}"
echo "[launch_lowdim_adaptive] MEM_PER_TRAIT_GB=${MEM_PER_TRAIT_GB}"
echo "[launch_lowdim_adaptive] THREADS_PER_TRAIT=${THREADS_PER_TRAIT}"
echo "[launch_lowdim_adaptive] MIN_THREADS_PER_TRAIT=${MIN_THREADS_PER_TRAIT}"
echo "[launch_lowdim_adaptive] MAX_CONCURRENT_TRAITS=${MAX_CONCURRENT_TRAITS}"

cd "${REPO_ROOT}"

exec python -m gpse.batch.adaptive_scheduler \
    --config "${CONFIG}" \
    --results-root "${RESULTS_ROOT}" \
    --max-threads "${MAX_THREADS}" \
    --min-free-mem-gb "${MIN_FREE_MEM_GB}" \
    --mem-per-trait-gb "${MEM_PER_TRAIT_GB}" \
    --threads-per-trait "${THREADS_PER_TRAIT}" \
    --min-threads-per-trait "${MIN_THREADS_PER_TRAIT}" \
    --max-concurrent-traits "${MAX_CONCURRENT_TRAITS}" \
    --max-mem-retries "${MAX_MEM_RETRIES}" \
    "$@"
