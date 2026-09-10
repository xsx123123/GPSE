#!/usr/bin/env bash
# Fast stacking smoke test for the GPSE pipeline, stage 2 (train) only.
#
# The full test (run_cucumber_pipeline_test.sh) verified convert and the
# rf_reg / svr_reg / xgboost_reg training paths, but was cut off by a wall
# clock limit before stacking ran (xgboost on 113k SNPs is slow). This
# script reuses the already-converted matrices and exercises the remaining
# wiring — model training + TOPSIS ranking + stacking ensemble — with three
# fast models and minimal trials, into a separate results directory.
#
# Usage:
#   bash tests/pipeline/run_cucumber_stacking_smoke.sh
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
OUT_DIR="${OUT_DIR:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE_test_result/pipeline}"
TRAIT="${TRAIT:-Days_to_flower}"

GENO="${OUT_DIR}/cucumber_${TRAIT}_genotype.parquet"
PHENO="${OUT_DIR}/cucumber_${TRAIT}_phenotype.parquet"
RESULTS_DIR="${OUT_DIR}/train_results_stacking_smoke"
LOG_FILE="${OUT_DIR}/run_cucumber_stacking_smoke.log"

cd "${GPSE_ROOT}"

gpse train \
  --geno_file "${GENO}" \
  --pheno_file "${PHENO}" \
  --target_trait "${TRAIT}" \
  --task_type regression \
  --models rf_reg knn_reg elasticnet_reg \
  --trials 2 \
  --train_folds 2 \
  --n_repeats 1 \
  --patience 2 \
  --random_seed 42 \
  --use_stacking \
  --top_n_models 3 \
  --cv_folds 2 \
  --threads 8 \
  --n_jobs 2 \
  --max_workers 3 \
  --repeat_workers 1 \
  --results_dir "${RESULTS_DIR}" >"${LOG_FILE}" 2>&1

echo "[DONE] stacking smoke test finished, results under ${RESULTS_DIR} (log: ${LOG_FILE})"
