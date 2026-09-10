#!/usr/bin/env bash
# GPSE pipeline module end-to-end smoke test (convert + train chained).
#
# Inputs:
#   - Genotype: raw_data/cucumber_raw_SNP.vcf.gz (113,854 SNPs x 1,234 samples)
#   - Phenotype: raw_data/cucumber_phenotype_format.csv (836 samples, all present in VCF)
#
# Outputs go to ${OUT_DIR}; logs are kept next to the outputs.
#
# Usage:
#   bash tests/pipeline/run_cucumber_pipeline_test.sh
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
RAW_DIR="${RAW_DIR:-/data/jzhang/project/Temp/gpse_26.6.4_test/raw_data}"
OUT_DIR="${OUT_DIR:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE_test_result/pipeline}"
TRAIT="${TRAIT:-Days_to_flower}"

VCF="${RAW_DIR}/cucumber_raw_SNP.vcf.gz"
PHENO="${RAW_DIR}/cucumber_phenotype_format.csv"
OUT_PREFIX="${OUT_DIR}/cucumber"
RESULTS_DIR="${OUT_DIR}/train_results_${TRAIT}"
LOG_FILE="${OUT_DIR}/run_cucumber_pipeline_test.log"

mkdir -p "${OUT_DIR}"
cd "${GPSE_ROOT}"

# Small, fast settings on purpose: this is a smoke test of the pipeline
# wiring (convert -> matrix lookup -> train), not a production run.
gpse pipeline \
  --vcf "${VCF}" \
  --pheno "${PHENO}" \
  --out-prefix "${OUT_PREFIX}" \
  --out-format parquet \
  --target_trait "${TRAIT}" \
  --task_type regression \
  --models rf_reg xgboost_reg svr_reg \
  --trials 5 \
  --train_folds 3 \
  --n_repeats 2 \
  --patience 5 \
  --random_seed 42 \
  --use_stacking \
  --top_n_models 3 \
  --cv_folds 3 \
  --threads 8 \
  --n_jobs 2 \
  --max_workers 2 \
  --repeat_workers 2 \
  --results_dir "${RESULTS_DIR}" >"${LOG_FILE}" 2>&1

echo "[DONE] pipeline test finished, outputs under ${OUT_DIR} (log: ${LOG_FILE})"
