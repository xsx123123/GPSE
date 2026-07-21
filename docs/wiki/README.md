# GPSE Wiki

> GPSE (Genomic Prediction with Stacking Ensemble) — a machine learning pipeline for genomic selection in horticultural crops.

This Wiki describes the current version (v0.0.3a1, Alpha): what the software does, what each subcommand offers, and the Python API.

## Contents

| Page | Topic |
|------|-------|
| [01. Overview](01-overview.md) | Purpose, key features, architecture, dependencies |
| [02. `gpse convert`](02-cli-convert.md) | Data conversion subcommand: pipeline / qc / recode / deps modes |
| [03. `gpse train`](03-cli-train.md) | Training subcommand: hyperparameter optimization, repeated CV, Stacking ensemble; plus `gpse batch` multi-trait training |
| [04. `gpse predict`](04-cli-predict.md) | Prediction subcommand: feature alignment, coverage checks, phenotype prediction |
| [05. Configuration](05-configuration.md) | `gpse.yaml` / package configs / TOPSIS config / batch YAML |
| [06. API Reference](06-api-reference.md) | Public classes and function signatures per module |

## Quick Start

```bash
# 1. Check external dependencies
gpse convert --check-deps

# 2. Convert data (VCF → training matrices)
gpse convert --vcf data.vcf.gz --pheno pheno.csv --out-prefix out/run

# 3. Train (regression, Optuna optimization + Stacking)
gpse train --task_type regression \
  --geno_file out/run_geno.csv --pheno_file out/run_pheno.csv \
  --target_trait yield --models xgboost_reg lightgbm_reg \
  --trials 100 --n_repeats 100 --use_stacking --threads 16

# 3b. Or train many traits at once from a YAML config
gpse batch --config batch_config.yaml [--dry_run]

# 4. Predict new samples
gpse predict --model optimization_results_v2/deployment_ensemble \
  --vcf-file new_samples.vcf.gz --out predictions.csv
```

### Splitting a held-out test set

Use `gpse tools split` to carve out a test set before training; score it later with `gpse predict`:

```bash
gpse tools split --geno out/run_geno.csv --pheno out/run_pheno.csv \
  --out-prefix data/split --test-ratio 0.2 --seed 42
```

This writes `data/split_train_geno`, `data/split_train_pheno`, `data/split_test_geno`, and `data/split_test_pheno`.
