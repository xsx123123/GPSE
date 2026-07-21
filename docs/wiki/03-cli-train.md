# 03. Subcommand `gpse train`

Trains genomic prediction models: Optuna hyperparameter optimization, repeated cross-validation, TOPSIS ranking, and Stacking ensemble. Implemented in `gpse/train/` (CLI: `gpse/train/cli.py`, argument parsing: `gpse/train/workflow.py`).

## Minimal Example

```bash
gpse train --task_type regression \
  --geno_file geno.csv --pheno_file pheno.csv --target_trait yield
```

Classification additionally requires `--n_classes`:

```bash
gpse train --task_type classification --n_classes 3 \
  --geno_file geno.csv --pheno_file pheno.csv --target_trait grade
```

## Option Groups

### Basic

| Option | Default | Description |
|--------|---------|-------------|
| `--task_type` | required | `regression` / `classification` |
| `--geno_file` `--pheno_file` `--target_trait` | required (normal mode) | Genotype matrix, phenotype file, target trait |
| `--n_classes` | — | Number of classes for classification |
| `--results_dir` | `optimization_results_v2` | Output directory |
| `--random_seed` | 42 | Random seed |

### Models & Optimization

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | all | See the model list below |
| `--trials` | 100 | Optuna trials |
| `--patience` | 20 | Early-stopping patience |
| `--use_default_params` | False | Skip optimization, use default parameters |
| `--optuna_per_repeat` / `--no-optuna_per_repeat` | False | Tune independently per repeat |
| `--cv_stability_penalty` | 0.5 | CV stability penalty coefficient |
| `--save_models` / `--save_representative` | — | Save models / representative model |

Regression models (15): `elasticnet_reg, gbdt_reg, svr_reg, mlp_reg, knn_reg, rf_reg, xgboost_reg, adaboost_reg, lightgbm_reg, catboost_reg, kernelridge_reg, gblup_reg, histgradientboost_reg, sgd_reg, lasso_reg`
Classification models (6): `rf_clf, xgboost_clf, lightgbm_clf, catboost_clf, svm_clf, mlp_clf`

### Cross-Validation

| Option | Default | Description |
|--------|---------|-------------|
| `--test_size` | 0.2 | Fixed hold-out fraction |
| `--train_folds` | 5 | Training-set CV folds (n_splits) |
| `--n_repeats` | 100 | Number of repeats |
| `--cv_file` / `--force_new_cv` | — | Reuse / force-regenerate CV splits |
| `--cv_id_column` | `ID` | ID column in the CV file |
| `--use_same_test_set` / `--no-use_same_test_set` | True | All models share the same test set |
| `--split_strategy` | `random` | `random` / `structure_aware` (with `--structure_clusters`) |

### Features & Imputation

| Option | Default | Description |
|--------|---------|-------------|
| `--feature_selection` | `none` | `variance / univariate / mutual_info / tree_importance / lasso / elasticnet / pca / svd` |
| `--select_k` | 5000 | Maximum number of features |
| `--select_percentile` / `--variance_threshold` | — | Percentile / variance threshold |
| `--genotype_imputation` | `none` | `none` / `mean` |
| `--missing_genotype_code` | 3 | Missing genotype code |
| `--standardize_phenotype` | False | Standardize the phenotype |

### Stacking Ensemble

| Option | Default | Description |
|--------|---------|-------------|
| `--use_stacking` | False | Enable Stacking |
| `--top_n_models` | 5 | Number of base models (selected by TOPSIS ranking) |
| `--cv_folds` | 5 | OOF meta-feature folds |

### Parallelism

| Option | Default | Description |
|--------|---------|-------------|
| `--threads` | — | One-shot derivation of the three options below |
| `--n_jobs` | 1 | Intra-model threads |
| `--max_workers` | 1 | Model-level parallelism |
| `--repeat_workers` | 1 | Repeat-level parallelism |

### Preprocessing Mode

`--enable_preprocess` runs the convert pipeline inline before training; `--preprocess_only` runs preprocessing only. Companion options mirror `gpse convert`: `--preprocess_prefix --vcf_file --bfile --ped_file --map_file --extract_file --snp_dir --direct_convert --matrix_file --raw_pheno_file --skip_matrix_conversion --skip_phenotype_match --skip_data_clean --load_matrix_info --plink_path --plink_out`.

## Training Artifacts

Training preserves the real SNP feature names (no renaming to `feature_0`/`feature_1`) and records them in `feature_manifest.json`. Under `results_dir` (see `train_pipeline.md` §12):

- `feature_manifest.json` / `split_manifest.json`: feature and data-split manifests;
- `deployment_ensemble/member_*.pkl`: deployable ensemble models;
- `model_comparison_cv*.csv`: model comparison leaderboards; `reports/`: evaluation reports.

## Batch Multi-Trait Training (`gpse batch`)

`gpse batch` trains many traits in one run from a single YAML config. Each trait executes the full `gpse train` workflow with its own output directory; one trait's failure does not stop the rest, and a per-trait summary is printed at the end. Implemented in `gpse/batch/`.

```bash
gpse batch --config batch_config.yaml
# Preview the generated per-trait commands without running them:
gpse batch --config batch_config.yaml --dry_run
```

The YAML has two sections. `defaults` accepts any `gpse train` option (same names as the command-line flags) plus the batch-only `results_root`. Each `traits` entry requires a `name` (used as `--target_trait`) and may override any option — including `task_type`, so regression and classification traits can be mixed in one batch:

```yaml
defaults:
  geno_file: maize_geno.csv
  pheno_file: maize_pheno.csv        # multi-trait table; `name` selects the column
  task_type: regression
  use_default_params: true
  use_stacking: true
  n_repeats: 2
  threads: 60
  results_root: results/             # each trait writes to results/<name>/

traits:
  - name: FT                         # inherits everything from defaults
    cv_file: cv_folds/FT_cv_50x5.csv
  - name: FW
    models: [rf_reg, xgboost_reg, gblup_reg]   # this trait only
  - name: color
    task_type: classification        # per-trait task type
    n_classes: 3
    results_dir: results_color/      # overrides results_root
  - name: DTF
    enabled: false                   # skipped but kept in the config
```

- Each trait writes to `<results_root>/<name>/` unless it sets `results_dir`;
- After the run, per-trait summary tables (`model_comparison*.csv` and the holdout summaries under `reports/`) are merged across traits — with a leading `Trait` column — into `<results_root>/merged/`;
- Option names that are not valid `gpse train` flags are rejected with an error; Ctrl+C aborts the whole batch;
- Template: `batch/batch_config.example.yaml`. Real-world validation configs: `tests/validation/*_batch_config.yaml`.

## Corresponding API

See [06. API Reference → train](06-api-reference.md#gpsetrain) and [→ batch](06-api-reference.md#gpsebatch).
