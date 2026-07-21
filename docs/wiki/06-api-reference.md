# 06. API Reference

Public classes and function signatures, listed by module. CLI entry point: `gpse = gpse.cli:main` (`gpse/cli.py:76`).

## `gpse.convert`

### `processor.py` — Orchestrator

```python
class GenomicDataProcessor(logger=None, plink_path="plink", config_path=None,
                           auto_project_config=False, allow_extra_chr=False)
    process_genomic_data(**kwargs) -> int              # full pipeline
    vcf_to_plink(vcf_file, out_prefix)
    extract_snps(bfile, extract_file, out_prefix)
    convert_bfile_to_ped(bfile, out_prefix)
    convert_to_matrix(fileprefix, out_file=None, out_format="parquet")
    process_snp_dir(bfile, snp_dir, out_dir)
    convert_phenotype(pheno_file, out_file=None, trait_name=None, trait_col=None)
    match_genotype_phenotype(pheno_df, geno_file, out_prefix, out_format="csv")
    standardize_phenotype(pheno_df, trait_col)
```

### `genotype_matrix.py` / `phenotype.py` — Functional API

```python
vcf_to_plink(vcf_file, out_prefix, *, plink_path, config_path,
             auto_project_config, allow_extra_chr, logger)
extract_snps(...); convert_bfile_to_ped(...)
convert_to_matrix(fileprefix, out_file=None, *, out_format="parquet", logger=None)
process_snp_dir(...)

convert_phenotype(...) -> pd.DataFrame
match_genotype_phenotype(pheno_df, geno_file, out_prefix, *, out_format, logger)
standardize_phenotype(pheno_df, trait_col, *, logger)
save_scaler_params(...)
detect_phenotype_type(series, max_classes=20, min_samples_per_class=5)
```

### `qc.py` / `external.py` / `workflow.py`

```python
recode_to_numeric(fileprefix)
filter_genotype(user_params, input_prefix, output_prefix, input_flag="--bfile")
impute_genotype_beagle(user_params, input_prefix, output_prefix)
analyze_and_prune(user_params, input_prefix, output_prefix, run_imputation=False) -> Tuple[str, str]

run_command(cmd_list, *, log_file=None, logger=None)
resolve_external_tool(...); check_configured_external_tools(...); get_convert_config(...)

validate_convert_mode(parser, args) -> str
run_convert_workflow(args, mode) -> int
```

## `gpse.train`

### `predictor.py` — Main Training Class

```python
class GenomicPredictorV2(random_seed=42, results_dir="optimization_results",
                         n_trials=100, n_threads=1, max_parallel_jobs=1,
                         repeat_workers=1, test_size=0.2, n_splits=5,
                         n_repeats=100, patience=20, use_default_params=False,
                         save_models=True, save_representative=False,
                         cv_file=None, force_new_cv=False, cv_id_column="ID",
                         task_type="regression", n_classes=None,
                         standardize_phenotype=False, cv_stability_penalty=0.5,
                         optuna_per_repeat=False, split_strategy="random",
                         structure_clusters=None, feature_selection="none",
                         select_k=5000, variance_threshold=0.0,
                         select_percentile=None, genotype_imputation="none",
                         missing_genotype_code=3.0)
```

Main entry point (`_pipeline.py:206`):

```python
run_all_models(geno_file, pheno_file, target_trait, models=None,
               use_stacking=False, top_n_models=5, cv_folds=5,
               use_same_test_set=True) -> Dict[str, Dict[str, Any]]
```

Key methods (bound from submodules):

- `_model_tools.py`: `create_model(model_name, params)`, `get_default_params(model_name)`, `filter_model_params_unified(...)`
- `_optimization.py`: `optimize_model_parameters(...)`
- `_cv_manager.py`: `prepare_cv_folds` / `generate_cv_folds_from_file` / `save_train_only_cv_folds`
- `_repeat_training.py`: `train_and_evaluate_model_for_repeat` / `run_model_multiple_repeats`
- `_ensemble.py`: `_compute_ensemble_predictions` / `_save_deployment_ensemble`
- `_results.py`: leaderboard / report generation

### Feature Selection & Pipeline

```python
@dataclass FeatureSelectionConfig(method="none", select_k=5000,
                                  variance_threshold=0.0, select_percentile=None)
@dataclass GenotypeImputationConfig(method="none", missing_genotype_code=3.0)
fit_feature_selector(...); transform_features(selector, X)
make_model_artifact(...); unpack_model_artifact(artifact)
build_training_pipeline(...)        # GenotypePreprocessor → model sklearn Pipeline
```

### `stacking.py` — Stacking Ensemble

```python
class StackingEnsemble(base_models_dir, model_factory, top_n_models=5,
                       meta_model_type="ridge", cv_folds=5, random_seed=42,
                       output_dir=None, n_threads=1, use_default_params=False,
                       task_type="regression", feature_selection_config=None,
                       genotype_imputation_config=None)
    create_meta_features(X_train, y_train, X_test=None) -> (meta_train, meta_test)  # strict OOF
```

### `topsis.py` — Multi-Criteria Evaluation

```python
class TOPSISEvaluator(logger=None)
    entropy_weight_method(data)
    topsis(data, weights, criteria_types, min_transform="reciprocal")
    evaluate(input_file, output_file, criteria=None, criteria_types=None,
             simple_output=None, manual_weights="0.8,0.2",
             min_transform="reciprocal", use_entropy_weights=False) -> DataFrame
```

## `gpse.models`

Canonical modules are `regression_model_optimizer.py` and `classification_model_optimizer.py`; `model_optimizers.py` and `classification_models.py` are backward-compatible import shims.

```python
class RegressionModelOptimizer(random_seed=None, n_threads=1,
                               random_state=None, catboost_train_dir=None)
    model_configs: Dict[str, ModelConfig]   # 15 regression models with Optuna search spaces (incl. gblup_reg)

class ClassificationModelOptimizer(...)
    create_classification_model(model_name, params)
    get_classification_default_params(model_name)
    get_available_models()
```

## `gpse.tasks`

```python
class GenomicClassifier(n_classes, results_dir=None, random_seed=None,
                        n_threads=1, random_state=None, classification_optimizer=None)
    prepare_classification_labels(y, results_dir) -> pd.Series   # LabelEncoder + label_encoder.pkl
    calculate_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]
```

## `gpse.predict`

```python
predict(model, genotype_file, output_file, *, missing_value=3.0,
        report_file=None, min_feature_coverage=0.0) -> dict   # returns alignment report, writes CSV+JSON
resolve_model_path(model) -> Path
load_genotype_matrix(path) -> DataFrame
load_vcf_matrix(path) -> (DataFrame, list[str])
align_features(genotype, model_features, *, missing_value=3.0) -> (DataFrame, dict)
```

## `gpse.batch`

`gpse/batch/` contains `cli.py` (`--config`, `--dry_run`), `runner.py`, and `merge.py`.

```python
# runner.py — YAML-driven multi-trait training
load_batch_config(config_path) -> (defaults, traits)
build_trait_argv(trait, defaults, actions) -> list[str]   # trait options → gpse train argv
run_batch(config_path, dry_run=False) -> int

# merge.py — merge per-trait summary tables into <results_root>/merged/
merge_trait_results(trait_dirs, output_dir) -> list[Path]   # adds a leading Trait column
```

## `gpse.tools`

`gpse/tools/` contains `cli.py` (the `split` subcommand), `split.py`, and `analyze_phenotypes.py`.

```python
# split.py — split matched genotype/phenotype samples into train/test subsets
split_dataset(geno_file, pheno_file, out_prefix, test_ratio=0.2, seed=42,
              stratify_column=None, id_column=None) -> dict
# writes <prefix>_train_geno, <prefix>_train_pheno, <prefix>_test_geno, <prefix>_test_pheno

# analyze_phenotypes.py — internal phenotype analysis helper (no CLI registration)
analyze_phenotype_file(file_path)
```

## `gpse.utils`

```python
# configuration.py — config loading (default.yaml → gpse.yaml → gpse.local.yaml → --config deep merge)
load_software_config(...); load_default_config(...); load_topsis_config(...)
get_loaded_project_configs(search_dir=None)

# genomic_utils.py — metrics and CV utilities
calculate_metrics(y_true, y_pred)
prepare_cv_folds(...); generate_cv_folds_from_file(...)
prepare_train_test_data(X, y, repeat_idx, ...)
train_fold_model(...); calculate_repeat_statistics(...); create_holdout_indices(...)

# snp_ids.py — canonical SNP IDs
canonical_snp_id(chromosome, position, reference=None) -> str
normalize_chromosome(...); canonical_ids_from_map_file(map_file)

# feature_manifest.py
write_feature_manifest(...); read_feature_manifest(path); find_feature_manifest(artifact_path)

# paralle.py — parallelism validation and derivation
validate_parallelism(n_jobs, max_workers, repeat_workers, logger=None)
derive_parallelism_from_threads(threads, n_models, n_repeats, ...)

# log_utils.py / dependency_checker.py
logger_init(logger_name=None, log_level="INFO", ...)
check_external_tool(...); check_all_external_tools(...)
```

## `gpse.config`

`gpse/config/__init__.py` exports: `ModelConstants`, `ModelConfig`, `ClassificationModelConfig`, `NumpyEncoder`.
