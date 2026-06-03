"""Global constant configuration and log message templates."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _ModelConstants:
    """Constants configuration related to model training"""
    # Optuna optimization related
    optuna_n_startup_trials: int = 10
    optuna_n_warmup_steps: int = 5
    optuna_interval_steps: int = 1
    optuna_n_jobs: int = 1  # Enforce Optuna serial execution

    # Thread environment variables
    thread_env_vars: tuple = field(default_factory=lambda: (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'
    ))

    # Default directory names
    default_results_dir: str = "optimization_results"
    default_logs_dir: str = "logs"
    default_cv_dir: str = "cv_folds"
    representative_model_dir: str = "representative_model"

    # Default file names
    main_log_file: str = "main.log"
    model_log_file: str = "model.log"
    repeat_results_file: str = "repeat_results.json"
    summary_results_file: str = "summary_results.json"
    all_predictions_file: str = "all_predictions.json"
    model_info_file: str = "info.json"
    model_pkl_file: str = "model.pkl"
    comparison_csv_file: str = "model_comparison.csv"

    # Log format templates
    log_separator_short: str = field(default_factory=lambda: "=" * 50)
    log_separator_long: str = field(default_factory=lambda: "=" * 70)

    # Numerical precision constants
    metrics_precision: int = 6
    time_precision: int = 2
    float_precision: int = 4
    correlation_min_threshold: float = 1e-10  # Threshold to avoid constant predictions


# NOTE: _LogMessages has been deprecated. All log messages are now inline strings.
# If you need to restore it, uncomment the block below.
#
# @dataclass(frozen=True)
# class _LogMessages:
#     """Log message template constants"""
#     logging_initialized: str = "Logging system initialization complete"
#     predictor_init: str = "Initializing GenomicPredictorV2, basic configuration:"
#     data_loading: str = "Loading data..."
#     data_loaded: str = "Successfully loaded genotype data: {}, phenotype data: {}"
#     common_samples: str = "Number of genotype samples: {}, number of phenotype samples: {}"
#     final_data_info: str = "Final data dimensions - Number of features: {}, number of samples: {}"
#     target_stats: str = "Target variable statistics - Mean: {:.4f}, Std: {:.4f}"
#     model_training_start: str = "Starting training for model: {} repeat: {}"
#     repeat_training_start: str = "Starting {} repetitions of training for model {}"
#     fold_training_start: str = "Training fold {}/{}"
#     parameter_optimization_start: str = "Starting parameter optimization for model {} (repeat {}/{})"
#     optimization_start: str = "Starting parameter optimization, max trials: {}"
#     fold_results: str = "Fold {} results:"
#     repeat_average_performance: str = "Repeat {} average performance:"
#     model_summary_performance: str = "Model {} - average performance across {} repeats:"
#     ensemble_test_performance: str = "Fold ensemble test set performance:"
#     data_file_not_found: str = "Failed to load data file: {}"
#     data_file_read_error: str = "Error reading data file: {}"
#     target_trait_not_found: str = "Target trait '{}' not found in phenotype data. Available columns: {}"
#     id_column_missing_geno: str = "ID column '{}' missing in genotype data"
#     id_column_missing_pheno: str = "ID column '{}' missing in phenotype data"
#     no_common_samples: str = "No common sample IDs between genotype and phenotype data"
#     dimension_mismatch: str = "Mismatch in number of samples between feature matrix and target variable: {} vs {}"


# Module-level singleton instance
ModelConstants = _ModelConstants()
# LogMessages = _LogMessages()  # deprecated
