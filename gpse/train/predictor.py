#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Genomic Prediction Standardized Pipeline Module v2
===================================================

Features:
- Support for regression and classification model evaluation and prediction
- Repeated K-fold cross-validation with reproducible fold splits
- Pearson correlation coefficient as the primary evaluation metric for regression
- Accuracy, F1 score, AUC, etc. for classification evaluation
- Support for default parameters and Optuna automatic hyperparameter tuning
- Parallel execution of multiple training repeats
- Each model limited to single-threaded execution
- Detailed saving of all training repeat and fold results
- Support for selecting best models for ensemble learning (Stacking)
- Hierarchical logging system with independent logs for parallel tasks
- Support for automatic classification label encoding and decoding
- TOPSIS multi-criteria decision model ranking
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union
import datetime

from gpse.config import ModelConstants
from gpse.models.regression_model_optimizer import RegressionModelOptimizer
from gpse.tasks.classification import GenomicClassifier
from gpse.utils.log_utils import logger_init


class _LazyMainLogger:
    """Delay logger_init() until the first actual log call.

    This prevents spurious "GPSE config loaded" messages when a user
    simply runs ``gpse train`` without arguments and only wants help.
    """
    _real = None

    @classmethod
    def _get(cls):
        if cls._real is None:
            from gpse.utils.log_utils import logger_init
            cls._real = logger_init()
        return cls._real

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        if name in ("_real",):
            super().__setattr__(name, value)
        else:
            setattr(self._get(), name, value)


main_logger = _LazyMainLogger()

# ---------------------------------------------------------------------------
# Import method implementations from sub-modules
# ---------------------------------------------------------------------------
from gpse.train._data_io import (
    load_data,
    _standardize_phenotype,
    _inverse_standardize_phenotype,
)
from gpse.train._model_tools import (
    create_model,
    get_default_params,
    get_param_func,
    filter_model_params_unified,
    _get_default_ensemble_metrics,
    _get_default_fold_metrics,
)
from gpse.train._fold_training import (
    _train_single_fold,
    _log_fold_results,
    _calculate_fold_average_metrics,
)
from gpse.train._ensemble import _compute_ensemble_predictions, _save_deployment_ensemble
from gpse.train._optimization import optimize_model_parameters
from gpse.train._repeat_training import (
    train_and_evaluate_model_for_repeat,
    run_model_multiple_repeats,
    _run_repeat_task,
)
from gpse.train._cv_manager import (
    prepare_cv_folds,
    generate_cv_folds_from_file,
    save_train_only_cv_folds,
)
from gpse.train._pipeline import run_all_models
from gpse.train._feature_selection import (
    validate_feature_selection_config,
    validate_genotype_imputation_config,
)
from gpse.config._topsis_config import (
    get_topsis_configuration,
    _save_representative_model,
    log_environment_settings,
)


class GenomicPredictorV2:
    """
    Genomic Predictor Class V2
    ==========================
    Supports repeated K-fold cross-validation and task-level logging.
    """

    def __init__(
        self,
        random_seed: int = 42,  # The answer to the ultimate question of life, the universe, and everything is 42.
        results_dir: str = "optimization_results",
        n_trials: int = 100,
        n_threads: int = 1,
        max_parallel_jobs: int = 1,
        repeat_workers: int = 1,
        test_size: float = 0.2,
        n_splits: int = 5,
        n_repeats: int = 100,
        patience: int = 20,
        use_default_params: bool = False,
        save_models: bool = True,
        save_representative: bool = False,
        cv_file: str = None,
        force_new_cv: bool = False,
        cv_id_column: str = "ID",
        task_type: str = "regression",
        n_classes: int = None,
        standardize_phenotype: bool = False,
        cv_stability_penalty: float = 0.5,
        optuna_per_repeat: bool = False,
        split_strategy: str = "random",
        structure_clusters: int = None,
        feature_selection: str = "none",
        select_k: int | None = 5000,
        variance_threshold: float = 0.0,
        select_percentile: float | None = None,
        genotype_imputation: str = "none",
        missing_genotype_code: float = 3.0,
        topsis_config: str = None,
    ):
        """
        Initialize the predictor.

        Parameters
        ----------
        random_seed : int
            Base random seed.
        results_dir : str
            Results output directory.
        n_trials : int
            Optuna optimization trial count.
        n_threads : int
            Threads per model (usually 1).
        max_parallel_jobs : int
            Maximum models trained in parallel.
        repeat_workers : int
            Maximum repeats trained in parallel within each model.
        test_size : float
            Test set proportion.
        n_splits : int
            Cross-validation fold count.
        n_repeats : int
            Number of repeats.
        patience : int
            Early stopping patience.
        use_default_params : bool
            Whether to use model default parameters.
        save_models : bool
            Whether to save trained models.
        save_representative : bool
            Whether to save the representative model closest to average performance.
        cv_file : str, optional
            Path to CV file; created if it does not exist.
        force_new_cv : bool
            Force generation of a new CV file even if one already exists.
        cv_id_column : str
            ID column name in phenotype data for CV file generation.
        task_type : str
            Task type ('regression' or 'classification').
        n_classes : int, optional
            Number of classes for classification (required for classification tasks).
        standardize_phenotype : bool
            Whether to standardize phenotype data (regression only).
        """
        self.random_seed = random_seed
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.n_threads = n_threads
        self.max_parallel_jobs = max_parallel_jobs
        self.repeat_workers = repeat_workers
        self.test_size = test_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.patience = patience
        self.use_default_params = use_default_params
        self.save_models = save_models
        self.save_representative = save_representative
        self.cv_file = cv_file
        self.force_new_cv = force_new_cv
        self.cv_id_column = cv_id_column
        self.task_type = task_type
        self.n_classes = n_classes
        self.standardize_phenotype = standardize_phenotype
        if cv_stability_penalty < 0:
            raise ValueError("cv_stability_penalty must be non-negative")
        self.cv_stability_penalty = cv_stability_penalty
        self.optuna_per_repeat = optuna_per_repeat
        if split_strategy not in {"random", "structure_aware"}:
            raise ValueError("split_strategy must be 'random' or 'structure_aware'")
        if structure_clusters is not None and structure_clusters < 2:
            raise ValueError("structure_clusters must be at least 2 when supplied")
        self.split_strategy = split_strategy
        self.structure_clusters = structure_clusters
        self.feature_selection_config = validate_feature_selection_config(
            feature_selection, select_k, variance_threshold, select_percentile
        )
        self.genotype_imputation_config = validate_genotype_imputation_config(
            genotype_imputation, missing_genotype_code
        )
        self.topsis_config = topsis_config
        self._optimization_cache = {}
        self.phenotype_scaler = None  # Store phenotype standardization parameters

        # Validate task type
        if task_type not in ["regression", "classification"]:
            raise ValueError("task_type must be 'regression' or 'classification'")

        if task_type == "classification" and n_classes is None:
            raise ValueError("n_classes must be specified for classification tasks")

        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Create logs directory
        self.logs_dir = self.results_dir / ModelConstants.default_logs_dir
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        # Generate timestamped log filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log_filename = f"gpse_{timestamp}.log"

        # Set main logger (unified output to a single file)
        global main_logger
        main_logger = logger_init(
            logger_name=str(self.logs_dir / self.main_log_filename),
            log_level="INFO",
        )

        # Initialize model optimizer
        if task_type == "regression":
            self.model_optimizer = RegressionModelOptimizer(
                random_seed=random_seed, n_threads=n_threads,
                catboost_train_dir=str(self.results_dir / "catboost_info"),
            )
            self.available_models = list(self.model_optimizer.model_configs.keys())
            self.classification_optimizer = None
            self.genomic_classifier = None
        else:
            self.genomic_classifier = GenomicClassifier(
                n_classes=n_classes,
                results_dir=str(self.results_dir),
                random_seed=random_seed,
                n_threads=n_threads,
            )
            self.classification_optimizer = self.genomic_classifier.classification_optimizer
            self.available_models = self.classification_optimizer.get_available_models()
            self.model_optimizer = None

        main_logger.info(f"Task type: {task_type}")
        main_logger.info(f"Available models: {', '.join(self.available_models)}")

        # Log environment settings
        self.log_environment_settings()

    # -----------------------------------------------------------------------
    # Bind imported functions as methods
    # -----------------------------------------------------------------------
    load_data = load_data
    _standardize_phenotype = _standardize_phenotype
    _inverse_standardize_phenotype = _inverse_standardize_phenotype
    save_train_only_cv_folds = save_train_only_cv_folds
    create_model = create_model
    get_default_params = get_default_params
    get_param_func = get_param_func
    filter_model_params_unified = filter_model_params_unified
    _get_default_ensemble_metrics = _get_default_ensemble_metrics
    _get_default_fold_metrics = _get_default_fold_metrics
    _train_single_fold = _train_single_fold
    _log_fold_results = _log_fold_results
    _calculate_fold_average_metrics = _calculate_fold_average_metrics
    _compute_ensemble_predictions = _compute_ensemble_predictions
    _save_deployment_ensemble = _save_deployment_ensemble
    optimize_model_parameters = optimize_model_parameters
    train_and_evaluate_model_for_repeat = train_and_evaluate_model_for_repeat
    run_model_multiple_repeats = run_model_multiple_repeats
    _run_repeat_task = _run_repeat_task
    prepare_cv_folds = prepare_cv_folds
    generate_cv_folds_from_file = generate_cv_folds_from_file
    run_all_models = run_all_models
    get_topsis_configuration = get_topsis_configuration
    _save_representative_model = _save_representative_model
    log_environment_settings = log_environment_settings
