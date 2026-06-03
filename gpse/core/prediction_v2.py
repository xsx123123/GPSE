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

from gpse.config import ModelConstants
from gpse.models.model_optimizers import ModelOptimizer
from gpse.models.classification_models import ClassificationModelOptimizer
from gpse.utils.log_utils import logger_init
from gpse.core.genomic_classification import GenomicClassifier

main_logger = logger_init()

# ---------------------------------------------------------------------------
# Import method implementations from sub-modules
# ---------------------------------------------------------------------------
from gpse.core._data_io import (
    load_data,
    _standardize_phenotype,
    _inverse_standardize_phenotype,
)
from gpse.core._model_tools import (
    create_model,
    get_default_params,
    get_param_func,
    filter_model_params_unified,
    _get_default_ensemble_metrics,
    _get_default_fold_metrics,
)
from gpse.core._fold_training import (
    _train_single_fold,
    _log_fold_results,
    _calculate_fold_average_metrics,
)
from gpse.core._ensemble import _compute_ensemble_predictions
from gpse.core._optimization import optimize_model_parameters
from gpse.core._repeat_training import (
    train_and_evaluate_model_for_repeat,
    run_model_multiple_repeats,
    _run_repeat_task,
)
from gpse.core._cv_manager import prepare_cv_folds, generate_cv_folds_from_file
from gpse.core._pipeline import run_all_models
from gpse.core._topsis_config import (
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
        random_seed: int = 42,
        results_dir: str = "optimization_results",
        n_trials: int = 100,
        n_threads: int = 1,
        max_parallel_jobs: int = 1,
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
            Maximum parallel jobs (multi-process).
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

        # Set main logger (unified output to a single file)
        global main_logger
        main_logger = logger_init(
            logger_name=str(self.logs_dir / "run.log"),
            log_level="INFO",
        )

        # Initialize model optimizer
        if task_type == "regression":
            self.model_optimizer = ModelOptimizer(
                random_state=random_seed, n_threads=n_threads
            )
            self.available_models = list(self.model_optimizer.model_configs.keys())
            self.classification_optimizer = None
            self.genomic_classifier = None
        else:
            self.classification_optimizer = ClassificationModelOptimizer(
                random_state=random_seed, n_threads=n_threads, n_classes=n_classes
            )
            self.genomic_classifier = GenomicClassifier(
                n_classes=n_classes,
                results_dir=str(self.results_dir),
                random_state=random_seed,
                n_threads=n_threads,
            )
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
