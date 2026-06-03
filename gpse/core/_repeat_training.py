#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Repeat Training Module
======================
Coordinates training and evaluation for individual repeats, including parallel execution.
"""

import os
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from loguru import logger as main_logger
except ImportError:
    main_logger = None

try:
    from ..config import ModelConstants
    from ..config.constants import NumpyEncoder
    from ..utils.genomic_utils import (
        generate_repeat_seed,
        generate_fold_seed,
        create_repeat_result_directory,
        prepare_train_test_data,
        create_model_result_directory,
        calculate_repeat_statistics,
        find_representative_repeat,
    )
    from ..utils.log_utils import logger_init
except ImportError:
    from config import ModelConstants
    from config.constants import NumpyEncoder
    from utils.genomic_utils import (
        generate_repeat_seed,
        generate_fold_seed,
        create_repeat_result_directory,
        prepare_train_test_data,
        create_model_result_directory,
        calculate_repeat_statistics,
        find_representative_repeat,
    )
    from utils.log_utils import logger_init


def train_and_evaluate_model_for_repeat(
    self,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    repeat_idx: int,
    test_indices: np.ndarray = None,
    task_logger=None,
    cv_pheno_data=None,
) -> Dict[str, Any]:
    """
    Train and evaluate a model for one repeat.

    Parameters
    ----------
    model_name : str
        Model name.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    repeat_idx : int
        Repeat index.
    test_indices : np.ndarray, optional
        Test set indices.
    task_logger
        Task-specific logger.
    cv_pheno_data : pd.DataFrame, optional
        Phenotype data with CV groups.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing training results.
    """
    if task_logger is None:
        task_logger = main_logger

    repeat_seed = generate_repeat_seed(self.random_seed, repeat_idx)
    np.random.seed(repeat_seed)

    task_logger.info(f"\n{'=' * 50}")
    task_logger.info(f"Model {model_name} - Repeat {repeat_idx + 1}/{self.n_repeats}")
    task_logger.info(f"{'=' * 50}")
    main_logger.info(f"Starting training Model:{model_name} Repeat:{repeat_idx + 1}")

    repeat_dir = create_repeat_result_directory(self.results_dir, model_name, repeat_idx)
    train_indices, X_train, y_train, X_test, y_test = prepare_train_test_data(
        X, y, repeat_idx, self.random_seed, self.test_size, test_indices
    )
    task_logger.info(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")

    folds = self.generate_cv_folds_from_file(
        X_train, y_train, cv_pheno_data, repeat_idx, task_logger
    )

    if self.use_default_params:
        params = self.get_default_params(model_name)
        task_logger.info(f"Using default parameters: {params}")
        optimization_info = {
            "best_params": params,
            "best_value": None,
            "n_trials": 0,
            "early_stopped": False,
        }
    else:
        optimization_info = self.optimize_model_parameters(
            X_train, y_train, model_name, repeat_idx, task_logger, cv_pheno_data
        )
        params = optimization_info["best_params"]

    model = self.create_model(model_name, params)
    fold_results = []
    all_predictions = {"train": [], "val": [], "test": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_seed = generate_fold_seed(self.random_seed, repeat_idx, fold_idx)
        np.random.seed(fold_seed)
        task_logger.info(f"Training fold {fold_idx + 1}/{self.n_splits}...")

        fold_result = self._train_single_fold(
            model,
            model_name,
            params,
            fold_idx,
            train_idx,
            val_idx,
            X_train,
            y_train,
            X_test,
            y_test,
            test_indices,
            repeat_idx,
            repeat_dir,
            all_predictions,
            task_logger,
        )
        if fold_result:
            fold_results.append(fold_result)

    import json

    predictions_path = repeat_dir / "all_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(all_predictions, f, indent=2, cls=NumpyEncoder)

    if fold_results:
        avg_metrics = self._calculate_fold_average_metrics(
            fold_results, repeat_idx, task_logger
        )
        ensemble_metrics = self._compute_ensemble_predictions(
            model_name, params, folds, X, X_train, y_train, X_test, y_test, repeat_dir, task_logger
        )
    else:
        task_logger.error("All folds failed, cannot compute average performance")
        avg_metrics = self._get_default_fold_metrics()
        ensemble_metrics = self._get_default_ensemble_metrics()

    repeat_results = {
        "model_name": model_name,
        "repeat_idx": repeat_idx,
        "params": params,
        "fold_results": fold_results,
        "avg_training_time": avg_metrics.get("avg_training_time", 0.0),
        "ensemble_metrics": ensemble_metrics,
        "optimization_info": optimization_info,
        "test_indices": (
            test_indices.tolist()
            if hasattr(test_indices, "tolist")
            else list(test_indices)
        ),
        "all_predictions_file": str(predictions_path),
    }
    repeat_results.update(avg_metrics)

    repeat_results_path = repeat_dir / "repeat_results.json"
    with open(repeat_results_path, "w") as f:
        json.dump(repeat_results, f, indent=2, cls=NumpyEncoder)

    task_logger.info(f"Repeat {repeat_idx + 1} results saved to {repeat_results_path}")
    return repeat_results


def run_model_multiple_repeats(
    self,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_pheno_data: pd.DataFrame,
    use_same_test_set: bool = True,
) -> Dict[str, Any]:
    """
    Execute multiple repeated training runs for a single model.

    Parameters
    ----------
    model_name : str
        Model name.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    cv_pheno_data : pd.DataFrame
        Phenotype data containing CV groups.
    use_same_test_set : bool
        Whether to use the same test set across all repeats.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all repeat results.
    """
    main_logger.info(f"\n{'=' * 70}")
    main_logger.info(
        f"Starting {self.n_repeats} repeated training runs for model {model_name}"
    )
    main_logger.info(f"{'=' * 70}")

    model_dir = create_model_result_directory(self.results_dir, model_name)

    test_indices = None
    if use_same_test_set:
        _, test_indices = train_test_split(
            range(len(X)), test_size=self.test_size, random_state=self.random_seed
        )
        main_logger.info(f"All repeats will use the same test set (size: {len(test_indices)})")

    all_repeat_results = []

    if self.max_parallel_jobs > 1:
        main_logger.info(f"Using {self.max_parallel_jobs} parallel tasks for training")

        with ProcessPoolExecutor(
            max_workers=self.max_parallel_jobs, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    self._run_repeat_task,
                    model_name,
                    X,
                    y,
                    repeat_idx,
                    test_indices,
                    cv_pheno_data,
                )
                for repeat_idx in range(self.n_repeats)
            ]

            for i, future in enumerate(as_completed(futures)):
                try:
                    repeat_results = future.result()
                    all_repeat_results.append(repeat_results)
                    main_logger.info(f"Completed repeat {i + 1}/{self.n_repeats}")
                except Exception as e:
                    main_logger.error(f"Repeat execution failed: {str(e)}")
    else:
        main_logger.info("Executing training sequentially")
        for repeat_idx in range(self.n_repeats):
            try:
                repeat_results = self.train_and_evaluate_model_for_repeat(
                    model_name,
                    X,
                    y,
                    repeat_idx,
                    test_indices,
                    task_logger=None,
                    cv_pheno_data=cv_pheno_data,
                )
                all_repeat_results.append(repeat_results)
            except Exception as e:
                main_logger.error(f"Repeat {repeat_idx + 1} execution failed: {str(e)}")

    if all_repeat_results:
        summary = calculate_repeat_statistics(all_repeat_results, self.task_type)

        if self.task_type == "classification":
            avg_ensemble_metric = summary["avg_ensemble_accuracy"]
            metric_name = "Accuracy"
            metric_key = "ensemble_accuracy_values"
        else:
            avg_ensemble_metric = summary["avg_ensemble_pearson"]
            metric_name = "Pearson correlation"
            metric_key = "ensemble_pearson_values"

        closest_repeat_idx, closest_repeat, difference = find_representative_repeat(
            all_repeat_results, avg_ensemble_metric
        )

        main_logger.info(
            f"Found representative repeat closest to average: Repeat {closest_repeat_idx + 1}"
        )
        main_logger.info(
            f"Its ensemble {metric_name}: "
            f"{summary['raw_values'][metric_key][closest_repeat_idx]:.{ModelConstants.metrics_precision}f}"
        )
        main_logger.info(
            f"Average ensemble {metric_name}: "
            f"{avg_ensemble_metric:.{ModelConstants.metrics_precision}f}"
        )
        main_logger.info(
            f"Difference: {difference:.{ModelConstants.metrics_precision}f}"
        )

        if self.task_type == "classification":
            repeat_info = {
                "repeat_idx": int(closest_repeat_idx),
                "ensemble_accuracy": float(
                    summary["raw_values"]["ensemble_accuracy_values"][closest_repeat_idx]
                ),
                "difference_from_avg": float(difference),
            }
        else:
            repeat_info = {
                "repeat_idx": int(closest_repeat_idx),
                "ensemble_pearson": float(
                    summary["raw_values"]["ensemble_pearson_values"][closest_repeat_idx]
                ),
                "difference_from_avg": float(difference),
            }

        model_path = self._save_representative_model(
            model_name, closest_repeat["params"], X, y, model_dir, repeat_info
        )

        if model_path:
            main_logger.info(f"Representative model saved to {model_path}")
            if self.task_type == "classification":
                summary["representative_model"] = {
                    "repeat_idx": int(closest_repeat_idx),
                    "ensemble_accuracy": float(
                        summary["raw_values"]["ensemble_accuracy_values"][closest_repeat_idx]
                    ),
                    "model_path": model_path,
                }
            else:
                summary["representative_model"] = {
                    "repeat_idx": int(closest_repeat_idx),
                    "ensemble_pearson": float(
                        summary["raw_values"]["ensemble_pearson_values"][closest_repeat_idx]
                    ),
                    "model_path": model_path,
                }

        model_name_summary = summary["model_name"]
        n_repeats = summary["n_repeats"]

        main_logger.info(f"\n{ModelConstants.log_separator_short}")
        main_logger.info(
            f"Model {model_name_summary} - Average performance over {n_repeats} repeats:"
        )

        if self.task_type == "classification":
            main_logger.info(
                f"Train Accuracy: {summary['avg_train_accuracy']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_train_accuracy']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Val Accuracy: {summary['avg_val_accuracy']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_val_accuracy']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Test Accuracy: {summary['avg_test_accuracy']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_test_accuracy']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Test F1: {summary['avg_test_f1']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_test_f1']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Test AUC: {summary['avg_test_auc']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_test_auc']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Ensemble Accuracy: {summary['avg_ensemble_accuracy']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_ensemble_accuracy']:.{ModelConstants.metrics_precision}f})"
            )
        else:
            main_logger.info(
                f"Train Pearson: {summary['avg_train_pearson']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_train_pearson']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Val Pearson: {summary['avg_val_pearson']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_val_pearson']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Test Pearson: {summary['avg_test_pearson']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_test_pearson']:.{ModelConstants.metrics_precision}f})"
            )
            main_logger.info(
                f"Ensemble Pearson: {summary['avg_ensemble_pearson']:.{ModelConstants.metrics_precision}f} "
                f"(±{summary['std_ensemble_pearson']:.{ModelConstants.metrics_precision}f})"
            )

        main_logger.info(
            f"Average training time: {summary['avg_training_time']:.{ModelConstants.time_precision}f}s"
        )
        main_logger.info(f"{ModelConstants.log_separator_short}")

        import json

        summary_path = model_dir / "summary_results.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        main_logger.info(
            f"Model {model_name_summary} summary results saved to {summary_path}"
        )

        return summary
    else:
        main_logger.error(f"Model {model_name} failed on all repeats")
        return {"model_name": model_name, "n_repeats": 0, "error": "All repeats failed"}


def _run_repeat_task(self, model_name, X, y, repeat_idx, test_indices, cv_pheno_data):
    """Wrapper function for parallel processing."""
    task_logger = logger_init(
        logger_name=str(self.results_dir / ModelConstants.default_logs_dir / "run.log"),
        log_level="DEBUG",
    )

    task_logger.info(
        f"Subprocess (PID={os.getpid()}) starting {model_name} model repeat {repeat_idx + 1}"
    )

    try:
        result = self.train_and_evaluate_model_for_repeat(
            model_name,
            X,
            y,
            repeat_idx,
            test_indices,
            task_logger=task_logger,
            cv_pheno_data=cv_pheno_data,
        )
        task_logger.info(
            f"Subprocess (PID={os.getpid()}) successfully completed {model_name} model repeat {repeat_idx + 1}"
        )
        return result
    except Exception as e:
        task_logger.error(f"Subprocess execution failed: {str(e)}")
        task_logger.error(traceback.format_exc())
        raise
