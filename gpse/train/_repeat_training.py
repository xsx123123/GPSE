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
from concurrent.futures import as_completed

from loguru import logger as main_logger

from gpse.config import ModelConstants
from gpse.config.constants import NumpyEncoder
from gpse.utils.genomic_utils import (
    generate_repeat_seed,
    generate_fold_seed,
    create_repeat_result_directory,
    prepare_train_test_data,
    create_model_result_directory,
    calculate_repeat_statistics,
)
from gpse.utils.log_utils import logger_init, setup_subprocess_logging, collect_subprocess_logs
from gpse.utils.paralle import graceful_process_pool


def _init_worker_threads(n_threads: int) -> None:
    """Initializer for ProcessPoolExecutor workers to enforce thread limits.

    Sets all 6 BLAS/OpenMP environment variables BEFORE any numpy/scipy
    imports happen in the child process (spawn context re-imports modules).
    """
    from gpse.config import ModelConstants
    for _env_var in ModelConstants.thread_env_vars:
        os.environ[_env_var] = str(n_threads)


def create_holdout_indices(
    y: pd.Series,
    test_size: float,
    random_seed: int,
    task_type: str,
    X: pd.DataFrame | None = None,
    split_strategy: str = "random",
    structure_clusters: int | None = None,
    min_class_count: int | None = None,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """Create random/stratified or PCA-cluster-aware reproducible hold-out indices."""
    indices = np.arange(len(y))
    metadata = {"split_strategy": split_strategy}
    if split_strategy == "structure_aware":
        if X is None:
            raise ValueError("structure_aware splitting requires the genotype matrix X")
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import StandardScaler

        n_components = min(10, X.shape[0] - 1, X.shape[1])
        if n_components < 1:
            raise ValueError("structure_aware splitting requires at least two samples and one feature")
        n_clusters = structure_clusters or min(max(2, round(1 / test_size)), max(2, len(X) // 3))
        n_clusters = min(n_clusters, len(X))
        pcs = PCA(n_components=n_components, random_state=random_seed).fit_transform(
            StandardScaler().fit_transform(X)
        )
        groups = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit_predict(pcs)
        splitter = GroupShuffleSplit(n_splits=32, test_size=test_size, random_state=random_seed)
        best_split = None
        best_distance = float("inf")
        overall_distribution = y.value_counts(normalize=True) if task_type == "classification" else None
        for train_indices, test_indices in splitter.split(X, y, groups):
            distance = abs(len(test_indices) / len(X) - test_size)
            if overall_distribution is not None:
                test_distribution = y.iloc[test_indices].value_counts(normalize=True)
                distance += sum(
                    abs(test_distribution.get(label, 0.0) - proportion)
                    for label, proportion in overall_distribution.items()
                )
            if distance < best_distance:
                best_split = (train_indices, test_indices)
                best_distance = distance
        if best_split is None:
            raise ValueError("Unable to create a structure-aware hold-out split")
        train_indices, test_indices = best_split
        metadata.update(
            {
                "pca_components": int(n_components),
                "n_clusters": int(n_clusters),
                "cluster_sizes": {
                    str(cluster): int((groups == cluster).sum())
                    for cluster in np.unique(groups)
                },
                "train_clusters": sorted(int(cluster) for cluster in np.unique(groups[train_indices])),
                "test_clusters": sorted(int(cluster) for cluster in np.unique(groups[test_indices])),
                "test_fraction": float(len(test_indices) / len(X)),
            }
        )
        result = (np.asarray(train_indices), np.asarray(test_indices))
        return (*result, metadata) if return_metadata else result
    if split_strategy != "random":
        raise ValueError("split_strategy must be 'random' or 'structure_aware'")
    stratify = y if task_type == "classification" else None
    if task_type == "classification" and min_class_count is not None:
        class_counts = y.value_counts()
        if class_counts.min() < min_class_count:
            raise ValueError(
                "Classification hold-out/CV requires every class to contain at least "
                f"{min_class_count} samples; class counts: {class_counts.to_dict()}"
            )
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError as exc:
        if task_type == "classification":
            counts = y.value_counts().to_dict()
            raise ValueError(
                "Unable to create a stratified hold-out split. Each class needs "
                "enough samples for the requested --test_size; class counts: "
                f"{counts}"
            ) from exc
        raise
    metadata["test_fraction"] = float(len(test_indices) / len(y))
    result = (np.asarray(train_indices), np.asarray(test_indices))
    return (*result, metadata) if return_metadata else result


def select_repeat_by_train_cv(
    all_repeat_results: list[Dict[str, Any]], task_type: str
) -> tuple[int, Dict[str, Any], Dict[str, float | str]]:
    """Choose saved-model parameters strictly from train-side CV results.

    Hold-out metrics remain reporting-only.  A higher validation mean wins;
    validation standard deviation and repeat index provide deterministic
    tie-breakers.
    """
    if not all_repeat_results:
        raise ValueError("Cannot select a repeat from an empty result list")

    if task_type == "classification":
        mean_key, std_key = "avg_val_accuracy", "std_val_accuracy"
    else:
        mean_key, std_key = "avg_val_pearson", "std_val_pearson"

    def ranking_key(item: tuple[int, Dict[str, Any]]) -> tuple[float, float, int]:
        repeat_idx, result = item
        mean = float(result.get(mean_key, float("-inf")))
        std = float(result.get(std_key, float("inf")))
        return (-mean, std, int(result.get("repeat_idx", repeat_idx)))

    selected_idx, selected_repeat = min(enumerate(all_repeat_results), key=ranking_key)
    return selected_idx, selected_repeat, {
        "metric": mean_key,
        "cv_mean": float(selected_repeat[mean_key]),
        "cv_std": float(selected_repeat[std_key]),
    }


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
    if test_indices is None:
        train_indices, test_indices = create_holdout_indices(
            y,
            self.test_size,
            repeat_seed,
            self.task_type,
            X=X,
            split_strategy=self.split_strategy,
            structure_clusters=self.structure_clusters,
            min_class_count=self.n_splits + 1 if self.task_type == "classification" else None,
        )
    else:
        test_indices = np.asarray(test_indices)
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices, assume_unique=False)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    y_test_original = y_test.copy()
    phenotype_scaler = None
    if self.task_type == "regression" and self.standardize_phenotype:
        y_train, phenotype_scaler = self._standardize_phenotype(y_train)
        if phenotype_scaler["applied"]:
            y_test = (y_test - phenotype_scaler["mean"]) / phenotype_scaler["std"]
        task_logger.info(
            "Fitted phenotype scaler on hold-out training labels only: "
            f"mean={phenotype_scaler['mean']:.6f}, std={phenotype_scaler['std']:.6f}"
        )
    task_logger.info(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")

    folds = self.generate_cv_folds_from_file(
        X_train, y_train, cv_pheno_data, repeat_idx, task_logger
    )
    cv_folds_path = self.save_train_only_cv_folds(X_train, folds, repeat_dir)

    if self.use_default_params or model_name == "gblup_reg":
        params = self.get_default_params(model_name)
        task_logger.info(f"Using fixed baseline/default parameters: {params}")
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
        import json

        with open(repeat_dir / "optuna_best_params.json", "w", encoding="utf-8") as handle:
            json.dump(optimization_info, handle, indent=2, cls=NumpyEncoder)

    fold_results = []
    all_predictions = {"train": [], "val": [], "test": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_seed = generate_fold_seed(self.random_seed, repeat_idx, fold_idx)
        np.random.seed(fold_seed)
        task_logger.info(f"Training fold {fold_idx + 1}/{self.n_splits}...")
        model = self.create_model(model_name, params)

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
            model_name,
            params,
            folds,
            X,
            X_train,
            y_train,
            X_test,
            y_test,
            repeat_dir,
            task_logger,
            y_test_original=y_test_original,
            phenotype_scaler=phenotype_scaler,
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
        "cv_folds_file": str(cv_folds_path),
        "phenotype_scaler": phenotype_scaler,
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
    test_indices: np.ndarray = None,
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

    if use_same_test_set and test_indices is None:
        _, test_indices = create_holdout_indices(
            y, self.test_size, self.random_seed, self.task_type
        )
        main_logger.info(f"All repeats will use the same test set (size: {len(test_indices)})")

    all_repeat_results = []

    if self.repeat_workers > 1 and self.optuna_per_repeat:
        main_logger.info(f"Using {self.repeat_workers} parallel repeat workers")

        with graceful_process_pool(
            max_workers=self.repeat_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker_threads,
            initargs=(self.n_threads,),
            logger=main_logger,
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

        # Collect subprocess logs into the main log after all parallel tasks finish
        collect_subprocess_logs(self.logs_dir, main_logger)
    else:
        if self.repeat_workers > 1 and not self.optuna_per_repeat:
            main_logger.info(
                "Executing repeats sequentially so frozen Optuna parameters can be reused "
                "after repeat 1; set --optuna_per_repeat to enable parallel independent searches"
            )
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
        selected_repeat_idx, selected_repeat, selection = select_repeat_by_train_cv(
            all_repeat_results, self.task_type
        )
        main_logger.info(
            "Selected representative/deployment parameters from train-only CV: "
            f"Repeat {selected_repeat_idx + 1}, {selection['metric']}="
            f"{selection['cv_mean']:.{ModelConstants.metrics_precision}f} "
            f"(±{selection['cv_std']:.{ModelConstants.metrics_precision}f})"
        )

        if self.task_type == "classification":
            repeat_info = {
                "repeat_idx": int(selected_repeat_idx),
                "selection_source": "train_only_cv",
                "cv_metric": selection["metric"],
                "cv_mean": selection["cv_mean"],
                "cv_std": selection["cv_std"],
                "ensemble_accuracy": float(
                    summary["raw_values"]["ensemble_accuracy_values"][selected_repeat_idx]
                ),
            }
        else:
            repeat_info = {
                "repeat_idx": int(selected_repeat_idx),
                "selection_source": "train_only_cv",
                "cv_metric": selection["metric"],
                "cv_mean": selection["cv_mean"],
                "cv_std": selection["cv_std"],
                "ensemble_pearson": float(
                    summary["raw_values"]["ensemble_pearson_values"][selected_repeat_idx]
                ),
            }

        model_path = self._save_representative_model(
            model_name, selected_repeat["params"], X, y, model_dir, repeat_info
        )
        deployment_path = self._save_deployment_ensemble(
            model_name, selected_repeat["params"], X, y, model_dir
        )

        if model_path:
            main_logger.info(f"Representative model saved to {model_path}")
            if self.task_type == "classification":
                summary["representative_model"] = {
                    "repeat_idx": int(selected_repeat_idx),
                    "selection_source": "train_only_cv",
                    "cv_metric": selection["metric"],
                    "cv_mean": selection["cv_mean"],
                    "cv_std": selection["cv_std"],
                    "ensemble_accuracy": float(
                        summary["raw_values"]["ensemble_accuracy_values"][selected_repeat_idx]
                    ),
                    "model_path": model_path,
                }
            else:
                summary["representative_model"] = {
                    "repeat_idx": int(selected_repeat_idx),
                    "selection_source": "train_only_cv",
                    "cv_metric": selection["metric"],
                    "cv_mean": selection["cv_mean"],
                    "cv_std": selection["cv_std"],
                    "ensemble_pearson": float(
                        summary["raw_values"]["ensemble_pearson_values"][selected_repeat_idx]
                    ),
                    "model_path": model_path,
                }

        if deployment_path:
            summary["deployment_ensemble"] = {
                "path": deployment_path,
                "n_members": self.n_splits,
                "trained_on": "all_samples",
                "performance_source": "holdout_fold_ensemble",
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

        summary["training_selection"] = {
            "selected_repeat_idx": int(selected_repeat_idx),
            "best_params": selected_repeat["params"],
            "cv_metric": selection["metric"],
            "cv_mean": selection["cv_mean"],
            "cv_std": selection["cv_std"],
            "source": "train_only_cv",
        }
        summary["phenotype_standardized"] = bool(
            self.task_type == "regression" and self.standardize_phenotype
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
    # Re-set all thread-control env vars explicitly in the child process.
    # The initializer (_init_worker_threads) handles this for spawn-context
    # processes, but we re-apply here as a belt-and-suspenders safeguard.
    for _env_var in ModelConstants.thread_env_vars:
        os.environ[_env_var] = str(self.n_threads)

    # Each subprocess writes to its own log file to avoid multi-process races
    task_logger = setup_subprocess_logging(
        model_name=model_name,
        repeat_idx=repeat_idx,
        log_dir=self.results_dir / ModelConstants.default_logs_dir,
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
