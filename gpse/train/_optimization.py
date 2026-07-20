#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization Module
===================================
Optuna-based hyperparameter optimization with cross-validation.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from optuna.pruners import MedianPruner
import optuna
from typing import Dict, Any

from loguru import logger as main_logger

from gpse.config import ModelConstants
from gpse.utils.genomic_utils import generate_optimization_seed
from threadpoolctl import threadpool_limits
from gpse.train._model_pipeline import build_training_pipeline


def optimize_model_parameters(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    repeat_idx: int,
    task_logger=None,
    cv_pheno_data=None,
) -> Dict[str, Any]:
    """
    Optimize model parameters using Optuna.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_name : str
        Model name.
    repeat_idx : int
        Repeat index.
    task_logger
        Task-specific logger.
    cv_pheno_data : pd.DataFrame, optional
        Phenotype data containing CV groups.

    Returns
    -------
    Dict[str, Any]
        Dictionary of best parameters and optimization info.
    """
    log = task_logger if task_logger else main_logger

    if not self.optuna_per_repeat and model_name in self._optimization_cache:
        cached = self._optimization_cache[model_name].copy()
        cached["reused_from_repeat"] = 0
        log.info(f"Reusing frozen Optuna parameters from repeat 1 for {model_name}")
        return cached

    log.info(
        f"Starting parameter optimization for model {model_name} "
        f"(repeat {repeat_idx + 1}/{self.n_repeats})..."
    )

    # Get CV folds
    folds = self.generate_cv_folds_from_file(X, y, cv_pheno_data, repeat_idx, task_logger)

    # Set random seed
    opt_seed = generate_optimization_seed(self.random_seed, repeat_idx)
    np.random.seed(opt_seed)

    # Define optimization objective
    def objective(trial):
        param_func = self.get_param_func(model_name)
        params = param_func(trial)

        filtered_params = self.filter_model_params_unified(model_name, params)

        cv_scores = []

        log.debug(f"Evaluating parameter set: {filtered_params}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            log.debug(f"Training fold {fold_idx + 1}...")
            try:
                fold_model = build_training_pipeline(
                    self.create_model(model_name, filtered_params),
                    task_type=self.task_type,
                    model_name=model_name,
                    feature_selection_config=self.feature_selection_config,
                    genotype_imputation_config=self.genotype_imputation_config,
                )
                with threadpool_limits(limits=self.n_threads):
                    fold_model.fit(X_fold_train, y_fold_train)

                if self.task_type == "classification":
                    (
                        y_fold_pred,
                        y_fold_proba,
                    ) = self.genomic_classifier.generate_classification_labels_from_model(
                        fold_model, X_fold_val
                    )
                    fold_metrics = self.genomic_classifier.calculate_classification_metrics(
                        y_fold_val, y_fold_pred, y_fold_proba
                    )
                    fold_score = balanced_accuracy_score(y_fold_val, y_fold_pred)
                else:
                    y_fold_pred = fold_model.predict(X_fold_val)

                    if (
                        np.isnan(y_fold_pred).any()
                        or np.std(y_fold_pred) < ModelConstants.correlation_min_threshold
                    ):
                        log.warning(
                            f"Warning: Fold {fold_idx + 1} produced invalid predictions "
                            f"(NaN or constant values)"
                        )
                        fold_score = 0.0
                    else:
                        fold_score = pearsonr(y_fold_val, y_fold_pred)[0]
                        if np.isnan(fold_score):
                            log.warning(
                                f"Warning: Fold {fold_idx + 1} Pearson correlation is NaN"
                            )
                            fold_score = 0.0

                cv_scores.append(fold_score)

            except Exception as e:
                log.error(f"Fold {fold_idx + 1} training failed: {str(e)}")
                cv_scores.append(0.0)

        if not cv_scores:
            log.error("All folds failed, returning a low score")
            return 0.0

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        objective_value = mean_score - self.cv_stability_penalty * std_score
        trial.set_user_attr("cv_mean", float(mean_score))
        trial.set_user_attr("cv_std", float(std_score))
        trial.set_user_attr("objective", float(objective_value))
        if self.task_type == "classification":
            log.debug(f"Average balanced accuracy: {mean_score:.6f} (±{std_score:.6f})")
        else:
            log.debug(f"Average Pearson correlation: {mean_score:.6f} (±{std_score:.6f})")
        return objective_value

    # Create early stopping callback
    def early_stopping_callback(study, trial):
        if trial.number >= self.patience:
            best_value = study.best_value
            recent_trials = [
                t.value
                for t in study.trials[-self.patience : -1]
                if t.value is not None
            ]
            if not recent_trials:
                recent_best_value = float("-inf")
            else:
                recent_best_value = max(recent_trials)

            if best_value is not None and recent_best_value is not None:
                if best_value <= recent_best_value:
                    log.info(f"\nEarly stopping: no improvement within {self.patience} trials")
                    study.stop()
        return False

    # Create Optuna study and optimize
    pruner = MedianPruner(
        n_startup_trials=ModelConstants.optuna_n_startup_trials,
        n_warmup_steps=ModelConstants.optuna_n_warmup_steps,
        interval_steps=ModelConstants.optuna_interval_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=opt_seed),
    )

    max_trials = min(self.n_trials, 100)
    log.info(
        "Starting robust parameter optimization, max trials: "
        f"{max_trials}, objective=mean(CV)-{self.cv_stability_penalty}*std(CV)"
    )
    study.optimize(
        objective,
        n_trials=max_trials,
        callbacks=[early_stopping_callback],
        show_progress_bar=True,
        n_jobs=ModelConstants.optuna_n_jobs,
    )

    best_params = study.best_params
    log.info(f"Best parameters: {best_params}")
    if self.task_type == "classification":
        log.info(f"Best robust CV balanced accuracy objective: {study.best_value:.6f}")
    else:
        log.info(f"Best CV Pearson correlation: {study.best_value:.6f}")

    filtered_best_params = self.filter_model_params_unified(model_name, best_params)

    top_trials = sorted(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: trial.value,
        reverse=True,
    )[:5]
    result = {
        "best_params": filtered_best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "requested_n_trials": self.n_trials,
        "effective_n_trials": max_trials,
        "early_stopped": len(study.trials) < max_trials,
        "objective": "mean_cv_score - cv_stability_penalty * std_cv_score",
        "cv_stability_penalty": self.cv_stability_penalty,
        "feature_selection": self.feature_selection_config.as_dict(),
        "genotype_imputation": self.genotype_imputation_config.as_dict(),
        "top_trials": [
            {
                "number": trial.number,
                "objective": trial.value,
                "cv_mean": trial.user_attrs.get("cv_mean"),
                "cv_std": trial.user_attrs.get("cv_std"),
                "params": trial.params,
            }
            for trial in top_trials
        ],
    }
    if not self.optuna_per_repeat:
        self._optimization_cache[model_name] = result.copy()
    return result
