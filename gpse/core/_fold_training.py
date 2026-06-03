#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fold Training Module
====================
Handles training and evaluation of individual CV folds and their aggregation.
"""

import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from loguru import logger as main_logger

from gpse.utils.genomic_utils import (
    prepare_fold_training_data,
    train_fold_model,
    predict_and_calculate_metrics,
    save_fold_predictions_and_plots,
)


def _train_single_fold(
    self,
    model,
    model_name: str,
    params: Dict,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_indices: np.ndarray,
    repeat_idx: int,
    repeat_dir,
    all_predictions: Dict,
    task_logger,
) -> Optional[Dict]:
    """Train a single fold and return the results."""
    try:
        # Prepare fold training data
        (
            X_fold_train_scaled,
            X_fold_val_scaled,
            X_test_scaled,
            y_fold_train,
            y_fold_val,
            scaler,
        ) = prepare_fold_training_data(X_train, y_train, train_idx, val_idx, X_test)

        # Train fold model
        model, training_time = train_fold_model(model, X_fold_train_scaled, y_fold_train)

        # Predict and calculate metrics
        if self.task_type == "classification":
            (
                y_fold_train_pred,
                y_fold_train_proba,
            ) = self.genomic_classifier.generate_classification_labels_from_model(
                model, X_fold_train_scaled
            )
            (
                y_fold_val_pred,
                y_fold_val_proba,
            ) = self.genomic_classifier.generate_classification_labels_from_model(
                model, X_fold_val_scaled
            )
            (
                y_test_pred,
                y_test_proba,
            ) = self.genomic_classifier.generate_classification_labels_from_model(
                model, X_test_scaled
            )

            train_metrics = self.genomic_classifier.calculate_classification_metrics(
                y_fold_train, y_fold_train_pred, y_fold_train_proba
            )
            val_metrics = self.genomic_classifier.calculate_classification_metrics(
                y_fold_val, y_fold_val_pred, y_fold_val_proba
            )
            test_metrics = self.genomic_classifier.calculate_classification_metrics(
                y_test, y_test_pred, y_test_proba
            )
            plots = {}
        else:
            (
                train_metrics,
                val_metrics,
                test_metrics,
                y_fold_train_pred,
                y_fold_val_pred,
                y_test_pred,
            ) = predict_and_calculate_metrics(
                model,
                X_fold_train_scaled,
                y_fold_train,
                X_fold_val_scaled,
                y_fold_val,
                X_test_scaled,
                y_test,
            )
            plots = save_fold_predictions_and_plots(
                train_idx,
                val_idx,
                test_indices,
                y_fold_train,
                y_fold_val,
                y_test,
                y_fold_train_pred,
                y_fold_val_pred,
                y_test_pred,
                all_predictions,
                model_name,
                repeat_idx,
                fold_idx,
                repeat_dir,
                self.results_dir,
            )

        # Store results
        fold_result = {
            "fold_idx": fold_idx,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "plots": plots,
        }

        # Save model
        if self.save_models:
            import joblib

            fold_model_path = repeat_dir / f"fold_{fold_idx + 1}_model.pkl"
            joblib.dump((model, scaler), fold_model_path)

        # Log results
        self._log_fold_results(
            fold_idx, train_metrics, val_metrics, test_metrics, training_time, task_logger
        )

        return fold_result

    except Exception as e:
        task_logger.error(f"Fold {fold_idx + 1} training failed: {str(e)}")
        task_logger.error(traceback.format_exc())
        return None


def _log_fold_results(
    self,
    fold_idx: int,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    training_time: float,
    task_logger,
) -> None:
    """Log fold results."""
    if self.task_type == "classification":
        self.genomic_classifier.log_classification_results(
            fold_idx, train_metrics, val_metrics, test_metrics, task_logger
        )
    else:
        task_logger.info(f"Fold {fold_idx + 1} results:")
        task_logger.info(f"  Train Pearson: {train_metrics['pearson']:.6f}")
        task_logger.info(f"  Train Spearman: {train_metrics['spearman']:.6f}")
        task_logger.info(f"  Val Pearson: {val_metrics['pearson']:.6f}")
        task_logger.info(f"  Val Spearman: {val_metrics['spearman']:.6f}")
        task_logger.info(f"  Test Pearson: {test_metrics['pearson']:.6f}")
        task_logger.info(f"  Test Spearman: {test_metrics['spearman']:.6f}")
        task_logger.info(f"  Test MSE: {test_metrics['mse']:.6f}")
    task_logger.info(f"  Training time: {training_time:.2f}s")


def _calculate_fold_average_metrics(
    self, fold_results: List[Dict], repeat_idx: int, task_logger
) -> Dict[str, float]:
    """Calculate average metrics across all folds."""
    metrics = {"avg_training_time": np.mean([r["training_time"] for r in fold_results])}

    if self.task_type == "classification":
        for metric_name in ["accuracy", "f1"]:
            for split in ["train", "val", "test"]:
                values = [r[f"{split}_metrics"][metric_name] for r in fold_results]
                metrics[f"avg_{split}_{metric_name}"] = np.mean(values)
                metrics[f"std_{split}_{metric_name}"] = np.std(values)

        # AUC metric
        if all("auc" in r["test_metrics"] for r in fold_results):
            auc_values = [r["test_metrics"]["auc"] for r in fold_results]
            metrics["avg_test_auc"] = np.mean(auc_values)
            metrics["std_test_auc"] = np.std(auc_values)
        else:
            metrics["avg_test_auc"] = 0.0
            metrics["std_test_auc"] = 0.0

        # Log output
        task_logger.info(f"\nRepeat {repeat_idx + 1} average performance:")
        task_logger.info(
            f"  Avg Train Accuracy: {metrics['avg_train_accuracy']:.6f} "
            f"(±{metrics['std_train_accuracy']:.6f})"
        )
        task_logger.info(
            f"  Avg Val Accuracy: {metrics['avg_val_accuracy']:.6f} "
            f"(±{metrics['std_val_accuracy']:.6f})"
        )
        task_logger.info(
            f"  Avg Test Accuracy: {metrics['avg_test_accuracy']:.6f} "
            f"(±{metrics['std_test_accuracy']:.6f})"
        )
        task_logger.info(
            f"  Avg Train F1: {metrics['avg_train_f1']:.6f} "
            f"(±{metrics['std_train_f1']:.6f})"
        )
        task_logger.info(
            f"  Avg Val F1: {metrics['avg_val_f1']:.6f} "
            f"(±{metrics['std_val_f1']:.6f})"
        )
        task_logger.info(
            f"  Avg Test F1: {metrics['avg_test_f1']:.6f} "
            f"(±{metrics['std_test_f1']:.6f})"
        )
        if metrics["avg_test_auc"] > 0:
            task_logger.info(
                f"  Avg Test AUC: {metrics['avg_test_auc']:.6f} "
                f"(±{metrics['std_test_auc']:.6f})"
            )
    else:
        # Regression task
        for split in ["train", "val", "test"]:
            pearson_values = [r[f"{split}_metrics"]["pearson"] for r in fold_results]
            metrics[f"avg_{split}_pearson"] = np.mean(pearson_values)
            metrics[f"std_{split}_pearson"] = np.std(pearson_values)

        spearman_values = [r["test_metrics"]["spearman"] for r in fold_results]
        mse_values = [r["test_metrics"]["mse"] for r in fold_results]
        metrics["avg_test_spearman"] = np.mean(spearman_values)
        metrics["std_test_spearman"] = np.std(spearman_values)
        metrics["avg_test_mse"] = np.mean(mse_values)
        metrics["std_test_mse"] = np.std(mse_values)

        task_logger.info(f"\nRepeat {repeat_idx + 1} average performance:")
        task_logger.info(
            f"  Avg Train Pearson: {metrics['avg_train_pearson']:.6f} "
            f"(±{metrics['std_train_pearson']:.6f})"
        )
        task_logger.info(
            f"  Avg Val Pearson: {metrics['avg_val_pearson']:.6f} "
            f"(±{metrics['std_val_pearson']:.6f})"
        )
        task_logger.info(
            f"  Avg Test Pearson: {metrics['avg_test_pearson']:.6f} "
            f"(±{metrics['std_test_pearson']:.6f})"
        )

    task_logger.info(f"  Avg training time: {metrics['avg_training_time']:.2f}s")
    return metrics
