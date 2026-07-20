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
    train_fold_model,
    predict_and_calculate_metrics,
    save_fold_predictions_and_plots,
)
from gpse.train._feature_selection import (
    make_model_artifact,
)
from gpse.train._model_pipeline import build_training_pipeline


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
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        pipeline = build_training_pipeline(
            model,
            task_type=self.task_type,
            model_name=model_name,
            feature_selection_config=self.feature_selection_config,
            genotype_imputation_config=self.genotype_imputation_config,
        )
        pipeline, training_time = train_fold_model(
            pipeline, X_fold_train, y_fold_train, n_threads=self.n_threads
        )
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        imputer = preprocessor.imputer_
        selector = preprocessor.selector_
        scaler = preprocessor.scaler_
        selected_features = preprocessor.selected_features_
        X_fold_train_scaled = preprocessor.transform(X_fold_train)
        X_fold_val_scaled = preprocessor.transform(X_fold_val)
        X_test_scaled = preprocessor.transform(X_test)

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
            "feature_selection": self.feature_selection_config.as_dict(),
            "genotype_imputation": self.genotype_imputation_config.as_dict(),
            "selected_feature_count": len(selected_features),
        }

        # Save model
        if self.save_models:
            import joblib

            fold_model_path = repeat_dir / f"fold_{fold_idx + 1}_model.pkl"
            joblib.dump(
                make_model_artifact(
                    model,
                    scaler,
                    selector,
                    self.feature_selection_config,
                    selected_features,
                    imputer=imputer,
                    imputation_config=self.genotype_imputation_config,
                    task_type=self.task_type,
                ),
                fold_model_path,
            )

        # Log results
        self._log_fold_results(
            fold_idx,
            train_metrics,
            val_metrics,
            test_metrics,
            training_time,
            task_logger,
            model_name=model_name,
            repeat_idx=repeat_idx,
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
    model_name: str = "",
    repeat_idx: int = 0,
) -> None:
    """Log fold results as a single tagged line (debug level; use -l DEBUG to see)."""
    tag = f"{model_name} R{repeat_idx + 1} F{fold_idx + 1}"
    if self.task_type == "classification":
        self.genomic_classifier.log_classification_results(
            fold_idx, train_metrics, val_metrics, test_metrics, task_logger, tag=tag, level="debug"
        )
        task_logger.debug(f"{tag} | {training_time:.2f}s")
    else:
        task_logger.debug(
            f"{tag} | "
            f"Train r={train_metrics['pearson']:.4f} ρ={train_metrics['spearman']:.4f} | "
            f"Val r={val_metrics['pearson']:.4f} ρ={val_metrics['spearman']:.4f} | "
            f"Test r={test_metrics['pearson']:.4f} ρ={test_metrics['spearman']:.4f} "
            f"MSE={test_metrics['mse']:.4g} | "
            f"{training_time:.2f}s"
        )


def _calculate_fold_average_metrics(
    self, fold_results: List[Dict], repeat_idx: int, task_logger, model_name: str = ""
) -> Dict[str, float]:
    """Calculate average metrics across all folds."""
    metrics = {"avg_training_time": np.mean([r["training_time"] for r in fold_results])}
    tag = f"{model_name} R{repeat_idx + 1}"

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

        if all("pr_auc" in r["test_metrics"] for r in fold_results):
            pr_auc_values = [r["test_metrics"]["pr_auc"] for r in fold_results]
            metrics["avg_test_pr_auc"] = np.mean(pr_auc_values)
            metrics["std_test_pr_auc"] = np.std(pr_auc_values)
        else:
            metrics["avg_test_pr_auc"] = 0.0
            metrics["std_test_pr_auc"] = 0.0

        # Log output
        line = (
            f"<cyan>{tag}</cyan> <b>avg</b> | "
            f"Train acc={metrics['avg_train_accuracy']:.4f}±{metrics['std_train_accuracy']:.4f} "
            f"f1={metrics['avg_train_f1']:.4f}±{metrics['std_train_f1']:.4f} | "
            f"Val acc={metrics['avg_val_accuracy']:.4f}±{metrics['std_val_accuracy']:.4f} "
            f"f1={metrics['avg_val_f1']:.4f}±{metrics['std_val_f1']:.4f} | "
            f"<green>Test acc={metrics['avg_test_accuracy']:.4f}±{metrics['std_test_accuracy']:.4f} "
            f"f1={metrics['avg_test_f1']:.4f}±{metrics['std_test_f1']:.4f}</green>"
        )
        if metrics["avg_test_auc"] > 0:
            line += f" auc={metrics['avg_test_auc']:.4f}±{metrics['std_test_auc']:.4f}"
        line += f" | {metrics['avg_training_time']:.2f}s"
        task_logger.opt(colors=True).info(line)
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

        task_logger.opt(colors=True).info(
            f"<cyan>{tag}</cyan> <b>avg</b> | "
            f"Train r={metrics['avg_train_pearson']:.4f}±{metrics['std_train_pearson']:.4f} | "
            f"Val r={metrics['avg_val_pearson']:.4f}±{metrics['std_val_pearson']:.4f} | "
            f"<green>Test r={metrics['avg_test_pearson']:.4f}±{metrics['std_test_pearson']:.4f}</green> | "
            f"{metrics['avg_training_time']:.2f}s"
        )

    return metrics
