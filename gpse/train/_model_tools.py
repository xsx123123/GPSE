#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Tools Module
==================
Provides unified model creation, parameter retrieval, and default metrics.
"""

from typing import Dict, Any

from loguru import logger as main_logger

from gpse.utils.genomic_utils import filter_model_params


def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
    """Create a model instance according to model name and parameters."""
    if self.task_type == "regression":
        return self.model_optimizer.create_model(model_name, params)
    else:
        return self.genomic_classifier.create_classification_model(model_name, params)


def get_default_params(self, model_name: str) -> Dict[str, Any]:
    """Get default parameters for a model."""
    if self.task_type == "regression":
        return self.model_optimizer.get_default_params(model_name)
    else:
        return self.genomic_classifier.get_classification_default_params(model_name)


def get_param_func(self, model_name: str) -> callable:
    """Get the parameter-suggestion function for a model."""
    if self.task_type == "regression":
        return self.model_optimizer.get_param_func(model_name)
    else:
        return self.genomic_classifier.get_classification_param_func(model_name)


def filter_model_params_unified(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Unified parameter filtering method."""
    if self.task_type == "regression":
        return filter_model_params(model_name, params)
    else:
        return self.genomic_classifier.filter_classification_params(model_name, params)


def _get_default_ensemble_metrics(self) -> Dict[str, float]:
    """Get default ensemble metrics (used when training fails)."""
    if self.task_type == "classification":
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0}
    return {
        "pearson": 0.0,
        "spearman": 0.0,
        "r2": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
    }


def _get_default_fold_metrics(self) -> Dict[str, float]:
    """Get default fold-average metrics (used when all folds fail)."""
    if self.task_type == "classification":
        return {
            "avg_train_accuracy": 0.0,
            "avg_val_accuracy": 0.0,
            "avg_test_accuracy": 0.0,
            "std_train_accuracy": 0.0,
            "std_val_accuracy": 0.0,
            "std_test_accuracy": 0.0,
            "avg_train_f1": 0.0,
            "avg_val_f1": 0.0,
            "avg_test_f1": 0.0,
            "std_train_f1": 0.0,
            "std_val_f1": 0.0,
            "std_test_f1": 0.0,
            "avg_test_auc": 0.0,
            "std_test_auc": 0.0,
            "avg_test_pr_auc": 0.0,
            "std_test_pr_auc": 0.0,
            "avg_training_time": 0.0,
        }
    return {
        "avg_train_pearson": 0.0,
        "avg_val_pearson": 0.0,
        "avg_test_pearson": 0.0,
        "std_train_pearson": 0.0,
        "std_val_pearson": 0.0,
        "std_test_pearson": 0.0,
        "avg_test_spearman": 0.0,
        "std_test_spearman": 0.0,
        "avg_test_mse": 0.0,
        "std_test_mse": 0.0,
        "avg_training_time": 0.0,
    }
