#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Predictions Module
============================
Computes fold-ensemble predictions and returns the corresponding metrics.
"""

import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict

try:
    from loguru import logger as main_logger
except ImportError:
    main_logger = None

try:
    from ..utils.genomic_utils import calculate_metrics
except ImportError:
    from utils.genomic_utils import calculate_metrics


def _compute_ensemble_predictions(
    self,
    model_name: str,
    params: Dict,
    folds: list,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    repeat_dir,
    task_logger,
) -> Dict[str, float]:
    """Compute fold-ensemble predictions and return metrics."""
    try:
        import joblib

        # Load or retrain models
        if self.save_models:
            ensemble_models = []
            for fold_idx in range(self.n_splits):
                fold_model_path = repeat_dir / f"fold_{fold_idx + 1}_model.pkl"
                if fold_model_path.exists():
                    model, scaler = joblib.load(fold_model_path)
                    ensemble_models.append((model, scaler))
        else:
            ensemble_models = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                fold_model = self.create_model(model_name, params)
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                fold_scaler = StandardScaler()
                X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                fold_model.fit(X_fold_train_scaled, y_fold_train)
                ensemble_models.append((fold_model, fold_scaler))

        # Compute ensemble predictions
        ensemble_preds = []
        for model, scaler in ensemble_models:
            X_test_scaled = scaler.transform(X_test)
            ensemble_preds.append(model.predict(X_test_scaled))

        if self.task_type == "classification":
            ensemble_pred = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
            ensemble_probas = []
            for model, scaler in ensemble_models:
                X_test_scaled = scaler.transform(X_test)
                if hasattr(model, "predict_proba"):
                    ensemble_probas.append(model.predict_proba(X_test_scaled))
                else:
                    pred = model.predict(X_test_scaled)
                    prob = np.zeros((len(pred), 2))
                    prob[np.arange(len(pred)), pred.astype(int)] = 1.0
                    ensemble_probas.append(prob)

            ensemble_proba = np.mean(ensemble_probas, axis=0) if ensemble_probas else None
            ensemble_metrics = self.genomic_classifier.calculate_classification_metrics(
                y_test, ensemble_pred, ensemble_proba
            )

            task_logger.info(f"\nFold-ensemble test performance:")
            task_logger.info(f"  Accuracy: {ensemble_metrics['accuracy']:.6f}")
            task_logger.info(f"  F1: {ensemble_metrics['f1']:.6f}")
            if "auc" in ensemble_metrics:
                task_logger.info(f"  AUC: {ensemble_metrics['auc']:.6f}")
        else:
            ensemble_pred = np.mean(ensemble_preds, axis=0)
            ensemble_metrics = calculate_metrics(y_test, ensemble_pred)

            task_logger.info(f"\nFold-ensemble test performance:")
            task_logger.info(f"  Pearson: {ensemble_metrics['pearson']:.6f}")
            task_logger.info(f"  Spearman: {ensemble_metrics['spearman']:.6f}")
            task_logger.info(f"  R²: {ensemble_metrics['r2']:.6f}")
            task_logger.info(f"  MSE: {ensemble_metrics['mse']:.6f}")
            task_logger.info(f"  RMSE: {ensemble_metrics['rmse']:.6f}")
            task_logger.info(f"  MAE: {ensemble_metrics['mae']:.6f}")

        return ensemble_metrics

    except Exception as e:
        task_logger.error(f"Computing fold-ensemble predictions failed: {str(e)}")
        task_logger.error(traceback.format_exc())
        return self._get_default_ensemble_metrics()
