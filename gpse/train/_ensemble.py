#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Predictions Module
============================
Computes fold-ensemble predictions and returns the corresponding metrics.
"""

import traceback
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict

from loguru import logger as main_logger

from gpse.utils.genomic_utils import calculate_metrics
from threadpoolctl import threadpool_limits
from gpse.train._feature_selection import (
    make_model_artifact,
    unpack_model_artifact,
    get_artifact_imputer,
    transform_features,
    transform_genotypes,
)
from gpse.train._model_pipeline import build_training_pipeline


def _save_deployment_ensemble(self, model_name, params, X, y, model_dir):
    """Train a full-data K-fold ensemble matching the evaluation artifact form."""
    deployment_dir = model_dir / "deployment_ensemble"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    try:
        if self.task_type == "classification":
            class_counts = y.value_counts()
            if class_counts.min() < self.n_splits:
                raise ValueError(
                    "Deployment StratifiedKFold requires every class to contain "
                    f"at least {self.n_splits} samples; counts: {class_counts.to_dict()}"
                )
            folds = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed + 10_000,
            ).split(X, y)
        else:
            folds = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed + 10_000,
            ).split(X)

        for member_idx, (train_idx, _) in enumerate(folds, start=1):
            X_member_train = X.iloc[train_idx]
            y_member_train = y.iloc[train_idx]
            pipeline = build_training_pipeline(
                self.create_model(model_name, params),
                task_type=self.task_type,
                model_name=model_name,
                feature_selection_config=self.feature_selection_config,
                genotype_imputation_config=self.genotype_imputation_config,
            )
            with threadpool_limits(limits=self.n_threads):
                pipeline.fit(X_member_train, y_member_train)
            preprocessor = pipeline.named_steps["preprocess"]
            joblib.dump(
                make_model_artifact(
                    pipeline.named_steps["model"],
                    preprocessor.scaler_,
                    preprocessor.selector_,
                    self.feature_selection_config,
                    preprocessor.selected_features_,
                    imputer=preprocessor.imputer_,
                    imputation_config=self.genotype_imputation_config,
                    task_type=self.task_type,
                ),
                deployment_dir / f"member_{member_idx}.pkl",
            )

        info = {
            "artifact_role": "deployment_ensemble",
            "n_members": self.n_splits,
            "trained_on": "all_samples",
            "performance_source": "holdout_fold_ensemble",
            "performance_belongs_to_saved_model": True,
            "task_type": self.task_type,
            "target_transform": "raw",
            "feature_selection": self.feature_selection_config.as_dict(),
            "genotype_imputation": self.genotype_imputation_config.as_dict(),
            "note": (
                "Hold-out metrics came from the same K-member fold-ensemble form; "
                "this deployment artifact was retrained on all samples and must not "
                "be evaluated on the former hold-out set."
            ),
        }
        with open(deployment_dir / "info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
        return str(deployment_dir)
    except Exception as exc:
        main_logger.error(f"Saving deployment ensemble failed: {exc}")
        main_logger.error(traceback.format_exc())
        return None


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
    y_test_original=None,
    phenotype_scaler=None,
) -> Dict[str, float]:
    """Compute fold-ensemble predictions and return metrics."""

    def transform_for_member(scaler, selector, imputer, features):
        transformed = transform_genotypes(
            imputer, features, self.genotype_imputation_config
        )
        transformed = transform_features(selector, transformed)
        return scaler.transform(transformed) if scaler is not None else transformed

    try:
        import joblib

        # Load or retrain models
        if self.save_models:
            ensemble_models = []
            for fold_idx in range(self.n_splits):
                fold_model_path = repeat_dir / f"fold_{fold_idx + 1}_model.pkl"
                if fold_model_path.exists():
                    artifact = joblib.load(fold_model_path)
                    model, scaler, selector = unpack_model_artifact(artifact)
                    ensemble_models.append((model, scaler, selector, get_artifact_imputer(artifact)))
        else:
            ensemble_models = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                pipeline = build_training_pipeline(
                    self.create_model(model_name, params),
                    task_type=self.task_type,
                    model_name=model_name,
                    feature_selection_config=self.feature_selection_config,
                    genotype_imputation_config=self.genotype_imputation_config,
                )
                with threadpool_limits(limits=self.n_threads):
                    pipeline.fit(X_fold_train, y_fold_train)
                preprocessor = pipeline.named_steps["preprocess"]
                ensemble_models.append(
                    (
                        pipeline.named_steps["model"],
                        preprocessor.scaler_,
                        preprocessor.selector_,
                        preprocessor.imputer_,
                    )
                )

        # Compute ensemble predictions
        ensemble_preds = []
        for model, scaler, selector, imputer in ensemble_models:
            X_test_transformed = transform_for_member(scaler, selector, imputer, X_test)
            ensemble_preds.append(model.predict(X_test_transformed))

        if self.task_type == "classification":
            ensemble_probas = []
            for model, scaler, selector, imputer in ensemble_models:
                X_test_transformed = transform_for_member(
                    scaler, selector, imputer, X_test
                )
                if hasattr(model, "predict_proba"):
                    ensemble_probas.append(model.predict_proba(X_test_transformed))
                else:
                    pred = model.predict(X_test_transformed)
                    n_classes = getattr(self, "n_classes", int(np.max(pred)) + 1)
                    prob = np.zeros((len(pred), n_classes))
                    prob[np.arange(len(pred)), pred.astype(int)] = 1.0
                    ensemble_probas.append(prob)

            ensemble_proba = np.mean(ensemble_probas, axis=0) if ensemble_probas else None
            if ensemble_proba is not None:
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
            else:
                pred_matrix = np.asarray(ensemble_preds)
                n_classes = getattr(self, "n_classes", int(np.max(pred_matrix)) + 1)
                ensemble_pred = np.array([
                    np.bincount(row.astype(int), minlength=n_classes).argmax()
                    for row in pred_matrix.T
                ])
            ensemble_metrics = self.genomic_classifier.calculate_classification_metrics(
                y_test, ensemble_pred, ensemble_proba
            )

            task_logger.info(f"\nFold-ensemble test performance:")
            task_logger.info(f"  Accuracy: {ensemble_metrics['accuracy']:.6f}")
            task_logger.info(f"  F1: {ensemble_metrics['f1']:.6f}")
            if "auc" in ensemble_metrics:
                task_logger.info(f"  AUC: {ensemble_metrics['auc']:.6f}")
            if "pr_auc" in ensemble_metrics:
                task_logger.info(f"  PR-AUC: {ensemble_metrics['pr_auc']:.6f}")
        else:
            ensemble_pred = np.mean(ensemble_preds, axis=0)
            metric_y_test = y_test_original if y_test_original is not None else y_test
            metric_predictions = ensemble_pred
            if phenotype_scaler and phenotype_scaler.get("applied", False):
                metric_predictions = (
                    ensemble_pred * phenotype_scaler["std"] + phenotype_scaler["mean"]
                )
            ensemble_metrics = calculate_metrics(metric_y_test, metric_predictions)

            task_logger.info("\nFold-ensemble test performance (original phenotype scale):")
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
