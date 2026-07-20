#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-free stacking ensemble based on train-side out-of-fold predictions."""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpse.utils.log_utils import logger
from gpse.train._feature_selection import (
    FeatureSelectionConfig,
    GenotypeImputationConfig,
)
from gpse.train._model_pipeline import build_training_pipeline


class StackingEnsemble:
    """Train a low-capacity meta-model on strictly out-of-fold base predictions."""

    def __init__(
        self,
        base_models_dir: str,
        model_factory: Callable[[str], Any],
        top_n_models: int = 5,
        meta_model_type: str = "ridge",
        cv_folds: int = 5,
        random_seed: int = 42,
        output_dir: Optional[str] = None,
        n_threads: int = 1,
        use_default_params: bool = False,
        task_type: str = "regression",
        feature_selection_config=None,
        genotype_imputation_config=None,
    ):
        self.base_models_dir = base_models_dir
        self._model_factory = model_factory
        self.top_n_models = top_n_models
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.output_dir = output_dir or os.path.join(base_models_dir, "ensemble_stacking")
        self.n_threads = n_threads
        self.use_default_params = use_default_params
        self.task_type = task_type
        self.feature_selection_config = feature_selection_config or FeatureSelectionConfig()
        self.genotype_imputation_config = genotype_imputation_config or GenotypeImputationConfig()
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_models: Dict[str, Any] = {}
        self.selected_models: List[str] = []
        self.meta_model = None
        self.meta_features_cols: List[str] = []

    def _prediction_feature(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
        return model.predict(X)

    def _splitter(self, X: pd.DataFrame, y: pd.Series):
        if self.task_type == "classification":
            counts = y.value_counts()
            if counts.min() < self.cv_folds:
                raise ValueError(
                    "Stacking OOF CV requires every class to contain at least "
                    f"cv_folds={self.cv_folds} samples; class counts: {counts.to_dict()}"
                )
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            ).split(X, y)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed).split(X)

    def create_meta_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Build OOF features; every training row is predicted by an unseen model."""
        meta_train = np.zeros((len(X_train), len(self.selected_models)))
        meta_test = np.zeros((len(X_test), len(self.selected_models))) if X_test is not None else None

        for column, model_name in enumerate(self.selected_models):
            logger.info(f"Generating strict OOF meta-feature for model {model_name}")
            for train_idx, val_idx in self._splitter(X_train, y_train):
                fold_pipeline = build_training_pipeline(
                    self._model_factory(model_name),
                    task_type=self.task_type,
                    model_name=model_name,
                    feature_selection_config=self.feature_selection_config,
                    genotype_imputation_config=self.genotype_imputation_config,
                )
                fold_pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                meta_train[val_idx, column] = self._prediction_feature(
                    fold_pipeline, X_train.iloc[val_idx]
                )

            full_pipeline = build_training_pipeline(
                self._model_factory(model_name),
                task_type=self.task_type,
                model_name=model_name,
                feature_selection_config=self.feature_selection_config,
                genotype_imputation_config=self.genotype_imputation_config,
            )
            full_pipeline.fit(X_train, y_train)
            self.base_models[model_name] = full_pipeline
            if X_test is not None:
                meta_test[:, column] = self._prediction_feature(
                    full_pipeline,
                    X_test,
                )

        return meta_train, meta_test

    def fit_meta_model(self, meta_train: np.ndarray, y_train: pd.Series) -> None:
        if self.task_type == "classification":
            estimator = LogisticRegression(
                C=1.0, random_state=self.random_seed, max_iter=1000, n_jobs=self.n_threads
            )
            self.meta_model = Pipeline([("scaler", StandardScaler()), ("logistic", estimator)])
        else:
            estimator = Ridge(alpha=1.0, random_state=self.random_seed)
            self.meta_model = Pipeline([("scaler", StandardScaler()), ("ridge", estimator)])
        self.meta_model.fit(meta_train, y_train)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.meta_model is None:
            raise ValueError("Meta-model is not trained; call fit first")
        meta_features = np.zeros((len(X), len(self.selected_models)))
        for column, model_name in enumerate(self.selected_models):
            model = self.base_models[model_name]
            meta_features[:, column] = self._prediction_feature(
                model, X
            )
        predictions = self.meta_model.predict(meta_features)
        probabilities = (
            self.meta_model.predict_proba(meta_features)
            if self.task_type == "classification" and hasattr(self.meta_model, "predict_proba")
            else None
        )
        return predictions, probabilities

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not model_names:
            raise ValueError("Stacking requires training-side selected base models")
        self.selected_models = list(model_names[: self.top_n_models])
        self.meta_features_cols = self.selected_models.copy()
        start_time = time.time()
        meta_train, meta_test = self.create_meta_features(X_train, y_train, X_test)
        self.fit_meta_model(meta_train, y_train)

        results: Dict[str, Any] = {
            "selected_models": self.selected_models,
            "train_metrics": self.calculate_metrics(y_train, self.meta_model.predict(meta_train), self._meta_probabilities(meta_train)),
            "training_time": time.time() - start_time,
        }
        if X_test is not None and y_test is not None and meta_test is not None:
            results["test_metrics"] = self.calculate_metrics(
                y_test, self.meta_model.predict(meta_test), self._meta_probabilities(meta_test)
            )

        factory = self._model_factory
        self._model_factory = None
        joblib.dump(self, os.path.join(self.output_dir, "stacking_ensemble_model.pkl"))
        self._model_factory = factory
        joblib.dump(results, os.path.join(self.output_dir, "stacking_results.pkl"))
        return results

    def _meta_probabilities(self, meta_features: np.ndarray) -> Optional[np.ndarray]:
        if self.task_type == "classification" and hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        return None

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if self.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
            if y_pred_proba is not None:
                try:
                    metrics["auc"] = float(
                        roc_auc_score(
                            y_true,
                            y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba,
                            multi_class="ovr" if y_pred_proba.shape[1] > 2 else "raise",
                            average="weighted" if y_pred_proba.shape[1] > 2 else "macro",
                        )
                    )
                except ValueError:
                    metrics["auc"] = 0.0
            return metrics

        pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
        spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
        mse = mean_squared_error(y_true, y_pred)
        return {
            "pearson": float(0.0 if np.isnan(pearson) else pearson),
            "spearman": float(0.0 if np.isnan(spearman) else spearman),
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
