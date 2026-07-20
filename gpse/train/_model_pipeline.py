"""Reusable sklearn-compatible preprocessing and model Pipeline for GPSE."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpse.train._feature_selection import (
    FeatureSelectionConfig,
    GenotypeImputationConfig,
    fit_feature_selector,
    fit_genotype_imputer,
    transform_features,
    transform_genotypes,
)


_TREE_MODEL_PREFIXES = (
    "rf_",
    "gbdt_",
    "xgboost_",
    "adaboost_",
    "lightgbm_",
    "catboost_",
    "histgradientboost_",
)


def model_uses_feature_scaling(model_name: str | None) -> bool:
    """Return whether a GPSE model benefits from StandardScaler preprocessing."""
    return not (model_name and model_name.startswith(_TREE_MODEL_PREFIXES))


class GenotypePreprocessor(BaseEstimator, TransformerMixin):
    """Fit imputation, SNP selection, and scaling on a single training partition."""

    def __init__(
        self,
        task_type: str,
        feature_selection_config: FeatureSelectionConfig | None = None,
        genotype_imputation_config: GenotypeImputationConfig | None = None,
        scale_features: bool = True,
    ):
        self.task_type = task_type
        self.feature_selection_config = feature_selection_config or FeatureSelectionConfig()
        self.genotype_imputation_config = genotype_imputation_config or GenotypeImputationConfig()
        self.scale_features = scale_features

    def fit(self, X, y=None):
        if y is None and self.feature_selection_config.method == "univariate":
            raise ValueError("univariate feature selection requires training labels")
        self.imputer_ = fit_genotype_imputer(X, self.genotype_imputation_config)
        X_imputed = transform_genotypes(self.imputer_, X, self.genotype_imputation_config)
        self.selector_, self.selected_features_ = fit_feature_selector(
            X_imputed, y, self.feature_selection_config, self.task_type
        )
        X_selected = transform_features(self.selector_, X_imputed)
        self.scaler_ = StandardScaler().fit(X_selected) if self.scale_features else None
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X):
        X_imputed = transform_genotypes(self.imputer_, X, self.genotype_imputation_config)
        X_selected = transform_features(self.selector_, X_imputed)
        return self.scaler_.transform(X_selected) if self.scaler_ is not None else X_selected

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_


def build_training_pipeline(
    model: Any,
    *,
    task_type: str,
    model_name: str | None = None,
    feature_selection_config: FeatureSelectionConfig | None = None,
    genotype_imputation_config: GenotypeImputationConfig | None = None,
) -> Pipeline:
    """Build an unfitted ``imputer -> selector -> scaler -> model`` Pipeline."""
    return Pipeline(
        [
            (
                "preprocess",
                GenotypePreprocessor(
                    task_type=task_type,
                    feature_selection_config=feature_selection_config,
                    genotype_imputation_config=genotype_imputation_config,
                    scale_features=model_uses_feature_scaling(model_name),
                ),
            ),
            ("model", model),
        ]
    )


def build_fitted_pipeline(
    model: Any,
    scaler: StandardScaler | None,
    selector: Any,
    imputer: Any,
    *,
    task_type: str,
    feature_selection_config: FeatureSelectionConfig,
    genotype_imputation_config: GenotypeImputationConfig,
    selected_features: list[str],
) -> Pipeline:
    """Wrap existing fitted GPSE components in the standard Pipeline contract."""
    preprocessor = GenotypePreprocessor(
        task_type=task_type,
        feature_selection_config=feature_selection_config,
        genotype_imputation_config=genotype_imputation_config,
        scale_features=scaler is not None,
    )
    preprocessor.imputer_ = imputer
    preprocessor.selector_ = selector
    preprocessor.scaler_ = scaler
    preprocessor.selected_features_ = list(selected_features)
    preprocessor.n_features_in_ = len(selected_features) if selector is None else None
    return Pipeline([("preprocess", preprocessor), ("model", model)])
