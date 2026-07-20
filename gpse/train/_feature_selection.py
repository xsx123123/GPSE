"""Leak-free feature-selection helpers for GPSE training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from functools import partial

from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Validated feature-selection settings shared by all training paths."""

    method: str = "none"
    select_k: int | None = 5000
    variance_threshold: float = 0.0
    select_percentile: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "select_k": self.select_k,
            "variance_threshold": self.variance_threshold,
            "select_percentile": self.select_percentile,
        }


@dataclass(frozen=True)
class GenotypeImputationConfig:
    """Fold-local missing-genotype handling with legacy-compatible defaults."""

    method: str = "none"
    missing_genotype_code: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "missing_genotype_code": self.missing_genotype_code,
        }


def validate_feature_selection_config(
    method: str,
    select_k: int | None,
    variance_threshold: float,
    select_percentile: float | None = None,
) -> FeatureSelectionConfig:
    if method not in {
        "none",
        "variance",
        "univariate",
        "mutual_info",
        "tree_importance",
        "lasso",
        "elasticnet",
        "pca",
        "svd",
    }:
        raise ValueError(
            "feature_selection must be 'none', 'variance', 'univariate', 'mutual_info', "
            "'tree_importance', 'lasso', 'elasticnet', 'pca', or 'svd'"
        )
    if select_k is not None and select_k < 1:
        raise ValueError("select_k must be at least 1 when supplied")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative")
    if select_percentile is not None and not 0 < select_percentile <= 100:
        raise ValueError("select_percentile must be greater than 0 and no more than 100")
    return FeatureSelectionConfig(method, select_k, variance_threshold, select_percentile)


def validate_genotype_imputation_config(
    method: str, missing_genotype_code: float
) -> GenotypeImputationConfig:
    if method not in {"none", "mean"}:
        raise ValueError("genotype_imputation must be 'none' or 'mean'")
    return GenotypeImputationConfig(method, float(missing_genotype_code))


def _replace_missing_genotype_code(X, config: GenotypeImputationConfig):
    if config.method == "none":
        return X
    if isinstance(X, pd.DataFrame):
        return X.replace(config.missing_genotype_code, np.nan)
    return np.where(np.asarray(X) == config.missing_genotype_code, np.nan, X)


def fit_genotype_imputer(X, config: GenotypeImputationConfig):
    """Fit a mean imputer on one training partition, never on validation/test data."""
    if config.method == "none":
        return None
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(_replace_missing_genotype_code(X, config))
    return imputer


def transform_genotypes(imputer, X, config: GenotypeImputationConfig):
    """Apply an already fitted imputer while preserving DataFrame IDs and SNP names."""
    if imputer is None:
        return X
    transformed = imputer.transform(_replace_missing_genotype_code(X, config))
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(transformed, index=X.index, columns=X.columns)
    return transformed


def fit_feature_selector(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config: FeatureSelectionConfig,
    task_type: str,
):
    """Fit a selector on one training partition only and return it with names."""
    feature_count = X.shape[1]
    selected_count = (
        max(1, int(np.ceil(feature_count * config.select_percentile / 100)))
        if config.select_percentile is not None
        else feature_count
        if config.select_k is None
        else min(config.select_k, feature_count)
    )
    if config.method == "none":
        return None, [str(name) for name in getattr(X, "columns", range(feature_count))]
    if config.method == "variance":
        selector = VarianceThreshold(threshold=config.variance_threshold)
    elif config.method == "univariate":
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=selected_count)
    elif config.method == "mutual_info":
        score_func = mutual_info_classif if task_type == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=partial(score_func, random_state=0), k=selected_count)
    elif config.method == "tree_importance":
        estimator_class = ExtraTreesClassifier if task_type == "classification" else ExtraTreesRegressor
        estimator = estimator_class(n_estimators=200, n_jobs=1, random_state=0)
        selector = SelectFromModel(estimator, threshold=-np.inf, max_features=selected_count)
    elif config.method in {"pca", "svd"}:
        maximum_components = min(len(X), feature_count)
        component_count = (
            maximum_components
            if config.select_percentile is None and config.select_k is None
            else min(selected_count, maximum_components)
        )
        selector = (
            PCA(n_components=component_count, random_state=0)
            if config.method == "pca"
            else TruncatedSVD(n_components=component_count, random_state=0)
        )
    else:
        if task_type == "classification":
            class_counts = pd.Series(y).value_counts()
            internal_cv = min(5, int(class_counts.min()))
            if internal_cv < 2:
                raise ValueError(
                    f"{config.method} feature selection requires at least two training samples per class; "
                    f"counts: {class_counts.to_dict()}"
                )
            estimator_args = {
                "Cs": 10,
                "cv": internal_cv,
                "solver": "saga",
                "max_iter": 5_000,
                "n_jobs": 1,
                "random_state": 0,
            }
            if config.method == "lasso":
                estimator = LogisticRegressionCV(penalty="l1", **estimator_args)
            else:
                estimator = LogisticRegressionCV(
                    penalty="elasticnet",
                    l1_ratios=[0.1, 0.5, 0.9],
                    **estimator_args,
                )
        else:
            internal_cv = min(5, len(X))
            if internal_cv < 2:
                raise ValueError(
                    f"{config.method} feature selection requires at least two training samples"
                )
            estimator_args = {
                "alphas": 100,
                "cv": internal_cv,
                "max_iter": 5_000,
                "n_jobs": 1,
                "random_state": 0,
            }
            estimator = (
                LassoCV(**estimator_args)
                if config.method == "lasso"
                else ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], **estimator_args)
            )
        selector = SelectFromModel(estimator, threshold=-np.inf, max_features=selected_count)
    selector.fit(X, y)
    names = getattr(X, "columns", range(feature_count))
    if not hasattr(selector, "get_support"):
        return selector, [str(name) for name in selector.get_feature_names_out(names)]

    support = selector.get_support()
    if not np.any(support):
        raise ValueError(
            "Feature selection removed every SNP; lower --variance_threshold or use --feature_selection none"
        )
    return selector, [str(name) for name, selected in zip(names, support) if selected]


def transform_features(selector, X: pd.DataFrame | np.ndarray):
    """Transform features with an already fitted selector, preserving none mode."""
    if selector is None:
        return X
    transformed = selector.transform(X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(
            transformed,
            index=X.index,
            columns=selector.get_feature_names_out(X.columns),
        )
    return transformed


def unpack_model_artifact(artifact):
    """Read current dict artifacts and legacy ``(model, scaler)`` tuples."""
    if isinstance(artifact, dict) and {"model", "scaler"}.issubset(artifact):
        return artifact["model"], artifact["scaler"], artifact.get("selector")
    if isinstance(artifact, tuple) and len(artifact) == 2:
        return artifact[0], artifact[1], None
    raise TypeError("Unsupported GPSE model artifact")


def get_artifact_imputer(artifact):
    """Return the optional fitted genotype imputer from a current artifact."""
    return artifact.get("imputer") if isinstance(artifact, dict) else None


def make_model_artifact(
    model,
    scaler,
    selector,
    config: FeatureSelectionConfig,
    selected_features,
    *,
    imputer=None,
    imputation_config: GenotypeImputationConfig | None = None,
    task_type: str = "regression",
):
    """Create a portable model artifact while retaining legacy-load compatibility."""
    resolved_imputation = imputation_config or GenotypeImputationConfig()
    from gpse.train._model_pipeline import build_fitted_pipeline

    pipeline = build_fitted_pipeline(
        model,
        scaler,
        selector,
        imputer,
        task_type=task_type,
        feature_selection_config=config,
        genotype_imputation_config=resolved_imputation,
        selected_features=list(selected_features),
    )
    return {
        "artifact_version": 3,
        "pipeline": pipeline,
        "task_type": task_type,
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "imputer": imputer,
        "feature_selection": config.as_dict(),
        "genotype_imputation": resolved_imputation.as_dict(),
        "feature_scaling": "standard" if scaler is not None else "passthrough",
        "selected_features": list(selected_features),
    }
