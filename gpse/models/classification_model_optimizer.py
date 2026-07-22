#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classification Model Configuration Module
=========================================

Contains parameter optimization configurations for 6 major classification models:
- RandomForest Classifier (rf_clf)
- XGBoost Classifier (xgboost_clf)
- LightGBM Classifier (lightgbm_clf)
- CatBoost Classifier (catboost_clf)
- SVM Classifier (svm_clf)
- Multi-layer Perceptron Classifier (mlp_clf)

Model registration is driven by ``gpse/config/models.yaml``. The param
functions below are referenced from that file via ``builtin:_xxx_params``.
"""

from typing import Dict, Any, Optional
import optuna
import os

from gpse.config import ModelConstants
from gpse.config._model_registry import ModelRegistry


class ClassificationModelOptimizer:
    """Classification model optimizer backed by the YAML model registry."""

    def __init__(
        self,
        random_seed: Optional[int] = None,
        n_threads: int = 1,
        n_classes: int = None,
        random_state: Optional[int] = None,
        catboost_train_dir: Optional[str] = None,
        model_config_path: Optional[str] = None,
    ):
        if random_seed is not None and random_state is not None and random_seed != random_state:
            raise ValueError("random_seed and random_state were both provided with different values")
        if random_seed is None:
            random_seed = random_state

        self.random_seed = random_seed
        self.random_state = random_seed
        self.n_threads = n_threads
        self.n_classes = n_classes
        self.catboost_train_dir = catboost_train_dir

        builtin_funcs = {
            name: getattr(self, name)
            for name in dir(self)
            if name.endswith("_params") and callable(getattr(self, name))
        }

        self._registry = ModelRegistry(
            user_config_path=model_config_path,
            random_seed=self.random_seed,
            n_threads=self.n_threads,
            n_classes=self.n_classes,
            catboost_train_dir=self.catboost_train_dir,
            builtin_param_funcs=builtin_funcs,
            task_filter="classification",
        )

        self.model_configs = self._registry.model_configs

        for _env_var in ModelConstants.thread_env_vars:
            os.environ[_env_var] = str(n_threads)

    def get_model_config(self, model_name: str):
        """Get model configuration."""
        if model_name not in self.model_configs:
            raise ValueError(f"Classification model {model_name} not found in configurations")
        return self.model_configs[model_name]

    def get_param_func(self, model_name: str) -> callable:
        """Get parameter function."""
        return self._registry.get_param_func(model_name)

    def create_classification_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create classification model instance via the registry."""
        return self._registry.create_model(model_name, params)

    def get_classification_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters via the registry."""
        return self._registry.get_default_params(model_name)

    def filter_classification_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter auxiliary params via the registry."""
        return self._registry.filter_model_params(model_name, params)

    def get_available_models(self) -> list:
        """Get list of available classification models."""
        return self._registry.get_available_models()

    # ─── Optuna search-space functions (referenced from models.yaml) ───────────

    def _rf_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """RandomForest classifier parameters."""
        param_dict = {
            'random_state': self.random_seed
        }

        param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 6)
        max_n_estimators = 800

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 10, max_n_estimators),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        })

        return param_dict

    def _xgboost_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """XGBoost classifier parameters."""
        param_dict = {
            'random_state': self.random_seed,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }

        booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
        param_dict['booster'] = booster

        if booster in ['gbtree', 'dart']:
            param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 6)
            max_n_estimators = 400
            min_eta = 1e-3

            param_dict.update({
                'n_estimators': trial.suggest_int('n_estimators', 20, max_n_estimators),
                'learning_rate': trial.suggest_float('learning_rate', min_eta, 1.0, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            })

        return param_dict

    def _lightgbm_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """LightGBM classifier parameters."""
        is_binary = (self.n_classes == 2)
        param_dict = {
            'random_state': self.random_seed,
            'verbosity': -1,
            'objective': 'binary' if is_binary else 'multiclass',
            'metric': 'binary_logloss' if is_binary else 'multi_logloss',
        }
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 6)
        max_n_estimators = 800
        min_learning_rate = 1e-3
        max_leaves = 64

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, max_n_estimators, log=True),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, max_leaves),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        })

        if is_binary:
            param_dict['is_unbalance'] = True
        else:
            param_dict['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])

        return param_dict

    def _catboost_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """CatBoost classifier parameters."""
        param_dict = {
            'logging_level': 'Silent',
            'random_seed': self.random_seed,
            'objective': 'MultiClass',
            'eval_metric': 'MultiClass'
        }

        param_dict['depth'] = trial.suggest_int('depth', 1, 10)

        if param_dict['depth'] >= 8:
            max_iterations = 300
            min_learning_rate = 1e-2
        elif param_dict['depth'] >= 6:
            max_iterations = 400
            min_learning_rate = 5e-3
        else:
            max_iterations = 500
            min_learning_rate = 1e-3

        param_dict.update({
            'iterations': trial.suggest_int('iterations', 20, max_iterations, log=True),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 1e0, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 1e1, log=True),
            'rsm': trial.suggest_float('rsm', 1e-2, 1e0, log=False),
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced', 'SqrtBalanced'])
        })

        return param_dict

    def _svc_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """SVM classifier parameters."""
        return {
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'dual': trial.suggest_categorical('dual', ['auto']),
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1e2, log=True),
            'random_state': self.random_seed
        }

    def _mlp_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Multi-layer Perceptron classifier parameters."""
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))

        return {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': self.random_seed
        }
