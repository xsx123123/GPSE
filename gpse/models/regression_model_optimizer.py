"""
Regression model hyperparameter optimization.

This module provides parameter ranges for Optuna hyperparameter optimization
and factories for regression models. Classification models live in
``gpse.models.classification_model_optimizer``.

Model registration is driven by ``gpse/config/models.yaml``. The param
functions below are referenced from that file via ``builtin:_xxx_params``.
"""

import os
import optuna
from typing import Dict, Any, Optional

from gpse.config import ModelConstants
from gpse.config._model_registry import ModelRegistry


LINEAR_SVR_MAX_ITER = 10_000


class RegressionModelOptimizer:
    """Regression model optimizer backed by the YAML model registry."""

    def __init__(
        self,
        random_seed: Optional[int] = None,
        n_threads: int = 1,
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
            catboost_train_dir=self.catboost_train_dir,
            builtin_param_funcs=builtin_funcs,
            task_filter="regression",
        )

        self.model_configs = self._registry.model_configs

        for _env_var in ModelConstants.thread_env_vars:
            os.environ[_env_var] = str(n_threads)

    def get_model_config(self, model_name: str):
        """Get model configuration by name."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configurations")
        return self.model_configs[model_name]

    def get_param_func(self, model_name: str) -> callable:
        """Get parameter function by model name."""
        return self._registry.get_param_func(model_name)

    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create model instance via the registry."""
        return self._registry.create_model(model_name, params)

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters via the registry."""
        return self._registry.get_default_params(model_name)

    def filter_model_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter auxiliary params via the registry."""
        return self._registry.filter_model_params(model_name, params)

    # ─── Optuna search-space functions (referenced from models.yaml) ───────────

    def _histgradientboost_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'max_iter': trial.suggest_int('max_iter', 10, 500, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 8, 64),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            'n_iter_no_change': 20,
            'random_state': self.random_seed
        }

    def _gbdt_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'random_state': self.random_seed
        }

        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 6)

        if param_dict['max_depth'] >= 5:
            max_n_estimators = 300
            min_learning_rate = 0.005
        else:
            max_n_estimators = 400
            min_learning_rate = 0.003

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, max_n_estimators),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 0.1),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        })

        return param_dict

    def _sgd_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
            'alpha': trial.suggest_float('alpha', 1e-8, 1e2, log=True),
            'random_state': self.random_seed
        }

    def _elasticnet_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'alpha': trial.suggest_float('alpha', 1e-8, 1e2, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 1e-5, 1.0, log=True),
            'random_state': self.random_seed
        }

    def _svr_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'loss': trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1e2, log=True),
            'max_iter': LINEAR_SVR_MAX_ITER,
            'random_state': self.random_seed
        }

    def _mlp_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))

        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': self.random_seed
        }

        params['_n_layers'] = n_layers

        return params

    def _knn_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 101, step=5),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'leaf_size': trial.suggest_int('leaf_size', 30, 100)
        }

    def _rf_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'random_state': self.random_seed
        }

        param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 6)
        max_n_estimators = 800

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 10, max_n_estimators),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        })

        return param_dict

    def _xgboost_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'random_state': self.random_seed,
            'verbosity': 0
        }

        if (param_dict['booster'] == 'gbtree') or (param_dict['booster'] == 'dart'):
            param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 6)
            max_n_estimators = 400
            min_eta = 1e-3

            param_dict['n_estimators'] = trial.suggest_int('n_estimators', 20, max_n_estimators)
            param_dict['eta'] = trial.suggest_float('eta', min_eta, 1.0, log=True)
            param_dict['min_child_weight'] = trial.suggest_float('min_child_weight', 1, 20)
            param_dict['gamma'] = trial.suggest_float('gamma', 0, 10)
            param_dict['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            param_dict['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            param_dict['max_bin'] = trial.suggest_categorical('max_bin', [64, 128, 256, 512, 1024])

            if param_dict['booster'] == 'dart':
                param_dict['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                param_dict['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                param_dict['rate_drop'] = trial.suggest_float('rate_drop', 0., 1.0)
                param_dict['one_drop'] = trial.suggest_categorical('one_drop', [0, 1])

        return param_dict

    def _adaboost_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e0, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
            'random_state': self.random_seed
        }

    def _lightgbm_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'objective': 'regression',
            'random_state': self.random_seed,
            'verbosity': -1
        }

        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 6)
        max_n_estimators = 800
        min_learning_rate = 1e-3
        max_leaves = 64

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, max_n_estimators, log=True),
            'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255, 511, 1023]),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, max_leaves),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        })

        return param_dict

    def _catboost_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'logging_level': 'Silent',
            'random_seed': self.random_seed
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
            'rsm': trial.suggest_float('rsm', 1e-2, 1e0, log=False)
        })

        return param_dict

    def _lasso_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'alpha': trial.suggest_float('alpha', 1e-8, 1e2, log=True),
            'random_state': self.random_seed
        }

    def _kernelridge_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'laplacian', 'polynomial', 'sigmoid'])

        params = {
            'alpha': trial.suggest_float('alpha', 1e-2, 10.0, log=True),
            'kernel': kernel,
        }

        if kernel in ['rbf', 'laplacian', 'polynomial', 'sigmoid']:
            params['gamma'] = trial.suggest_float('gamma', 1e-6, 1.0, log=True)

        if kernel == 'polynomial':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        if kernel == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        return params

    def _gblup_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Fixed linear-kernel ridge baseline (GBLUP-equivalent after marker scaling)."""
        return {'alpha': 1.0, 'kernel': 'linear'}


# Backward-compatible public name.
ModelOptimizer = RegressionModelOptimizer
