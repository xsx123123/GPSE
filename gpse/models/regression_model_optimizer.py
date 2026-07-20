"""
Regression model hyperparameter optimization.

This module provides parameter ranges for Optuna hyperparameter optimization
and factories for regression models. Classification models live in
``gpse.models.classification_model_optimizer``.
"""

import os
import optuna
from typing import Dict, Any, Optional

# Import config functions
from gpse.config import ModelConfig, ModelConstants


LINEAR_SVR_MAX_ITER = 10_000


class RegressionModelOptimizer:
    """Regression model optimizer."""

    def __init__(
        self,
        random_seed: Optional[int] = None,
        n_threads: int = 1,
        random_state: Optional[int] = None,
    ):
        if random_seed is not None and random_state is not None and random_seed != random_state:
            raise ValueError("random_seed and random_state were both provided with different values")
        if random_seed is None:
            random_seed = random_state

        self.random_seed = random_seed                   # random seed for reproducibility
        self.random_state = random_seed                  # backward-compatible alias
        self.n_threads = n_threads                       # Set number of threads
        self.model_configs = {}                          # Initialize empty dict first
        self.model_configs = self._init_model_configs()  # Then populate it

        # Set environment variables for multi-threading (all 6 BLAS/OpenMP backends)
        for _env_var in ModelConstants.thread_env_vars:
            os.environ[_env_var] = str(n_threads)

    def _init_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize regression model configurations."""
        configs = {
            # Regression models
            'elasticnet_reg': ModelConfig(
                model_class='ElasticNet',
                param_func=self._elasticnet_reg_params,
                is_regression=True
            ),
            'gbdt_reg': ModelConfig(
                model_class='GradientBoostingRegressor',
                param_func=self._gbdt_reg_params,
                is_regression=True
            ),
            'svr_reg': ModelConfig(
                model_class='LinearSVR',
                param_func=self._svr_reg_params,
                is_regression=True
            ),
            'mlp_reg': ModelConfig(
                model_class='MLPRegressor',
                param_func=self._mlp_reg_params,
                is_regression=True
            ),
            'knn_reg': ModelConfig(
                model_class='KNeighborsRegressor',
                param_func=self._knn_reg_params,
                is_regression=True
            ),
            'rf_reg': ModelConfig(
                model_class='RandomForestRegressor',
                param_func=self._rf_reg_params,
                is_regression=True
            ),
            'xgboost_reg': ModelConfig(
                model_class='XGBRegressor',
                param_func=self._xgboost_reg_params,
                is_regression=True
            ),
            'adaboost_reg': ModelConfig(
                model_class='AdaBoostRegressor',
                param_func=self._adaboost_reg_params,
                is_regression=True
            ),
            'lightgbm_reg': ModelConfig(
                model_class='LGBMRegressor',
                param_func=self._lightgbm_reg_params,
                is_regression=True
            ),
            'catboost_reg': ModelConfig(
                model_class='CatBoostRegressor',
                param_func=self._catboost_reg_params,
                is_regression=True
            ),
            'kernelridge_reg': ModelConfig(
                model_class='KernelRidge',
                param_func=self._kernelridge_reg_params,
                is_regression=True
            ),
            'gblup_reg': ModelConfig(
                model_class='KernelRidge',
                param_func=self._gblup_reg_params,
                is_regression=True
            ),
            'histgradientboost_reg': ModelConfig(
                model_class='HistGradientBoostingRegressor',
                param_func=self._histgradientboost_reg_params,
                is_regression=True
            ),
            'sgd_reg': ModelConfig(
                model_class='SGDRegressor',
                param_func=self._sgd_reg_params,
                is_regression=True
            ),
            'lasso_reg': ModelConfig(
                model_class='Lasso',
                param_func=self._lasso_reg_params,
                is_regression=True
            )
        }

        return configs

    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get model configuration by name
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configurations")
        return self.model_configs[model_name]

    def get_param_func(self, model_name: str) -> callable:
        """Get parameter function by model name"""
        return self.get_model_config(model_name).param_func

    # Regression parameter functions
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

        # 限制最大深度，减少过深的树
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 6)

        # 反转逻辑，深树使用更少的数量
        if param_dict['max_depth'] >= 5:
            max_n_estimators = 300
            min_learning_rate = 0.005
        else:
            max_n_estimators = 400  # 大幅降低上限
            min_learning_rate = 0.003

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, max_n_estimators),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 0.1),
            # 缩小样本分裂和叶节点的搜索范围
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)  # 提高最小子采样比例
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


    def _gradientboost_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        param_dict = {
            'random_state': self.random_seed,
            'n_iter_no_change': 20
        }

        param_dict['max_depth'] = trial.suggest_int('max_depth', 1, 6)

        if param_dict['max_depth'] >= 5:
            max_n_estimators = 250
            min_learning_rate = 1e-2
        else:
            max_n_estimators = 300
            min_learning_rate = 5e-3

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 10, max_n_estimators),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 1e0, log=True),
            'subsample': trial.suggest_float('subsample', 1e-2, 1.0, log=False),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
        })

        return param_dict

    def _svr_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'loss': trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1e2, log=True),
            'max_iter': LINEAR_SVR_MAX_ITER,
            'random_state': self.random_seed
        }

    def _mlp_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # 标记为临时参数，这样不会被传递给最终模型
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))

        # 只返回MLPRegressor实际需要的参数
        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': self.random_seed
        }

        # 将辅助参数作为元数据存储，防止传递给模型
        params['_n_layers'] = n_layers  # 添加下划线前缀标记为辅助参数

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

    # 添加RKHS (KernelRidge)模型的参数函数
    def _kernelridge_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        KernelRidge参数优化

        参数:
        - alpha: 正则化强度，值越大正则化越强
        - kernel: 核函数类型
        - gamma: RBF, laplacian, polynomial, exponential, sigmoid核的参数
        - degree: 多项式核的度
        - coef0: 多项式和sigmoid核的独立项
        """
        kernel = trial.suggest_categorical('kernel', ['rbf', 'laplacian', 'polynomial', 'sigmoid'])

        params = {
            'alpha': trial.suggest_float('alpha', 1e-6, 10.0, log=True),
            'kernel': kernel,
        }

        # 根据不同核函数添加特定参数
        if kernel in ['rbf', 'laplacian', 'polynomial', 'sigmoid']:
            params['gamma'] = trial.suggest_float('gamma', 1e-6, 10.0, log=True)

        if kernel == 'polynomial':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        if kernel == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        return params

    def _gblup_reg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Fixed linear-kernel ridge baseline (GBLUP-equivalent after marker scaling)."""
        return {'alpha': 1.0, 'kernel': 'linear'}

    def filter_model_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤掉不是实际模型参数的辅助参数

        参数:
            model_name: 模型名称
            params: 原始参数字典

        返回:
            过滤后的参数字典
        """
        # 复制参数字典，避免修改原始参数
        filtered_params = params.copy()

        # 通用过滤：移除所有以下划线开头的辅助参数
        filtered_params = {k: v for k, v in filtered_params.items() if not k.startswith('_')}

        # 模型特定过滤
        if model_name == 'mlp_reg':
            # 移除MLP特有的辅助参数
            if 'n_layers' in filtered_params:
                del filtered_params['n_layers']

            # 移除所有形如 n_units_lX 的参数
            filtered_params = {k: v for k, v in filtered_params.items()
                              if not k.startswith('n_units_l')}

        # 处理ngboost特有的参数
        elif model_name == 'ngboost_reg':
            # 移除特定的嵌套参数标记
            if 'Base' in filtered_params:
                del filtered_params['Base']
            # 移除以Base__开头的参数，这些需要特殊处理
            filtered_params = {k: v for k, v in filtered_params.items()
                              if not k.startswith('Base__')}

        return filtered_params

    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """创建模型实例并设置线程参数"""
        # 创建参数副本
        params = params.copy()

        if model_name == 'xgboost_reg':
            from xgboost import XGBRegressor
            params['n_jobs'] = self.n_threads
            params['nthread'] = self.n_threads
            return XGBRegressor(**params)
        elif model_name == 'histgradientboost_reg':
            from sklearn.ensemble import HistGradientBoostingRegressor
            params['n_jobs'] = self.n_threads
            return HistGradientBoostingRegressor(**params)
        elif model_name == 'sgd_reg':
            from sklearn.linear_model import SGDRegressor
            return SGDRegressor(**params)
        elif model_name == 'ngboost_reg':
            from ngboost import NGBRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.linear_model import LinearRegression

            # 处理基学习器
            base_type = params.get('Base', 'DecisionTreeRegressor')
            if 'Base' in params:
                del params['Base']

            # 创建基学习器
            if base_type == 'DecisionTreeRegressor':
                # 收集所有Base__前缀的参数
                base_params = {}
                keys_to_delete = []
                for key in params:
                    if key.startswith('Base__'):
                        param_name = key.replace('Base__', '')
                        base_params[param_name] = params[key]
                        keys_to_delete.append(key)

                # 删除已处理的参数
                for key in keys_to_delete:
                    del params[key]

                # 创建决策树基学习器
                params['Base'] = DecisionTreeRegressor(**base_params)

            elif base_type == 'LinearRegressor':
                params['Base'] = LinearRegression()

            # 设置线程
            params['n_jobs'] = self.n_threads

            return NGBRegressor(**params)
        elif model_name == 'lightgbm_reg':
            from lightgbm import LGBMRegressor
            params['n_jobs'] = self.n_threads
            return LGBMRegressor(**params)
        elif model_name == 'elasticnet_reg':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(**params)
        elif model_name == 'gbdt_reg':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
        elif model_name == 'svr_reg':
            from sklearn.svm import LinearSVR
            return LinearSVR(**params)
        elif model_name == 'mlp_reg':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(**params)
        elif model_name == 'knn_reg':
            from sklearn.neighbors import KNeighborsRegressor
            params['n_jobs'] = self.n_threads
            return KNeighborsRegressor(**params)
        elif model_name == 'rf_reg':
            from sklearn.ensemble import RandomForestRegressor
            params['n_jobs'] = self.n_threads
            return RandomForestRegressor(**params)
        elif model_name == 'adaboost_reg':
            from sklearn.ensemble import AdaBoostRegressor
            return AdaBoostRegressor(**params)
        elif model_name == 'catboost_reg':
            from catboost import CatBoostRegressor
            params['thread_count'] = self.n_threads
            return CatBoostRegressor(**params)
        elif model_name == 'kernelridge_reg':
            from sklearn.kernel_ridge import KernelRidge
            return KernelRidge(**params)
        elif model_name == 'gblup_reg':
            from sklearn.kernel_ridge import KernelRidge
            return KernelRidge(**params)
        elif model_name == 'lasso_reg':
            from sklearn.linear_model import Lasso
            return Lasso(**params)
        else:
            # 如果没有特殊处理的模型，抛出异常而不是返回None
            raise ValueError(f"不支持的模型类型: {model_name}")

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型的默认参数"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configurations")

        # 为不同模型设置默认参数
        if model_name == 'elasticnet_reg':
            return {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': self.random_seed
            }
        elif model_name == 'gbdt_reg':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 1.0,
                'random_state': self.random_seed
            }
        elif model_name == 'svr_reg':
            return {
                'loss': 'epsilon_insensitive',
                'C': 1.0,
                'tol': 1e-4,
                'max_iter': LINEAR_SVR_MAX_ITER,
                'random_state': self.random_seed
            }
        elif model_name == 'mlp_reg':
            return {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 200,
                'random_state': self.random_seed
            }
        elif model_name == 'knn_reg':
            return {
                'n_neighbors': 5,
                'weights': 'uniform',
                'leaf_size': 30
            }
        elif model_name == 'rf_reg':
            return {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'random_state': self.random_seed
            }
        elif model_name == 'xgboost_reg':
            return {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'booster': 'gbtree',
                'random_state': self.random_seed,
                'verbosity': 0
            }
        elif model_name == 'lightgbm_reg':
            return {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'min_child_samples': 20,
                'objective': 'regression',
                'random_state': self.random_seed,
                'verbosity': -1
            }
        elif model_name == 'catboost_reg':
            return {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'logging_level': 'Silent',
                'random_seed': self.random_seed
            }
        elif model_name == 'adaboost_reg':
            return {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'loss': 'linear',
                'random_state': self.random_seed
            }
        elif model_name == 'kernelridge_reg':
            return {
                'alpha': 1.0,
                'kernel': 'rbf',
                'gamma': None
            }
        elif model_name == 'gblup_reg':
            return {'alpha': 1.0, 'kernel': 'linear'}
        elif model_name == 'histgradientboost_reg':
            return {
                'max_depth': None,
                'max_iter': 100,
                'max_leaf_nodes': 31,
                'min_samples_leaf': 20,
                'learning_rate': 0.1,
                'random_state': self.random_seed
            }
        elif model_name == 'sgd_reg':
            return {
                'loss': 'squared_error',
                'penalty': 'l2',
                'alpha': 0.0001,
                'random_state': self.random_seed
            }
        elif model_name == 'lasso_reg':
            return {
                'alpha': 1.0,
                'random_state': self.random_seed
            }
        else:
            # 如果没有特定的默认参数，返回一个空字典
            return {}


# Backward-compatible public name. New code should prefer
# ``RegressionModelOptimizer`` to make the task type explicit.
ModelOptimizer = RegressionModelOptimizer
