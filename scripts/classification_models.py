#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分类模型配置模块
===============

包含6个主要分类模型的参数优化配置:
- RandomForest分类器 (rf_clf)
- XGBoost分类器 (xgboost_clf) 
- LightGBM分类器 (lightgbm_clf)
- CatBoost分类器 (catboost_clf)
- SVM分类器 (svm_clf)
- 多层感知机分类器 (mlp_clf)

从 model_optimizers_class.py 中提取的纯分类模型配置
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import optuna
import os
import numpy as np

@dataclass
class ClassificationModelConfig:
    """分类模型配置"""
    model_class: Any
    param_func: callable

class ClassificationModelOptimizer:
    """分类模型优化器"""
    
    def __init__(self, random_state: Optional[int] = None, n_threads: int = 1, n_classes: int = None):
        self.random_state = random_state
        self.n_threads = n_threads
        self.n_classes = n_classes
        self.model_configs = self._init_classification_model_configs()
        
        # 设置线程环境变量
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    
    def _init_classification_model_configs(self) -> Dict[str, ClassificationModelConfig]:
        """初始化分类模型配置"""
        configs = {
            'rf_clf': ClassificationModelConfig(
                model_class='RandomForestClassifier',
                param_func=self._rf_clf_params
            ),
            'xgboost_clf': ClassificationModelConfig(
                model_class='XGBClassifier',
                param_func=self._xgboost_clf_params
            ),
            'lightgbm_clf': ClassificationModelConfig(
                model_class='LGBMClassifier',
                param_func=self._lightgbm_clf_params
            ),
            'catboost_clf': ClassificationModelConfig(
                model_class='CatBoostClassifier',
                param_func=self._catboost_clf_params
            ),
            'svm_clf': ClassificationModelConfig(
                model_class='SVC',
                param_func=self._svc_clf_params
            ),
            'mlp_clf': ClassificationModelConfig(
                model_class='MLPClassifier',
                param_func=self._mlp_clf_params
            )
        }
        return configs
    
    def get_model_config(self, model_name: str) -> ClassificationModelConfig:
        """获取模型配置"""
        if model_name not in self.model_configs:
            raise ValueError(f"Classification model {model_name} not found in configurations")
        return self.model_configs[model_name]
    
    def get_param_func(self, model_name: str) -> callable:
        """获取参数函数"""
        return self.get_model_config(model_name).param_func
    
    # 分类模型参数函数
    def _rf_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """随机森林分类器参数"""
        param_dict = {
            'random_state': self.random_state
        }
        
        param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 32)
        
        if param_dict['max_depth'] >= 25:
            max_n_estimators = 500
        elif param_dict['max_depth'] >= 15:
            max_n_estimators = 1000
        else:
            max_n_estimators = 2000
            
        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 10, max_n_estimators),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        })
        
        return param_dict
    
    def _xgboost_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """XGBoost分类器参数"""
        param_dict = {
            'random_state': self.random_state,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }
        
        booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
        param_dict['booster'] = booster
        
        if booster in ['gbtree', 'dart']:
            param_dict['max_depth'] = trial.suggest_int('max_depth', 1, 14)
            
            if param_dict['max_depth'] >= 12:
                max_n_estimators = 200
                min_eta = 1e-2
            elif param_dict['max_depth'] >= 10:
                max_n_estimators = 300
                min_eta = 1e-3
            else:
                max_n_estimators = 400
                min_eta = 1e-4
                
            param_dict.update({
                'n_estimators': trial.suggest_int('n_estimators', 20, max_n_estimators),
                'learning_rate': trial.suggest_float('learning_rate', min_eta, 1.0, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            })
        
        return param_dict
    
    def _lightgbm_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """LightGBM分类器参数"""
        is_binary = (self.n_classes == 2)
        param_dict = {
            'random_state': self.random_state,
            'verbosity': -1,
            'objective': 'binary' if is_binary else 'multiclass',
            'metric': 'binary_logloss' if is_binary else 'multi_logloss',
        }
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 20)

        if param_dict['max_depth'] >= 15:
            max_n_estimators = 500; min_learning_rate = 1e-2; max_leaves = 100
        elif param_dict['max_depth'] >= 10:
            max_n_estimators = 1000; min_learning_rate = 1e-3; max_leaves = 200
        else:
            max_n_estimators = 2000; min_learning_rate = 1e-4; max_leaves = 256

        param_dict.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, max_n_estimators, log=True),
            'learning_rate': trial.suggest_float('learning_rate', min_learning_rate, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, max_leaves),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        })

        if is_binary:
            # 二分类自动权重
            param_dict['is_unbalance'] = True
        else:
            # 多分类的权重可选
            param_dict['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])

        return param_dict
    
    def _catboost_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """CatBoost分类器参数"""
        param_dict = {
            'logging_level': 'Silent',
            'random_seed': self.random_state,
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
        """SVM分类器参数"""
        return {
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'dual': trial.suggest_categorical('dual', ['auto']),
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1e2, log=True),
            'random_state': self.random_state
        }
    
    def _mlp_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """多层感知机分类器参数"""
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
            'random_state': self.random_state
        }
    
    def filter_classification_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """过滤分类模型参数"""
        # 复制参数字典，避免修改原始参数
        filtered_params = params.copy()
        
        # 通用过滤：移除所有以下划线开头的辅助参数
        filtered_params = {k: v for k, v in filtered_params.items() if not k.startswith('_')}
        
        # 模型特定过滤
        if model_name == 'mlp_clf':
            # 移除MLP特有的辅助参数
            if 'n_layers' in filtered_params:
                del filtered_params['n_layers']
            
            # 移除所有形如 n_units_lX 的参数
            filtered_params = {k: v for k, v in filtered_params.items() 
                              if not k.startswith('n_units_l')}
        
        return filtered_params
    
    def create_classification_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """创建分类模型实例"""
        params = params.copy()
        
        if model_name == 'rf_clf':
            from sklearn.ensemble import RandomForestClassifier
            params['n_jobs'] = self.n_threads
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost_clf':
            from xgboost import XGBClassifier
            params['n_jobs'] = self.n_threads
            params['nthread'] = self.n_threads
            # 添加类别数参数
            if self.n_classes is not None and self.n_classes > 1:
                params['num_class'] = self.n_classes
            return XGBClassifier(**params)
        elif model_name == 'lightgbm_clf':
            from lightgbm import LGBMClassifier
            p = params.copy()
            p['n_jobs'] = self.n_threads
            # 只有多分类才传 num_class
            if self.n_classes is not None and self.n_classes > 2:
                p['num_class'] = self.n_classes
            else:
                p.pop('num_class', None)
                # 确保目标/评估为二分类设定（若外部默认参数未覆盖）
                p.setdefault('objective', 'binary')
                p.setdefault('metric', 'binary_logloss')
                p.setdefault('is_unbalance', True)
            return LGBMClassifier(**p)
        elif model_name == 'catboost_clf':
            from catboost import CatBoostClassifier
            params['thread_count'] = self.n_threads
            return CatBoostClassifier(**params)
        elif model_name == 'svm_clf':
            from sklearn.svm import SVC
            return SVC(**params)
        elif model_name == 'mlp_clf':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**params)
        else:
            raise ValueError(f"不支持的分类模型类型: {model_name}")
    
    def get_classification_default_params(self, model_name: str) -> Dict[str, Any]:
        """获取分类模型的默认参数"""
        if model_name == 'rf_clf':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'criterion': 'gini',
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': self.n_threads
            }
        elif model_name == 'xgboost_clf':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': self.random_state,
                'n_jobs': self.n_threads,
                'verbosity': 0
            }
        elif model_name == 'lightgbm_clf':
            if self.n_classes == 2:
                return {
                    'n_estimators': 300,
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'min_child_samples': 20,
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'is_unbalance': True,
                    'random_state': self.random_state,
                    'n_jobs': self.n_threads,
                    'verbosity': -1
                }
            else:
                return {
                    'n_estimators': 300,
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'min_child_samples': 20,
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'random_state': self.random_state,
                    'n_jobs': self.n_threads,
                    'verbosity': -1,
                    'num_class': self.n_classes
                }
        elif model_name == 'catboost_clf':
            return {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'logging_level': 'Silent',
                'random_seed': self.random_state,
                'thread_count': self.n_threads
            }
        elif model_name == 'svm_clf':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'degree': 3,
                'probability': True,
                'random_state': self.random_state
            }
        elif model_name == 'mlp_clf':
            return {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 500,
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            }
        else:
            return {}
    
    def get_available_models(self) -> list:
        """获取可用的分类模型列表"""
        return list(self.model_configs.keys())