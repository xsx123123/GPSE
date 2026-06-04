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

Pure classification model configuration extracted from model_optimizers_class.py
"""

from typing import Dict, Any, Optional
import optuna
import os
import numpy as np

from gpse.config import ClassificationModelConfig, ModelConstants

class ClassificationModelOptimizer:
    """Classification Model Optimizer"""
    
    def __init__(
        self,
        random_seed: Optional[int] = None,
        n_threads: int = 1,
        n_classes: int = None,
        random_state: Optional[int] = None,
    ):
        if random_seed is not None and random_state is not None and random_seed != random_state:
            raise ValueError("random_seed and random_state were both provided with different values")
        if random_seed is None:
            random_seed = random_state

        self.random_seed = random_seed
        self.random_state = random_seed
        self.n_threads = n_threads
        self.n_classes = n_classes
        self.model_configs = self._init_classification_model_configs()
        
        # Set environment variables for multi-threading (all 6 BLAS/OpenMP backends)
        for _env_var in ModelConstants.thread_env_vars:
            os.environ[_env_var] = str(n_threads)
    
    def _init_classification_model_configs(self) -> Dict[str, ClassificationModelConfig]:
        """Initialize classification model configurations"""
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
        """Get model configuration"""
        if model_name not in self.model_configs:
            raise ValueError(f"Classification model {model_name} not found in configurations")
        return self.model_configs[model_name]
    
    def get_param_func(self, model_name: str) -> callable:
        """Get parameter function"""
        return self.get_model_config(model_name).param_func
    
    # Classification model parameter functions
    def _rf_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """RandomForest classifier parameters"""
        param_dict = {
            'random_state': self.random_seed
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
        """XGBoost classifier parameters"""
        param_dict = {
            'random_state': self.random_seed,
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
        """LightGBM classifier parameters"""
        is_binary = (self.n_classes == 2)
        param_dict = {
            'random_state': self.random_seed,
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
            # Automatic weights for binary classification
            param_dict['is_unbalance'] = True
        else:
            # Optional weights for multi-class classification
            param_dict['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])

        return param_dict
    
    def _catboost_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """CatBoost classifier parameters"""
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
        """SVM classifier parameters"""
        return {
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'dual': trial.suggest_categorical('dual', ['auto']),
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1e2, log=True),
            'random_state': self.random_seed
        }
    
    def _mlp_clf_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Multi-layer Perceptron classifier parameters"""
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
    
    def filter_classification_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter classification model parameters"""
        # Copy parameter dictionary to avoid modifying original
        filtered_params = params.copy()
        
        # General filtering: remove all auxiliary parameters starting with an underscore
        filtered_params = {k: v for k, v in filtered_params.items() if not k.startswith('_')}
        
        # Model-specific filtering
        if model_name == 'mlp_clf':
            # Remove MLP-specific auxiliary parameters
            if 'n_layers' in filtered_params:
                del filtered_params['n_layers']
            
            # Remove all parameters in the form of n_units_lX
            filtered_params = {k: v for k, v in filtered_params.items() 
                              if not k.startswith('n_units_l')}
        
        return filtered_params
    
    def create_classification_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create classification model instance"""
        params = params.copy()
        
        if model_name == 'rf_clf':
            from sklearn.ensemble import RandomForestClassifier
            params['n_jobs'] = self.n_threads
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost_clf':
            from xgboost import XGBClassifier
            params['n_jobs'] = self.n_threads
            params['nthread'] = self.n_threads
            # Add number of classes parameter
            if self.n_classes is not None and self.n_classes > 1:
                params['num_class'] = self.n_classes
            return XGBClassifier(**params)
        elif model_name == 'lightgbm_clf':
            from lightgbm import LGBMClassifier
            p = params.copy()
            p['n_jobs'] = self.n_threads
            # Pass num_class only for multi-class classification
            if self.n_classes is not None and self.n_classes > 2:
                p['num_class'] = self.n_classes
            else:
                p.pop('num_class', None)
                # Ensure target/metric is set for binary classification (if not overridden by external default params)
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
            raise ValueError(f"Unsupported classification model type: {model_name}")
    
    def get_classification_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for classification models"""
        if model_name == 'rf_clf':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'criterion': 'gini',
                'max_features': 'sqrt',
                'random_state': self.random_seed,
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
                'random_state': self.random_seed,
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
                    'random_state': self.random_seed,
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
                    'random_state': self.random_seed,
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
                'random_seed': self.random_seed,
                'thread_count': self.n_threads
            }
        elif model_name == 'svm_clf':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'degree': 3,
                'probability': True,
                'random_state': self.random_seed
            }
        elif model_name == 'mlp_clf':
            return {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 500,
                'random_state': self.random_seed,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            }
        else:
            return {}
    
    def get_available_models(self) -> list:
        """Get list of available classification models"""
        return list(self.model_configs.keys())
