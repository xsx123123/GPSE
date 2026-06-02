#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基因组预测集成学习模块
======================

功能特性:
- 基于现有已训练模型生成Stacking集成模型
- 以Pearson相关系数为主要评估和选择指标
- 自动选择性能最佳的N个基础模型
- 支持交叉验证生成元特征
- 可视化集成模型性能
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StackingEnsemble:
    """Stacking集成学习模型类"""
    
    def __init__(
        self,
        base_models_dir: str,
        top_n_models: int = 5,
        meta_model_type: str = 'ridge',
        cv_folds: int = 5,
        random_seed: int = 42,
        output_dir: Optional[str] = None,
        n_threads: int = 1,
        use_default_params: bool = False,
        task_type: str = 'regression'  # 新增：任务类型
    ):
        """
        初始化Stacking集成学习模型
        
        参数:
            base_models_dir: 基础模型存储目录
            top_n_models: 选择性能最佳的前N个模型
            meta_model_type: 元模型类型 (目前支持'ridge')
            cv_folds: 生成元特征的交叉验证折数
            random_seed: 随机种子
            output_dir: 结果输出目录，默认为base_models_dir/ensemble_stacking
            n_threads: 线程数
            use_default_params: 是否使用默认参数
            task_type: 任务类型 ('regression' 或 'classification')
        """
        self.base_models_dir = base_models_dir
        self.top_n_models = top_n_models
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.output_dir = output_dir or os.path.join(base_models_dir, 'ensemble_stacking')
        self.n_threads = n_threads
        self.use_default_params = use_default_params
        self.task_type = task_type  # 新增：保存任务类型
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型和结果容器
        self.base_models = {}
        self.selected_models = []
        self.meta_model = None
        self.meta_features_cols = []
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        logger.info(f"初始化Stacking集成学习模型，选择前{top_n_models}个模型")
    
    def setup_logging(self):
        """配置日志记录"""
        log_file = os.path.join(self.output_dir, 'stacking_ensemble.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def load_and_select_models(self, model_names: Optional[List[str]] = None) -> List[str]:
        """加载并选择性能最佳的模型"""
        logger.info("开始加载和选择模型...")
        
        # 查找模型比较结果
        comparison_file = os.path.join(self.base_models_dir, 'model_comparison.csv')
        if not os.path.exists(comparison_file):
            raise FileNotFoundError(f"找不到模型比较文件: {comparison_file}")
        
        # 加载模型比较结果
        comparison_df = pd.read_csv(comparison_file)
        logger.info(f"找到{len(comparison_df)}个已训练模型")
        
        # 打印CSV中的模型名称，用于调试
        logger.info(f"CSV中的模型名称: {comparison_df['Model'].tolist()}")
        
        # 如果传入了指定的模型名称，直接使用
        if model_names:
            logger.info(f"使用指定的模型名称: {model_names}")
            
            # 验证指定的模型是否存在于比较文件中
            available_models = comparison_df['Model'].tolist()
            valid_models = []
            
            for name in model_names:
                # 去掉可能的_reg后缀
                base_name = name.replace('_reg', '')
                # 在CSV中查找匹配的模型名称
                matches = comparison_df[comparison_df['Model'].str.lower() == base_name.lower()]
                if not matches.empty:
                    valid_models.append(name)
                else:
                    logger.warning(f"指定的模型 {name} 不在比较文件中，跳过")
            
            if not valid_models:
                raise ValueError(f"没有找到任何有效的指定模型。可用模型: {available_models}")
            
            self.selected_models = valid_models
            logger.info(f"使用指定的{len(self.selected_models)}个模型: {', '.join(self.selected_models)}")
            
        else:
            # 如果没有指定模型名称，根据任务类型按相应指标排序选择
            if self.task_type == 'classification':
                # 分类任务：按Ensemble Accuracy排序，如果没有则按Test Accuracy排序
                if 'Ensemble Accuracy' in comparison_df.columns:
                    sort_column = 'Ensemble Accuracy'
                    logger.info("按Ensemble Accuracy排序选择分类模型")
                else:
                    sort_column = 'Test Accuracy'
                    logger.info("按Test Accuracy排序选择分类模型")
            else:
                # 回归任务：按Ensemble Pearson排序，如果没有则按Test Pearson排序
                if 'Ensemble Pearson' in comparison_df.columns:
                    sort_column = 'Ensemble Pearson'
                    logger.info("按Ensemble Pearson相关系数排序选择回归模型")
                else:
                    sort_column = 'Test Pearson'
                    logger.info("按Test Pearson相关系数排序选择回归模型")
            
            comparison_df = comparison_df.sort_values(sort_column, ascending=False)
            
            # 选择前N个模型
            n_models = min(self.top_n_models, len(comparison_df))
            self.selected_models = comparison_df['Model'].iloc[:n_models].tolist()
            logger.info(f"选择的{n_models}个性能最佳模型: {', '.join(self.selected_models)}")
        
        # 加载选定的模型
        for model_name in self.selected_models:
            # 根据任务类型确定模型后缀和搜索路径
            if self.task_type == 'classification':
                # 分类模型：模型名称 + _clf 后缀
                model_with_suffix = f"{model_name}_clf"
            else:
                # 回归模型：模型名称 + _reg 后缀  
                model_with_suffix = f"{model_name}_reg"
            
            # 构建搜索路径列表（按优先级排序）
            model_paths = [
                # 1. 代表性模型路径（最优先）
                os.path.join(self.base_models_dir, model_with_suffix, "representative_model", "model.pkl"),
                
                # 2. 直接模型路径
                os.path.join(self.base_models_dir, model_with_suffix, 'model.pkl'),
                
                # 3. 备选：无后缀的模型路径（兼容性）
                os.path.join(self.base_models_dir, model_name, "representative_model", "model.pkl"),
                os.path.join(self.base_models_dir, model_name, 'model.pkl'),
            ]
            
            # 4. 添加重复实验路径
            for repeat_idx in range(1, 51):
                model_paths.extend([
                    os.path.join(self.base_models_dir, model_with_suffix, f"repeat_{repeat_idx}", "model.pkl"),
                    os.path.join(self.base_models_dir, model_name, f"repeat_{repeat_idx}", "model.pkl")
                ])
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.base_models[model_name] = joblib.load(model_path)
                    logger.info(f"成功加载模型: {model_name}，路径: {model_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                logger.warning(f"找不到模型文件，尝试过的路径: {model_paths}")
                self.selected_models.remove(model_name)
        
        if not self.base_models:
            raise ValueError("没有成功加载任何模型")
        
        logger.info(f"最终选择的{len(self.selected_models)}个模型: {', '.join(self.selected_models)}")
        self.meta_features_cols = self.selected_models.copy()
        
        return self.selected_models
    
    def create_meta_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建元特征矩阵
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征，可选
            
        返回:
            训练集元特征和测试集元特征(如果提供了X_test)
        """
        logger.info(f"开始创建元特征，使用{self.cv_folds}折交叉验证...")
        
        # 创建用于训练的元特征矩阵
        meta_train = np.zeros((X_train.shape[0], len(self.selected_models)))
        
        # 如果有测试集，创建测试集的元特征矩阵
        meta_test = None
        if X_test is not None:
            meta_test = np.zeros((X_test.shape[0], len(self.selected_models)))
        
        # 为每个基础模型生成元特征
        for i, model_name in enumerate(self.selected_models):
            logger.info(f"为模型 {model_name} 生成元特征")
            
            # 使用已加载的模型，而不是重新加载
            model = self.base_models[model_name]
            
            # 检查模型是否为元组(model, scaler)
            is_tuple_model = isinstance(model, tuple) and len(model) == 2
            
            # 使用K折交叉验证生成训练集元特征
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            
            for train_idx, val_idx in kf.split(X_train):
                # 划分数据
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                if is_tuple_model:
                    # 如果是元组，解包模型和缩放器
                    actual_model, scaler = model
                    # 克隆模型
                    fold_model = clone(actual_model)
                    # 缩放数据
                    X_fold_train_scaled = scaler.transform(X_fold_train)
                    X_fold_val_scaled = scaler.transform(X_fold_val)
                    # 训练模型
                    fold_model.fit(X_fold_train_scaled, y_fold_train)
                    # 预测
                    meta_train[val_idx, i] = fold_model.predict(X_fold_val_scaled)
                else:
                    # 原始代码逻辑
                    fold_model = clone(model)
                    fold_model.fit(X_fold_train, y_fold_train)
                    meta_train[val_idx, i] = fold_model.predict(X_fold_val)
            
            # 如果有测试集，使用完整训练集训练模型并预测测试集
            if X_test is not None:
                if is_tuple_model:
                    # 解包模型和缩放器
                    actual_model, scaler = model
                    # 缩放训练和测试数据
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    # 训练模型
                    actual_model.fit(X_train_scaled, y_train)
                    # 预测
                    meta_test[:, i] = actual_model.predict(X_test_scaled)
                else:
                    # 原始代码逻辑
                    model.fit(X_train, y_train)
                    meta_test[:, i] = model.predict(X_test)
        
        logger.info(f"元特征创建完成，训练集元特征形状: {meta_train.shape}")
        if meta_test is not None:
            logger.info(f"测试集元特征形状: {meta_test.shape}")
        
        return meta_train, meta_test
    
    def fit_meta_model(self, meta_train: np.ndarray, y_train: pd.Series):
        """
        训练元模型
        
        参数:
            meta_train: 训练集元特征
            y_train: 训练集标签
        """
        logger.info(f"开始训练{self.task_type}任务的元模型 ({self.meta_model_type})，使用{self.n_threads}个线程...")
        
        # 根据任务类型创建不同的元模型
        if self.task_type == 'classification':
            # 分类任务：使用逻辑回归作为元模型
            if self.meta_model_type == 'ridge':
                from sklearn.linear_model import LogisticRegression
                
                if self.use_default_params:
                    logger.info("使用LogisticRegression元模型默认参数...")
                    lr_model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
                else:
                    lr_model = LogisticRegression(
                        C=1.0, 
                        random_state=self.random_seed, 
                        max_iter=1000,
                        multi_class='ovr'  # 多分类策略
                    )
                
                # 检查是否支持n_jobs参数
                if 'n_jobs' in LogisticRegression.__init__.__code__.co_varnames:
                    if self.use_default_params:
                        lr_model = LogisticRegression(
                            random_state=self.random_seed, 
                            max_iter=1000,
                            n_jobs=self.n_threads
                        )
                    else:
                        lr_model = LogisticRegression(
                            C=1.0, 
                            random_state=self.random_seed, 
                            max_iter=1000,
                            multi_class='ovr',
                            n_jobs=self.n_threads
                        )
                
                self.meta_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logistic', lr_model)
                ])
            else:
                raise ValueError(f"分类任务不支持的元模型类型: {self.meta_model_type}")
        else:
            # 回归任务（原有逻辑）
            if self.meta_model_type == 'ridge':
                from sklearn.linear_model import Ridge
                
                if self.use_default_params:
                    logger.info("使用Ridge元模型默认参数...")
                    ridge_model = Ridge(random_state=self.random_seed)
                else:
                    ridge_model = Ridge(alpha=1.0, random_state=self.random_seed)
                    
                # 如果Ridge支持n_jobs参数
                if 'n_jobs' in Ridge.__init__.__code__.co_varnames:
                    if self.use_default_params:
                        ridge_model = Ridge(random_state=self.random_seed, n_jobs=self.n_threads)
                    else:
                        ridge_model = Ridge(alpha=1.0, random_state=self.random_seed, n_jobs=self.n_threads)
                    
                self.meta_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', ridge_model)
                ])
            else:
                raise ValueError(f"回归任务不支持的元模型类型: {self.meta_model_type}")
        
        # 训练元模型
        self.meta_model.fit(meta_train, y_train)
        
        logger.info("元模型训练完成")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        使用Stacking集成模型进行预测
        
        参数:
            X: 特征矩阵
            
        返回:
            (预测值, 预测概率) - 分类任务返回概率，回归任务概率为None
        """
        if self.meta_model is None:
            raise ValueError("元模型尚未训练，请先调用fit方法")
        
        # 创建元特征
        meta_features = np.zeros((X.shape[0], len(self.selected_models)))
        
        # 使用每个基础模型生成预测作为元特征
        for i, model_name in enumerate(self.selected_models):
            model = self.base_models[model_name]
            
            # 检查模型是否为元组(model, scaler)
            if isinstance(model, tuple) and len(model) == 2:
                actual_model, scaler = model
                X_scaled = scaler.transform(X)
                if self.task_type == 'classification' and hasattr(actual_model, 'predict_proba'):
                    # 分类任务：使用概率预测的最大值作为元特征
                    probas = actual_model.predict_proba(X_scaled)
                    meta_features[:, i] = np.max(probas, axis=1)
                else:
                    meta_features[:, i] = actual_model.predict(X_scaled)
            else:
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    # 分类任务：使用概率预测的最大值作为元特征
                    probas = model.predict_proba(X)
                    meta_features[:, i] = np.max(probas, axis=1)
                else:
                    meta_features[:, i] = model.predict(X)
        
        # 使用元模型进行最终预测
        if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
            # 分类任务：返回预测标签和概率
            y_pred_proba = self.meta_model.predict_proba(meta_features)
            y_pred = self.meta_model.predict(meta_features)
            return y_pred, y_pred_proba
        else:
            # 回归任务：只返回预测值
            y_pred = self.meta_model.predict(meta_features)
            return y_pred, None
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: Optional[pd.DataFrame] = None, 
        y_test: Optional[pd.Series] = None,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        训练Stacking集成模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征，可选
            y_test: 测试集标签，可选
            model_names: 要考虑的模型名称列表，None表示考虑所有已训练模型
            
        返回:
            评估结果
        """
        # 记录开始时间
        start_time = time.time()
        
        # 加载和选择模型
        self.load_and_select_models(model_names)
        
        # 创建元特征
        meta_train, meta_test = self.create_meta_features(X_train, y_train, X_test)
        
        # 训练元模型
        self.fit_meta_model(meta_train, y_train)
        
        # 保存模型
        logger.info("保存Stacking集成模型...")
        joblib.dump(self, os.path.join(self.output_dir, 'stacking_ensemble_model.pkl'))
        
        # 评估性能
        results = {}
        
        # 在训练集上评估
        if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
            y_train_pred = self.meta_model.predict(meta_train)
            y_train_proba = self.meta_model.predict_proba(meta_train)
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
        else:
            y_train_pred = self.meta_model.predict(meta_train)
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
        
        results['train_metrics'] = train_metrics
        
        logger.info("训练集性能:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.6f}")
        
        # 如果有测试集，在测试集上评估
        if X_test is not None and y_test is not None:
            if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
                y_test_pred = self.meta_model.predict(meta_test)
                y_test_proba = self.meta_model.predict_proba(meta_test)
                test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
            else:
                y_test_pred = self.meta_model.predict(meta_test)
                test_metrics = self.calculate_metrics(y_test, y_test_pred)
                
            results['test_metrics'] = test_metrics
            
            logger.info("测试集性能:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.6f}")
            
            # 注释: 已移除绘图功能以加速训练
        
        # 特征重要性分析
        if hasattr(self.meta_model[-1], 'coef_'):
            importances = np.abs(self.meta_model[-1].coef_)
            sorted_idx = np.argsort(importances)[::-1]
            model_importances = {
                self.meta_features_cols[i]: importances[i] for i in sorted_idx
            }
            
            results['model_importances'] = model_importances
            logger.info("模型重要性:")
            for model, importance in model_importances.items():
                logger.info(f"{model}: {importance:.6f}")
            
            # 注释: 已移除绘图功能以加速训练
        
        # 添加训练时间
        training_time = time.time() - start_time
        results['training_time'] = training_time
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        # 添加空的cv_scores，使用None而不是NaN
        results['cv_scores'] = {'mean': None, 'std': None}
        
        # 保存结果
        joblib.dump(results, os.path.join(self.output_dir, 'stacking_results.pkl'))
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        根据任务类型计算性能指标
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            y_pred_proba: 预测概率（分类任务可选）
            
        返回:
            性能指标字典
        """
        if self.task_type == 'classification':
            # 分类任务指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            try:
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }
                
                # 如果提供了预测概率，计算AUC
                if y_pred_proba is not None:
                    from sklearn.metrics import roc_auc_score
                    try:
                        n_classes = len(np.unique(y_true))
                        if n_classes == 2:  # 二分类
                            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:  # 多分类
                            metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    except Exception as e:
                        logger.warning(f"计算AUC失败: {e}")
                        metrics['auc'] = 0.0
                        
                return metrics
            except Exception as e:
                logger.error(f"计算分类指标失败: {e}")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        else:
            # 回归任务指标（原有逻辑）
            return {
                'pearson': pearsonr(y_true, y_pred)[0],
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }
    
