#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基因组分类预测模块
================

功能特性:
- 分类任务的性能指标计算 (准确率、精确度、召回率、F1分数、AUC等)
- 分类标签编码和预处理
- 分类模型预测结果处理
- 概率到标签的稳健转换
- 分类散点图生成 (如果需要)

从 genomic_prediction_v2_class.py 中提取的纯分类功能模块
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
import joblib
import logging
from pathlib import Path

# 获取主日志记录器
main_logger = logging.getLogger("main")

class GenomicClassifier:
    """基因组分类预测工具类"""
    
    def __init__(self, n_classes: int, results_dir: str = None, random_state: int = None, n_threads: int = 1):
        """
        初始化分类器
        
        参数:
            n_classes: 类别数量
            results_dir: 结果保存目录
            random_state: 随机状态
            n_threads: 线程数
        """
        self.n_classes = n_classes
        self.results_dir = Path(results_dir) if results_dir else Path(".")
        self.label_encoder = None
        self.random_state = random_state
        self.n_threads = n_threads
        
        # 初始化分类模型优化器
        from classification_models import ClassificationModelOptimizer
        self.classification_optimizer = ClassificationModelOptimizer(
            random_state=random_state, 
            n_threads=n_threads, 
            n_classes=n_classes
        )
        
    def prepare_classification_labels(self, y: pd.Series, results_dir: Path) -> pd.Series:
        """
        准备分类标签，进行必要的编码
        
        参数:
            y: 原始标签
            results_dir: 结果保存目录
            
        返回:
            编码后的标签
        """
        # 检查标签是否为数值型
        unique_labels = y.unique()
        main_logger.info(f"原始分类标签: {sorted(unique_labels)}")
        
        # 检查标签是否为从0开始的连续整数
        sorted_labels = sorted(unique_labels)
        expected_labels = list(range(len(sorted_labels)))
        
        # 强制进行标签编码以确保从0开始
        # 这解决了XGBoost等模型要求标签从0开始的问题
        main_logger.info("对标签进行编码以确保从0开始的连续整数...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_result = pd.Series(y_encoded, index=y.index)
        
        # 保存标签编码器
        encoder_path = results_dir / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        main_logger.info(f"标签编码器已保存到: {encoder_path}")
        main_logger.info(f"编码后的标签: {sorted(y_result.unique())}")
        main_logger.info(f"标签映射: {dict(zip(sorted_labels, sorted(y_result.unique())))}")
        
        # 验证类别数
        actual_n_classes = len(unique_labels)
        if self.n_classes != actual_n_classes:
            main_logger.warning(f"指定的类别数({self.n_classes})与实际类别数({actual_n_classes})不符，使用实际类别数")
            self.n_classes = actual_n_classes
            
        return y_result
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        计算分类性能指标
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率（可选）
            
        返回:
            性能指标字典
        """
        # 确保输入是正确的格式
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # 如果y_pred是概率矩阵(n_samples, n_classes)，自动转为标签，并补充y_pred_proba
        if y_pred.ndim == 2 and y_pred.shape[0] == y_true.shape[0]:
            if y_pred_proba is None:
                y_pred_proba = y_pred
            y_pred = np.argmax(y_pred, axis=1)

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        # 如果预测结果是浮点数，转换为整数
        if y_pred.dtype.kind == 'f':
            y_pred = y_pred.astype(int)
            
        # 检查数据形状是否匹配
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")
        
        # 分类指标
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            # 如果计算指标失败，返回默认值
            print(f"计算分类指标失败: {e}")
            print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
            print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            print(f"y_true unique: {np.unique(y_true)}")
            print(f"y_pred unique: {np.unique(y_pred)}")
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # 如果提供了预测概率，计算AUC和log loss
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:  # 二分类
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # 多分类
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception as e:
                main_logger.warning(f"计算AUC或log_loss失败: {e}")
                metrics['auc'] = 0.0
                metrics['log_loss'] = float('inf')
        
        return metrics
    
    def proba_to_labels_robust(
        self, 
        proba: np.ndarray, 
        n_samples: int, 
        n_classes_hint: int = None
    ) -> np.ndarray:
        """
        将概率稳健地转换为标签
        
        参数:
            proba: 预测概率
            n_samples: 期望的样本数
            n_classes_hint: 类别数提示
            
        返回:
            预测标签数组
        """
        # 统一为 ndarray
        if isinstance(proba, (list, tuple)):
            try:
                proba = np.column_stack(proba)
            except Exception:
                proba = np.asarray(proba)
        arr = np.asarray(proba)
        
        # 直接可用形状
        if arr.ndim == 2:
            if arr.shape[0] == n_samples:
                P = arr
            elif arr.shape[1] == n_samples:
                P = arr.T
            else:
                # 试图按样本数重塑
                total = arr.size
                if total % n_samples == 0:
                    k = total // n_samples
                    P = arr.reshape(n_samples, k)
                else:
                    # 强制按行优先截断/填充
                    k = n_classes_hint if n_classes_hint else max(2, arr.shape[-1])
                    flat = arr.ravel()
                    need = n_samples * k
                    if flat.size >= need:
                        P = flat[:need].reshape(n_samples, k)
                    else:
                        pad = np.tile(flat[-1], need - flat.size)
                        P = np.concatenate([flat, pad]).reshape(n_samples, k)
        elif arr.ndim == 1:
            # 二分类一维概率
            if arr.shape[0] == n_samples:
                P = np.vstack([1 - arr, arr]).T
            else:
                # 长度异常，拉平重塑
                k = n_classes_hint if n_classes_hint else 2
                need = n_samples * k
                flat = arr.ravel()
                if flat.size >= need:
                    P = flat[:need].reshape(n_samples, k)
                else:
                    pad = np.tile(flat[-1], need - flat.size)
                    P = np.concatenate([flat, pad]).reshape(n_samples, k)
        else:
            # 其他非常规维度，兜底为二分类均匀概率
            P = np.full((n_samples, max(2, n_classes_hint or 2)), 1.0)
        
        idx = np.argmax(P, axis=1)
        return idx.astype(int)
    
    def ensure_prediction_length(
        self, 
        pred: np.ndarray, 
        proba: np.ndarray, 
        n_samples: int
    ) -> np.ndarray:
        """
        确保预测结果长度正确
        
        参数:
            pred: 原始预测结果
            proba: 预测概率
            n_samples: 期望样本数
            
        返回:
            长度正确的预测结果
        """
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 1 and pred_arr.shape[0] == n_samples:
            return pred_arr
        if proba is not None:
            # 直接从概率稳健生成
            return self.proba_to_labels_robust(proba, n_samples, self.n_classes)
        # 兜底：截断或填充到正确长度
        flat = pred_arr.ravel()
        if flat.size >= n_samples:
            return flat[:n_samples]
        else:
            pad = np.tile(flat[-1] if flat.size > 0 else 0, n_samples - flat.size)
            return np.concatenate([flat, pad])
    
    def generate_classification_labels_from_model(
        self, 
        model: Any, 
        X_scaled: np.ndarray
    ) -> tuple:
        """
        从模型生成分类标签和概率
        
        参数:
            model: 训练好的模型
            X_scaled: 标准化后的特征
            
        返回:
            (预测标签, 预测概率) 元组
        """
        y_pred_proba = None
        
        # 优先使用predict_proba；失败则回退到predict
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_scaled)
                y_pred = self.proba_to_labels_robust(y_pred_proba, len(X_scaled), self.n_classes)
            except Exception:
                pred = model.predict(X_scaled)
                y_pred = np.asarray(pred).ravel()
                if y_pred.dtype.kind == 'f':
                    y_pred = y_pred.astype(int)
        else:
            pred = model.predict(X_scaled)
            y_pred = np.asarray(pred).ravel()
            if y_pred.dtype.kind == 'f':
                y_pred = y_pred.astype(int)
        
        # 确保预测长度正确
        y_pred = self.ensure_prediction_length(y_pred, y_pred_proba, len(X_scaled))
        
        return y_pred, y_pred_proba
    
    def decode_labels_if_needed(self, labels: np.ndarray) -> np.ndarray:
        """
        如果使用了标签编码器，将标签解码回原始形式
        
        参数:
            labels: 编码后的标签
            
        返回:
            解码后的标签
        """
        if self.label_encoder is not None:
            try:
                return self.label_encoder.inverse_transform(labels)
            except Exception as e:
                main_logger.warning(f"标签解码失败: {e}")
                return labels
        return labels
    
    def create_classification_summary(
        self, 
        all_repeat_results: list,
        model_name: str
    ) -> Dict[str, Any]:
        """
        创建分类任务的结果摘要
        
        参数:
            all_repeat_results: 所有重复的结果列表
            model_name: 模型名称
            
        返回:
            分类摘要字典
        """
        if not all_repeat_results:
            return {
                'model_name': model_name,
                'n_repeats': 0,
                'avg_test_accuracy': 0.0,
                'std_test_accuracy': 0.0,
                'avg_test_f1': 0.0,
                'std_test_f1': 0.0,
                'avg_training_time': 0.0
            }
        
        # 分类任务摘要
        avg_results = {
            'test_accuracy_values': [r['avg_test_accuracy'] for r in all_repeat_results if 'avg_test_accuracy' in r],
            'test_f1_values': [r['avg_test_f1'] for r in all_repeat_results if 'avg_test_f1' in r],
            'test_auc_values': [r['avg_test_auc'] for r in all_repeat_results if 'avg_test_auc' in r],
            'training_time_values': [r['avg_training_time'] for r in all_repeat_results if 'avg_training_time' in r]
        }
        
        summary = {
            'model_name': model_name,
            'n_repeats': len(all_repeat_results),
            'avg_test_accuracy': np.mean(avg_results['test_accuracy_values']) if avg_results['test_accuracy_values'] else 0.0,
            'std_test_accuracy': np.std(avg_results['test_accuracy_values']) if avg_results['test_accuracy_values'] else 0.0,
            'avg_test_f1': np.mean(avg_results['test_f1_values']) if avg_results['test_f1_values'] else 0.0, 
            'std_test_f1': np.std(avg_results['test_f1_values']) if avg_results['test_f1_values'] else 0.0,
            'avg_test_auc': np.mean(avg_results['test_auc_values']) if avg_results['test_auc_values'] else 0.0,
            'std_test_auc': np.std(avg_results['test_auc_values']) if avg_results['test_auc_values'] else 0.0,
            'avg_training_time': np.mean(avg_results['training_time_values']),
            'raw_values': avg_results
        }
        
        return summary
    
    def log_classification_results(
        self, 
        fold_idx: int, 
        train_metrics: dict, 
        val_metrics: dict, 
        test_metrics: dict, 
        logger=None
    ):
        """
        记录分类结果到日志
        
        参数:
            fold_idx: 折索引
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            test_metrics: 测试集指标
            logger: 日志记录器
        """
        if logger is None:
            logger = main_logger
            
        logger.info(f"折 {fold_idx+1} 结果:")
        logger.info(f"  训练集 Accuracy: {train_metrics['accuracy']:.6f}")
        logger.info(f"  训练集 F1: {train_metrics['f1']:.6f}")
        logger.info(f"  验证集 Accuracy: {val_metrics['accuracy']:.6f}")
        logger.info(f"  验证集 F1: {val_metrics['f1']:.6f}")
        logger.info(f"  测试集 Accuracy: {test_metrics['accuracy']:.6f}")
        logger.info(f"  测试集 F1: {test_metrics['f1']:.6f}")
        if 'auc' in test_metrics:
            logger.info(f"  测试集 AUC: {test_metrics['auc']:.6f}")
    
    def create_classification_comparison_row(
        self, 
        model_name: str, 
        results: dict
    ) -> dict:
        """
        创建分类模型比较表格的行数据
        
        参数:
            model_name: 模型名称
            results: 结果字典
            
        返回:
            比较表格行字典
        """
        # 清理模型名称
        display_name = model_name.replace('_clf', '')
        if display_name == 'stacking_ensemble':
            display_name = 'Ensemble'
        
        row = {
            'Model': display_name,
            'Test Accuracy': results.get('avg_test_accuracy', 0.0),
            'Test Accuracy (std)': results.get('std_test_accuracy', 0.0),
            'Test F1': results.get('avg_test_f1', 0.0),
            'Test F1 (std)': results.get('std_test_f1', 0.0),
            'Test Precision': results.get('avg_test_precision', 0.0),
            'Test Recall': results.get('avg_test_recall', 0.0),
            'Validation Accuracy': results.get('avg_val_accuracy', 0.0),
            'Training Time (s)': results.get('avg_training_time', 0.0)
        }
        
        # 如果有AUC指标也包含
        if 'avg_test_auc' in results:
            row['Test AUC'] = results.get('avg_test_auc', 0.0)
            
        return row
    
    def create_classification_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        创建分类模型实例
        
        参数:
            model_name: 模型名称
            params: 模型参数
            
        返回:
            模型实例
        """
        return self.classification_optimizer.create_classification_model(model_name, params)

    def get_classification_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        获取分类模型的默认参数
        
        参数:
            model_name: 模型名称
            
        返回:
            默认参数字典
        """
        return self.classification_optimizer.get_classification_default_params(model_name)

    def get_classification_param_func(self, model_name: str) -> callable:
        """
        获取分类模型的参数函数
        
        参数:
            model_name: 模型名称
            
        返回:
            参数函数
        """
        return self.classification_optimizer.get_param_func(model_name)

    def filter_classification_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤分类模型参数
        
        参数:
            model_name: 模型名称
            params: 原始参数
            
        返回:
            过滤后的参数
        """
        return self.classification_optimizer.filter_classification_params(model_name, params)