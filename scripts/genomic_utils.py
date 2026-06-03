#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基因组预测工具函数模块
====================

包含基因组预测流程中使用的工具函数、辅助类和独立功能函数。
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算所有性能指标"""
    try:
        pearson_corr = pearsonr(y_true, y_pred)[0]
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
    except:
        pearson_corr = 0.0
        
    try:
        spearman_corr = spearmanr(y_true, y_pred)[0]
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0
        
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def prepare_cv_data(phe_data, save_path, cv_times, cvfold):
    """
    生成并保存交叉验证分组信息
    
    参数:
        phe_data: 表型数据DataFrame
        save_path: CV文件保存路径
        cv_times: 重复次数
        cvfold: 交叉验证折数
    
    返回:
        添加了CV分组列的表型数据
    """
    sample_block = int(phe_data.shape[0] / cvfold)
    phe_index = phe_data.index.to_numpy(copy=True)
    for cvi in range(cv_times):
        random.shuffle(phe_index)
        for i in range(cvfold):
            if i == cvfold - 1:
                phe_data.loc[phe_index[sample_block * i:], f'cv{cvi}'] = i
            else:
                phe_data.loc[phe_index[sample_block * i: sample_block * (i + 1)], f'cv{cvi}'] = i
    phe_data.sort_index(inplace=True)
    phe_data.to_csv(save_path, header=True, index=True)
    return phe_data


def call_topsis_evaluator(
    comparison_csv, 
    output_csv, 
    criteria=None, 
    criteria_types=None,
    manual_weights: str = None,
    min_transform: str = 'reciprocal',
    simple_output: str = None,
    logger=None
):
    """Call TOPSIS evaluator for model comprehensive evaluation."""
    try:
        # Import TOPSIS evaluator from scripts directory
        try:
            from topsis import TOPSISEvaluator
        except ImportError:
            # Fallback for different naming if necessary
            from .topsis import TOPSISEvaluator
        
        # Create TOPSIS evaluator
        evaluator = TOPSISEvaluator(logger=logger)
        
        # Execute TOPSIS evaluation
        evaluator.evaluate(
            input_file=comparison_csv,
            output_file=output_csv,
            criteria=criteria,
            criteria_types=criteria_types,
            simple_output=simple_output,
            manual_weights=manual_weights,
            min_transform=min_transform,
            use_entropy_weights=False
        )
        
        msg = f"TOPSIS evaluation completed: {output_csv}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
    except Exception as e:
        import traceback
        err_msg = f"TOPSIS evaluation failed: {e}\n{traceback.format_exc()}"
        if logger:
            logger.error(err_msg)
        else:
            print(err_msg)
        raise


def create_comparison_table(all_model_results: Dict[str, Dict[str, Any]], results_dir: Path, logger=None):
    """创建模型性能比较表格"""
    # 准备比较数据
    comparison_data = []
    
    # 检查任务类型
    first_result = next(iter(all_model_results.values()))
    is_classification = 'avg_test_accuracy' in first_result
    
    for model_name, results in all_model_results.items():
        # 清理模型名称，去除_clf和_reg后缀
        display_name = model_name.replace('_reg', '').replace('_clf', '')
        # 为集成模型使用特殊名称
        if display_name == 'stacking_ensemble':
            display_name = 'Ensemble'
        
        # 根据任务类型收集性能指标
        try:
            if is_classification:
                # 分类任务指标
                row = {
                    'Model': display_name,
                    'Test Accuracy': results.get('avg_test_accuracy', 0.0),
                    'Test Accuracy (std)': results.get('std_test_accuracy', 0.0),
                    'Test F1': results.get('avg_test_f1', 0.0),
                    'Test F1 (std)': results.get('std_test_f1', 0.0),
                    'Test AUC': results.get('avg_test_auc', 0.0),
                    'Test AUC (std)': results.get('std_test_auc', 0.0),
                    'Validation Accuracy': results.get('avg_val_accuracy', 0.0),
                    'Validation Accuracy (std)': results.get('std_val_accuracy', 0.0),
                    'Ensemble Accuracy': results.get('avg_ensemble_accuracy', 0.0),
                    'Ensemble Accuracy (std)': results.get('std_ensemble_accuracy', 0.0),
                    'Training Time (s)': results.get('avg_training_time', 0.0)
                }
                sort_column = 'Ensemble Accuracy'
            else:
                # 回归任务指标
                row = {
                    'Model': display_name,
                    'Test Pearson': results.get('avg_test_pearson', 0.0),
                    'Test Pearson (std)': results.get('std_test_pearson', 0.0),
                    'Test Spearman': results.get('avg_test_spearman', 0.0),
                    'Test Spearman (std)': results.get('std_test_spearman', 0.0),
                    'Validation Pearson': results.get('avg_val_pearson', 0.0),
                    'Validation Pearson (std)': results.get('std_val_pearson', 0.0),
                    'Test MSE': results.get('avg_test_mse', 0.0),
                    'Test MSE (std)': results.get('std_test_mse', 0.0),
                    'Ensemble Pearson': results.get('avg_ensemble_pearson', 0.0),
                    'Ensemble Pearson (std)': results.get('std_ensemble_pearson', 0.0),
                    'Training Time (s)': results.get('avg_training_time', 0.0)
                }
                sort_column = 'Ensemble Pearson'
                
            comparison_data.append(row)
        except Exception as e:
            error_msg = f"处理模型 {model_name} 的结果时出错: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
    
    # 创建数据框
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按主要性能指标排序
        comparison_df = comparison_df.sort_values(sort_column, ascending=False)
        
        # 保存到CSV
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        # 记录结果
        if logger:
            logger.info(f"\n模型性能比较:")
            logger.info("\n" + comparison_df.to_string(index=False))
            logger.info(f"\n比较表格已保存到 {comparison_path}")
        else:
            print(f"\n模型性能比较:")
            print("\n" + comparison_df.to_string(index=False))
            print(f"\n比较表格已保存到 {comparison_path}")
    else:
        warning_msg = "没有足够的结果来创建比较表格"
        if logger:
            logger.warning(warning_msg)
        else:
            print(f"警告: {warning_msg}")


def generate_cv_folds_from_file(
    X: pd.DataFrame, 
    y: pd.Series, 
    cv_pheno_data: pd.DataFrame, 
    repeat_idx: int, 
    n_splits: int,
    task_logger=None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    从CV文件生成交叉验证折
    
    参数:
        X: 特征矩阵
        y: 目标变量
        cv_pheno_data: 包含CV分组的表型数据
        repeat_idx: 重复索引
        n_splits: 交叉验证折数
        task_logger: 任务专用日志记录器
        
    返回:
        包含训练和验证索引的列表
    """
    # 使用提供的记录器或默认记录器
    log = task_logger if task_logger else None
    
    # 获取当前重复的CV列名
    cv_col = f'cv{repeat_idx}'
    
    # 确保CV列存在
    if cv_col not in cv_pheno_data.columns:
        error_msg = f"CV文件中不存在列 {cv_col}，无法生成交叉验证折"
        if log:
            log.error(error_msg)
        else:
            print(f"错误: {error_msg}")
        raise ValueError(f"CV文件中不存在列 {cv_col}")
    
    # 生成所有折
    folds = []
    for fold_idx in range(n_splits):
        # 获取验证集索引
        val_mask = cv_pheno_data[cv_col] == fold_idx
        val_indices = cv_pheno_data.index[val_mask].tolist()
        
        # 获取训练集索引
        train_mask = cv_pheno_data[cv_col] != fold_idx
        train_indices = cv_pheno_data.index[train_mask].tolist()
        
        # 将索引转换为X中的位置索引
        X_index_map = {idx: i for i, idx in enumerate(X.index)}
        train_idx = np.array([X_index_map[idx] for idx in train_indices if idx in X_index_map])
        val_idx = np.array([X_index_map[idx] for idx in val_indices if idx in X_index_map])
        
        folds.append((train_idx, val_idx))
        
        # 记录折的大小
        if log:
            log.debug(f"重复 {repeat_idx+1}, 折 {fold_idx+1}/{n_splits}: "
                     f"训练集大小 {len(train_idx)}, 验证集大小 {len(val_idx)}")
    
    return folds


def filter_model_params(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
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
    
    return filtered_params


def generate_cv_folds(
    X: pd.DataFrame, 
    y: pd.Series, 
    repeat_idx: int, 
    n_splits: int,
    random_seed: int,
    n_repeats: int = None,
    task_logger=None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成可重复的交叉验证折
    
    参数:
        X: 特征矩阵
        y: 目标变量
        repeat_idx: 重复索引
        n_splits: 交叉验证折数
        random_seed: 基础随机种子
        n_repeats: 总重复次数(用于日志)
        task_logger: 任务专用日志记录器
        
    返回:
        包含训练和验证索引的列表
    """
    # 使用提供的记录器或默认记录器
    log = task_logger if task_logger else None
    
    # 使用重复索引作为随机种子，确保可复现性
    fold_seed = random_seed + repeat_idx
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    
    # 生成所有折
    folds = list(kf.split(X))
    
    # 记录折的大小
    if log:
        for i, (train_idx, val_idx) in enumerate(folds):
            if n_repeats:
                log.debug(f"重复 {repeat_idx+1}/{n_repeats}, 折 {i+1}/{n_splits}: "
                         f"训练集大小 {len(train_idx)}, 验证集大小 {len(val_idx)}")
            else:
                log.debug(f"重复 {repeat_idx+1}, 折 {i+1}/{n_splits}: "
                         f"训练集大小 {len(train_idx)}, 验证集大小 {len(val_idx)}")
    
    return folds


# 种子生成函数
def generate_optimization_seed(random_seed: int, repeat_idx: int) -> int:
    """生成Optuna优化使用的随机种子"""
    return random_seed + repeat_idx * 100  # 使用常量100作为种子乘数

def generate_repeat_seed(random_seed: int, repeat_idx: int) -> int:
    """生成重复训练使用的随机种子"""
    return random_seed + repeat_idx

def generate_fold_seed(random_seed: int, repeat_idx: int, fold_idx: int) -> int:
    """生成折训练使用的随机种子"""
    repeat_seed = generate_repeat_seed(random_seed, repeat_idx)
    return repeat_seed + fold_idx

# 目录创建函数
def create_model_result_directory(results_dir: Path, model_name: str) -> Path:
    """创建模型结果主目录"""
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir

def create_repeat_result_directory(results_dir: Path, model_name: str, repeat_idx: int) -> Path:
    """创建重复训练结果目录"""
    repeat_dir = results_dir / model_name / f"repeat_{repeat_idx+1}"
    repeat_dir.mkdir(exist_ok=True, parents=True)
    return repeat_dir

def create_plots_directory(repeat_dir: Path) -> Path:
    """创建散点图保存目录"""
    plots_dir = repeat_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    return plots_dir

def create_representative_model_directory(model_dir: Path) -> Path:
    """创建代表性模型保存目录"""
    representative_dir = model_dir / "representative_model"
    representative_dir.mkdir(exist_ok=True, parents=True)
    return representative_dir

# 数据处理函数
def prepare_train_test_data(X: pd.DataFrame, y: pd.Series, repeat_idx: int, 
                           random_seed: int, test_size: float, 
                           test_indices: np.ndarray = None) -> Tuple[np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    准备训练测试数据分割
    
    参数:
        X: 特征矩阵
        y: 目标变量
        repeat_idx: 重复索引
        random_seed: 基础随机种子
        test_size: 测试集比例
        test_indices: 测试集索引(如果提供)
        
    返回:
        train_indices, X_train, y_train, X_test, y_test
    """
    # 分割训练集和测试集(如果没有提供测试集索引)
    if test_indices is None:
        repeat_seed = generate_repeat_seed(random_seed, repeat_idx)
        _, test_indices = train_test_split(
            range(len(X)), 
            test_size=test_size, 
            random_state=repeat_seed
        )
    
    train_indices = np.array([i for i in range(len(X)) if i not in test_indices])
    
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    
    return train_indices, X_train, y_train, X_test, y_test

def prepare_fold_training_data(X_train: pd.DataFrame, y_train: pd.Series, 
                              train_idx: np.ndarray, val_idx: np.ndarray, 
                              X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    为单个折准备训练数据，包含标准化
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练目标变量
        train_idx: 折内训练索引
        val_idx: 折内验证索引
        X_test: 测试特征矩阵
        
    返回:
        X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, y_fold_train, y_fold_val, scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    # 获取折数据
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # 标准化特征
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, y_fold_train, y_fold_val, scaler

def train_fold_model(model: Any, X_fold_train_scaled: np.ndarray, 
                    y_fold_train: pd.Series) -> Tuple[Any, float]:
    """
    训练单个折的模型
    
    参数:
        model: 模型实例
        X_fold_train_scaled: 标准化后的训练特征
        y_fold_train: 训练目标变量
        
    返回:
        trained_model, training_time
    """
    import time
    start_time = time.time()
    model.fit(X_fold_train_scaled, y_fold_train)
    training_time = time.time() - start_time
    return model, training_time

def predict_and_calculate_metrics(model: Any, X_fold_train_scaled: np.ndarray,
                                 y_fold_train: pd.Series, X_fold_val_scaled: np.ndarray,
                                 y_fold_val: pd.Series, X_test_scaled: np.ndarray,
                                 y_test: pd.Series) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    执行预测并计算所有性能指标
    
    参数:
        model: 训练好的模型
        X_fold_train_scaled, y_fold_train: 训练数据
        X_fold_val_scaled, y_fold_val: 验证数据  
        X_test_scaled, y_test: 测试数据
        
    返回:
        train_metrics, val_metrics, test_metrics, y_fold_train_pred, y_fold_val_pred, y_test_pred
    """
    # 预测
    y_fold_train_pred = model.predict(X_fold_train_scaled)
    y_fold_val_pred = model.predict(X_fold_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 计算性能指标
    train_metrics = calculate_metrics(y_fold_train, y_fold_train_pred)
    val_metrics = calculate_metrics(y_fold_val, y_fold_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    return train_metrics, val_metrics, test_metrics, y_fold_train_pred, y_fold_val_pred, y_test_pred

def save_fold_predictions_and_plots(train_idx: np.ndarray, val_idx: np.ndarray, 
                                   test_indices: np.ndarray, y_fold_train: pd.Series,
                                   y_fold_val: pd.Series, y_test: pd.Series,
                                   y_fold_train_pred: np.ndarray, y_fold_val_pred: np.ndarray,
                                   y_test_pred: np.ndarray, all_predictions: Dict,
                                   model_name: str, repeat_idx: int, fold_idx: int,
                                   repeat_dir: Path, results_dir: Path) -> Dict:
    """
    保存折预测结果（已移除绘图功能以加速训练）
    
    返回:
        空字典（不再生成图片）
    """
    # 保存预测结果
    train_predictions = {
        'indices': train_idx.tolist(),
        'true_values': y_fold_train.tolist(),
        'predicted_values': y_fold_train_pred.tolist()
    }
    
    val_predictions = {
        'indices': val_idx.tolist(),
        'true_values': y_fold_val.tolist(),
        'predicted_values': y_fold_val_pred.tolist()
    }
    
    test_predictions = {
        'indices': test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices),
        'true_values': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
        'predicted_values': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred)
    }
    
    all_predictions['train'].append(train_predictions)
    all_predictions['val'].append(val_predictions)
    all_predictions['test'].append(test_predictions)
    
    # 注释: 已移除绘图功能以加速训练
    return {}

# 统计计算函数
def calculate_repeat_statistics(all_repeat_results: List[Dict], task_type: str = None) -> Dict[str, Any]:
    """
    计算所有重复结果的统计信息
    
    参数:
        all_repeat_results: 所有重复的结果列表
        
    返回:
        包含平均值、标准差和原始数据的统计字典
    """
    if not all_repeat_results:
        return None
        
    # 检查任务类型 - 优先使用传入的参数，其次使用数据推断
    if task_type:
        is_classification = (task_type == 'classification')
    else:
        # 使用多种方式确保准确性
        first_result = all_repeat_results[0]
        is_classification = (
            'avg_test_accuracy' in first_result or 
            'avg_train_accuracy' in first_result or
            ('ensemble_metrics' in first_result and 'accuracy' in first_result.get('ensemble_metrics', {}))
        )
    
    # 根据任务类型提取相应指标的值
    avg_results = {
        'training_time_values': [r['avg_training_time'] for r in all_repeat_results],
    }
    
    if is_classification:
        # 分类任务指标
        avg_results.update({
            'test_accuracy_values': [r.get('avg_test_accuracy', 0.0) for r in all_repeat_results],
            'val_accuracy_values': [r.get('avg_val_accuracy', 0.0) for r in all_repeat_results],
            'train_accuracy_values': [r.get('avg_train_accuracy', 0.0) for r in all_repeat_results],
            'test_f1_values': [r.get('avg_test_f1', 0.0) for r in all_repeat_results],
            'val_f1_values': [r.get('avg_val_f1', 0.0) for r in all_repeat_results],
            'train_f1_values': [r.get('avg_train_f1', 0.0) for r in all_repeat_results],
            'test_auc_values': [r.get('avg_test_auc', 0.0) for r in all_repeat_results],
            'ensemble_accuracy_values': [r.get('ensemble_metrics', {}).get('accuracy', 0.0) for r in all_repeat_results],
        })
    else:
        # 回归任务指标
        avg_results.update({
            'test_pearson_values': [r.get('avg_test_pearson', 0.0) for r in all_repeat_results],
            'val_pearson_values': [r.get('avg_val_pearson', 0.0) for r in all_repeat_results],
            'train_pearson_values': [r.get('avg_train_pearson', 0.0) for r in all_repeat_results],
            'ensemble_pearson_values': [r.get('ensemble_metrics', {}).get('pearson', 0.0) for r in all_repeat_results],
            'test_spearman_values': [r.get('avg_test_spearman', 0.0) for r in all_repeat_results],
            'test_mse_values': [r.get('avg_test_mse', 0.0) for r in all_repeat_results]
        })
    
    # 计算统计摘要
    summary = {
        'model_name': all_repeat_results[0]['model_name'],
        'n_repeats': len(all_repeat_results),
        'avg_training_time': np.mean(avg_results['training_time_values']),
        'raw_values': avg_results
    }
    
    if is_classification:
        # 分类任务统计
        summary.update({
            'avg_test_accuracy': np.mean(avg_results['test_accuracy_values']),
            'std_test_accuracy': np.std(avg_results['test_accuracy_values']),
            'avg_val_accuracy': np.mean(avg_results['val_accuracy_values']),
            'std_val_accuracy': np.std(avg_results['val_accuracy_values']),
            'avg_train_accuracy': np.mean(avg_results['train_accuracy_values']),
            'std_train_accuracy': np.std(avg_results['train_accuracy_values']),
            'avg_test_f1': np.mean(avg_results['test_f1_values']),
            'std_test_f1': np.std(avg_results['test_f1_values']),
            'avg_val_f1': np.mean(avg_results['val_f1_values']),
            'std_val_f1': np.std(avg_results['val_f1_values']),
            'avg_train_f1': np.mean(avg_results['train_f1_values']),
            'std_train_f1': np.std(avg_results['train_f1_values']),
            'avg_ensemble_accuracy': np.mean(avg_results['ensemble_accuracy_values']),
            'std_ensemble_accuracy': np.std(avg_results['ensemble_accuracy_values']),
            'avg_test_auc': np.mean(avg_results['test_auc_values']),
            'std_test_auc': np.std(avg_results['test_auc_values']),
        })
    else:
        # 回归任务统计
        summary.update({
            'avg_test_pearson': np.mean(avg_results['test_pearson_values']),
            'std_test_pearson': np.std(avg_results['test_pearson_values']),
            'avg_val_pearson': np.mean(avg_results['val_pearson_values']),
            'std_val_pearson': np.std(avg_results['val_pearson_values']),
            'avg_train_pearson': np.mean(avg_results['train_pearson_values']),
            'std_train_pearson': np.std(avg_results['train_pearson_values']),
            'avg_ensemble_pearson': np.mean(avg_results['ensemble_pearson_values']),
            'std_ensemble_pearson': np.std(avg_results['ensemble_pearson_values']),
            'avg_test_spearman': np.mean(avg_results['test_spearman_values']),
            'std_test_spearman': np.std(avg_results['test_spearman_values']),
            'avg_test_mse': np.mean(avg_results['test_mse_values']),
            'std_test_mse': np.std(avg_results['test_mse_values']),
        })
    
    return summary

def find_representative_repeat(all_repeat_results: List[Dict], avg_ensemble_metric: float) -> Tuple[int, Dict, float]:
    """
    找出性能最接近平均值的重复
    
    参数:
        all_repeat_results: 所有重复的结果列表
        avg_ensemble_metric: 平均集成性能（分类：准确率，回归：Pearson相关系数）
        
    返回:
        closest_repeat_idx, closest_repeat, difference
    """
    # 检查任务类型并提取相应的集成指标
    first_result = all_repeat_results[0]
    is_classification = 'avg_test_accuracy' in first_result
    
    if is_classification:
        ensemble_values = [r['ensemble_metrics']['accuracy'] for r in all_repeat_results]
    else:
        ensemble_values = [r['ensemble_metrics']['pearson'] for r in all_repeat_results]
    
    # 计算每个重复的性能与平均性能的差距
    metric_diffs = [abs(val - avg_ensemble_metric) for val in ensemble_values]
    
    # 找出差距最小的重复索引
    closest_repeat_idx = np.argmin(metric_diffs)
    closest_repeat = all_repeat_results[closest_repeat_idx]
    difference = metric_diffs[closest_repeat_idx]
    
    return closest_repeat_idx, closest_repeat, difference

def prepare_cv_folds(
    pheno_data: pd.DataFrame, 
    target_trait: str,
    cv_file: str = None,
    force_new_cv: bool = False,
    cv_id_column: str = 'ID',
    n_repeats: int = 100,
    n_splits: int = 5,
    results_dir: str = "optimization_results",
    logger=None
) -> pd.DataFrame:
    """
    Prepare cross-validation fold information and save to file.
    """
    results_dir = Path(results_dir)
    
    # Determine CV file path
    if cv_file is not None:
        cv_file_path = Path(cv_file)
    else:
        # Create CV file directory
        cv_dir = results_dir / "cv_folds"
        cv_dir.mkdir(exist_ok=True, parents=True)
        
        # Default CV file path
        cv_file_path = cv_dir / f"{target_trait}_cv_{n_repeats}x{n_splits}.csv"
    
    # Check if CV file exists and should not be regenerated
    if cv_file_path.exists() and not force_new_cv:
        msg = f"Loading existing CV grouping file: {cv_file_path}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        cv_pheno_data = pd.read_csv(cv_file_path, index_col=0)
    else:
        msg = f"Generating new CV groupings and saving to: {cv_file_path}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # Ensure pheno_data has an index column
        pheno_data_copy = pheno_data.copy()
        
        # If ID column is in data, set it as index
        if cv_id_column in pheno_data_copy.columns:
            pheno_data_copy.set_index(cv_id_column, inplace=True)
        
        # Generate CV groupings
        cv_pheno_data = prepare_cv_data(
            pheno_data_copy, 
            str(cv_file_path), 
            n_repeats, 
            n_splits
        )
    
    return cv_pheno_data