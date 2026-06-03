#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基因组预测标准化流程模块 v2
=======================

功能特性:
- 支持回归和分类模型的评估和预测
- 多次循环的K折交叉验证，支持可复现的折划分
- 回归任务使用Pearson相关系数作为主要评估指标
- 分类任务使用准确率、F1分数、AUC等评估指标
- 支持默认参数和Optuna自动调参
- 并行执行多个训练循环
- 每个模型限制使用单线程
- 详细保存所有训练循环和折的结果
- 支持选择最佳模型进行集成学习(Stacking)
- 分级日志系统，支持并行任务独立日志
- 支持分类标签自动编码和解码
- TOPSIS多准则决策模型排名
"""

import os
import sys
import logging
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union

# 导入依赖库
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.pruners import MedianPruner
import joblib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from model_optimizers import ModelOptimizer
from ensemble_stacking import StackingEnsemble
from genomic_classification import GenomicClassifier
from classification_models import ClassificationModelOptimizer
from genomic_utils import (
    calculate_metrics,
    prepare_cv_data, call_topsis_evaluator, 
    create_comparison_table, filter_model_params,
    generate_optimization_seed, generate_repeat_seed, generate_fold_seed,
    create_model_result_directory, create_repeat_result_directory,
    create_representative_model_directory,
    prepare_train_test_data, prepare_fold_training_data, train_fold_model,
    predict_and_calculate_metrics, save_fold_predictions_and_plots,
    calculate_repeat_statistics, find_representative_repeat
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 配置全局日志级别，只添加一个控制台处理器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class ModelConstants:
    """模型训练相关常量配置"""
    # Optuna优化相关
    OPTUNA_N_STARTUP_TRIALS = 10
    OPTUNA_N_WARMUP_STEPS = 5
    OPTUNA_INTERVAL_STEPS = 1
    OPTUNA_N_JOBS = 1  # 强制Optuna串行执行
    
    # 线程环境变量
    THREAD_ENV_VARS = [
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'
    ]
    
    # 默认目录名称
    DEFAULT_RESULTS_DIR = "optimization_results"
    DEFAULT_LOGS_DIR = "logs"
    DEFAULT_CV_DIR = "cv_folds"
    REPRESENTATIVE_MODEL_DIR = "representative_model"
    
    # 默认文件名称
    MAIN_LOG_FILE = "main.log"
    MODEL_LOG_FILE = "model.log"
    REPEAT_RESULTS_FILE = "repeat_results.json"
    SUMMARY_RESULTS_FILE = "summary_results.json"
    ALL_PREDICTIONS_FILE = "all_predictions.json"
    MODEL_INFO_FILE = "info.json"
    MODEL_PKL_FILE = "model.pkl"
    COMPARISON_CSV_FILE = "model_comparison.csv"
    
    # 日志格式模板
    LOG_SEPARATOR_SHORT = "=" * 50
    LOG_SEPARATOR_LONG = "=" * 70
    
    # 数值精度常量
    METRICS_PRECISION = 6
    TIME_PRECISION = 2
    FLOAT_PRECISION = 4
    CORRELATION_MIN_THRESHOLD = 1e-10  # 避免常量预测的阈值


class LogMessages:
    """日志消息模板常量"""
    # 初始化消息
    LOGGING_INITIALIZED = "日志系统初始化完成"
    PREDICTOR_INIT = "初始化GenomicPredictorV2，基础配置:"
    
    # 数据加载消息
    DATA_LOADING = "正在加载数据..."
    DATA_LOADED = "成功加载基因型数据: {}, 表型数据: {}"
    COMMON_SAMPLES = "基因型样本数: {}, 表型样本数: {}"
    FINAL_DATA_INFO = "最终数据维度 - 特征数量: {}, 样本数量: {}"
    TARGET_STATS = "目标变量统计 - 均值: {:.4f}, 标准差: {:.4f}"
    
    # 训练消息
    MODEL_TRAINING_START = "开始训练 模型:{} 重复:{}"
    REPEAT_TRAINING_START = "开始为模型 {} 执行 {} 次重复训练"
    FOLD_TRAINING_START = "训练折 {}/{}"
    PARAMETER_OPTIMIZATION_START = "开始为模型 {} 优化参数 (重复 {}/{})"
    OPTIMIZATION_START = "开始参数优化，最大试验次数: {}"
    
    # 结果消息
    FOLD_RESULTS = "折 {} 结果:"
    REPEAT_AVERAGE_PERFORMANCE = "重复 {} 平均性能:"
    MODEL_SUMMARY_PERFORMANCE = "模型 {} - {} 次重复的平均性能:"
    ENSEMBLE_TEST_PERFORMANCE = "折集成测试集性能:"
    
    # 错误消息模板
    DATA_FILE_NOT_FOUND = "数据文件加载失败: {}"
    DATA_FILE_READ_ERROR = "数据文件读取错误: {}"
    TARGET_TRAIT_NOT_FOUND = "目标性状 '{}' 不存在于表型数据中。可用列: {}"
    ID_COLUMN_MISSING_GENO = "基因型数据中缺少ID列 '{}'"
    ID_COLUMN_MISSING_PHENO = "表型数据中缺少ID列 '{}'"
    NO_COMMON_SAMPLES = "基因型和表型数据没有共同的样本ID"
    DIMENSION_MISMATCH = "特征矩阵和目标变量样本数不匹配: {} vs {}"


# 创建主日志记录器
main_logger = create_logger(name="main")

class GenomicPredictorV2:
    """
    基因组预测器类 V2 - 支持多次循环的K折交叉验证和任务级日志
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        results_dir: str = "optimization_results",
        n_trials: int = 100,
        n_threads: int = 1,
        max_parallel_jobs: int = 1,
        test_size: float = 0.2,
        n_splits: int = 5,
        n_repeats: int = 100,
        patience: int = 20,
        use_default_params: bool = False,
        save_models: bool = True,
        save_representative: bool = False,
        cv_file: str = None,
        force_new_cv: bool = False,
        cv_id_column: str = 'ID',
        task_type: str = 'regression',  # 新增：'regression' 或 'classification'
        n_classes: int = None,  # 新增：分类任务的类别数
        standardize_phenotype: bool = False  # 新增：是否标准化表型数据
    ):
        """
        初始化预测器
        
        参数:
            random_seed: 基础随机种子
            results_dir: 结果保存目录
            n_trials: Optuna优化试验次数
            n_threads: 每个模型使用的线程数（通常设为1）
            max_parallel_jobs: 最大并行任务数（多进程并行）
            test_size: 测试集比例
            n_splits: 交叉验证折数
            n_repeats: 重复执行次数
            patience: 提前停止的耐心值
            use_default_params: 是否使用模型默认参数
            save_models: 是否保存训练好的模型
            save_representative: 是否保存性能最接近平均值的代表性模型
            cv_file: 指定CV文件路径，如果不存在则创建
            force_new_cv: 强制生成新的CV文件，即使已存在
            cv_id_column: 表型数据中的ID列名，用于CV文件生成
            task_type: 任务类型 ('regression' 或 'classification')
            n_classes: 分类任务的类别数（分类任务必须指定）
            standardize_phenotype: 是否标准化表型数据（仅回归任务有效）
        """
        self.random_seed = random_seed
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.n_threads = n_threads
        self.max_parallel_jobs = max_parallel_jobs
        self.test_size = test_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.patience = patience
        self.use_default_params = use_default_params
        self.save_models = save_models
        self.save_representative = save_representative
        self.cv_file = cv_file
        self.force_new_cv = force_new_cv
        self.cv_id_column = cv_id_column
        self.task_type = task_type
        self.n_classes = n_classes
        self.standardize_phenotype = standardize_phenotype
        self.phenotype_scaler = None  # 保存表型标准化参数
        
        # 验证任务类型
        if task_type not in ['regression', 'classification']:
            raise ValueError("task_type must be 'regression' or 'classification'")
        
        if task_type == 'classification' and n_classes is None:
            raise ValueError("n_classes must be specified for classification tasks")
        
        # 创建结果目录
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建日志目录
        self.logs_dir = self.results_dir / ModelConstants.DEFAULT_LOGS_DIR
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置主日志
        self.setup_logging()
        
        # 初始化模型优化器
        if task_type == 'regression':
            # 回归任务，使用原有的ModelOptimizer
            self.model_optimizer = ModelOptimizer(random_state=random_seed, n_threads=n_threads)
            self.available_models = list(self.model_optimizer.model_configs.keys())
            self.classification_optimizer = None
            self.genomic_classifier = None
        else:
            # 分类任务，使用分类模型优化器
            self.classification_optimizer = ClassificationModelOptimizer(
                random_state=random_seed, n_threads=n_threads, n_classes=n_classes
            )
            self.genomic_classifier = GenomicClassifier(
                n_classes=n_classes, 
                results_dir=str(self.results_dir),
                random_state=random_seed,
                n_threads=n_threads
            )
            self.available_models = self.classification_optimizer.get_available_models()
            self.model_optimizer = None
        
        main_logger.info(f"任务类型: {task_type}")
        main_logger.info(f"可用模型: {', '.join(self.available_models)}")
        
        # 记录环境设置
        self.log_environment_settings()
    
    def get_topsis_configuration(self) -> tuple:
        """
        根据任务类型获取TOPSIS评价配置
        
        返回:
            (criteria, criteria_types, manual_weights) 元组
        """
        if self.task_type == 'classification':
            # 分类任务：使用准确率和准确率标准差，权重比8:2
            criteria = ['Test Accuracy', 'Test Accuracy (std)']
            criteria_types = ['max', 'min']  # 准确率越大越好，标准差越小越好
            manual_weights = '0.8,0.2'  # 8:2权重比，重视准确率
            main_logger.info("TOPSIS配置：分类任务 - Test Accuracy:Test Accuracy (std) = 8:2")
        else:
            # 回归任务：使用Pearson相关系数和标准差，权重比8:2  
            criteria = ['Test Pearson', 'Test Pearson (std)']
            criteria_types = ['max', 'min']  # 相关系数越大越好，标准差越小越好
            manual_weights = '0.8,0.2'  # 8:2权重比，重视准确性
            main_logger.info("TOPSIS配置：回归任务 - Test Pearson:Test Pearson (std) = 8:2")
            
        return criteria, criteria_types, manual_weights
    
    def _save_representative_model(self, model_name: str, params: Dict, X: pd.DataFrame, 
                                 y: pd.Series, model_dir: Path, repeat_info: Dict) -> str:
        """
        训练并保存代表性模型
        
        参数:
            model_name: 模型名称
            params: 模型参数
            X, y: 训练数据
            model_dir: 模型目录
            repeat_info: 重复信息字典
            
        返回:
            模型保存路径
        """
        # 创建代表性模型目录
        representative_model_dir = create_representative_model_directory(model_dir)
        
        try:
            # 创建模型
            model = self.create_model(model_name, params)
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 训练模型
            model.fit(X_scaled, y)
            
            # 保存模型
            model_path = representative_model_dir / ModelConstants.MODEL_PKL_FILE
            joblib.dump((model, scaler), model_path)
            
            # 保存重复索引信息
            info_path = representative_model_dir / ModelConstants.MODEL_INFO_FILE
            with open(info_path, 'w') as f:
                json.dump(repeat_info, f, indent=2)
            
            return str(model_path)
            
        except Exception as e:
            main_logger.error(f"保存代表性模型失败: {str(e)}")
            main_logger.error(traceback.format_exc())
            return None
    
    def setup_logging(self) -> None:
        """配置主日志记录"""
        # 主日志文件
        main_log_file = self.logs_dir / ModelConstants.MAIN_LOG_FILE
        
        # 创建主日志记录器
        global main_logger
        main_logger = create_logger(log_file=str(main_log_file), name="main", level=logging.INFO)
        
        main_logger.info(LogMessages.LOGGING_INITIALIZED)
    
    def get_task_logger(self, model_name: str, repeat_idx: Optional[int] = None):
        """
        为特定任务创建专用日志记录器
        
        参数:
            model_name: 模型名称
            repeat_idx: 重复索引(如果有)
            
        返回:
            任务专用日志记录器
        """
        # 创建模型日志目录
        model_log_dir = self.logs_dir / model_name
        model_log_dir.mkdir(exist_ok=True, parents=True)
        
        if repeat_idx is not None:
            # 为特定重复创建日志
            log_file = model_log_dir / f"repeat_{repeat_idx+1}.log"
            logger_name = f"{model_name}_repeat_{repeat_idx+1}"
        else:
            # 为模型创建通用日志
            log_file = model_log_dir / ModelConstants.MODEL_LOG_FILE
            logger_name = f"{model_name}"
        
        # 返回配置好的日志记录器
        return create_logger(log_file=str(log_file), name=logger_name, level=logging.DEBUG)
    
    def log_environment_settings(self) -> None:
        """记录环境设置"""
        main_logger.info(LogMessages.PREDICTOR_INIT)
        main_logger.info(f"- 结果目录: {self.results_dir}")
        main_logger.info(f"- 日志目录: {self.logs_dir}")
        main_logger.info(f"- 随机种子: {self.random_seed}")
        main_logger.info(f"- 每个模型线程数: {self.n_threads}")
        main_logger.info(f"- 并行任务数: {self.max_parallel_jobs}")
        main_logger.info(f"- 交叉验证折数: {self.n_splits}")
        main_logger.info(f"- 重复执行次数: {self.n_repeats}")
        main_logger.info(f"- 使用默认参数: {self.use_default_params}")
        
        # 记录线程环境变量
        for env_var in ModelConstants.THREAD_ENV_VARS:
            main_logger.info(f"{env_var}={os.environ.get(env_var, 'not set')}")
    
    def load_data(self, geno_file: str, pheno_file: str, target_trait: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        加载基因型和表型数据
        
        参数:
            geno_file: 基因型数据文件路径
            pheno_file: 表型数据文件路径  
            target_trait: 目标性状名称
            
        返回:
            特征矩阵、目标变量和完整的表型数据
            
        异常:
            FileNotFoundError: 当数据文件不存在时
            KeyError: 当目标性状列不存在时
            ValueError: 当没有共同样本时
        """
        main_logger.info(LogMessages.DATA_LOADING)
        
        # 步骤1: 加载原始数据文件
        try:
            geno_data = pd.read_csv(geno_file)
            pheno_data = pd.read_csv(pheno_file)
            main_logger.info(f"成功加载基因型数据: {geno_data.shape}, 表型数据: {pheno_data.shape}")
        except FileNotFoundError as e:
            main_logger.error(f"数据文件加载失败: {e}")
            raise
        except Exception as e:
            main_logger.error(f"数据文件读取错误: {e}")
            raise
        
        # 步骤2: 验证目标性状是否存在
        if target_trait not in pheno_data.columns:
            error_msg = f"目标性状 '{target_trait}' 不存在于表型数据中。可用列: {list(pheno_data.columns)}"
            main_logger.error(error_msg)
            raise KeyError(error_msg)
        
        # 步骤3: 确定ID列并验证
        id_col = 'ID' if 'ID' in geno_data.columns else self.cv_id_column
        
        if id_col not in geno_data.columns:
            error_msg = f"基因型数据中缺少ID列 '{id_col}'"
            main_logger.error(error_msg)
            raise KeyError(error_msg)
        
        if id_col not in pheno_data.columns:
            error_msg = f"表型数据中缺少ID列 '{id_col}'"
            main_logger.error(error_msg)
            raise KeyError(error_msg)
        
        # 步骤4: 找到共同样本并验证数量
        geno_ids = set(geno_data[id_col])
        pheno_ids = set(pheno_data[id_col])
        common_ids = geno_ids.intersection(pheno_ids)
        
        if len(common_ids) == 0:
            error_msg = "基因型和表型数据没有共同的样本ID"
            main_logger.error(error_msg)
            raise ValueError(error_msg)
        
        main_logger.info(f"基因型样本数: {len(geno_ids)}, 表型样本数: {len(pheno_ids)}")
        main_logger.info(f"共有样本数: {len(common_ids)}")
        
        # 步骤5: 筛选共同样本数据
        geno_data = geno_data[geno_data[id_col].isin(common_ids)]
        pheno_data = pheno_data[pheno_data[id_col].isin(common_ids)]
        
        # 处理表型数据中的重复样本ID（保留第一条记录）
        if pheno_data[id_col].duplicated().any():
            dup_count = pheno_data[id_col].duplicated().sum()
            main_logger.warning(f"表型数据中存在 {dup_count} 个重复样本ID，将保留每个ID的第一条记录")
            pheno_data = pheno_data.drop_duplicates(subset=id_col, keep='first')
        
        # 步骤6: 设置索引并排序以确保数据对齐
        geno_data.set_index(id_col, inplace=True)
        pheno_data.set_index(id_col, inplace=True)
        
        # 确保两个数据集的样本顺序一致
        geno_data = geno_data.sort_index()
        pheno_data = pheno_data.sort_index()
        
        # 步骤7: 提取特征矩阵和目标变量
        X = geno_data.copy()  # 创建副本避免意外修改原始数据
        y = pheno_data[target_trait].copy()
        
        # 步骤8: 标准化特征列名
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 步骤9: 验证最终数据质量
        if X.shape[0] != y.shape[0]:
            error_msg = f"特征矩阵和目标变量样本数不匹配: {X.shape[0]} vs {y.shape[0]}"
            main_logger.error(error_msg)
            raise ValueError(error_msg)
        
        if X.isnull().sum().sum() > 0:
            null_count = X.isnull().sum().sum()
            main_logger.warning(f"特征矩阵中存在 {null_count} 个缺失值")
        
        if y.isnull().sum() > 0:
            null_count = y.isnull().sum()
            main_logger.warning(f"目标变量中存在 {null_count} 个缺失值")
        
        # 步骤10: 分类任务的特殊处理
        if self.task_type == 'classification':
            # 对分类标签进行编码处理
            main_logger.info("正在处理分类标签...")
            y = self.genomic_classifier.prepare_classification_labels(y, self.results_dir)
        
        # 步骤11: 表型数据标准化（仅回归任务）
        if self.task_type == 'regression' and self.standardize_phenotype:
            main_logger.info("正在标准化表型数据...")
            y, self.phenotype_scaler = self._standardize_phenotype(y)
            main_logger.info(f"表型标准化完成 - 原始均值: {self.phenotype_scaler['mean']:.4f}, 原始标准差: {self.phenotype_scaler['std']:.4f}")
            
            # 保存标准化参数到文件
            scaler_path = self.results_dir / 'phenotype_scaler.json'
            with open(scaler_path, 'w') as f:
                json.dump(self.phenotype_scaler, f, indent=2)
            main_logger.info(f"表型标准化参数已保存到: {scaler_path}")
        
        # 步骤12: 记录最终数据信息
        main_logger.info(f"最终数据维度 - 特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        main_logger.info(f"目标变量统计 - 均值: {y.mean():.4f}, 标准差: {y.std():.4f}")
        
        return X, y, pheno_data

    def _standardize_phenotype(self, y: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """
        对表型数据进行Z-score标准化
        
        参数:
            y: 原始表型数据
            
        返回:
            标准化后的表型数据和标准化参数字典
        """
        mean_val = float(y.mean())
        std_val = float(y.std())
        
        # 避免除以零
        if std_val < 1e-10:
            main_logger.warning("表型数据标准差接近零，跳过标准化")
            return y, {'mean': mean_val, 'std': 1.0, 'applied': False}
        
        y_standardized = (y - mean_val) / std_val
        scaler_params = {
            'mean': mean_val,
            'std': std_val,
            'applied': True
        }
        
        return y_standardized, scaler_params

    def _inverse_standardize_phenotype(self, y: np.ndarray) -> np.ndarray:
        """
        对标准化后的预测值进行反标准化
        
        参数:
            y: 标准化后的预测值
            
        返回:
            原始尺度的预测值
        """
        if self.phenotype_scaler is None or not self.phenotype_scaler.get('applied', False):
            return y
        
        return y * self.phenotype_scaler['std'] + self.phenotype_scaler['mean']

    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """根据模型名称和参数创建模型实例"""
        if self.task_type == 'regression':
            return self.model_optimizer.create_model(model_name, params)
        else:
            return self.genomic_classifier.create_classification_model(model_name, params)

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型的默认参数"""
        if self.task_type == 'regression':
            return self.model_optimizer.get_default_params(model_name)
        else:
            return self.genomic_classifier.get_classification_default_params(model_name)

    def get_param_func(self, model_name: str) -> callable:
        """获取参数函数"""
        if self.task_type == 'regression':
            return self.model_optimizer.get_param_func(model_name)
        else:
            return self.genomic_classifier.get_classification_param_func(model_name)

    def filter_model_params_unified(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """统一的参数过滤方法"""
        if self.task_type == 'regression':
            return filter_model_params(model_name, params)
        else:
            return self.genomic_classifier.filter_classification_params(model_name, params)

    def _get_default_ensemble_metrics(self) -> Dict[str, float]:
        """获取默认的集成指标（当训练失败时使用）"""
        if self.task_type == 'classification':
            return {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.0}
        return {'pearson': 0.0, 'spearman': 0.0, 'r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}

    def _get_default_fold_metrics(self) -> Dict[str, float]:
        """获取默认的折平均指标（当所有折失败时使用）"""
        if self.task_type == 'classification':
            return {
                'avg_train_accuracy': 0.0, 'avg_val_accuracy': 0.0, 'avg_test_accuracy': 0.0,
                'std_train_accuracy': 0.0, 'std_val_accuracy': 0.0, 'std_test_accuracy': 0.0,
                'avg_train_f1': 0.0, 'avg_val_f1': 0.0, 'avg_test_f1': 0.0,
                'std_train_f1': 0.0, 'std_val_f1': 0.0, 'std_test_f1': 0.0,
                'avg_test_auc': 0.0, 'std_test_auc': 0.0, 'avg_training_time': 0.0
            }
        return {
            'avg_train_pearson': 0.0, 'avg_val_pearson': 0.0, 'avg_test_pearson': 0.0,
            'std_train_pearson': 0.0, 'std_val_pearson': 0.0, 'std_test_pearson': 0.0,
            'avg_test_spearman': 0.0, 'std_test_spearman': 0.0,
            'avg_test_mse': 0.0, 'std_test_mse': 0.0, 'avg_training_time': 0.0
        }

    def _train_single_fold(
        self, model, model_name: str, params: Dict, 
        fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray,
        X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
        test_indices: np.ndarray, repeat_idx: int, repeat_dir: Path,
        all_predictions: Dict, task_logger
    ) -> Optional[Dict]:
        """训练单个折并返回结果"""
        try:
            # 准备折训练数据
            X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, y_fold_train, y_fold_val, scaler = prepare_fold_training_data(
                X_train, y_train, train_idx, val_idx, X_test
            )
            
            # 训练折模型
            model, training_time = train_fold_model(model, X_fold_train_scaled, y_fold_train)
            
            # 预测并计算指标
            if self.task_type == 'classification':
                y_fold_train_pred, y_fold_train_proba = self.genomic_classifier.generate_classification_labels_from_model(model, X_fold_train_scaled)
                y_fold_val_pred, y_fold_val_proba = self.genomic_classifier.generate_classification_labels_from_model(model, X_fold_val_scaled)
                y_test_pred, y_test_proba = self.genomic_classifier.generate_classification_labels_from_model(model, X_test_scaled)
                
                train_metrics = self.genomic_classifier.calculate_classification_metrics(y_fold_train, y_fold_train_pred, y_fold_train_proba)
                val_metrics = self.genomic_classifier.calculate_classification_metrics(y_fold_val, y_fold_val_pred, y_fold_val_proba)
                test_metrics = self.genomic_classifier.calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
                plots = {}
            else:
                train_metrics, val_metrics, test_metrics, y_fold_train_pred, y_fold_val_pred, y_test_pred = predict_and_calculate_metrics(
                    model, X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val, X_test_scaled, y_test
                )
                plots = save_fold_predictions_and_plots(
                    train_idx, val_idx, test_indices, y_fold_train, y_fold_val, y_test,
                    y_fold_train_pred, y_fold_val_pred, y_test_pred, all_predictions,
                    model_name, repeat_idx, fold_idx, repeat_dir, self.results_dir
                )
            
            # 存储结果
            fold_result = {
                'fold_idx': fold_idx,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'plots': plots
            }
            
            # 保存模型
            if self.save_models:
                fold_model_path = repeat_dir / f"fold_{fold_idx+1}_model.pkl"
                joblib.dump((model, scaler), fold_model_path)
            
            # 记录结果
            self._log_fold_results(fold_idx, train_metrics, val_metrics, test_metrics, training_time, task_logger)
            
            return fold_result
            
        except Exception as e:
            task_logger.error(f"折 {fold_idx+1} 训练失败: {str(e)}")
            task_logger.error(traceback.format_exc())
            return None

    def _log_fold_results(self, fold_idx: int, train_metrics: Dict, val_metrics: Dict, 
                          test_metrics: Dict, training_time: float, task_logger) -> None:
        """记录折结果到日志"""
        if self.task_type == 'classification':
            self.genomic_classifier.log_classification_results(fold_idx, train_metrics, val_metrics, test_metrics, task_logger)
        else:
            task_logger.info(f"折 {fold_idx+1} 结果:")
            task_logger.info(f"  训练集 Pearson: {train_metrics['pearson']:.6f}")
            task_logger.info(f"  训练集 Spearman: {train_metrics['spearman']:.6f}")
            task_logger.info(f"  验证集 Pearson: {val_metrics['pearson']:.6f}")
            task_logger.info(f"  验证集 Spearman: {val_metrics['spearman']:.6f}")
            task_logger.info(f"  测试集 Pearson: {test_metrics['pearson']:.6f}")
            task_logger.info(f"  测试集 Spearman: {test_metrics['spearman']:.6f}")
            task_logger.info(f"  测试集 MSE: {test_metrics['mse']:.6f}")
        task_logger.info(f"  训练时间: {training_time:.2f}秒")

    def _calculate_fold_average_metrics(self, fold_results: List[Dict], repeat_idx: int, task_logger) -> Dict[str, float]:
        """计算所有折的平均指标"""
        metrics = {'avg_training_time': np.mean([r['training_time'] for r in fold_results])}
        
        if self.task_type == 'classification':
            for metric_name in ['accuracy', 'f1']:
                for split in ['train', 'val', 'test']:
                    values = [r[f'{split}_metrics'][metric_name] for r in fold_results]
                    metrics[f'avg_{split}_{metric_name}'] = np.mean(values)
                    metrics[f'std_{split}_{metric_name}'] = np.std(values)
            
            # AUC指标
            if all('auc' in r['test_metrics'] for r in fold_results):
                auc_values = [r['test_metrics']['auc'] for r in fold_results]
                metrics['avg_test_auc'] = np.mean(auc_values)
                metrics['std_test_auc'] = np.std(auc_values)
            else:
                metrics['avg_test_auc'] = 0.0
                metrics['std_test_auc'] = 0.0
            
            # 日志输出
            task_logger.info(f"\n重复 {repeat_idx+1} 平均性能:")
            task_logger.info(f"  平均训练集 Accuracy: {metrics['avg_train_accuracy']:.6f} (±{metrics['std_train_accuracy']:.6f})")
            task_logger.info(f"  平均验证集 Accuracy: {metrics['avg_val_accuracy']:.6f} (±{metrics['std_val_accuracy']:.6f})")
            task_logger.info(f"  平均测试集 Accuracy: {metrics['avg_test_accuracy']:.6f} (±{metrics['std_test_accuracy']:.6f})")
            task_logger.info(f"  平均训练集 F1: {metrics['avg_train_f1']:.6f} (±{metrics['std_train_f1']:.6f})")
            task_logger.info(f"  平均验证集 F1: {metrics['avg_val_f1']:.6f} (±{metrics['std_val_f1']:.6f})")
            task_logger.info(f"  平均测试集 F1: {metrics['avg_test_f1']:.6f} (±{metrics['std_test_f1']:.6f})")
            if metrics['avg_test_auc'] > 0:
                task_logger.info(f"  平均测试集 AUC: {metrics['avg_test_auc']:.6f} (±{metrics['std_test_auc']:.6f})")
        else:
            # 回归任务
            for split in ['train', 'val', 'test']:
                pearson_values = [r[f'{split}_metrics']['pearson'] for r in fold_results]
                metrics[f'avg_{split}_pearson'] = np.mean(pearson_values)
                metrics[f'std_{split}_pearson'] = np.std(pearson_values)
            
            spearman_values = [r['test_metrics']['spearman'] for r in fold_results]
            mse_values = [r['test_metrics']['mse'] for r in fold_results]
            metrics['avg_test_spearman'] = np.mean(spearman_values)
            metrics['std_test_spearman'] = np.std(spearman_values)
            metrics['avg_test_mse'] = np.mean(mse_values)
            metrics['std_test_mse'] = np.std(mse_values)
            
            task_logger.info(f"\n重复 {repeat_idx+1} 平均性能:")
            task_logger.info(f"  平均训练集 Pearson: {metrics['avg_train_pearson']:.6f} (±{metrics['std_train_pearson']:.6f})")
            task_logger.info(f"  平均验证集 Pearson: {metrics['avg_val_pearson']:.6f} (±{metrics['std_val_pearson']:.6f})")
            task_logger.info(f"  平均测试集 Pearson: {metrics['avg_test_pearson']:.6f} (±{metrics['std_test_pearson']:.6f})")
        
        task_logger.info(f"  平均训练时间: {metrics['avg_training_time']:.2f}秒")
        return metrics

    def _compute_ensemble_predictions(
        self, model_name: str, params: Dict, folds: List, 
        X: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, 
        X_test: pd.DataFrame, y_test: pd.Series, repeat_dir: Path, task_logger
    ) -> Dict[str, float]:
        """计算折集成预测并返回指标"""
        try:
            # 加载或重新训练模型
            if self.save_models:
                ensemble_models = []
                for fold_idx in range(self.n_splits):
                    fold_model_path = repeat_dir / f"fold_{fold_idx+1}_model.pkl"
                    if fold_model_path.exists():
                        model, scaler = joblib.load(fold_model_path)
                        ensemble_models.append((model, scaler))
            else:
                ensemble_models = []
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    fold_model = self.create_model(model_name, params)
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    fold_scaler = StandardScaler()
                    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                    fold_model.fit(X_fold_train_scaled, y_fold_train)
                    ensemble_models.append((fold_model, fold_scaler))
            
            # 计算集成预测
            ensemble_preds = []
            for model, scaler in ensemble_models:
                X_test_scaled = scaler.transform(X_test)
                ensemble_preds.append(model.predict(X_test_scaled))
            
            if self.task_type == 'classification':
                ensemble_pred = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
                ensemble_probas = []
                for model, scaler in ensemble_models:
                    X_test_scaled = scaler.transform(X_test)
                    if hasattr(model, 'predict_proba'):
                        ensemble_probas.append(model.predict_proba(X_test_scaled))
                    else:
                        pred = model.predict(X_test_scaled)
                        prob = np.zeros((len(pred), 2))
                        prob[np.arange(len(pred)), pred.astype(int)] = 1.0
                        ensemble_probas.append(prob)
                
                ensemble_proba = np.mean(ensemble_probas, axis=0) if ensemble_probas else None
                ensemble_metrics = self.genomic_classifier.calculate_classification_metrics(y_test, ensemble_pred, ensemble_proba)
                
                task_logger.info(f"\n折集成测试集性能:")
                task_logger.info(f"  准确率: {ensemble_metrics['accuracy']:.6f}")
                task_logger.info(f"  F1: {ensemble_metrics['f1']:.6f}")
                if 'auc' in ensemble_metrics:
                    task_logger.info(f"  AUC: {ensemble_metrics['auc']:.6f}")
            else:
                ensemble_pred = np.mean(ensemble_preds, axis=0)
                ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
                
                task_logger.info(f"\n折集成测试集性能:")
                task_logger.info(f"  Pearson: {ensemble_metrics['pearson']:.6f}")
                task_logger.info(f"  Spearman: {ensemble_metrics['spearman']:.6f}")
                task_logger.info(f"  R²: {ensemble_metrics['r2']:.6f}")
                task_logger.info(f"  MSE: {ensemble_metrics['mse']:.6f}")
                task_logger.info(f"  RMSE: {ensemble_metrics['rmse']:.6f}")
                task_logger.info(f"  MAE: {ensemble_metrics['mae']:.6f}")
            
            return ensemble_metrics
            
        except Exception as e:
            task_logger.error(f"计算折集成预测失败: {str(e)}")
            task_logger.error(traceback.format_exc())
            return self._get_default_ensemble_metrics()

    def optimize_model_parameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_name: str, 
        repeat_idx: int,
        task_logger=None,
        cv_pheno_data=None
    ) -> Dict[str, Any]:
        """
        使用Optuna优化模型参数
        
        参数:
            X: 特征矩阵
            y: 目标变量
            model_name: 模型名称
            repeat_idx: 重复索引
            task_logger: 任务专用日志记录器
            cv_pheno_data: 包含CV分组的表型数据
            
        返回:
            最佳参数字典
        """
        # 使用提供的记录器或默认记录器
        log = task_logger if task_logger else main_logger
        
        log.info(f"开始为模型 {model_name} 优化参数 (重复 {repeat_idx+1}/{self.n_repeats})...")
        
        # 获取交叉验证折
        folds = self.generate_cv_folds_from_file(X, y, cv_pheno_data, repeat_idx, task_logger)
        
        # 设置随机种子
        opt_seed = generate_optimization_seed(self.random_seed, repeat_idx)
        np.random.seed(opt_seed)
        
        # 定义优化目标函数
        def objective(trial):
            # 获取模型参数
            param_func = self.get_param_func(model_name)
            params = param_func(trial)
            
            # 过滤掉辅助参数
            filtered_params = self.filter_model_params_unified(model_name, params)
            
            # 创建模型
            model = self.create_model(model_name, filtered_params)
            
            # 使用交叉验证评估模型
            cv_scores = []
            
            log.debug(f"评估参数组合: {filtered_params}")
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # 获取当前折的数据
                X_fold_train = X.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_fold_train = y.iloc[train_idx]
                y_fold_val = y.iloc[val_idx]
                
                # 标准化特征
                scaler = StandardScaler()
                X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                X_fold_val_scaled = scaler.transform(X_fold_val)
                
                # 训练模型
                log.debug(f"训练第 {fold_idx+1} 折...")
                try:
                    model.fit(X_fold_train_scaled, y_fold_train)
                    
                    # 预测并计算得分
                    if self.task_type == 'classification':
                        # 分类任务：使用准确率作为优化目标
                        y_fold_pred, y_fold_proba = self.genomic_classifier.generate_classification_labels_from_model(model, X_fold_val_scaled)
                        fold_metrics = self.genomic_classifier.calculate_classification_metrics(y_fold_val, y_fold_pred, y_fold_proba)
                        fold_score = fold_metrics['accuracy']  # 使用准确率作为优化目标
                    else:
                        # 回归任务：使用Pearson相关系数
                        y_fold_pred = model.predict(X_fold_val_scaled)
                        
                        # 安全计算Pearson相关系数
                        if np.isnan(y_fold_pred).any() or np.std(y_fold_pred) < ModelConstants.CORRELATION_MIN_THRESHOLD:
                            log.warning(f"警告: 第 {fold_idx+1} 折产生了无效的预测 (NaN 或常量值)")
                            fold_score = 0.0
                        else:
                            fold_score = pearsonr(y_fold_val, y_fold_pred)[0]
                            if np.isnan(fold_score):
                                log.warning(f"警告: 第 {fold_idx+1} 折的Pearson相关系数计算结果为NaN")
                                fold_score = 0.0
                    
                    cv_scores.append(fold_score)
                
                except Exception as e:
                    log.error(f"第 {fold_idx+1} 折训练失败: {str(e)}")
                    cv_scores.append(0.0)
            
            if not cv_scores:
                log.error("所有折都失败了，返回一个低分")
                return 0.0
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            if self.task_type == 'classification':
                log.debug(f"平均准确率: {mean_score:.6f} (±{std_score:.6f})")
            else:
                log.debug(f"平均 Pearson相关系数: {mean_score:.6f} (±{std_score:.6f})")
            return mean_score
        
        # 创建提前停止回调
        def early_stopping_callback(study, trial):
            if trial.number >= self.patience:
                best_value = study.best_value
                # 安全获取最近的值，避免None值比较
                recent_trials = [t.value for t in study.trials[-self.patience:-1] if t.value is not None]
                if not recent_trials:  # 如果没有有效值，使用一个低值
                    recent_best_value = float('-inf')
                else:
                    recent_best_value = max(recent_trials)
                
                # 安全比较，确保best_value不是None
                if best_value is not None and recent_best_value is not None:
                    if best_value > recent_best_value:
                        log.info(f"\n提前停止：{self.patience} 次试验内没有改进")
                        return True
            return False
        
        # 创建Optuna研究并优化
        pruner = MedianPruner(
            n_startup_trials=ModelConstants.OPTUNA_N_STARTUP_TRIALS, 
            n_warmup_steps=ModelConstants.OPTUNA_N_WARMUP_STEPS, 
            interval_steps=ModelConstants.OPTUNA_INTERVAL_STEPS
        )
        study = optuna.create_study(direction='maximize', pruner=pruner, sampler=optuna.samplers.TPESampler(seed=opt_seed))
        
        # 优化 - 设置n_jobs=1确保串行执行
        log.info(f"开始参数优化，最大试验次数: {self.n_trials}")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[early_stopping_callback],
            show_progress_bar=True,
            n_jobs=ModelConstants.OPTUNA_N_JOBS  # 强制Optuna串行执行
        )
        
        # 获取最佳参数
        best_params = study.best_params
        log.info(f"最佳参数: {best_params}")
        if self.task_type == 'classification':
            log.info(f"最佳交叉验证准确率: {study.best_value:.6f}")
        else:
            log.info(f"最佳交叉验证Pearson相关系数: {study.best_value:.6f}")
        
        # 过滤参数，移除辅助参数
        filtered_best_params = self.filter_model_params_unified(model_name, best_params)
        
        # 返回包含优化信息的字典
        return {
            'best_params': filtered_best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'early_stopped': len(study.trials) < self.n_trials
        }
    
    def train_and_evaluate_model_for_repeat(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        repeat_idx: int,
        test_indices: np.ndarray = None,
        task_logger=None,
        cv_pheno_data=None
    ) -> Dict[str, Any]:
        """
        对一个重复周期训练和评估模型
        
        参数:
            model_name: 模型名称
            X: 特征矩阵
            y: 目标变量
            repeat_idx: 重复索引
            test_indices: 测试集索引(如果提供)
            task_logger: 任务专用日志记录器
            cv_pheno_data: 包含CV分组的表型数据
            
        返回:
            包含训练结果的字典
        """
        # 初始化日志和种子
        if task_logger is None:
            task_logger = self.get_task_logger(model_name, repeat_idx)
        
        repeat_seed = generate_repeat_seed(self.random_seed, repeat_idx)
        np.random.seed(repeat_seed)
        
        task_logger.info(f"\n{'='*50}")
        task_logger.info(f"模型 {model_name} - 重复 {repeat_idx+1}/{self.n_repeats}")
        task_logger.info(f"{'='*50}")
        main_logger.info(f"开始训练 模型:{model_name} 重复:{repeat_idx+1}")
        
        # 创建目录并准备数据
        repeat_dir = create_repeat_result_directory(self.results_dir, model_name, repeat_idx)
        train_indices, X_train, y_train, X_test, y_test = prepare_train_test_data(
            X, y, repeat_idx, self.random_seed, self.test_size, test_indices
        )
        task_logger.info(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")
        
        # 获取交叉验证折和模型参数
        folds = self.generate_cv_folds_from_file(X_train, y_train, cv_pheno_data, repeat_idx, task_logger)
        
        if self.use_default_params:
            params = self.get_default_params(model_name)
            task_logger.info(f"使用默认参数: {params}")
            optimization_info = {'best_params': params, 'best_value': None, 'n_trials': 0, 'early_stopped': False}
        else:
            optimization_info = self.optimize_model_parameters(X_train, y_train, model_name, repeat_idx, task_logger, cv_pheno_data)
            params = optimization_info['best_params']
        
        # 训练各折模型
        model = self.create_model(model_name, params)
        fold_results = []
        all_predictions = {'train': [], 'val': [], 'test': []}
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold_seed = generate_fold_seed(self.random_seed, repeat_idx, fold_idx)
            np.random.seed(fold_seed)
            task_logger.info(f"训练折 {fold_idx+1}/{self.n_splits}...")
            
            fold_result = self._train_single_fold(
                model, model_name, params, fold_idx, train_idx, val_idx,
                X_train, y_train, X_test, y_test, test_indices, repeat_idx,
                repeat_dir, all_predictions, task_logger
            )
            if fold_result:
                fold_results.append(fold_result)
        
        # 保存预测结果
        predictions_path = repeat_dir / 'all_predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(all_predictions, f, indent=2, cls=NumpyEncoder)
        
        # 计算指标
        if fold_results:
            avg_metrics = self._calculate_fold_average_metrics(fold_results, repeat_idx, task_logger)
            ensemble_metrics = self._compute_ensemble_predictions(
                model_name, params, folds, X, X_train, y_train, X_test, y_test, repeat_dir, task_logger
            )
        else:
            task_logger.error("所有折都训练失败，无法计算平均性能")
            avg_metrics = self._get_default_fold_metrics()
            ensemble_metrics = self._get_default_ensemble_metrics()
        
        # 构建并保存结果
        repeat_results = {
            'model_name': model_name,
            'repeat_idx': repeat_idx,
            'params': params,
            'fold_results': fold_results,
            'avg_training_time': avg_metrics.get('avg_training_time', 0.0),
            'ensemble_metrics': ensemble_metrics,
            'optimization_info': optimization_info,
            'test_indices': test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices),
            'all_predictions_file': str(predictions_path),
        }
        repeat_results.update(avg_metrics)
        
        repeat_results_path = repeat_dir / 'repeat_results.json'
        with open(repeat_results_path, 'w') as f:
            json.dump(repeat_results, f, indent=2, cls=NumpyEncoder)
        
        task_logger.info(f"重复 {repeat_idx+1} 结果已保存到 {repeat_results_path}")
        return repeat_results

    def run_model_multiple_repeats(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_pheno_data: pd.DataFrame,
        use_same_test_set: bool = True
    ) -> Dict[str, Any]:
        """
        执行多次重复训练同一个模型
        
        参数:
            model_name: 模型名称
            X: 特征矩阵
            y: 目标变量
            cv_pheno_data: 包含CV分组的表型数据
            use_same_test_set: 是否在所有重复中使用相同的测试集
            
        返回:
            包含所有重复结果的字典
        """
        main_logger.info(f"\n{'='*70}")
        main_logger.info(f"开始为模型 {model_name} 执行 {self.n_repeats} 次重复训练")
        main_logger.info(f"{'='*70}")
        
        # 创建模型结果目录
        model_dir = create_model_result_directory(self.results_dir, model_name)
        
        # 如果使用相同的测试集，预先生成测试集索引
        test_indices = None
        if use_same_test_set:
            _, test_indices = train_test_split(
                range(len(X)), 
                test_size=self.test_size, 
                random_state=self.random_seed
            )
            main_logger.info(f"所有重复将使用相同的测试集 (大小: {len(test_indices)})")
        
        # 存储所有重复的结果
        all_repeat_results = []
        
        # 决定是否并行
        if self.max_parallel_jobs > 1:
            # 并行处理多个重复
            main_logger.info(f"使用 {self.max_parallel_jobs} 个并行任务执行训练")
            
            # 并行执行任务
            with ProcessPoolExecutor(max_workers=self.max_parallel_jobs, 
                                    mp_context=mp.get_context('spawn')) as executor:
                futures = [
                    executor.submit(
                        self._run_repeat_task, model_name, X, y, repeat_idx, test_indices, cv_pheno_data
                    ) for repeat_idx in range(self.n_repeats)
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        repeat_results = future.result()
                        all_repeat_results.append(repeat_results)
                        main_logger.info(f"完成重复 {i+1}/{self.n_repeats}")
                    except Exception as e:
                        main_logger.error(f"重复执行失败: {str(e)}")
        else:
            # 串行处理
            main_logger.info("串行执行训练")
            for repeat_idx in range(self.n_repeats):
                try:
                    repeat_results = self.train_and_evaluate_model_for_repeat(
                        model_name, X, y, repeat_idx, test_indices, task_logger=None, cv_pheno_data=cv_pheno_data
                    )
                    all_repeat_results.append(repeat_results)
                except Exception as e:
                    main_logger.error(f"重复 {repeat_idx+1} 执行失败: {str(e)}")
        
        # 计算所有重复的平均性能
        if all_repeat_results:
            # 使用新的统计计算方法
            summary = calculate_repeat_statistics(all_repeat_results, self.task_type)
            
            # 找出性能最接近平均值的重复
            if self.task_type == 'classification':
                avg_ensemble_metric = summary['avg_ensemble_accuracy']
                metric_name = "准确率"
                metric_key = 'ensemble_accuracy_values'
            else:
                avg_ensemble_metric = summary['avg_ensemble_pearson']
                metric_name = "Pearson相关系数"
                metric_key = 'ensemble_pearson_values'
                
            closest_repeat_idx, closest_repeat, difference = find_representative_repeat(
                all_repeat_results, avg_ensemble_metric
            )
            
            main_logger.info(f"找到性能最接近平均值的重复: 重复 {closest_repeat_idx+1}")
            main_logger.info(f"该重复的集成{metric_name}: {summary['raw_values'][metric_key][closest_repeat_idx]:.{ModelConstants.METRICS_PRECISION}f}")
            main_logger.info(f"平均集成{metric_name}: {avg_ensemble_metric:.{ModelConstants.METRICS_PRECISION}f}")
            main_logger.info(f"差距: {difference:.{ModelConstants.METRICS_PRECISION}f}")
            
            # 保存代表性模型
            if self.task_type == 'classification':
                repeat_info = {
                    'repeat_idx': int(closest_repeat_idx),
                    'ensemble_accuracy': float(summary['raw_values']['ensemble_accuracy_values'][closest_repeat_idx]),
                    'difference_from_avg': float(difference)
                }
            else:
                repeat_info = {
                    'repeat_idx': int(closest_repeat_idx),
                    'ensemble_pearson': float(summary['raw_values']['ensemble_pearson_values'][closest_repeat_idx]),
                    'difference_from_avg': float(difference)
                }
            
            model_path = self._save_representative_model(
                model_name, closest_repeat['params'], X, y, model_dir, repeat_info
            )
            
            if model_path:
                main_logger.info(f"已保存代表性模型到 {model_path}")
                # 在summary中添加代表性模型信息
                if self.task_type == 'classification':
                    summary['representative_model'] = {
                        'repeat_idx': int(closest_repeat_idx),
                        'ensemble_accuracy': float(summary['raw_values']['ensemble_accuracy_values'][closest_repeat_idx]),
                        'model_path': model_path
                    }
                else:
                    summary['representative_model'] = {
                        'repeat_idx': int(closest_repeat_idx),
                        'ensemble_pearson': float(summary['raw_values']['ensemble_pearson_values'][closest_repeat_idx]),
                        'model_path': model_path
                    }
            
            # 打印结果摘要
            # 打印模型性能摘要
            model_name = summary['model_name']
            n_repeats = summary['n_repeats']
            
            main_logger.info(f"\n{ModelConstants.LOG_SEPARATOR_SHORT}")
            main_logger.info(f"模型 {model_name} - {n_repeats} 次重复的平均性能:")
            
            if self.task_type == 'classification':
                main_logger.info(f"训练集 准确率: {summary['avg_train_accuracy']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_train_accuracy']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"验证集 准确率: {summary['avg_val_accuracy']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_val_accuracy']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"测试集 准确率: {summary['avg_test_accuracy']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_test_accuracy']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"测试集 F1: {summary['avg_test_f1']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_test_f1']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"测试集 AUC: {summary['avg_test_auc']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_test_auc']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"折集成 准确率: {summary['avg_ensemble_accuracy']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_ensemble_accuracy']:.{ModelConstants.METRICS_PRECISION}f})")
            else:
                main_logger.info(f"训练集 Pearson: {summary['avg_train_pearson']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_train_pearson']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"验证集 Pearson: {summary['avg_val_pearson']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_val_pearson']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"测试集 Pearson: {summary['avg_test_pearson']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_test_pearson']:.{ModelConstants.METRICS_PRECISION}f})")
                main_logger.info(f"折集成 Pearson: {summary['avg_ensemble_pearson']:.{ModelConstants.METRICS_PRECISION}f} (±{summary['std_ensemble_pearson']:.{ModelConstants.METRICS_PRECISION}f})")
                
            main_logger.info(f"平均训练时间: {summary['avg_training_time']:.{ModelConstants.TIME_PRECISION}f}秒")
            main_logger.info(f"{ModelConstants.LOG_SEPARATOR_SHORT}")
            
            # 保存总结果
            summary_path = model_dir / 'summary_results.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
            main_logger.info(f"模型 {model_name} 的所有重复结果摘要已保存到 {summary_path}")
            
            return summary
        else:
            main_logger.error(f"模型 {model_name} 的所有重复都失败了")
            return {
                'model_name': model_name,
                'n_repeats': 0,
                'error': "所有重复都失败了"
            }
    
    def _run_repeat_task(self, model_name, X, y, repeat_idx, test_indices, cv_pheno_data):
        """为并行处理提供的包装器函数"""
        # 在子进程中创建新的日志目录和日志记录器
        # 注意：子进程需要重新创建日志目录，因为它们是独立的进程
        logs_dir = Path(self.results_dir) / "logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建模型日志目录
        model_log_dir = logs_dir / model_name
        model_log_dir.mkdir(exist_ok=True, parents=True)
        
        # 为这个重复任务创建日志文件
        log_file = model_log_dir / f"repeat_{repeat_idx+1}.log"
        task_logger = create_logger(
            log_file=str(log_file),
            name=f"{model_name}_repeat_{repeat_idx+1}_pid{os.getpid()}",
            level=logging.DEBUG,
            propagate=False
        )
        
        # 记录子进程开始信息
        task_logger.info(f"子进程(PID={os.getpid()})开始执行 {model_name} 模型的重复 {repeat_idx+1}")
        
        try:
            # 使用任务专用日志记录器执行任务
            result = self.train_and_evaluate_model_for_repeat(
                model_name, X, y, repeat_idx, test_indices, task_logger=task_logger, cv_pheno_data=cv_pheno_data
            )
            task_logger.info(f"子进程(PID={os.getpid()})成功完成 {model_name} 模型的重复 {repeat_idx+1}")
            return result
        except Exception as e:
            # 记录异常详情到任务日志
            task_logger.error(f"子进程执行失败: {str(e)}")
            task_logger.error(traceback.format_exc())
            # 重新抛出异常以通知主进程
            raise
    
    def prepare_cv_folds(self, pheno_data: pd.DataFrame, target_trait: str) -> pd.DataFrame:
        """
        准备交叉验证折分组信息并保存到文件
        
        参数:
            pheno_data: 表型数据DataFrame
            target_trait: 目标性状名称
            
        返回:
            添加了CV分组列的表型数据
        """
        # 确定CV文件路径
        if self.cv_file is not None:
            cv_file_path = Path(self.cv_file)
        else:
            # 创建CV文件目录
            cv_dir = self.results_dir / "cv_folds"
            cv_dir.mkdir(exist_ok=True, parents=True)
            
            # 默认CV文件路径
            cv_file_path = cv_dir / f"{target_trait}_cv_{self.n_repeats}x{self.n_splits}.csv"
        
        # 检查是否已存在CV文件且不需要强制重新生成
        if cv_file_path.exists() and not self.force_new_cv:
            main_logger.info(f"加载已有的CV分组文件: {cv_file_path}")
            cv_pheno_data = pd.read_csv(cv_file_path, index_col=0)
        else:
            main_logger.info(f"生成新的CV分组并保存到: {cv_file_path}")
            # 确保pheno_data有索引列
            pheno_data_copy = pheno_data.copy()
            
            # 如果ID列在数据中，将其设置为索引
            if self.cv_id_column in pheno_data_copy.columns:
                pheno_data_copy.set_index(self.cv_id_column, inplace=True)
            
            # 生成CV分组
            cv_pheno_data = prepare_cv_data(
                pheno_data_copy, 
                str(cv_file_path), 
                self.n_repeats, 
                self.n_splits
            )
        
        return cv_pheno_data

    def generate_cv_folds_from_file(self, X: pd.DataFrame, y: pd.Series, cv_pheno_data: pd.DataFrame, repeat_idx: int, task_logger=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        从CV文件生成交叉验证折
        
        参数:
            X: 特征矩阵
            y: 目标变量
            cv_pheno_data: 包含CV分组的表型数据
            repeat_idx: 重复索引
            task_logger: 任务专用日志记录器
            
        返回:
            包含训练和验证索引的列表
        """
        # 使用提供的记录器或默认记录器
        log = task_logger if task_logger else main_logger
        
        # 获取当前重复的CV列名
        cv_col = f'cv{repeat_idx}'
        
        # 确保CV列存在
        if cv_col not in cv_pheno_data.columns:
            log.error(f"CV文件中不存在列 {cv_col}，无法生成交叉验证折")
            raise ValueError(f"CV文件中不存在列 {cv_col}")
        
        # 生成所有折
        folds = []
        for fold_idx in range(self.n_splits):
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
            log.debug(f"重复 {repeat_idx+1}/{self.n_repeats}, 折 {fold_idx+1}/{self.n_splits}: "
                     f"训练集大小 {len(train_idx)}, 验证集大小 {len(val_idx)}")
        
        return folds

    def run_all_models(
        self,
        geno_file: str,
        pheno_file: str,
        target_trait: str,
        models: Optional[List[str]] = None,
        use_stacking: bool = False,
        top_n_models: int = 5,
        cv_folds: int = 5,
        use_same_test_set: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        运行所有或指定的模型，并可选择性地进行Stacking集成
        
        参数:
            geno_file: 基因型数据文件路径
            pheno_file: 表型数据文件路径
            target_trait: 目标性状名称
            models: 模型名称列表，None表示运行所有可用模型
            use_stacking: 是否使用Stacking集成
            top_n_models: Stacking集成使用的顶级模型数量
            cv_folds: Stacking集成交叉验证折数
            use_same_test_set: 是否在所有模型和重复中使用相同的测试集
            
        返回:
            包含所有模型结果的字典
        """
        # 加载数据
        X, y, pheno_data = self.load_data(geno_file, pheno_file, target_trait)
        
        # 准备交叉验证折
        cv_pheno_data = self.prepare_cv_folds(pheno_data, target_trait)
        
        # 确定要运行的模型
        if models is None:
            models = self.available_models
        
        # 记录要运行的模型
        main_logger.info(f"将要运行以下模型: {', '.join(models)}")
        
        # 存储所有模型的结果
        all_model_results = {}
        
        # 运行每个模型
        for model_name in models:
            try:
                model_summary = self.run_model_multiple_repeats(
                    model_name, X, y, cv_pheno_data, use_same_test_set
                )
                all_model_results[model_name] = model_summary
            except Exception as e:
                main_logger.error(f"模型 {model_name} 运行失败: {str(e)}")
                main_logger.error(traceback.format_exc())
        
        # 第一次创建模型比较表格（集成前）
        main_logger.info("=" * 50)
        main_logger.info("集成学习前的模型性能比较：")
        create_comparison_table(all_model_results, self.results_dir, main_logger)
        
        # 新增：在Stacking集成之前运行TOPSIS评价
        selected_models_for_stacking = None
        if use_stacking and len(all_model_results) >= 2:
            # 运行TOPSIS评价
            comparison_path = self.results_dir / 'model_comparison.csv'
            if comparison_path.exists():
                out_base = comparison_path.stem + "_topsis"
                topsis_out = comparison_path.with_name(f"{out_base}.csv")
                topsis_simple = comparison_path.with_name(f"{out_base}_simple.csv")
                
                try:
                    # 根据任务类型获取TOPSIS配置
                    criteria, criteria_types, manual_weights = self.get_topsis_configuration()
                    
                    call_topsis_evaluator(
                        comparison_csv=str(comparison_path),
                        output_csv=str(topsis_out),
                        criteria=criteria,
                        criteria_types=criteria_types,
                        manual_weights=manual_weights,
                        min_transform='neglog',
                        simple_output=str(topsis_simple),
                        logger=main_logger
                    )
                    
                    # 读取TOPSIS结果，选择前N个模型进行集成
                    topsis_df = pd.read_csv(topsis_simple)
                    selected_models_for_stacking = topsis_df.head(top_n_models)['Model'].tolist()
                    main_logger.info(f"基于TOPSIS排名选择的前{top_n_models}个模型: {selected_models_for_stacking}")
                    
                except Exception as e:
                    main_logger.error(f"TOPSIS评价失败: {str(e)}")
                    selected_models_for_stacking = None
        
        # 如果需要，运行Stacking集成
        if use_stacking and len(all_model_results) >= 2:
            try:
                main_logger.info(f"\n{'='*70}")
                main_logger.info(f"开始运行Stacking集成")
                main_logger.info(f"{'='*70}")
                
                # 选择模型进行集成
                if selected_models_for_stacking:
                    # 使用TOPSIS排名选择的模型
                    selected_models = selected_models_for_stacking
                    main_logger.info(f"使用TOPSIS排名选择的前 {len(selected_models)} 个模型进行集成: {', '.join(selected_models)}")
                else:
                    # 回退到原来的方法：根据测试集性能选择
                    if self.task_type == 'classification':
                        model_performances = [
                            (name, results.get('avg_ensemble_accuracy', results.get('avg_test_accuracy', 0.0))) 
                            for name, results in all_model_results.items()
                        ]
                    else:
                        model_performances = [
                            (name, results.get('avg_ensemble_pearson', results.get('avg_test_pearson', 0.0))) 
                            for name, results in all_model_results.items()
                        ]
                    model_performances.sort(key=lambda x: x[1], reverse=True)
                    selected_models = [name for name, _ in model_performances[:top_n_models]]
                    main_logger.info(f"使用测试集性能选择的前 {len(selected_models)} 个模型进行集成: {', '.join(selected_models)}")
                
                # 准备Stacking训练数据
                if use_same_test_set:
                    _, test_indices = train_test_split(
                        range(len(X)), 
                        test_size=self.test_size, 
                        random_state=self.random_seed
                    )
                    train_indices = np.array([i for i in range(len(X)) if i not in test_indices])
                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=self.test_size, random_state=self.random_seed
                    )
                
                # 创建Stacking集成模型
                stacking_model = StackingEnsemble(
                    base_models_dir=str(self.results_dir),
                    top_n_models=top_n_models,
                    cv_folds=cv_folds,
                    random_seed=self.random_seed,
                    n_threads=self.n_threads,
                    use_default_params=self.use_default_params,
                    task_type=self.task_type  # 新增：传递任务类型
                )
                
                # 训练集成模型 - 传入TOPSIS选择的模型名称
                stacking_results = stacking_model.fit(
                    X_train, y_train, X_test, y_test, model_names=selected_models
                )
                
                # 添加Stacking结果
                if self.task_type == 'classification':
                    all_model_results['stacking_ensemble'] = {
                        'model_name': 'stacking_ensemble',
                        'avg_test_accuracy': stacking_results['test_metrics']['accuracy'],
                        'std_test_accuracy': 0.0,  # 只运行一次，没有标准差
                        'avg_test_f1': stacking_results['test_metrics']['f1'],
                        'std_test_f1': 0.0,
                        'avg_test_auc': stacking_results['test_metrics'].get('auc', 0.0),
                        'std_test_auc': 0.0,
                        'selected_models': selected_models
                    }
                else:
                    all_model_results['stacking_ensemble'] = {
                        'model_name': 'stacking_ensemble',
                        'avg_test_pearson': stacking_results['test_metrics']['pearson'],
                        'std_test_pearson': 0.0,  # 只运行一次，没有标准差
                        'selected_models': selected_models
                    }
                
                # 第二次更新比较表格（集成后）
                main_logger.info("=" * 50)
                main_logger.info("集成学习后的模型性能比较：")
                create_comparison_table(all_model_results, self.results_dir, main_logger)
                
            except Exception as e:
                main_logger.error(f"Stacking集成运行失败: {str(e)}")
                main_logger.error(traceback.format_exc())
        
        return all_model_results


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description='基因组预测标准化流程V2 - 支持多次循环的K折交叉验证')
    parser.add_argument('--geno_file', type=str, 
                      help='基因型数据文件路径')
    parser.add_argument('--pheno_file', type=str, 
                      help='表型数据文件路径')
    parser.add_argument('--target_trait', type=str, 
                      help='目标性状名称')
    parser.add_argument('--models', type=str, nargs='+',
                      help='要运行的模型列表，不指定则运行所有可用模型')
    parser.add_argument('--trials', type=int, default=100,
                      help='每个模型的Optuna优化最大试验次数')
    parser.add_argument('--threads', type=int, default=1,
                      help='每个模型使用的线程数（建议设为1）')
    parser.add_argument('--parallel_jobs', type=int, default=1,
                      help='并行执行的任务数')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='测试集比例')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='交叉验证折数')
    parser.add_argument('--n_repeats', type=int, default=100,
                      help='重复执行次数')
    parser.add_argument('--patience', type=int, default=20,
                      help='提前停止的耐心值')
    parser.add_argument('--results_dir', type=str, default="optimization_results_v2",
                      help='结果保存目录')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--use_stacking', action='store_true',
                      help='是否使用Stacking集成学习')
    parser.add_argument('--top_n_models', type=int, default=5,
                      help='Stacking集成中使用的顶级模型数量')
    parser.add_argument('--cv_folds', type=int, default=5,
                      help='生成元特征的交叉验证折数')
    parser.add_argument('--use_default_params', action='store_true',
                      help='是否使用模型的默认参数而不进行优化')
    parser.add_argument('--save_models', action='store_true',
                      help='是否保存训练好的模型')
    parser.add_argument('--use_same_test_set', action='store_true', default=True,
                      help='是否在所有重复中使用相同的测试集')
    parser.add_argument('--save_representative', action='store_true',
                      help='是否保存性能最接近平均值的代表性模型')
    parser.add_argument('--cv_file', type=str, default=None,
                      help='指定CV文件路径，如果不存在则创建')
    parser.add_argument('--force_new_cv', action='store_true',
                      help='强制生成新的CV文件，即使已存在')
    parser.add_argument('--cv_id_column', type=str, default='ID',
                      help='表型数据中的ID列名，用于CV文件生成')
    # TOPSIS参数现在通过get_topsis_configuration()自动配置，不需要手动指定
    # parser.add_argument('--topsis_criteria', type=str, default='Test Pearson,Test Pearson (std)', help='TOPSIS评价指标')  
    # parser.add_argument('--topsis_criteria_types', type=str, default='max,min', help='TOPSIS指标类型')
    
    # 新增分类相关参数
    parser.add_argument('--task_type', type=str, default='regression', 
                      choices=['regression', 'classification'],
                      help='任务类型：regression（回归）或 classification（分类）')
    parser.add_argument('--n_classes', type=int, default=None,
                      help='分类任务的类别数（分类任务必须指定）')
    parser.add_argument('--standardize_phenotype', action='store_true',
                      help='是否对表型数据进行Z-score标准化（仅回归任务有效）')
    
    # 数据预处理参数组
    preprocess_group = parser.add_argument_group('数据预处理选项', '基因组数据预处理相关参数')
    preprocess_group.add_argument('--enable_preprocess', action='store_true',
                                help='启用数据预处理功能')
    preprocess_group.add_argument('--preprocess_only', action='store_true',
                                help='仅执行数据预处理，不进行模型训练')
    
    # VCF和PLINK相关参数
    preprocess_group.add_argument('--vcf_file', type=str, help='VCF格式文件路径')
    preprocess_group.add_argument('--bfile', type=str, help='输入 PLINK 二进制格式文件 (BED/BIM/FAM) 的前缀 (不含扩展名)')
    preprocess_group.add_argument('--ped_file', type=str, help='PLINK PED文件路径')
    preprocess_group.add_argument('--map_file', type=str, help='PLINK MAP文件路径')
    preprocess_group.add_argument('--plink_path', type=str, default='plink', help='PLINK可执行文件路径')
    preprocess_group.add_argument('--plink_out', type=str, help='PLINK输出前缀（VCF转换时使用）')
    
    # SNP提取参数
    preprocess_group.add_argument('--extract_file', type=str, help='包含要提取的SNP ID的文件')
    preprocess_group.add_argument('--snp_dir', type=str, help='包含SNP列表文件的目录')
    preprocess_group.add_argument('--direct_convert', action='store_true', 
                                help='直接将整个bfile转换为矩阵，不进行SNP提取')
    preprocess_group.add_argument('--matrix_file', type=str, 
                                help='指定已有的基因型矩阵文件路径，跳过矩阵生成步骤')
    
    # 表型处理参数
    preprocess_group.add_argument('--raw_pheno_file', type=str, help='原始表型TXT文件路径')
    
    # 输出和控制参数
    preprocess_group.add_argument('--preprocess_prefix', type=str, help='预处理输出文件前缀')
    preprocess_group.add_argument('--skip_matrix_conversion', action='store_true', 
                                help='跳过SNP提取和矩阵转换步骤')
    preprocess_group.add_argument('--skip_phenotype_match', action='store_true', 
                                help='跳过表型基因型匹配步骤')
    preprocess_group.add_argument('--skip_data_clean', action='store_true', 
                                help='跳过数据清理步骤')
    preprocess_group.add_argument('--load_matrix_info', action='store_true', 
                                help='加载并显示矩阵信息')
    
    args = parser.parse_args()
    
    # 验证分类参数
    if args.task_type == 'classification' and args.n_classes is None:
        main_logger.error("分类任务必须指定 --n_classes 参数")
        return 1
    
    # 参数验证
    if args.preprocess_only:
        # 仅预处理模式的参数验证
        if not args.preprocess_prefix:
            main_logger.error("预处理模式下必须提供 --preprocess_prefix 参数")
            return 1
    elif args.enable_preprocess:
        # 预处理+训练模式的参数验证
        if not args.preprocess_prefix:
            main_logger.error("启用数据预处理时必须提供 --preprocess_prefix 参数")
            return 1
        if not args.target_trait:
            main_logger.error("训练模式下必须提供 --target_trait 参数")
            return 1
    else:
        # 正常训练模式的参数验证  
        if not args.geno_file or not args.pheno_file:
            main_logger.error("正常模式下必须提供 --geno_file 和 --pheno_file 参数")
            return 1
        if not args.target_trait:
            main_logger.error("训练模式下必须提供 --target_trait 参数")
            return 1
    
    # 打印配置信息
    main_logger.info("基因组预测系统配置:")
    for arg, value in vars(args).items():
        main_logger.info(f"{arg}: {value}")
    
    # 数据预处理阶段
    processed_geno_file = args.geno_file
    processed_pheno_file = args.pheno_file
    
    if args.enable_preprocess or args.preprocess_only:
        main_logger.info("="*70)
        main_logger.info("开始数据预处理阶段")
        main_logger.info("="*70)
        
        # 导入数据处理器
        from genomic_data_pipeline import GenomicDataProcessor
        
        # 创建数据处理器
        processor = GenomicDataProcessor(logger=main_logger, plink_path=args.plink_path)
        
        # 准备预处理参数
        preprocess_kwargs = {
            'out_prefix': args.preprocess_prefix,
            'vcf': args.vcf_file,
            'bfile': args.bfile,
            'ped_file': args.ped_file,
            'map_file': args.map_file,
            'extract': args.extract_file,
            'snp_dir': args.snp_dir,
            'direct': args.direct_convert,
            'matrix_file': args.matrix_file,
            'pheno': args.raw_pheno_file,
            'plink_out': args.plink_out,
            'skip_matrix': args.skip_matrix_conversion,
            'skip_match': args.skip_phenotype_match,
            'skip_clean': args.skip_data_clean,
            'load': args.load_matrix_info,
            'trait_name': args.target_trait,  # Pass the target trait name
            'standardize_phenotype': args.standardize_phenotype  # 传递标准化参数
        }
        
        # 执行数据预处理
        try:
            result = processor.process_genomic_data(**preprocess_kwargs)
            if result != 0:
                main_logger.error("数据预处理失败")
                return result
                
            main_logger.info("数据预处理完成!")
            
            # 设置处理后的文件路径（新的统一命名格式）
            if not args.skip_phenotype_match and args.raw_pheno_file:
                # 使用一体化处理后的最终文件
                processed_geno_file = f"{args.preprocess_prefix}_genotype.csv"
                processed_pheno_file = f"{args.preprocess_prefix}_phenotype.csv"
            else:
                # 使用原始处理的文件
                processed_geno_file = args.preprocess_prefix + ".csv"
                processed_pheno_file = args.raw_pheno_file
                
        except Exception as e:
            main_logger.error(f"数据预处理过程中发生错误: {str(e)}")
            main_logger.error(traceback.format_exc())
            return 1
    
    # 如果只是预处理模式，则结束程序
    if args.preprocess_only:
        main_logger.info("预处理完成，程序结束")
        return 0
    
    # 验证处理后的文件是否存在
    if not os.path.exists(processed_geno_file):
        main_logger.error(f"基因型文件不存在: {processed_geno_file}")
        return 1
    if not os.path.exists(processed_pheno_file):
        main_logger.error(f"表型文件不存在: {processed_pheno_file}")
        return 1
    
    main_logger.info("="*70)
    main_logger.info("开始模型训练阶段")
    main_logger.info("="*70)
    main_logger.info(f"使用基因型文件: {processed_geno_file}")
    main_logger.info(f"使用表型文件: {processed_pheno_file}")
    
    # 如果预处理阶段已经标准化，训练阶段不再重复标准化
    training_standardize = False  # 预处理阶段已完成标准化
    if args.enable_preprocess and args.standardize_phenotype:
        main_logger.info("表型数据已在预处理阶段完成标准化，训练阶段跳过标准化")
    elif not args.enable_preprocess:
        # 如果未启用预处理，则使用用户指定的标准化设置
        training_standardize = args.standardize_phenotype
    
    # 创建预测器实例，传递CV相关参数
    predictor = GenomicPredictorV2(
        random_seed=args.random_seed,
        results_dir=args.results_dir,
        n_trials=args.trials,
        n_threads=args.threads,
        max_parallel_jobs=args.parallel_jobs,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        patience=args.patience,
        use_default_params=args.use_default_params,
        save_models=args.save_models,
        save_representative=args.save_representative,
        cv_file=args.cv_file,
        force_new_cv=args.force_new_cv,
        cv_id_column=args.cv_id_column,
        task_type=args.task_type,
        n_classes=args.n_classes,
        standardize_phenotype=training_standardize  # 根据预处理情况决定
    )
    
    # 运行模型，使用处理后的文件（如果进行了预处理）或原始文件
    predictor.run_all_models(
        geno_file=processed_geno_file,
        pheno_file=processed_pheno_file,
        target_trait=args.target_trait,
        models=args.models,
        use_stacking=args.use_stacking,
        top_n_models=args.top_n_models,
        cv_folds=args.cv_folds,
        use_same_test_set=args.use_same_test_set
    )

if __name__ == "__main__":
    main() 