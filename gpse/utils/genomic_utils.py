#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Genomic Prediction Utility Functions Module
===========================================

Contains utility functions, helper classes, and independent functional functions 
used in the genomic prediction pipeline.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all performance metrics."""
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


def prepare_cv_data(phe_data: pd.DataFrame, save_path: str, cv_times: int, cvfold: int) -> pd.DataFrame:
    """
    Generate and save cross-validation grouping information.
    
    Args:
        phe_data: Phenotype data DataFrame.
        save_path: Path to save the CV file.
        cv_times: Number of repetitions.
        cvfold: Number of CV folds.
    
    Returns:
        Phenotype data with added CV grouping columns.
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
        from gpse.train.topsis import TOPSISEvaluator
        
        # Use provided logger or try to get one from log_utils
        if logger is None:
            try:
                from .log_utils import logger as logutils_logger
                logger = logutils_logger
            except (ImportError, ValueError):
                pass
        
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
    """Create a comparison table for model performance."""
    # Prepare comparison data
    comparison_data = []
    
    # Check task type
    first_result = next(iter(all_model_results.values()))
    is_classification = 'avg_test_accuracy' in first_result
    
    for model_name, results in all_model_results.items():
        # Clean model names, removing _clf and _reg suffixes
        display_name = model_name.replace('_reg', '').replace('_clf', '')
        # Use special name for ensemble models
        if display_name == 'stacking_ensemble':
            display_name = 'Ensemble'
        
        # Collect performance metrics based on task type
        try:
            if is_classification:
                # Classification task metrics
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
                # Regression task metrics
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
            error_msg = f"Error processing results for model {model_name}: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
    
    # Create DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary performance metric
        comparison_df = comparison_df.sort_values(sort_column, ascending=False)
        
        # Save to CSV
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        # Log results
        if logger:
            logger.info(f"\nModel Performance Comparison:")
            logger.info("\n" + comparison_df.to_string(index=False))
            logger.info(f"\nComparison table saved to {comparison_path}")
        else:
            print(f"\nModel Performance Comparison:")
            print("\n" + comparison_df.to_string(index=False))
            print(f"\nComparison table saved to {comparison_path}")
    else:
        warning_msg = "Not enough results to create comparison table"
        if logger:
            logger.warning(warning_msg)
        else:
            print(f"Warning: {warning_msg}")


def generate_cv_folds_from_file(
    X: pd.DataFrame, 
    y: pd.Series, 
    cv_pheno_data: pd.DataFrame, 
    repeat_idx: int, 
    n_splits: int,
    task_logger=None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate cross-validation folds from a CV file.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        cv_pheno_data: Phenotype data containing CV groupings.
        repeat_idx: Index of the current repetition.
        n_splits: Number of CV folds.
        task_logger: Logger for specific tasks.
        
    Returns:
        List containing training and validation indices.
    """
    # Use provided logger or default to None
    log = task_logger if task_logger else None
    
    # Get CV column name for the current repetition
    cv_col = f'cv{repeat_idx}'
    
    # Ensure CV column exists
    if cv_col not in cv_pheno_data.columns:
        error_msg = f"Column {cv_col} not found in CV file, cannot generate cross-validation folds"
        if log:
            log.error(error_msg)
        else:
            print(f"Error: {error_msg}")
        raise ValueError(f"Column {cv_col} missing in CV file")
    
    # Generate all folds
    folds = []
    for fold_idx in range(n_splits):
        # Get validation set indices
        val_mask = cv_pheno_data[cv_col] == fold_idx
        val_indices = cv_pheno_data.index[val_mask].tolist()
        
        # Get training set indices
        train_mask = cv_pheno_data[cv_col] != fold_idx
        train_indices = cv_pheno_data.index[train_mask].tolist()
        
        # Convert indices to positional indices in X
        X_index_map = {idx: i for i, idx in enumerate(X.index)}
        train_idx = np.array([X_index_map[idx] for idx in train_indices if idx in X_index_map])
        val_idx = np.array([X_index_map[idx] for idx in val_indices if idx in X_index_map])
        
        folds.append((train_idx, val_idx))
        
        # Log fold size
        if log:
            log.debug(f"Repetition {repeat_idx+1}, Fold {fold_idx+1}/{n_splits}: "
                     f"Training set size {len(train_idx)}, Validation set size {len(val_idx)}")
    
    return folds


def filter_model_params(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out auxiliary parameters that are not actual model parameters.
    
    Args:
        model_name: Model name.
        params: Original parameter dictionary.
        
    Returns:
        Filtered parameter dictionary.
    """
    # Copy parameter dictionary to avoid modifying original
    filtered_params = params.copy()
    
    # General filtering: remove all auxiliary parameters starting with underscore
    filtered_params = {k: v for k, v in filtered_params.items() if not k.startswith('_')}
    
    # Model-specific filtering
    if model_name == 'mlp_reg':
        # Remove MLP-specific auxiliary parameters
        if 'n_layers' in filtered_params:
            del filtered_params['n_layers']
        
        # Remove all parameters in the form of n_units_lX
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
    Generate reproducible cross-validation folds.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        repeat_idx: Index of the current repetition.
        n_splits: Number of CV folds.
        random_seed: Base random seed.
        n_repeats: Total number of repetitions (for logging).
        task_logger: Logger for specific tasks.
        
    Returns:
        List containing training and validation indices.
    """
    # Use provided logger or default to None
    log = task_logger if task_logger else None
    
    # Use repetition index as random seed to ensure reproducibility
    fold_seed = random_seed + repeat_idx
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    
    # Generate all folds
    folds = list(kf.split(X))
    
    # Log fold size
    if log:
        for i, (train_idx, val_idx) in enumerate(folds):
            if n_repeats:
                log.debug(f"Repetition {repeat_idx+1}/{n_repeats}, Fold {i+1}/{n_splits}: "
                         f"Training set size {len(train_idx)}, Validation set size {len(val_idx)}")
            else:
                log.debug(f"Repetition {repeat_idx+1}, Fold {i+1}/{n_splits}: "
                         f"Training set size {len(train_idx)}, Validation set size {len(val_idx)}")
    
    return folds


# Seed generation functions
def generate_optimization_seed(random_seed: int, repeat_idx: int) -> int:
    """Generate random seed for Optuna optimization."""
    return random_seed + repeat_idx * 100  # Use 100 as constant multiplier for seed

def generate_repeat_seed(random_seed: int, repeat_idx: int) -> int:
    """Generate random seed for repetition training."""
    return random_seed + repeat_idx

def generate_fold_seed(random_seed: int, repeat_idx: int, fold_idx: int) -> int:
    """Generate random seed for fold training."""
    repeat_seed = generate_repeat_seed(random_seed, repeat_idx)
    return repeat_seed + fold_idx


# Directory creation functions
def create_model_result_directory(results_dir: Path, model_name: str) -> Path:
    """Create main directory for model results."""
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir

def create_repeat_result_directory(results_dir: Path, model_name: str, repeat_idx: int) -> Path:
    """Create directory for repetition training results."""
    repeat_dir = results_dir / model_name / f"repeat_{repeat_idx+1}"
    repeat_dir.mkdir(exist_ok=True, parents=True)
    return repeat_dir

def create_plots_directory(repeat_dir: Path) -> Path:
    """Create directory for scatter plots."""
    plots_dir = repeat_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    return plots_dir

def create_representative_model_directory(model_dir: Path) -> Path:
    """Create directory for representative model storage."""
    representative_dir = model_dir / "representative_model"
    representative_dir.mkdir(exist_ok=True, parents=True)
    return representative_dir


# Data processing functions
def prepare_train_test_data(X: pd.DataFrame, y: pd.Series, repeat_idx: int, 
                           random_seed: int, test_size: float, 
                           test_indices: np.ndarray = None) -> Tuple[np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare training and test data split.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        repeat_idx: Index of current repetition.
        random_seed: Base random seed.
        test_size: Proportion for test set.
        test_indices: Test set indices (if provided).
        
    Returns:
        train_indices, X_train, y_train, X_test, y_test
    """
    # Split training and test sets if no test indices are provided
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
    Prepare training data for a single fold, including standardization.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target variable.
        train_idx: Indices for training within the fold.
        val_idx: Indices for validation within the fold.
        X_test: Test feature matrix.
        
    Returns:
        X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, y_fold_train, y_fold_val, scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    # Get fold data
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, y_fold_train, y_fold_val, scaler

def train_fold_model(model: Any, X_fold_train_scaled: np.ndarray,
                    y_fold_train: pd.Series, n_threads: int = 1) -> Tuple[Any, float]:
    """
    Train a model for a single fold.

    Args:
        model: Model instance.
        X_fold_train_scaled: Standardized training features.
        y_fold_train: Training target variable.
        n_threads: Max threads for BLAS/OpenMP (threadpoolctl safety net).

    Returns:
        trained_model, training_time
    """
    import time
    from threadpoolctl import threadpool_limits

    start_time = time.time()
    with threadpool_limits(limits=n_threads):
        model.fit(X_fold_train_scaled, y_fold_train)
    training_time = time.time() - start_time
    return model, training_time

def predict_and_calculate_metrics(model: Any, X_fold_train_scaled: np.ndarray,
                                 y_fold_train: pd.Series, X_fold_val_scaled: np.ndarray,
                                 y_fold_val: pd.Series, X_test_scaled: np.ndarray,
                                 y_test: pd.Series) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform prediction and calculate all performance metrics.
    
    Args:
        model: Trained model.
        X_fold_train_scaled, y_fold_train: Training data.
        X_fold_val_scaled, y_fold_val: Validation data.
        X_test_scaled, y_test: Test data.
        
    Returns:
        train_metrics, val_metrics, test_metrics, y_fold_train_pred, y_fold_val_pred, y_test_pred
    """
    # Prediction
    y_fold_train_pred = model.predict(X_fold_train_scaled)
    y_fold_val_pred = model.predict(X_fold_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
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
    Save fold prediction results (plotting functionality removed to accelerate training).
    
    Returns:
        Empty dictionary (no images generated).
    """
    # Save prediction results
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
    
    return {}

# Statistical calculation functions
def calculate_repeat_statistics(all_repeat_results: List[Dict], task_type: str = None) -> Dict[str, Any]:
    """
    Calculate statistical information for all repetition results.
    
    Args:
        all_repeat_results: List containing results from all repetitions.
        task_type: Type of task (regression/classification).
        
    Returns:
        Statistical dictionary containing mean, standard deviation, and raw data.
    """
    if not all_repeat_results:
        return None
        
    # Check task type - priority to passed parameter, followed by data inference
    if task_type:
        is_classification = (task_type == 'classification')
    else:
        # Multiple ways to ensure accuracy
        first_result = all_repeat_results[0]
        is_classification = (
            'avg_test_accuracy' in first_result or 
            'avg_train_accuracy' in first_result or
            ('ensemble_metrics' in first_result and 'accuracy' in first_result.get('ensemble_metrics', {}))
        )
    
    # Extract metric values based on task type
    avg_results = {
        'training_time_values': [r['avg_training_time'] for r in all_repeat_results],
    }
    
    if is_classification:
        # Classification task metrics
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
        # Regression task metrics
        avg_results.update({
            'test_pearson_values': [r.get('avg_test_pearson', 0.0) for r in all_repeat_results],
            'val_pearson_values': [r.get('avg_val_pearson', 0.0) for r in all_repeat_results],
            'train_pearson_values': [r.get('avg_train_pearson', 0.0) for r in all_repeat_results],
            'ensemble_pearson_values': [r.get('ensemble_metrics', {}).get('pearson', 0.0) for r in all_repeat_results],
            'test_spearman_values': [r.get('avg_test_spearman', 0.0) for r in all_repeat_results],
            'test_mse_values': [r.get('avg_test_mse', 0.0) for r in all_repeat_results]
        })
    
    # Calculate statistical summary
    summary = {
        'model_name': all_repeat_results[0]['model_name'],
        'n_repeats': len(all_repeat_results),
        'avg_training_time': np.mean(avg_results['training_time_values']),
        'raw_values': avg_results
    }
    
    if is_classification:
        # Classification task statistics
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
        # Regression task statistics
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
    Find the repetition whose performance is closest to the average.
    
    Args:
        all_repeat_results: List containing results from all repetitions.
        avg_ensemble_metric: Average ensemble performance (classification: accuracy, regression: Pearson).
        
    Returns:
        closest_repeat_idx, closest_repeat, difference
    """
    # Check task type and extract corresponding ensemble metrics
    first_result = all_repeat_results[0]
    is_classification = 'avg_test_accuracy' in first_result
    
    if is_classification:
        ensemble_values = [r['ensemble_metrics']['accuracy'] for r in all_repeat_results]
    else:
        ensemble_values = [r['ensemble_metrics']['pearson'] for r in all_repeat_results]
    
    # Calculate gap between each repetition's performance and the average
    metric_diffs = [abs(val - avg_ensemble_metric) for val in ensemble_values]
    
    # Find the repetition index with the smallest gap
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
    
    Args:
        pheno_data: Phenotype data DataFrame
        target_trait: Target trait name
        cv_file: Path to CV file (optional)
        force_new_cv: Whether to force regeneration of CV folds
        cv_id_column: ID column name in phenotype data
        n_repeats: Number of repeats
        n_splits: Number of CV folds
        results_dir: Results directory
        logger: Logger instance (optional)
        
    Returns:
        Phenotype data with CV grouping columns
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
    
    # Try to get logger if not provided
    if logger is None:
        try:
            from .log_utils import logger as logutils_logger
            logger = logutils_logger
        except (ImportError, ValueError):
            pass

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
