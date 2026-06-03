#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Genomic Classification Prediction Module
========================================

Features:
- Performance metric calculation for classification tasks (Accuracy, Precision, Recall, F1-score, AUC, etc.)
- Classification label encoding and preprocessing
- Classification model prediction result processing
- Robust conversion from probabilities to labels
- Classification scatter plot generation (if needed)

Pure classification function module extracted from genomic_prediction_v2_class.py
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
from pathlib import Path

from loguru import logger as main_logger

from gpse.models.classification_models import ClassificationModelOptimizer

class GenomicClassifier:
    """Genomic Classification Prediction Utility Class"""
    
    def __init__(self, n_classes: int, results_dir: str = None, random_state: int = None, n_threads: int = 1):
        """
        Initialize the classifier.
        
        Args:
            n_classes: Number of classes
            results_dir: Directory to save results
            random_state: Random state for reproducibility
            n_threads: Number of threads
        """
        self.n_classes = n_classes
        self.results_dir = Path(results_dir) if results_dir else Path(".")
        self.label_encoder = None
        self.random_state = random_state
        self.n_threads = n_threads
        
        # Initialize classification model optimizer
        self.classification_optimizer = ClassificationModelOptimizer(
            random_state=random_state, 
            n_threads=n_threads, 
            n_classes=n_classes
        )
        
    def prepare_classification_labels(self, y: pd.Series, results_dir: Path) -> pd.Series:
        """
        Prepare classification labels, performing necessary encoding.
        
        Args:
            y: Original labels
            results_dir: Directory to save results
            
        Returns:
            Encoded labels
        """
        # Check if labels are numeric
        unique_labels = y.unique()
        main_logger.info(f"Original classification labels: {sorted(unique_labels)}")
        
        # Check if labels are continuous integers starting from 0
        sorted_labels = sorted(unique_labels)
        expected_labels = list(range(len(sorted_labels)))
        
        # Force label encoding to ensure they start from 0
        # This solves issues where models like XGBoost require labels to start from 0
        main_logger.info("Encoding labels to ensure they are continuous integers starting from 0...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_result = pd.Series(y_encoded, index=y.index)
        
        # Save label encoder
        encoder_path = results_dir / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        main_logger.info(f"Label encoder saved to: {encoder_path}")
        main_logger.info(f"Encoded labels: {sorted(y_result.unique())}")
        main_logger.info(f"Label mapping: {dict(zip(sorted_labels, sorted(y_result.unique())))}")
        
        # Validate number of classes
        actual_n_classes = len(unique_labels)
        if self.n_classes != actual_n_classes:
            main_logger.warning(f"Specified number of classes ({self.n_classes}) does not match actual ({actual_n_classes}), using actual")
            self.n_classes = actual_n_classes
            
        return y_result
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate classification performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        # Ensure input is in correct format
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # If y_pred is a probability matrix (n_samples, n_classes), automatically convert to labels and fill y_pred_proba
        if y_pred.ndim == 2 and y_pred.shape[0] == y_true.shape[0]:
            if y_pred_proba is None:
                y_pred_proba = y_pred
            y_pred = np.argmax(y_pred, axis=1)

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        # Convert floating point predictions to integers
        if y_pred.dtype.kind == 'f':
            y_pred = y_pred.astype(int)
            
        # Check if shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")
        
        # Classification metrics
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            # If metric calculation fails, return default values
            main_logger.error(f"Failed to calculate classification metrics: {e}")
            main_logger.debug(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
            main_logger.debug(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            main_logger.debug(f"y_true unique: {np.unique(y_true)}")
            main_logger.debug(f"y_pred unique: {np.unique(y_pred)}")
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # If predicted probabilities are provided, calculate AUC and log loss
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:  # Binary classification
                    if y_pred_proba.ndim == 2:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                else:  # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception as e:
                main_logger.warning(f"Failed to calculate AUC or log_loss: {e}")
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
        Robustly convert probabilities to labels.
        
        Args:
            proba: Predicted probabilities
            n_samples: Expected number of samples
            n_classes_hint: Hint for the number of classes
            
        Returns:
            Array of predicted labels
        """
        # Unify as ndarray
        if isinstance(proba, (list, tuple)):
            try:
                proba = np.column_stack(proba)
            except Exception:
                proba = np.asarray(proba)
        arr = np.asarray(proba)
        
        # Check shape suitability
        if arr.ndim == 2:
            if arr.shape[0] == n_samples:
                P = arr
            elif arr.shape[1] == n_samples:
                P = arr.T
            else:
                # Try to reshape based on sample count
                total = arr.size
                if total % n_samples == 0:
                    k = total // n_samples
                    P = arr.reshape(n_samples, k)
                else:
                    # Force truncation/padding row-wise
                    k = n_classes_hint if n_classes_hint else max(2, arr.shape[-1])
                    flat = arr.ravel()
                    need = n_samples * k
                    if flat.size >= need:
                        P = flat[:need].reshape(n_samples, k)
                    else:
                        pad = np.tile(flat[-1], need - flat.size)
                        P = np.concatenate([flat, pad]).reshape(n_samples, k)
        elif arr.ndim == 1:
            # Binary classification 1D probability
            if arr.shape[0] == n_samples:
                P = np.vstack([1 - arr, arr]).T
            else:
                # Abnormal length, flatten and reshape
                k = n_classes_hint if n_classes_hint else 2
                need = n_samples * k
                flat = arr.ravel()
                if flat.size >= need:
                    P = flat[:need].reshape(n_samples, k)
                else:
                    pad = np.tile(flat[-1], need - flat.size)
                    P = np.concatenate([flat, pad]).reshape(n_samples, k)
        else:
            # Other unconventional dimensions, fallback to binary uniform probability
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
        Ensure the prediction result length is correct.
        
        Args:
            pred: Original predicted results
            proba: Predicted probabilities
            n_samples: Expected number of samples
            
        Returns:
            Predicted results with correct length
        """
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 1 and pred_arr.shape[0] == n_samples:
            return pred_arr
        if proba is not None:
            # Generate robustly from probabilities
            return self.proba_to_labels_robust(proba, n_samples, self.n_classes)
        # Fallback: truncate or pad to correct length
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
        Generate classification labels and probabilities from the model.
        
        Args:
            model: Trained model
            X_scaled: Standardized features
            
        Returns:
            Tuple of (Predicted Labels, Predicted Probabilities)
        """
        y_pred_proba = None
        
        # Prefer predict_proba; fall back to predict on failure
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
        
        # Ensure prediction length is correct
        y_pred = self.ensure_prediction_length(y_pred, y_pred_proba, len(X_scaled))
        
        return y_pred, y_pred_proba
    
    def decode_labels_if_needed(self, labels: np.ndarray) -> np.ndarray:
        """
        If a label encoder was used, decode labels back to their original form.
        
        Args:
            labels: Encoded labels
            
        Returns:
            Decoded labels
        """
        if self.label_encoder is not None:
            try:
                return self.label_encoder.inverse_transform(labels)
            except Exception as e:
                main_logger.warning(f"Label decoding failed: {e}")
                return labels
        return labels
    
    def create_classification_summary(
        self, 
        all_repeat_results: list,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Create a results summary for classification tasks.
        
        Args:
            all_repeat_results: List of results from all repetitions
            model_name: Model name
            
        Returns:
            Classification summary dictionary
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
        
        # Summary for classification task
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
        Log classification results.
        
        Args:
            fold_idx: Fold index
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            test_metrics: Test set metrics
            logger: Logger instance
        """
        if logger is None:
            logger = main_logger
            
        logger.info(f"Fold {fold_idx+1} results:")
        logger.info(f"  Training set Accuracy: {train_metrics['accuracy']:.6f}")
        logger.info(f"  Training set F1: {train_metrics['f1']:.6f}")
        logger.info(f"  Validation set Accuracy: {val_metrics['accuracy']:.6f}")
        logger.info(f"  Validation set F1: {val_metrics['f1']:.6f}")
        logger.info(f"  Test set Accuracy: {test_metrics['accuracy']:.6f}")
        logger.info(f"  Test set F1: {test_metrics['f1']:.6f}")
        if 'auc' in test_metrics:
            logger.info(f"  Test set AUC: {test_metrics['auc']:.6f}")
    
    def create_classification_comparison_row(
        self, 
        model_name: str, 
        results: dict
    ) -> dict:
        """
        Create row data for the classification model comparison table.
        
        Args:
            model_name: Model name
            results: Results dictionary
            
        Returns:
            Comparison table row dictionary
        """
        # Clean model name
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
        
        # Include AUC if available
        if 'avg_test_auc' in results:
            row['Test AUC'] = results.get('avg_test_auc', 0.0)
            
        return row
    
    def create_classification_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        Create classification model instance.
        
        Args:
            model_name: Model name
            params: Model parameters
            
        Returns:
            Model instance
        """
        return self.classification_optimizer.create_classification_model(model_name, params)

    def get_classification_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for classification models.
        
        Args:
            model_name: Model name
            
        Returns:
            Default parameters dictionary
        """
        return self.classification_optimizer.get_classification_default_params(model_name)

    def get_classification_param_func(self, model_name: str) -> callable:
        """
        Get parameter function for classification models.
        
        Args:
            model_name: Model name
            
        Returns:
            Parameter function
        """
        return self.classification_optimizer.get_param_func(model_name)

    def filter_classification_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter classification model parameters.
        
        Args:
            model_name: Model name
            params: Original parameters
            
        Returns:
            Filtered parameters
        """
        return self.classification_optimizer.filter_classification_params(model_name, params)
