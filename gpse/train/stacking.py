#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Genomic Prediction Stacking Ensemble Module
===========================================

Features:
- Build a stacking ensemble from existing trained models
- Use Pearson correlation as the primary regression evaluation and selection metric
- Automatically select the top N base models
- Generate meta-features with cross-validation
- Evaluate ensemble model performance
"""

import os
import sys
import time
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

from gpse.utils.log_utils import logger

class StackingEnsemble:
    """Stacking ensemble model."""
    
    def __init__(
        self,
        base_models_dir: str,
        top_n_models: int = 5,
        meta_model_type: str = 'ridge',
        cv_folds: int = 5,
        random_seed: int = 42,   # The answer to the ultimate question of life, the universe, and everything is 42.
        output_dir: Optional[str] = None,
        n_threads: int = 1,
        use_default_params: bool = False,
        task_type: str = 'regression'  # Task type
    ):
        """
        Initialize the stacking ensemble model.

        Parameters:
            base_models_dir: Directory containing base models.
            top_n_models: Number of top-performing models to select.
            meta_model_type: Meta-model type, currently supports 'ridge'.
            cv_folds: Number of cross-validation folds for meta-feature generation.
            random_seed: Random seed.
            output_dir: Output directory, defaults to base_models_dir/ensemble_stacking.
            n_threads: Number of threads.
            use_default_params: Whether to use default parameters.
            task_type: Task type ('regression' or 'classification').
        """
        self.base_models_dir = base_models_dir
        self.top_n_models = top_n_models
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.output_dir = output_dir or os.path.join(base_models_dir, 'ensemble_stacking')
        self.n_threads = n_threads
        self.use_default_params = use_default_params
        self.task_type = task_type  # Store task type

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize model and result containers
        self.base_models = {}
        self.selected_models = []
        self.meta_model = None
        self.meta_features_cols = []

        # Set random seed
        np.random.seed(self.random_seed)

        logger.info(f"Initialized stacking ensemble, selecting top {top_n_models} models")

    def load_and_select_models(self, model_names: Optional[List[str]] = None) -> List[str]:
        """Load and select the best-performing models."""
        logger.info("Loading and selecting models...")

        # Find model comparison results
        comparison_file = os.path.join(self.base_models_dir, 'model_comparison.csv')
        if not os.path.exists(comparison_file):
            raise FileNotFoundError(f"Model comparison file not found: {comparison_file}")

        # Load model comparison results
        comparison_df = pd.read_csv(comparison_file)
        logger.info(f"Found {len(comparison_df)} trained models")

        # Log model names from the CSV for debugging
        logger.info(f"Model names in CSV: {comparison_df['Model'].tolist()}")

        # Use explicitly provided model names when available
        if model_names:
            logger.info(f"Using specified model names: {model_names}")

            # Validate that specified models exist in the comparison file
            available_models = comparison_df['Model'].tolist()
            valid_models = []

            for name in model_names:
                # Remove a possible _reg suffix
                base_name = name.replace('_reg', '')
                # Look for a matching model name in the CSV
                matches = comparison_df[comparison_df['Model'].str.lower() == base_name.lower()]
                if not matches.empty:
                    valid_models.append(name)
                else:
                    logger.warning(f"Specified model {name} is not in the comparison file; skipping")

            if not valid_models:
                raise ValueError(f"No valid specified models found. Available models: {available_models}")

            self.selected_models = valid_models
            logger.info(f"Using {len(self.selected_models)} specified models: {', '.join(self.selected_models)}")

        else:
            # If no model names are specified, sort by task-specific metrics
            if self.task_type == 'classification':
                # Classification: sort by Ensemble Accuracy, falling back to Test Accuracy
                if 'Ensemble Accuracy' in comparison_df.columns:
                    sort_column = 'Ensemble Accuracy'
                    logger.info("Selecting classification models sorted by Ensemble Accuracy")
                else:
                    sort_column = 'Test Accuracy'
                    logger.info("Selecting classification models sorted by Test Accuracy")
            else:
                # Regression: sort by Ensemble Pearson, falling back to Test Pearson
                if 'Ensemble Pearson' in comparison_df.columns:
                    sort_column = 'Ensemble Pearson'
                    logger.info("Selecting regression models sorted by Ensemble Pearson")
                else:
                    sort_column = 'Test Pearson'
                    logger.info("Selecting regression models sorted by Test Pearson")

            comparison_df = comparison_df.sort_values(sort_column, ascending=False)

            # Select the top N models
            n_models = min(self.top_n_models, len(comparison_df))
            self.selected_models = comparison_df['Model'].iloc[:n_models].tolist()
            logger.info(f"Selected top {n_models} models: {', '.join(self.selected_models)}")

        # Load selected models
        for model_name in self.selected_models:
            # Determine model suffix and search paths by task type
            if self.task_type == 'classification':
                # Classification model: model name + _clf suffix
                model_with_suffix = f"{model_name}_clf"
            else:
                # Regression model: model name + _reg suffix
                model_with_suffix = f"{model_name}_reg"

            # Build search paths in priority order
            model_paths = [
                # 1. Representative model path, highest priority
                os.path.join(self.base_models_dir, model_with_suffix, "representative_model", "model.pkl"),

                # 2. Direct model path
                os.path.join(self.base_models_dir, model_with_suffix, 'model.pkl'),

                # 3. Fallback: model path without suffix
                os.path.join(self.base_models_dir, model_name, "representative_model", "model.pkl"),
                os.path.join(self.base_models_dir, model_name, 'model.pkl'),
            ]

            # 4. Add repeated experiment paths
            for repeat_idx in range(1, 51):
                model_paths.extend([
                    os.path.join(self.base_models_dir, model_with_suffix, f"repeat_{repeat_idx}", "model.pkl"),
                    os.path.join(self.base_models_dir, model_name, f"repeat_{repeat_idx}", "model.pkl")
                ])
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.base_models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model successfully: {model_name}, path: {model_path}")
                    model_loaded = True
                    break

            if not model_loaded:
                logger.warning(f"Model file not found. Tried paths: {model_paths}")
                self.selected_models.remove(model_name)

        if not self.base_models:
            raise ValueError("No models were loaded successfully")

        logger.info(f"Final selected models ({len(self.selected_models)}): {', '.join(self.selected_models)}")
        self.meta_features_cols = self.selected_models.copy()
        
        return self.selected_models
    
    def create_meta_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create the meta-feature matrix.

        Parameters:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Optional test feature matrix.

        Returns:
            Training meta-features and optional test meta-features.
        """
        logger.info(f"Creating meta-features with {self.cv_folds}-fold cross-validation...")

        # Create meta-feature matrix for training
        meta_train = np.zeros((X_train.shape[0], len(self.selected_models)))

        # Create test meta-feature matrix when test data is provided
        meta_test = None
        if X_test is not None:
            meta_test = np.zeros((X_test.shape[0], len(self.selected_models)))

        # Generate meta-features for each base model
        for i, model_name in enumerate(self.selected_models):
            logger.info(f"Generating meta-features for model {model_name}")

            # Use the already-loaded model instead of loading it again
            model = self.base_models[model_name]

            # Check whether the model is a (model, scaler) tuple
            is_tuple_model = isinstance(model, tuple) and len(model) == 2

            # Use K-fold cross-validation to generate training meta-features
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

            for train_idx, val_idx in kf.split(X_train):
                # Split data
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]

                if is_tuple_model:
                    # Unpack model and scaler
                    actual_model, scaler = model
                    # Clone model
                    fold_model = clone(actual_model)
                    # Scale data
                    X_fold_train_scaled = scaler.transform(X_fold_train)
                    X_fold_val_scaled = scaler.transform(X_fold_val)
                    # Train model
                    fold_model.fit(X_fold_train_scaled, y_fold_train)
                    # Predict
                    meta_train[val_idx, i] = fold_model.predict(X_fold_val_scaled)
                else:
                    # Original model path without scaler
                    fold_model = clone(model)
                    fold_model.fit(X_fold_train, y_fold_train)
                    meta_train[val_idx, i] = fold_model.predict(X_fold_val)

            # If test data is provided, train on the full training set and predict test data
            if X_test is not None:
                if is_tuple_model:
                    # Unpack model and scaler
                    actual_model, scaler = model
                    # Scale training and test data
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    # Train model
                    actual_model.fit(X_train_scaled, y_train)
                    # Predict
                    meta_test[:, i] = actual_model.predict(X_test_scaled)
                else:
                    # Original model path without scaler
                    model.fit(X_train, y_train)
                    meta_test[:, i] = model.predict(X_test)

        logger.info(f"Meta-feature creation completed. Training meta-feature shape: {meta_train.shape}")
        if meta_test is not None:
            logger.info(f"Test meta-feature shape: {meta_test.shape}")
        
        return meta_train, meta_test
    
    def fit_meta_model(self, meta_train: np.ndarray, y_train: pd.Series):
        """
        Train the meta-model.

        Parameters:
            meta_train: Training meta-features.
            y_train: Training labels.
        """
        logger.info(
            f"Training {self.task_type} meta-model ({self.meta_model_type}) "
            f"with {self.n_threads} threads..."
        )

        # Create task-specific meta-models
        if self.task_type == 'classification':
            # Classification: use logistic regression as the meta-model
            if self.meta_model_type == 'ridge':
                from sklearn.linear_model import LogisticRegression

                if self.use_default_params:
                    logger.info("Using default LogisticRegression meta-model parameters...")
                    lr_model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
                else:
                    lr_model = LogisticRegression(
                        C=1.0,
                        random_state=self.random_seed,
                        max_iter=1000,
                        multi_class='ovr'  # Multiclass strategy
                    )

                # Check whether n_jobs is supported
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
                raise ValueError(f"Unsupported meta-model type for classification: {self.meta_model_type}")
        else:
            # Regression path
            if self.meta_model_type == 'ridge':
                from sklearn.linear_model import Ridge

                if self.use_default_params:
                    logger.info("Using default Ridge meta-model parameters...")
                    ridge_model = Ridge(random_state=self.random_seed)
                else:
                    ridge_model = Ridge(alpha=1.0, random_state=self.random_seed)

                # Use n_jobs when Ridge supports it
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
                raise ValueError(f"Unsupported meta-model type for regression: {self.meta_model_type}")

        # Train meta-model
        self.meta_model.fit(meta_train, y_train)

        logger.info("Meta-model training completed")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with the stacking ensemble.

        Parameters:
            X: Feature matrix.

        Returns:
            (predictions, predicted probabilities). Classification returns probabilities;
            regression returns None for probabilities.
        """
        if self.meta_model is None:
            raise ValueError("Meta-model is not trained; call fit first")

        # Create meta-features
        meta_features = np.zeros((X.shape[0], len(self.selected_models)))

        # Use predictions from each base model as meta-features
        for i, model_name in enumerate(self.selected_models):
            model = self.base_models[model_name]

            # Check whether the model is a (model, scaler) tuple
            if isinstance(model, tuple) and len(model) == 2:
                actual_model, scaler = model
                X_scaled = scaler.transform(X)
                if self.task_type == 'classification' and hasattr(actual_model, 'predict_proba'):
                    # Classification: use the maximum predicted probability as the meta-feature
                    probas = actual_model.predict_proba(X_scaled)
                    meta_features[:, i] = np.max(probas, axis=1)
                else:
                    meta_features[:, i] = actual_model.predict(X_scaled)
            else:
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    # Classification: use the maximum predicted probability as the meta-feature
                    probas = model.predict_proba(X)
                    meta_features[:, i] = np.max(probas, axis=1)
                else:
                    meta_features[:, i] = model.predict(X)

        # Use the meta-model for final prediction
        if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
            # Classification: return predicted labels and probabilities
            y_pred_proba = self.meta_model.predict_proba(meta_features)
            y_pred = self.meta_model.predict(meta_features)
            return y_pred, y_pred_proba
        else:
            # Regression: return predictions only
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
        Train the stacking ensemble model.

        Parameters:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Optional test feature matrix.
            y_test: Optional test labels.
            model_names: Optional list of model names to consider. None means all trained models.

        Returns:
            Evaluation results.
        """
        # Record start time
        start_time = time.time()

        # Load and select models
        self.load_and_select_models(model_names)

        # Create meta-features
        meta_train, meta_test = self.create_meta_features(X_train, y_train, X_test)

        # Train meta-model
        self.fit_meta_model(meta_train, y_train)

        # Save model
        logger.info("Saving stacking ensemble model...")
        joblib.dump(self, os.path.join(self.output_dir, 'stacking_ensemble_model.pkl'))

        # Evaluate performance
        results = {}

        # Evaluate on the training set
        if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
            y_train_pred = self.meta_model.predict(meta_train)
            y_train_proba = self.meta_model.predict_proba(meta_train)
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
        else:
            y_train_pred = self.meta_model.predict(meta_train)
            train_metrics = self.calculate_metrics(y_train, y_train_pred)

        results['train_metrics'] = train_metrics

        logger.info("Training set performance:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.6f}")

        # Evaluate on the test set when provided
        if X_test is not None and y_test is not None:
            if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
                y_test_pred = self.meta_model.predict(meta_test)
                y_test_proba = self.meta_model.predict_proba(meta_test)
                test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
            else:
                y_test_pred = self.meta_model.predict(meta_test)
                test_metrics = self.calculate_metrics(y_test, y_test_pred)

            results['test_metrics'] = test_metrics

            logger.info("Test set performance:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.6f}")

            # Plotting has been removed to speed up training

        # Feature importance analysis
        if hasattr(self.meta_model[-1], 'coef_'):
            importances = np.abs(self.meta_model[-1].coef_)
            sorted_idx = np.argsort(importances)[::-1]
            model_importances = {
                self.meta_features_cols[i]: importances[i] for i in sorted_idx
            }

            results['model_importances'] = model_importances
            logger.info("Model importance:")
            for model, importance in model_importances.items():
                logger.info(f"{model}: {importance:.6f}")

            # Plotting has been removed to speed up training

        # Add training time
        training_time = time.time() - start_time
        results['training_time'] = training_time
        logger.info(f"Training time: {training_time:.2f} seconds")

        # Add empty cv_scores, using None instead of NaN
        results['cv_scores'] = {'mean': None, 'std': None}

        # Save results
        joblib.dump(results, os.path.join(self.output_dir, 'stacking_results.pkl'))
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate performance metrics by task type.

        Parameters:
            y_true: True values.
            y_pred: Predicted values.
            y_pred_proba: Predicted probabilities, optional for classification.

        Returns:
            Performance metric dictionary.
        """
        if self.task_type == 'classification':
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            try:
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }

                # Calculate AUC when predicted probabilities are provided
                if y_pred_proba is not None:
                    from sklearn.metrics import roc_auc_score
                    try:
                        n_classes = len(np.unique(y_true))
                        if n_classes == 2:  # Binary classification
                            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:  # Multiclass classification
                            metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    except Exception as e:
                        logger.warning(f"Failed to calculate AUC: {e}")
                        metrics['auc'] = 0.0

                return metrics
            except Exception as e:
                logger.error(f"Failed to calculate classification metrics: {e}")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        else:
            # Regression metrics
            return {
                'pearson': pearsonr(y_true, y_pred)[0],
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }
    
