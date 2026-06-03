#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TOPSIS Configuration and Environment Logging Module
====================================================
Provides TOPSIS evaluation setup, representative model saving, and environment logging.
"""

import os
import traceback
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple

try:
    from loguru import logger as main_logger
except ImportError:
    main_logger = None

try:
    from ..config import ModelConstants
    from ..utils.genomic_utils import create_representative_model_directory
except ImportError:
    from config import ModelConstants
    from utils.genomic_utils import create_representative_model_directory


def get_topsis_configuration(self) -> Tuple[list, list, str]:
    """
    Get TOPSIS evaluation configuration based on task type.

    Returns
    -------
    Tuple[list, list, str]
        (criteria, criteria_types, manual_weights)
    """
    if self.task_type == "classification":
        criteria = ["Test Accuracy", "Test Accuracy (std)"]
        criteria_types = ["max", "min"]
        manual_weights = "0.8,0.2"
        main_logger.info(
            "TOPSIS config: Classification task - "
            "Test Accuracy : Test Accuracy (std) = 8:2"
        )
    else:
        criteria = ["Test Pearson", "Test Pearson (std)"]
        criteria_types = ["max", "min"]
        manual_weights = "0.8,0.2"
        main_logger.info(
            "TOPSIS config: Regression task - "
            "Test Pearson : Test Pearson (std) = 8:2"
        )

    return criteria, criteria_types, manual_weights


def _save_representative_model(
    self,
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Path,
    repeat_info: dict,
) -> str:
    """
    Train and save a representative model.

    Parameters
    ----------
    model_name : str
        Model name.
    params : dict
        Model parameters.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_dir : Path
        Model directory.
    repeat_info : dict
        Repeat information dictionary.

    Returns
    -------
    str
        Model save path, or None if saving failed.
    """
    representative_model_dir = create_representative_model_directory(model_dir)

    try:
        model = self.create_model(model_name, params)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model.fit(X_scaled, y)

        model_path = representative_model_dir / ModelConstants.model_pkl_file
        joblib.dump((model, scaler), model_path)

        info_path = representative_model_dir / ModelConstants.model_info_file
        with open(info_path, "w") as f:
            json.dump(repeat_info, f, indent=2)

        return str(model_path)

    except Exception as e:
        main_logger.error(f"Saving representative model failed: {str(e)}")
        main_logger.error(traceback.format_exc())
        return None


def log_environment_settings(self) -> None:
    """Log environment settings."""
    main_logger.info("Initializing GenomicPredictorV2, base configuration:")
    main_logger.info(f"- Results directory: {self.results_dir}")
    main_logger.info(f"- Logs directory: {self.logs_dir}")
    main_logger.info(f"- Random seed: {self.random_seed}")
    main_logger.info(f"- Threads per model: {self.n_threads}")
    main_logger.info(f"- Parallel jobs: {self.max_parallel_jobs}")
    main_logger.info(f"- CV folds: {self.n_splits}")
    main_logger.info(f"- Repeats: {self.n_repeats}")
    main_logger.info(f"- Use default params: {self.use_default_params}")

    for env_var in ModelConstants.thread_env_vars:
        main_logger.info(f"{env_var}={os.environ.get(env_var, 'not set')}")
