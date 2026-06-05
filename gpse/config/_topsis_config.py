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
from typing import Any, Tuple

from loguru import logger as main_logger

from gpse.config.constants import ModelConstants
from gpse.utils.configuration import load_topsis_config
from gpse.utils.genomic_utils import create_representative_model_directory
from threadpoolctl import threadpool_limits


def _format_weight(weight: Any) -> str:
    """Format a YAML weight value for TOPSISEvaluator's comma-string interface."""
    if isinstance(weight, float):
        return f"{weight:g}"
    return str(weight)


def _parse_topsis_task_config(task_type: str) -> Tuple[list, list, str]:
    """Parse and validate the TOPSIS config for a task type."""
    config = load_topsis_config()
    task_configs = config.get("tasks", {})
    if not isinstance(task_configs, dict):
        raise ValueError("TOPSIS config must define 'tasks' as a mapping")

    task_key = "classification" if task_type == "classification" else "regression"
    task_config = task_configs.get(task_key)

    if not isinstance(task_config, dict):
        raise ValueError(f"Missing TOPSIS configuration for task '{task_key}'")

    criteria_config = task_config.get("criteria")
    if not isinstance(criteria_config, list) or not criteria_config:
        raise ValueError(f"TOPSIS task '{task_key}' must define a non-empty criteria list")

    criteria = []
    criteria_types = []
    weights = []
    for index, item in enumerate(criteria_config, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"TOPSIS criterion #{index} for task '{task_key}' must be a mapping")

        name = item.get("name")
        criterion_type = item.get("type")
        weight = item.get("weight")

        if not name:
            raise ValueError(f"TOPSIS criterion #{index} for task '{task_key}' is missing 'name'")
        if criterion_type not in {"max", "min"}:
            raise ValueError(
                f"TOPSIS criterion '{name}' for task '{task_key}' must use type 'max' or 'min'"
            )
        if weight is None:
            raise ValueError(f"TOPSIS criterion '{name}' for task '{task_key}' is missing 'weight'")
        try:
            float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TOPSIS criterion '{name}' for task '{task_key}' has a non-numeric weight"
            ) from exc

        criteria.append(str(name))
        criteria_types.append(str(criterion_type))
        weights.append(weight)

    manual_weights = ",".join(_format_weight(weight) for weight in weights)
    return criteria, criteria_types, manual_weights


def get_topsis_configuration(self) -> Tuple[list, list, str]:
    """
    Get TOPSIS evaluation configuration based on task type.

    Returns
    -------
    Tuple[list, list, str]
        (criteria, criteria_types, manual_weights)
    """
    criteria, criteria_types, manual_weights = _parse_topsis_task_config(self.task_type)
    task_name = "Classification" if self.task_type == "classification" else "Regression"
    main_logger.info(
        f"TOPSIS config: {task_name} task - "
        f"{' : '.join(criteria)}; weights={manual_weights}"
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

        with threadpool_limits(limits=self.n_threads):
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
