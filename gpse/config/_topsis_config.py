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
from gpse.train._feature_selection import (
    make_model_artifact,
)
from gpse.train._model_pipeline import build_training_pipeline


def _format_weight(weight: Any) -> str:
    """Format a YAML weight value for TOPSISEvaluator's comma-string interface."""
    if isinstance(weight, float):
        return f"{weight:g}"
    return str(weight)


def _parse_topsis_task_config(
    task_type: str, user_config_path: str | None = None
) -> Tuple[list, list, str]:
    """Parse and validate the TOPSIS config for a task type.

    Only criteria with weight > 0 are returned for the actual TOPSIS
    computation. Weight-0 criteria are reference-only.
    """
    config = load_topsis_config(user_config_path)
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
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TOPSIS criterion '{name}' for task '{task_key}' has a non-numeric weight"
            ) from exc

        if weight > 0:
            criteria.append(str(name))
            criteria_types.append(str(criterion_type))
            weights.append(weight)

    if not criteria:
        raise ValueError(
            f"TOPSIS task '{task_key}' has no criteria with weight > 0; "
            "at least one criterion must have a positive weight"
        )

    manual_weights = ",".join(_format_weight(weight) for weight in weights)
    return criteria, criteria_types, manual_weights


def get_all_topsis_criteria(task_type: str, user_config_path: str | None = None) -> list[dict]:
    """Return the full criteria list (including weight-0) for display."""
    config = load_topsis_config(user_config_path)
    task_key = "classification" if task_type == "classification" else "regression"
    task_config = config.get("tasks", {}).get(task_key, {})
    return task_config.get("criteria", [])


def get_topsis_configuration(self) -> Tuple[list, list, str]:
    """
    Get TOPSIS evaluation configuration based on task type.

    Returns
    -------
    Tuple[list, list, str]
        (criteria, criteria_types, manual_weights)
    """
    user_config = getattr(self, "topsis_config", None)
    criteria, criteria_types, manual_weights = _parse_topsis_task_config(
        self.task_type, user_config
    )
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
        pipeline = build_training_pipeline(
            self.create_model(model_name, params),
            task_type=self.task_type,
            model_name=model_name,
            feature_selection_config=self.feature_selection_config,
            genotype_imputation_config=self.genotype_imputation_config,
        )

        with threadpool_limits(limits=self.n_threads):
            pipeline.fit(X, y)
        preprocessor = pipeline.named_steps["preprocess"]

        model_path = representative_model_dir / ModelConstants.model_pkl_file
        joblib.dump(
            make_model_artifact(
                pipeline.named_steps["model"],
                preprocessor.scaler_,
                preprocessor.selector_,
                self.feature_selection_config,
                preprocessor.selected_features_,
                imputer=preprocessor.imputer_,
                imputation_config=self.genotype_imputation_config,
                task_type=self.task_type,
            ),
            model_path,
        )

        info_path = representative_model_dir / ModelConstants.model_info_file
        with open(info_path, "w") as f:
            json.dump(
                {
                    **repeat_info,
                    "feature_selection": self.feature_selection_config.as_dict(),
                    "genotype_imputation": self.genotype_imputation_config.as_dict(),
                    "selected_feature_count": len(preprocessor.selected_features_),
                },
                f,
                indent=2,
            )

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
    main_logger.info(f"- Model workers: {self.max_parallel_jobs}")
    main_logger.info(f"- Repeat workers: {self.repeat_workers}")
    main_logger.info(f"- Train folds: {self.n_splits}")
    main_logger.info(f"- Repeats: {self.n_repeats}")
    main_logger.info(f"- Use default params: {self.use_default_params}")

    for env_var in ModelConstants.thread_env_vars:
        main_logger.info(f"{env_var}={os.environ.get(env_var, 'not set')}")

    user_config = getattr(self, "topsis_config", None)
    if user_config:
        main_logger.info(f"- TOPSIS config: {user_config} (user-supplied)")
    else:
        main_logger.info("- TOPSIS config: built-in default (gpse/config/topsis.yaml)")
    try:
        all_criteria = get_all_topsis_criteria(self.task_type, user_config)
        active = [c for c in all_criteria if float(c.get("weight", 0)) > 0]
        inactive = [c for c in all_criteria if float(c.get("weight", 0)) == 0]
        main_logger.info(
            f"- TOPSIS active criteria ({len(active)}): "
            + ", ".join(f"{c['name']}({c['type']},w={c['weight']})" for c in active)
        )
        if inactive:
            main_logger.info(
                f"- TOPSIS reference criteria ({len(inactive)}, weight=0): "
                + ", ".join(c["name"] for c in inactive)
            )
    except Exception:
        main_logger.debug("Could not load TOPSIS criteria for display")
