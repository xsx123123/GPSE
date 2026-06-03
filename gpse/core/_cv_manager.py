#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Validation Manager Module
================================
Manages CV fold preparation and generation from files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    from loguru import logger as main_logger
except ImportError:
    main_logger = None

try:
    from ..utils.genomic_utils import prepare_cv_data
except ImportError:
    from utils.genomic_utils import prepare_cv_data


def prepare_cv_folds(self, pheno_data: pd.DataFrame, target_trait: str) -> pd.DataFrame:
    """
    Prepare cross-validation fold group information and save to file.

    Parameters
    ----------
    pheno_data : pd.DataFrame
        Phenotype DataFrame.
    target_trait : str
        Target trait name.

    Returns
    -------
    pd.DataFrame
        Phenotype data with added CV group columns.
    """
    if self.cv_file is not None:
        cv_file_path = Path(self.cv_file)
    else:
        cv_dir = self.results_dir / "cv_folds"
        cv_dir.mkdir(exist_ok=True, parents=True)
        cv_file_path = cv_dir / f"{target_trait}_cv_{self.n_repeats}x{self.n_splits}.csv"

    if cv_file_path.exists() and not self.force_new_cv:
        main_logger.info(f"Loading existing CV fold file: {cv_file_path}")
        cv_pheno_data = pd.read_csv(cv_file_path, index_col=0)
    else:
        main_logger.info(f"Generating new CV folds and saving to: {cv_file_path}")
        pheno_data_copy = pheno_data.copy()

        if self.cv_id_column in pheno_data_copy.columns:
            pheno_data_copy.set_index(self.cv_id_column, inplace=True)

        cv_pheno_data = prepare_cv_data(
            pheno_data_copy, str(cv_file_path), self.n_repeats, self.n_splits
        )

    return cv_pheno_data


def generate_cv_folds_from_file(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    cv_pheno_data: pd.DataFrame,
    repeat_idx: int,
    task_logger=None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate cross-validation folds from a CV file.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    cv_pheno_data : pd.DataFrame
        Phenotype data containing CV groups.
    repeat_idx : int
        Repeat index.
    task_logger
        Task-specific logger.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_idx, val_idx) tuples.
    """
    log = task_logger if task_logger else main_logger

    cv_col = f"cv{repeat_idx}"

    if cv_col not in cv_pheno_data.columns:
        log.error(
            f"CV column {cv_col} does not exist in CV file, cannot generate folds"
        )
        raise ValueError(f"CV column {cv_col} does not exist")

    folds = []
    for fold_idx in range(self.n_splits):
        val_mask = cv_pheno_data[cv_col] == fold_idx
        val_indices = cv_pheno_data.index[val_mask].tolist()

        train_mask = cv_pheno_data[cv_col] != fold_idx
        train_indices = cv_pheno_data.index[train_mask].tolist()

        X_index_map = {idx: i for i, idx in enumerate(X.index)}
        train_idx = np.array(
            [X_index_map[idx] for idx in train_indices if idx in X_index_map]
        )
        val_idx = np.array(
            [X_index_map[idx] for idx in val_indices if idx in X_index_map]
        )

        folds.append((train_idx, val_idx))

        log.debug(
            f"Repeat {repeat_idx + 1}/{self.n_repeats}, "
            f"Fold {fold_idx + 1}/{self.n_splits}: "
            f"Train size {len(train_idx)}, Val size {len(val_idx)}"
        )

    return folds
