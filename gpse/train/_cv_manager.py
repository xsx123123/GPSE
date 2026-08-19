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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from loguru import logger as main_logger

from gpse.utils.genomic_utils import prepare_cv_data


def _build_structure_groups(X: pd.DataFrame, n_splits: int, random_seed: int, requested_clusters=None):
    """Cluster training samples from genotype PCs for group-aware CV."""
    n_components = min(10, X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        raise ValueError("Structure-aware CV requires at least two samples and one feature")
    n_clusters = max(n_splits, requested_clusters or n_splits)
    if n_clusters > len(X):
        raise ValueError(
            "Structure-aware CV needs at least n_splits samples/clusters; "
            f"got samples={len(X)}, n_splits={n_splits}"
        )
    pcs = PCA(n_components=n_components, random_state=random_seed).fit_transform(
        StandardScaler().fit_transform(X)
    )
    groups = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit_predict(pcs)
    return groups


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
        main_logger.info(f"Loaded CV folds from: {cv_file_path}")
    else:
        main_logger.info(f"Generating new CV folds and saving to: {cv_file_path}")
        pheno_data_copy = pheno_data.copy()

        if self.cv_id_column in pheno_data_copy.columns:
            pheno_data_copy.set_index(self.cv_id_column, inplace=True)

        cv_pheno_data = prepare_cv_data(
            pheno_data_copy, str(cv_file_path), self.n_repeats, self.n_splits, seed=42
        )
        main_logger.info(f"Saved new CV folds to: {cv_file_path}")

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

    # Default CV is generated directly from the post-hold-out training set.
    # A user supplied --cv_file remains an explicit override for reproducible
    # externally-defined folds.
    if getattr(self, "cv_file", None) is None:
        if getattr(self, "split_strategy", "random") == "structure_aware":
            groups = _build_structure_groups(
                X,
                self.n_splits,
                self.random_seed + repeat_idx,
                getattr(self, "structure_clusters", None),
            )
            if self.task_type == "classification":
                splitter = StratifiedGroupKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_seed + repeat_idx,
                )
                folds = list(splitter.split(X, y, groups))
            else:
                folds = list(GroupKFold(n_splits=self.n_splits).split(X, y, groups))
            log.debug(
                "Generated structure-aware train-only CV with "
                f"{len(np.unique(groups))} genotype clusters"
            )
        elif self.task_type == "classification":
            class_counts = y.value_counts()
            if class_counts.min() < self.n_splits:
                raise ValueError(
                    "Stratified CV requires every class to contain at least "
                    f"n_splits={self.n_splits} samples; class counts: {class_counts.to_dict()}"
                )
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed + repeat_idx,
            )
            folds = list(splitter.split(X, y))
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed + repeat_idx,
            )
            folds = list(splitter.split(X))
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            log.debug(
                f"Repeat {repeat_idx + 1}/{self.n_repeats}, "
                f"train-only fold {fold_idx + 1}/{self.n_splits}: "
                f"Train size {len(train_idx)}, Val size {len(val_idx)}"
            )
        return folds

    if self.task_type == "classification":
        class_counts = y.value_counts()
        if class_counts.min() < self.n_splits:
            raise ValueError(
                "Stratified CV requires every class to contain at least "
                f"n_splits={self.n_splits} samples; class counts: {class_counts.to_dict()}"
            )
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_seed + repeat_idx,
        )
        folds = list(splitter.split(X, y))
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            log.debug(
                f"Repeat {repeat_idx + 1}/{self.n_repeats}, "
                f"stratified fold {fold_idx + 1}/{self.n_splits}: "
                f"Train size {len(train_idx)}, Val size {len(val_idx)}"
            )
        return folds

    cv_col = f"cv{repeat_idx}"

    if cv_col not in cv_pheno_data.columns:
        log.error(
            f"CV column {cv_col} does not exist in CV file, cannot generate folds"
        )
        raise ValueError(f"CV column {cv_col} does not exist")

    X_index_map = {idx: i for i, idx in enumerate(X.index)}
    folds = []
    for fold_idx in range(self.n_splits):
        val_mask = cv_pheno_data[cv_col] == fold_idx
        val_indices = cv_pheno_data.index[val_mask].tolist()

        train_mask = cv_pheno_data[cv_col] != fold_idx
        train_indices = cv_pheno_data.index[train_mask].tolist()

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


def save_train_only_cv_folds(self, X: pd.DataFrame, folds, repeat_dir) -> Path:
    """Persist the exact post-hold-out CV assignment for auditability."""
    assignments = np.full(len(X), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(folds):
        assignments[val_idx] = fold_idx
    if (assignments < 0).any():
        raise ValueError("Train-only CV assignment did not cover every training sample")
    output_path = Path(repeat_dir) / "cv_train_only.csv"
    pd.DataFrame({"sample_id": X.index.astype(str), "fold": assignments}).to_csv(
        output_path, index=False
    )
    return output_path
