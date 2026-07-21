#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data I/O and Preprocessing Module
=================================
Handles genotype / phenotype data loading, phenotype standardization,
and related preprocessing utilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

from loguru import logger as main_logger

from gpse.utils.feature_manifest import write_feature_manifest


def _read_file_auto(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif suffix == '.feather':
        return pd.read_feather(file_path)
    else:
        return pd.read_csv(file_path)


def load_data(self, geno_file: str, pheno_file: str, target_trait: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load genotype and phenotype data.

    Parameters
    ----------
    geno_file : str
        Path to the genotype data file.
    pheno_file : str
        Path to the phenotype data file.
    target_trait : str
        Name of the target trait column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        Feature matrix, target variable, and the full phenotype DataFrame.

    Raises
    ------
    FileNotFoundError
        When data files do not exist.
    KeyError
        When the target trait column does not exist.
    ValueError
        When there are no common samples between genotype and phenotype.
    """
    main_logger.info("Loading data...")

    # Step 1: Load raw data files
    try:
        geno_data = _read_file_auto(geno_file)
        pheno_data = _read_file_auto(pheno_file)
        main_logger.info(
            f"Successfully loaded genotype data: {geno_data.shape}, "
            f"phenotype data: {pheno_data.shape}"
        )
    except FileNotFoundError as e:
        main_logger.error(f"Data file loading failed: {e}")
        raise
    except Exception as e:
        main_logger.error(f"Data file read error: {e}")
        raise

    # Step 2: Verify that the target trait exists
    if target_trait not in pheno_data.columns:
        error_msg = (
            f"Target trait '{target_trait}' does not exist in phenotype data. "
            f"Available columns: {list(pheno_data.columns)}"
        )
        main_logger.error(error_msg)
        raise KeyError(error_msg)

    # Step 3: Determine ID column and validate presence
    id_col = "ID" if "ID" in geno_data.columns else self.cv_id_column

    if id_col not in geno_data.columns:
        error_msg = f"Genotype data is missing ID column '{id_col}'"
        main_logger.error(error_msg)
        raise KeyError(error_msg)

    if id_col not in pheno_data.columns:
        error_msg = f"Phenotype data is missing ID column '{id_col}'"
        main_logger.error(error_msg)
        raise KeyError(error_msg)

    # Step 4: Find common samples and validate count
    geno_ids = set(geno_data[id_col])
    pheno_ids = set(pheno_data[id_col])
    common_ids = geno_ids.intersection(pheno_ids)

    if len(common_ids) == 0:
        error_msg = "Genotype and phenotype data have no common sample IDs"
        main_logger.error(error_msg)
        raise ValueError(error_msg)

    main_logger.info(f"Genotype samples: {len(geno_ids)}, Phenotype samples: {len(pheno_ids)}")
    main_logger.info(f"Common samples: {len(common_ids)}")

    main_logger.info("Filtering common samples and aligning index...")
    # Step 5: Retain only common samples
    geno_data = geno_data[geno_data[id_col].isin(common_ids)]
    pheno_data = pheno_data[pheno_data[id_col].isin(common_ids)]

    # Handle duplicate sample IDs in phenotype data (keep first record)
    if pheno_data[id_col].duplicated().any():
        dup_count = pheno_data[id_col].duplicated().sum()
        main_logger.warning(
            f"Found {dup_count} duplicate sample IDs in phenotype data; "
            f"keeping the first record for each ID"
        )
        pheno_data = pheno_data.drop_duplicates(subset=id_col, keep="first")

    # Step 6: Set index and sort to ensure alignment
    geno_data.set_index(id_col, inplace=True)
    pheno_data.set_index(id_col, inplace=True)

    geno_data = geno_data.sort_index()
    pheno_data = pheno_data.sort_index()

    # Step 7: Extract feature matrix and target variable
    X = geno_data
    y = pheno_data[target_trait].copy()

    # Step 7b: For regression, coerce the target to numeric so that string
    # placeholders like "--" or "" become NaN instead of crashing later stats.
    if self.task_type == "regression" and y.dtype == object:
        main_logger.info("Coercing phenotype target to numeric (non-numeric values -> NaN)...")
        y = pd.to_numeric(y, errors="coerce")

    # Step 8: Preserve the ordered SNP IDs from convert.  Legacy matrices may
    # still use arbitrary names, but they are retained verbatim in the manifest.
    X.columns = [str(column) for column in X.columns]
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Genotype matrix contains duplicate feature IDs: {duplicates[:5]}")

    main_logger.info("Validating final data quality...")
    # Step 9: Validate final data quality
    if X.shape[0] != y.shape[0]:
        error_msg = (
            f"Feature matrix and target variable sample counts do not match: "
            f"{X.shape[0]} vs {y.shape[0]}"
        )
        main_logger.error(error_msg)
        raise ValueError(error_msg)

    # Single-pass null scan on the underlying numpy array (much faster than
    # the double pandas aggregation X.isnull().sum().sum() for wide frames).
    null_count = int(pd.isnull(X.to_numpy()).sum())
    if null_count > 0:
        main_logger.warning(f"Feature matrix contains {null_count} missing values")

    if y.isnull().sum() > 0:
        null_count = int(y.isnull().sum())
        main_logger.warning(
            f"Target variable contains {null_count} missing values; "
            f"dropping these samples before training"
        )
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        pheno_data = pheno_data.loc[valid_mask]
        main_logger.info(
            f"After removing missing targets - Features: {X.shape[1]}, "
            f"Samples: {X.shape[0]}"
        )
        if X.shape[0] == 0:
            raise ValueError(
                "All samples were removed due to missing target values; "
                "please check the phenotype data"
            )

    # Step 10: Special handling for classification tasks
    if self.task_type == "classification":
        main_logger.info("Processing classification labels...")
        y = self.genomic_classifier.prepare_classification_labels(y, self.results_dir)

    # Step 11: Preserve raw phenotypes.  When requested, standardization is
    # fitted only after the hold-out split from y_train in repeat training.
    if self.task_type == "regression" and self.standardize_phenotype:
        main_logger.info(
            "Phenotype standardization is deferred until after hold-out splitting "
            "so test labels cannot affect scaler parameters"
        )

    # Step 12: Persist the exact feature order used by every trained model.
    manifest_path = write_feature_manifest(
        self.results_dir, X.columns, source_file=geno_file
    )
    main_logger.info(f"Feature manifest saved to: {manifest_path}")

    # Step 13: Record final data information
    main_logger.info(
        f"Final data dimensions - Features: {X.shape[1]}, Samples: {X.shape[0]}"
    )
    main_logger.info(
        f"Target variable statistics - Mean: {y.mean():.4f}, Std: {y.std():.4f}"
    )

    return X, y, pheno_data


def _standardize_phenotype(self, y: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Z-score standardization of phenotype data.

    Parameters
    ----------
    y : pd.Series
        Raw phenotype data.

    Returns
    -------
    Tuple[pd.Series, Dict[str, float]]
        Standardized phenotype data and a dictionary of standardization parameters.
    """
    mean_val = float(y.mean())
    std_val = float(y.std())

    # Avoid division by zero
    if std_val < 1e-10:
        main_logger.warning("Phenotype data standard deviation is near zero, skipping standardization")
        return y, {"mean": mean_val, "std": 1.0, "applied": False}

    y_standardized = (y - mean_val) / std_val
    scaler_params = {"mean": mean_val, "std": std_val, "applied": True}

    return y_standardized, scaler_params


def _inverse_standardize_phenotype(self, y: np.ndarray) -> np.ndarray:
    """
    Inverse standardization of standardized predictions.

    Parameters
    ----------
    y : np.ndarray
        Standardized predictions.

    Returns
    -------
    np.ndarray
        Predictions in the original scale.
    """
    if self.phenotype_scaler is None or not self.phenotype_scaler.get("applied", False):
        return y

    return y * self.phenotype_scaler["std"] + self.phenotype_scaler["mean"]
