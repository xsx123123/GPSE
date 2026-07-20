#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phenotype processing utilities for GPSE.

Pure functions for phenotype file conversion, genotype-phenotype sample
matching, and optional z-score standardization.
"""

import os
import re
import json

import pandas as pd

try:
    from gpse.utils.log_utils import logger as _default_logger
except Exception:  # pragma: no cover
    import logging
    _default_logger = logging.getLogger(__name__)


def convert_phenotype(
    pheno_file,
    out_file=None,
    *,
    trait_name=None,
    trait_col=None,
    clean_column_names_fn=None,
    logger=None,
):
    """Convert a phenotype file to CSV and remove missing phenotype values.

    Parameters
    ----------
    pheno_file : str
        Path to the input phenotype file (tab- or comma-separated).
    out_file : str, optional
        Output CSV path.  Defaults to ``{base}.csv`` (or ``{base}_converted.csv``
        when the input is already ``.csv``).
    trait_name : str, optional
        Rename the selected trait column to this value in the output.
    trait_col : str, optional
        Name of the trait column to extract.  When omitted the second column
        is used.
    clean_column_names_fn : callable, optional
        A function that accepts a list of column names and returns cleaned
        names.  When omitted a no-op is used.
    logger
        Logger instance.  Falls back to the shared GPSE logger.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with ``ID`` and the selected trait.
    """
    log = logger or _default_logger
    log.info(f"Processing phenotype file: {pheno_file}")

    # --- Read input ---
    try:
        df = pd.read_csv(pheno_file, sep='\t')
        if df.shape[1] < 2:
            df = pd.read_csv(pheno_file, sep=',')
    except Exception as e:
        log.warning(f"pandas failed to read phenotype file; falling back to raw parsing: {e}")
        pheno_data = []
        with open(pheno_file, 'r') as f:
            next(f)  # Skip header.
            for line in f:
                line = line.strip().split('\t')
                if len(line) >= 2:
                    pheno_data.append([line[0], line[1]])
        df = pd.DataFrame(pheno_data, columns=['ID', 'Phenotype'])

    if df.shape[1] < 2:
        raise ValueError(f"Phenotype file {pheno_file} must contain at least two columns")

    # --- Determine columns ---
    original_id_col = df.columns[0]
    if trait_col is not None:
        if trait_col not in df.columns:
            raise ValueError(
                f"Trait column '{trait_col}' not found in {pheno_file}. "
                f"Available columns: {list(df.columns)}"
            )
        original_trait_col = trait_col
    else:
        original_trait_col = df.columns[1]

    log.info(f"Original columns: ID={original_id_col}, Trait={original_trait_col}")

    # --- Rename columns ---
    rename_dict = {original_id_col: 'ID'}

    target_col_name = original_trait_col
    if trait_name:
        rename_dict[original_trait_col] = trait_name
        target_col_name = trait_name
        log.info(f"Renaming phenotype column '{original_trait_col}' to '{trait_name}'")
    elif clean_column_names_fn:
        cleaned_col = clean_column_names_fn([original_trait_col])[0]
        if cleaned_col != original_trait_col:
            rename_dict[original_trait_col] = cleaned_col
            target_col_name = cleaned_col

    df = df.rename(columns=rename_dict)
    df = df[['ID', target_col_name]]

    # --- Drop missing values ---
    df = df.dropna(subset=[target_col_name])
    if df[target_col_name].dtype == object:
        df = df[df[target_col_name] != 'NA']

    log.info(f"Remaining samples after dropping missing phenotypes: {len(df)}")

    # --- Save CSV ---
    if out_file is None:
        base, ext = os.path.splitext(pheno_file)
        if ext.lower() == '.csv':
            out_file = f"{base}_converted.csv"
        else:
            out_file = f"{base}.csv"

    df.to_csv(out_file, index=False)
    log.info(f"Phenotype CSV saved to: {out_file}")

    return df


def match_genotype_phenotype(pheno_df, geno_file, out_prefix, *, out_format='csv', logger=None):
    """Match genotype and phenotype samples and preserve a shared order.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype DataFrame with an ``ID`` column.
    geno_file : str
        Path to the genotype matrix (first column = sample IDs).
    out_prefix : str
        Output prefix for the matched files.
    out_format : str
        Output format: ``csv``, ``parquet``, or ``feather``. Default ``csv``.
    logger
        Logger instance.

    Returns
    -------
    tuple[str, str]
        Paths to the matched phenotype and genotype files.
    """
    log = logger or _default_logger
    log.info(f"Reading genotype file: {geno_file}")

    geno_df = pd.read_csv(geno_file, index_col=0)

    geno_samples = set(geno_df.index)
    log.info(f"Genotype file contains {len(geno_samples)} samples")

    pheno_samples = set(pheno_df['ID'])
    log.info(f"Phenotype file contains {len(pheno_samples)} samples")

    common_samples = sorted(list(geno_samples.intersection(pheno_samples)))
    log.info(f"Found {len(common_samples)} shared samples")

    if len(common_samples) == 0:
        raise ValueError("No shared sample IDs found. Check sample ID formatting.")

    # Filter and sort by shared sample IDs.
    pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
    pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
    geno_filtered = geno_df.loc[common_samples]

    out_format = out_format.lower()
    ext = '.parquet' if out_format == 'parquet' else '.feather' if out_format == 'feather' else '.csv'
    pheno_out_file = f"{out_prefix}_phenotype" + ext
    geno_out_file = f"{out_prefix}_genotype" + ext

    if out_format == 'parquet':
        pheno_filtered.to_parquet(pheno_out_file, index=False)
        geno_filtered.reset_index().to_parquet(geno_out_file, index=False)
    elif out_format == 'feather':
        pheno_filtered.to_feather(pheno_out_file)
        geno_filtered.reset_index().to_feather(geno_out_file)
    else:
        pheno_filtered.to_csv(pheno_out_file, index=False)
        geno_filtered.to_csv(geno_out_file)

    log.info(f"Matched phenotype file saved to: {pheno_out_file}")
    log.info(f"Matched genotype file saved to: {geno_out_file}")

    return pheno_out_file, geno_out_file


def standardize_phenotype(pheno_df, trait_col, *, logger=None):
    """Apply z-score standardization to a phenotype column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype DataFrame.
    trait_col : str
        Column to standardize.
    logger
        Logger instance.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Standardized DataFrame and scaler parameter dictionary.
    """
    log = logger or _default_logger

    y = pheno_df[trait_col]
    mean_val = float(y.mean())
    std_val = float(y.std())

    if std_val < 1e-10:
        log.warning("Phenotype standard deviation is near zero; skipping standardization")
        return pheno_df, {'mean': mean_val, 'std': 1.0, 'applied': False,
                          'method': 'zscore', 'trait': trait_col}

    pheno_standardized = pheno_df.copy()
    pheno_standardized[trait_col] = (y - mean_val) / std_val

    scaler_params = {
        'mean': mean_val,
        'std': std_val,
        'applied': True,
        'method': 'zscore',
        'trait': trait_col,
    }

    log.info(f"Phenotype standardization completed: mean={mean_val:.4f}, std={std_val:.4f}")
    return pheno_standardized, scaler_params


def minmax_normalize_phenotype(pheno_df, trait_col, *, logger=None):
    """Apply min-max normalization to a phenotype column, scaling to [0, 1].

    This matches the phenotype normalization described in Azodi et al. 2019
    (https://doi.org/10.1534/g3.119.400498).

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype DataFrame.
    trait_col : str
        Column to normalize.
    logger
        Logger instance.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Normalized DataFrame and scaler parameter dictionary.
    """
    log = logger or _default_logger

    y = pd.to_numeric(pheno_df[trait_col], errors='coerce')
    min_val = float(y.min())
    max_val = float(y.max())

    if max_val - min_val < 1e-10:
        log.warning("Phenotype range is near zero; skipping min-max normalization")
        return pheno_df, {'min': min_val, 'max': max_val, 'applied': False,
                          'method': 'minmax', 'trait': trait_col}

    pheno_normalized = pheno_df.copy()
    pheno_normalized[trait_col] = (y - min_val) / (max_val - min_val)

    scaler_params = {
        'min': min_val,
        'max': max_val,
        'applied': True,
        'method': 'minmax',
        'trait': trait_col,
    }

    log.info(f"Phenotype min-max normalization completed: min={min_val:.4f}, max={max_val:.4f}")
    return pheno_normalized, scaler_params


def save_scaler_params(scaler_params, scaler_file, *, logger=None):
    """Persist scaler parameters to a JSON file."""
    log = logger or _default_logger
    with open(scaler_file, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    log.info(f"Phenotype scaler parameters saved to: {scaler_file}")


def detect_phenotype_type(series, max_classes=20, min_samples_per_class=5):
    """Automatically detect whether a phenotype is regression or classification.

    Parameters
    ----------
    series : pandas.Series
        The phenotype column to analyse (after dropping NA values).
    max_classes : int, default 20
        Maximum number of unique values allowed for a classification trait.
    min_samples_per_class : int, default 5
        Minimum samples required in each class for a valid classification trait.

    Returns
    -------
    tuple[str, int | None, dict]
        (task_type, n_classes, info_dict)
        * task_type -- ``"regression"`` or ``"classification"``
        * n_classes -- number of classes (``None`` for regression)
        * info_dict -- extra metadata (class distribution, reason, stats, ...)
    """
    # Treat common missing-value markers as NA before deciding on the type.
    missing_markers = ["", " ", "--", ".", "?", "NA", "N/A", "na", "n/a",
                       "NULL", "null", "None", "none", "NaN", "nan"]
    s = series.replace(missing_markers, np.nan).dropna()
    n_total = len(s)

    if n_total == 0:
        return "regression", None, {"error": "all values are NA"}

    # Try numeric conversion
    numeric_s = pd.to_numeric(s, errors='coerce').dropna()
    n_numeric = len(numeric_s)

    # Case 1: contains non-numeric (string) labels -> classification
    if n_numeric < n_total:
        n_unique = int(s.nunique())
        value_counts = s.value_counts().to_dict()
        # JSON-safe keys
        value_counts = {str(k): int(v) for k, v in value_counts.items()}
        return (
            "classification",
            n_unique,
            {
                "reason": "contains non-numeric values",
                "n_samples": n_total,
                "class_distribution": value_counts,
            },
        )

    # Case 2: all values are numeric
    n_unique = int(numeric_s.nunique())

    # Binary (<=2 unique values) -> always classification
    if n_unique <= 2:
        value_counts = numeric_s.value_counts().to_dict()
        value_counts = {str(k): int(v) for k, v in value_counts.items()}
        return (
            "classification",
            n_unique,
            {
                "reason": "binary values",
                "n_samples": n_total,
                "class_distribution": value_counts,
            },
        )

    # Multi-class: unique values <= max_classes AND all integer-like
    if n_unique <= max_classes:
        is_integer_like = numeric_s.apply(
            lambda x: abs(float(x) - round(float(x))) < 1e-10
        ).all()
        if is_integer_like:
            value_counts = numeric_s.value_counts().to_dict()
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            min_count = min(value_counts.values())
            if min_count >= min_samples_per_class:
                return (
                    "classification",
                    n_unique,
                    {
                        "reason": "integer-encoded with sufficient samples per class",
                        "n_samples": n_total,
                        "class_distribution": value_counts,
                    },
                )

    # Case 3: continuous / too many unique values -> regression
    return (
        "regression",
        None,
        {
            "reason": "continuous numeric values",
            "n_samples": n_total,
            "n_unique": n_unique,
            "mean": float(numeric_s.mean()),
            "std": float(numeric_s.std()),
            "min": float(numeric_s.min()),
            "max": float(numeric_s.max()),
        },
    )


def save_phenotype_info(info, info_file, *, logger=None):
    """Persist phenotype type-detection metadata to a JSON file.

    Parameters
    ----------
    info : dict
        Detection result (as returned by :func:`detect_phenotype_type`).
    info_file : str
        Destination JSON path.
    logger
        Logger instance.
    """
    log = logger or _default_logger
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log.info(f"Phenotype info saved to: {info_file}")
