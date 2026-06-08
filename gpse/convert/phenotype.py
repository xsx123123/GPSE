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


def match_genotype_phenotype(pheno_df, geno_file, out_prefix, *, logger=None):
    """Match genotype and phenotype samples and preserve a shared order.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype DataFrame with an ``ID`` column.
    geno_file : str
        Path to the genotype CSV matrix (first column = sample IDs).
    out_prefix : str
        Output prefix for the matched files.
    logger
        Logger instance.

    Returns
    -------
    tuple[str, str]
        Paths to the matched phenotype and genotype CSV files.
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

    pheno_out_file = f"{out_prefix}_phenotype.csv"
    geno_out_file = f"{out_prefix}_genotype.csv"

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
        return pheno_df, {'mean': mean_val, 'std': 1.0, 'applied': False}

    pheno_standardized = pheno_df.copy()
    pheno_standardized[trait_col] = (y - mean_val) / std_val

    scaler_params = {
        'mean': mean_val,
        'std': std_val,
        'applied': True,
        'trait': trait_col,
    }

    log.info(f"Phenotype standardization completed: mean={mean_val:.4f}, std={std_val:.4f}")
    return pheno_standardized, scaler_params


def save_scaler_params(scaler_params, scaler_file, *, logger=None):
    """Persist scaler parameters to a JSON file."""
    log = logger or _default_logger
    with open(scaler_file, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    log.info(f"Phenotype scaler parameters saved to: {scaler_file}")
