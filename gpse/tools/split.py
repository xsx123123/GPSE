#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset splitting for genomic prediction.

Splits matched genotype / phenotype samples into training and test
(prediction) subsets so that ``gpse train`` and ``gpse predict`` can
operate on disjoint sample sets.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def _read_file_auto(file_path: str) -> pd.DataFrame:
    """Read a tabular file, dispatching on extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    return pd.read_csv(file_path)


def _write_file_auto(df: pd.DataFrame, file_path: str) -> None:
    """Write a DataFrame, dispatching on extension."""
    suffix = Path(file_path).suffix.lower()
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    elif suffix == ".feather":
        df.reset_index(drop=True).to_feather(file_path)
    else:
        df.to_csv(file_path, index=False)


def split_dataset(
    geno_file: str,
    pheno_file: str,
    out_prefix: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_column: str | None = None,
    id_column: str | None = None,
) -> dict:
    """
    Split genotype and phenotype data into train / test subsets.

    Parameters
    ----------
    geno_file : str
        Path to the genotype matrix (CSV / Parquet / Feather).
    pheno_file : str
        Path to the phenotype table (CSV / Parquet / Feather).
    out_prefix : str
        Output prefix.  Four files are written:
        ``<prefix>_train_geno.<ext>``, ``<prefix>_train_pheno.<ext>``,
        ``<prefix>_test_geno.<ext>``, ``<prefix>_test_pheno.<ext>``.
    test_ratio : float
        Fraction of samples reserved for the test (prediction) set.
    seed : int
        Random seed for reproducible splitting.
    stratify_column : str or None
        Phenotype column used for stratified splitting (classification).
    id_column : str or None
        Name of the sample-ID column.  Auto-detected as ``ID`` or the
        first column when *None*.

    Returns
    -------
    dict
        Summary with sample counts and output paths.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info(f"Loading genotype: {geno_file}")
    geno = _read_file_auto(geno_file)
    logger.info(f"Loading phenotype: {pheno_file}")
    pheno = _read_file_auto(pheno_file)

    logger.info(f"Genotype shape: {geno.shape}  |  Phenotype shape: {pheno.shape}")

    # ------------------------------------------------------------------
    # 2. Resolve ID column
    # ------------------------------------------------------------------
    if id_column is None:
        if "ID" in geno.columns:
            id_column = "ID"
        else:
            id_column = geno.columns[0]
    if id_column not in geno.columns:
        raise KeyError(f"ID column '{id_column}' not found in genotype file")
    if id_column not in pheno.columns:
        raise KeyError(f"ID column '{id_column}' not found in phenotype file")

    # ------------------------------------------------------------------
    # 3. Intersect samples
    # ------------------------------------------------------------------
    geno_ids = set(geno[id_column])
    pheno_ids = set(pheno[id_column])
    common_ids = geno_ids & pheno_ids

    if not common_ids:
        raise ValueError("No common sample IDs between genotype and phenotype")

    logger.info(
        f"Samples — geno: {len(geno_ids)}, pheno: {len(pheno_ids)}, "
        f"common: {len(common_ids)}"
    )

    geno = geno[geno[id_column].isin(common_ids)].drop_duplicates(subset=id_column, keep="first")
    pheno = pheno[pheno[id_column].isin(common_ids)].drop_duplicates(subset=id_column, keep="first")

    # Align on ID so row order matches
    geno = geno.set_index(id_column).sort_index().reset_index()
    pheno = pheno.set_index(id_column).sort_index().reset_index()

    total = len(geno)
    logger.info(f"Aligned samples: {total}")

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    stratify_labels = None
    if stratify_column is not None:
        if stratify_column not in pheno.columns:
            raise KeyError(
                f"Stratify column '{stratify_column}' not found in phenotype"
            )
        stratify_labels = pheno[stratify_column]
        logger.info(f"Stratified split on column '{stratify_column}'")

    indices = np.arange(total)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    train_geno = geno.iloc[train_idx].reset_index(drop=True)
    test_geno = geno.iloc[test_idx].reset_index(drop=True)
    train_pheno = pheno.iloc[train_idx].reset_index(drop=True)
    test_pheno = pheno.iloc[test_idx].reset_index(drop=True)

    logger.info(f"Train samples: {len(train_geno)}  |  Test samples: {len(test_geno)}")

    # ------------------------------------------------------------------
    # 5. Write outputs (same format as input genotype file)
    # ------------------------------------------------------------------
    ext = Path(geno_file).suffix.lower() or ".csv"

    train_geno_path = f"{out_prefix}_train_geno{ext}"
    train_pheno_path = f"{out_prefix}_train_pheno{ext}"
    test_geno_path = f"{out_prefix}_test_geno{ext}"
    test_pheno_path = f"{out_prefix}_test_pheno{ext}"

    _write_file_auto(train_geno, train_geno_path)
    _write_file_auto(train_pheno, train_pheno_path)
    _write_file_auto(test_geno, test_geno_path)
    _write_file_auto(test_pheno, test_pheno_path)

    logger.info(f"Written: {train_geno_path}")
    logger.info(f"Written: {train_pheno_path}")
    logger.info(f"Written: {test_geno_path}")
    logger.info(f"Written: {test_pheno_path}")

    summary = {
        "total_samples": total,
        "train_samples": int(len(train_geno)),
        "test_samples": int(len(test_geno)),
        "test_ratio": test_ratio,
        "seed": seed,
        "stratify_column": stratify_column,
        "files": {
            "train_geno": train_geno_path,
            "train_pheno": train_pheno_path,
            "test_geno": test_geno_path,
            "test_pheno": test_pheno_path,
        },
    }
    return summary
