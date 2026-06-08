#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data validation and column-name sanitization utilities for GPSE.

Checks and cleans feature names so they are compatible with downstream
ML frameworks (e.g. LightGBM).
"""

import pandas as pd

try:
    from gpse.utils.log_utils import logger as _default_logger
except Exception:  # pragma: no cover
    import logging
    _default_logger = logging.getLogger(__name__)

# Characters unsupported by LightGBM feature names
SPECIAL_CHARS = [
    ':', '|', '[', ']', '{', '}', '"', '\\', ',', ' '
]

# Characters that are never allowed in trait (phenotype column) names.
# These cause problems in filenames, shell commands, and downstream ML tools.
TRAIT_INVALID_CHARS = [
    ' ', '%', ':', '/', '\\', '|', '[', ']', '{', '}', '"', ',',
    '\t', '\n', '\r',
]


def validate_trait_names(trait_names, *, logger=None):
    """Validate trait names before any conversion work.

    Raises
    ------
    ValueError
        If any trait name contains invalid characters (spaces, ``%``,
        ``:``, ``/``, etc.) or is empty.

    Parameters
    ----------
    trait_names : list[str]
        Trait (phenotype column) names to validate.
    logger
        Logger instance for diagnostic messages.
    """
    log = logger or _default_logger
    errors = []

    for name in trait_names:
        if not name or not name.strip():
            errors.append((repr(name), "empty or whitespace-only"))
            continue

        bad_chars = [ch for ch in TRAIT_INVALID_CHARS if ch in name]
        if bad_chars:
            display = ", ".join(repr(ch) for ch in bad_chars)
            errors.append((name, f"invalid characters: {display}"))

    if errors:
        log.error("Trait name validation failed:")
        for name, reason in errors:
            log.error(f"  {name!s:30s} → {reason}")
        raise ValueError(
            f"Invalid trait name(s) found: {', '.join(name for name, _ in errors)}. "
            f"Trait names must not contain spaces, %, or other special characters."
        )

    log.info(f"Trait name validation passed: {trait_names}")


def check_special_chars(column_names):
    """Return a dict of columns containing unsupported characters.

    Parameters
    ----------
    column_names : list[str]
        Column names to check.

    Returns
    -------
    dict[str, list[str]]
        Mapping of problematic column name → list of offending characters.
    """
    problematic = {}
    for col in column_names:
        chars_found = [ch for ch in SPECIAL_CHARS if ch in col]
        if chars_found:
            problematic[col] = chars_found
    return problematic


def clean_column_names(column_names):
    """Replace unsupported characters in column names with underscores.

    Parameters
    ----------
    column_names : list[str]
        Column names to clean.

    Returns
    -------
    list[str]
        Cleaned column names.
    """
    cleaned = []
    for col in column_names:
        cleaned_col = col
        for ch in SPECIAL_CHARS:
            cleaned_col = cleaned_col.replace(ch, '_')
        cleaned.append(cleaned_col)
    return cleaned


def process_file(file_path, output_path, *, logger=None):
    """Sanitize column names in a CSV file and write the result.

    Parameters
    ----------
    file_path : str
        Input CSV file.
    output_path : str
        Output CSV file with cleaned column names.
    logger
        Logger instance.

    Returns
    -------
    bool
        ``True`` if any columns were renamed, ``False`` otherwise.
    """
    log = logger or _default_logger
    log.info(f"Processing file: {file_path}")

    df = pd.read_csv(file_path)
    problematic = check_special_chars(df.columns)

    if problematic:
        log.warning(f"Found {len(problematic)} column name(s) with unsupported characters")
        for col, chars in problematic.items():
            log.warning(f"Column '{col}' contains unsupported characters: {', '.join(chars)}")

        column_mapping = dict(zip(df.columns, clean_column_names(df.columns)))
        df = df.rename(columns=column_mapping)

    else:
        log.info("No unsupported characters found in column names")

    df.to_csv(output_path, index=False)
    log.info(f"Sanitized data saved to: {output_path}")
    return bool(problematic)


def load_matrix(matrix_file, *, logger=None):
    """Load a genotype matrix CSV and log summary statistics.

    Parameters
    ----------
    matrix_file : str
        Path to the CSV matrix (first column = sample IDs).
    logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        The loaded matrix.
    """
    log = logger or _default_logger
    log.info(f"Loading matrix file: {matrix_file}")

    matrix_df = pd.read_csv(matrix_file, index_col=0)

    log.info(f"Matrix shape: {matrix_df.shape}")
    log.info(f"Sample count: {len(matrix_df.index)}")
    log.info(f"SNP count: {len(matrix_df.columns)}")

    missing_values = matrix_df.isnull().sum().sum()
    log.info(f"Missing value count: {missing_values}")

    return matrix_df
