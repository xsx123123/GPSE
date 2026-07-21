#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-trait summary tables after a batch run.

Each ``gpse train`` run writes small summary CSVs into its results
directory (``model_comparison*.csv`` at the top level and holdout
summaries under ``reports/``). This module concatenates the tables of
every trait into one file per summary kind, adding a ``Trait`` column
as the first column, so cross-trait comparison needs no ad-hoc script.

Merged files are written to ``<results_root>/merged/``.
"""

from __future__ import annotations

from pathlib import Path

# Summary tables merged across traits, relative to each trait's results
# directory. Only the basename is kept for the merged output, so the
# ``reports/`` entries must not collide with the top-level ones.
SUMMARY_FILES = (
    "model_comparison.csv",
    "model_comparison_cv.csv",
    "model_comparison_cv_topsis.csv",
    "model_comparison_cv_topsis_simple.csv",
    "reports/model_comparison_holdout.csv",
    "reports/holdout_metrics.csv",
)


def merge_trait_results(
    trait_dirs: dict[str, Path],
    output_dir: str | Path,
) -> list[Path]:
    """Merge the summary CSVs of every trait into ``output_dir``.

    ``trait_dirs`` maps trait name -> results directory. Missing files
    are skipped; a merged file is only written when at least one trait
    produced it. Columns are unioned across traits (traits trained with
    different task types may have different metrics). Returns the list
    of written file paths.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    written: list[Path] = []
    for relative in SUMMARY_FILES:
        frames = []
        for name, results_dir in trait_dirs.items():
            path = Path(results_dir) / relative
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "Trait", name)
            frames.append(frame)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True, sort=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / Path(relative).name
        merged.to_csv(out_path, index=False)
        written.append(out_path)
    return written
