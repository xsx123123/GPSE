#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for the GPSE train module.

Defines the `gpse train` argument parser and runs the training workflow.
The top-level `gpse.cli` module only routes to this module.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from gpse.convert.processor import GenomicDataProcessor
from gpse.train.predictor import GenomicPredictorV2, main_logger
from gpse.train.workflow import (
    _build_parser,
    _configure_logging,
    _log_stage,
    _show_version,
)
from gpse.utils.paralle import validate_parallelism

try:
    from rich_argparse import RichHelpFormatter
except ImportError:  # pragma: no cover
    RichHelpFormatter = argparse.HelpFormatter

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _console = Console()
except ImportError:  # pragma: no cover
    _console = None
    Table = None
    Text = None


def _log_config(args) -> None:
    """Pretty-print the active GPSE configuration, grouped and filtered.

    Skips flags the user didn't touch (``version``/``help``), drops ``None``
    values for optional inputs, and only shows preprocessing/convert-related
    args when the user is actually in that mode. Renders a rich table on the
    terminal and writes a compact flat copy to the log file.
    """
    ns = vars(args)

    groups: list[tuple[str, list[tuple[str, object]]]] = [
        ("Data", [
            ("geno_file", ns.get("geno_file")),
            ("pheno_file", ns.get("pheno_file")),
            ("target_trait", ns.get("target_trait")),
            ("task_type", ns.get("task_type")),
            ("n_classes", ns.get("n_classes")),
            ("standardize_phenotype", ns.get("standardize_phenotype")),
        ]),
        ("Training", [
            ("models", ns.get("models")),
            ("trials", ns.get("trials")),
            ("test_size", ns.get("test_size")),
            ("n_splits", ns.get("n_splits")),
            ("n_repeats", ns.get("n_repeats")),
            ("patience", ns.get("patience")),
            ("use_default_params", ns.get("use_default_params")),
            ("random_seed", ns.get("random_seed")),
            ("cv_folds", ns.get("cv_folds")),
            ("use_same_test_set", ns.get("use_same_test_set")),
            ("use_stacking", ns.get("use_stacking")),
            ("top_n_models", ns.get("top_n_models")),
        ]),
        ("CV", [
            ("cv_file", ns.get("cv_file")),
            ("force_new_cv", ns.get("force_new_cv")),
            ("cv_id_column", ns.get("cv_id_column")),
        ]),
        ("Parallelism", [
            ("n_jobs", ns.get("n_jobs")),
            ("max_workers", ns.get("max_workers")),
        ]),
        ("Output", [
            ("results_dir", ns.get("results_dir")),
            ("save_models", ns.get("save_models")),
            ("save_representative", ns.get("save_representative")),
            ("log_level", ns.get("log_level")),
        ]),
    ]

    if ns.get("enable_preprocess") or ns.get("preprocess_only"):
        groups.append(("Preprocess", [
            ("preprocess_only", ns.get("preprocess_only")),
            ("preprocess_prefix", ns.get("preprocess_prefix")),
            ("vcf_file", ns.get("vcf_file")),
            ("bfile", ns.get("bfile")),
            ("ped_file", ns.get("ped_file")),
            ("map_file", ns.get("map_file")),
            ("plink_path", ns.get("plink_path")),
            ("plink_out", ns.get("plink_out")),
            ("extract_file", ns.get("extract_file")),
            ("snp_dir", ns.get("snp_dir")),
            ("direct_convert", ns.get("direct_convert")),
            ("matrix_file", ns.get("matrix_file")),
            ("raw_pheno_file", ns.get("raw_pheno_file")),
            ("skip_matrix_conversion", ns.get("skip_matrix_conversion")),
            ("skip_phenotype_match", ns.get("skip_phenotype_match")),
            ("skip_data_clean", ns.get("skip_data_clean")),
            ("load_matrix_info", ns.get("load_matrix_info")),
        ]))

    # Filter Nones up front so widths and log output only reflect real rows.
    filtered: list[tuple[str, list[tuple[str, object]]]] = []
    for name, items in groups:
        rows = [(k, v) for k, v in items if v is not None]
        if rows:
            filtered.append((name, rows))

    # Compact flat copy to the log file (always).
    for name, rows in filtered:
        main_logger.info(f"[{name}]")
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            main_logger.info(f"  {k.ljust(width)} : {v}")

    # Pretty rich table to the terminal (when available).
    if _console is not None and Table is not None:
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            show_edge=False,
            show_lines=False,
        )
        table.add_column("key", style="cyan", no_wrap=True)
        table.add_column("value", style="white")

        for name, rows in filtered:
            table.add_row(Text(name, style="bold magenta"), "")
            width = max(len(k) for k, _ in rows)
            for k, v in rows:
                table.add_row(f"  {k.ljust(width)}", str(v))
            table.add_row("", "")

        _console.print(table)


def _try_infer_task_type(geno_file, pheno_file, target_trait, preprocess_prefix,
                         enable_preprocess):
    """Try to infer task_type and n_classes from a phenotype_info.json file.

    Returns
    -------
    tuple[str | None, int | None, str | None]
        (task_type, n_classes, info_file_path) — all ``None`` when no info file
        is found or it cannot be parsed.
    """
    import json
    import re

    candidates = []

    if enable_preprocess and preprocess_prefix and target_trait:
        safe_trait = re.sub(r'[^\w\-]', '_', target_trait)
        candidates.append(f"{preprocess_prefix}_{safe_trait}_phenotype_info.json")

    if pheno_file:
        base = os.path.splitext(pheno_file)[0]
        candidates.append(f"{base}_info.json")
        # e.g. .../yield_phenotype.csv -> .../yield_phenotype_info.json
        candidates.append(base.replace('_phenotype', '_phenotype_info') + '.json')

    if geno_file:
        base = os.path.splitext(geno_file)[0]
        # e.g. .../yield_genotype.csv -> .../yield_phenotype_info.json
        candidates.append(base.replace('_genotype', '_phenotype_info') + '.json')

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    info = json.load(f)
                task_type = info.get('task_type')
                n_classes = info.get('n_classes')
                if task_type in ('regression', 'classification'):
                    return task_type, n_classes, path
            except Exception:
                continue

    return None, None, None


def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    """Run the GPSE training CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    parser = _build_parser(
        formatter_class=formatter_class or RichHelpFormatter,
        prog=prog or "python -m gpse.train.cli",
        help_action=help_action,
        parents=parents,
    )
    if not raw_args:
        parser.error("train requires arguments. Use 'gpse train -h' for help.")

    args = parser.parse_args(raw_args)

    if args.version:
        _show_version()
        return 0

    _configure_logging(args.log_level)

    try:
        args.n_jobs, args.max_workers = validate_parallelism(
            args.n_jobs,
            args.max_workers,
            logger=main_logger,
        )
    except ValueError:
        return 1

    if args.task_type == "classification":
        if args.n_classes is None:
            main_logger.error("Classification tasks require the --n_classes argument")
            return 1
        if args.n_classes < 2:
            main_logger.error("--n_classes must be >= 2 for classification")
            return 1
    elif args.n_classes is not None:
        main_logger.warning("--n_classes is ignored for regression tasks")

    if args.preprocess_only:
        if not args.preprocess_prefix:
            main_logger.error("--preprocess_prefix is required when --preprocess_only is set")
            return 1
    elif args.enable_preprocess:
        if not args.preprocess_prefix:
            main_logger.error("--preprocess_prefix is required when --enable_preprocess is set")
            return 1
        if not args.target_trait:
            main_logger.error("--target_trait is required for training")
            return 1
    else:
        if not args.geno_file or not args.pheno_file:
            main_logger.error("--geno_file and --pheno_file are required in normal training mode")
            return 1
        if not args.target_trait:
            main_logger.error("--target_trait is required for training")
            return 1

    _log_config(args)

    processed_geno_file = args.geno_file
    processed_pheno_file = args.pheno_file

    if args.enable_preprocess or args.preprocess_only:
        _log_stage("Starting data preprocessing")

        processor = GenomicDataProcessor(logger=main_logger, plink_path=args.plink_path)
        preprocess_kwargs = {
            "out_prefix": args.preprocess_prefix,
            "vcf": args.vcf_file,
            "bfile": args.bfile,
            "ped_file": args.ped_file,
            "map_file": args.map_file,
            "extract": args.extract_file,
            "snp_dir": args.snp_dir,
            "direct": args.direct_convert,
            "matrix_file": args.matrix_file,
            "pheno": args.raw_pheno_file,
            "plink_out": args.plink_out,
            "skip_matrix": args.skip_matrix_conversion,
            "skip_match": args.skip_phenotype_match,
            "skip_clean": args.skip_data_clean,
            "load": args.load_matrix_info,
            "trait_name": args.target_trait,
            "standardize_phenotype": args.standardize_phenotype,
        }

        try:
            result = processor.process_genomic_data(**preprocess_kwargs)
            if result != 0:
                main_logger.error("Data preprocessing failed")
                return result

            main_logger.info("Data preprocessing completed successfully!")
            if not args.skip_phenotype_match and args.raw_pheno_file:
                for ext in [".parquet", ".feather", ".csv"]:
                    test_file = f"{args.preprocess_prefix}_genotype{ext}"
                    if os.path.exists(test_file):
                        processed_geno_file = test_file
                        break
                else:
                    processed_geno_file = f"{args.preprocess_prefix}_genotype.csv"
                processed_pheno_file = f"{args.preprocess_prefix}_phenotype.csv"
            else:
                for ext in [".parquet", ".feather", ".csv"]:
                    test_file = args.preprocess_prefix + ext
                    if os.path.exists(test_file):
                        processed_geno_file = test_file
                        break
                else:
                    processed_geno_file = args.preprocess_prefix + ".csv"
                processed_pheno_file = args.raw_pheno_file


        except Exception as e:
            main_logger.error(f"Error during data preprocessing: {e}")
            main_logger.error(traceback.format_exc())
            return 1

    if args.preprocess_only:
        main_logger.info("Preprocessing finished; exiting without training")
        return 0

    if not os.path.exists(processed_geno_file):
        main_logger.error(f"Genotype file not found: {processed_geno_file}")
        return 1
    if not os.path.exists(processed_pheno_file):
        main_logger.error(f"Phenotype file not found: {processed_pheno_file}")
        return 1

    # ---- Auto-detect task_type from phenotype_info.json ----
    task_type_explicit = any(
        arg == '--task_type' or arg.startswith('--task_type=')
        for arg in raw_args
    )

    inferred_task, inferred_n_classes, info_path = _try_infer_task_type(
        processed_geno_file,
        processed_pheno_file,
        args.target_trait,
        args.preprocess_prefix,
        args.enable_preprocess,
    )

    if inferred_task:
        if not task_type_explicit:
            main_logger.info(
                f"Auto-detected task type from {os.path.basename(info_path)}: "
                f"{inferred_task}"
                + (f" (n_classes={inferred_n_classes})" if inferred_n_classes else "")
            )
            args.task_type = inferred_task
            if inferred_n_classes is not None:
                args.n_classes = inferred_n_classes
        elif args.task_type != inferred_task:
            main_logger.warning(
                f"User specified --task_type={args.task_type}, but "
                f"{os.path.basename(info_path)} suggests {inferred_task}. "
                f"Using user-specified value."
            )

    if args.task_type == "classification" and args.n_classes is None:
        main_logger.error(
            "Classification tasks require the --n_classes argument "
            "(auto-detection failed)"
        )
        return 1
    if args.task_type == "classification" and args.n_classes < 2:
        main_logger.error("--n_classes must be >= 2 for classification")
        return 1

    _log_stage("Starting model training")
    main_logger.info(f"Genotype file: {processed_geno_file}")
    main_logger.info(f"Phenotype file: {processed_pheno_file}")

    training_standardize = False
    if args.enable_preprocess and args.standardize_phenotype:
        main_logger.info("Phenotype already standardized during preprocessing; skipping standardization in training")
    elif not args.enable_preprocess:
        training_standardize = args.standardize_phenotype

    predictor = GenomicPredictorV2(
        random_seed=args.random_seed,
        results_dir=args.results_dir,
        n_trials=args.trials,
        n_threads=args.n_jobs,
        max_parallel_jobs=args.max_workers,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        patience=args.patience,
        use_default_params=args.use_default_params,
        save_models=args.save_models,
        save_representative=args.save_representative,
        cv_file=args.cv_file,
        force_new_cv=args.force_new_cv,
        cv_id_column=args.cv_id_column,
        task_type=args.task_type,
        n_classes=args.n_classes,
        standardize_phenotype=training_standardize,
    )

    predictor.run_all_models(
        geno_file=processed_geno_file,
        pheno_file=processed_pheno_file,
        target_trait=args.target_trait,
        models=args.models,
        use_stacking=args.use_stacking,
        top_n_models=args.top_n_models,
        cv_folds=args.cv_folds,
        use_same_test_set=args.use_same_test_set,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
