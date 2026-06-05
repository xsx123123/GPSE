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

try:
    from rich_argparse import RichHelpFormatter
except ImportError:  # pragma: no cover
    RichHelpFormatter = argparse.HelpFormatter

try:
    from rich.console import Console
    _console = Console()
except ImportError:  # pragma: no cover
    _console = None

try:
    from gpse.utils.configuration import load_software_config

    _software_conf = load_software_config()
    __version__ = _software_conf.get("software", {}).get("version", "unknown")
except Exception:
    __version__ = "unknown"


def _log_stage(title: str) -> None:
    """Print a pretty stage separator to terminal and log to file."""
    if _console is not None:
        _console.rule(f"[bold blue]{title}[/bold blue]")
    main_logger.info(title)


def _print_easter_egg(show_question: bool = False) -> None:
    """Display a compact easter egg for the ultimate answer."""
    import time

    question = "The ultimate question of life, the universe, and everything"
    text = "The answer to the ultimate question of life, the universe, and everything is 42"
    type_delay = 0.014
    answer_segments = [
        ("The answer", "bold cyan"),
        (" to the ", "white"),
        ("ultimate question", "bold magenta"),
        (" of ", "white"),
        ("life, the universe, and everything", "bold green"),
        (" is ", "white"),
        ("42", "bold yellow"),
    ]

    if _console is None:
        if show_question:
            print(f"\nGPSE 42\n{'-' * 7}\nQ: {question}\nA: 42")
        else:
            print(f"\nGPSE 42\n{'-' * 7}")
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(type_delay)
        print("\n")
        return

    from rich.align import Align
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    def append_styled_answer(body: Text, answer_text: str) -> None:
        offset = 0
        for segment, style in answer_segments:
            if offset >= len(answer_text):
                break
            visible = answer_text[offset : offset + len(segment)]
            if visible:
                body.append(visible, style=style)
            offset += len(segment)

    def render_panel(answer_text: str) -> Panel:
        body = Text()
        if show_question:
            body.append("Q: ", style="bold cyan")
            body.append(f"{question}\n\n", style="white")
        body.append("42\n", style="bold bright_cyan")
        append_styled_answer(body, answer_text)
        return Panel(
            Align.center(body),
            title="[bold cyan]GPSE 42[/bold cyan]",
            subtitle="[dim]Don't Panic[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )

    _console.print()
    answer = ""
    with Live(render_panel(answer), console=_console, refresh_per_second=30) as live:
        for char in text:
            answer += char
            live.update(render_panel(answer))
            time.sleep(type_delay)
    _console.print("\n")


def _build_parser(
    formatter_class=argparse.HelpFormatter,
    prog: str = "python -m gpse.train.cli",
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> argparse.ArgumentParser:
    """Build the parser for the train workflow."""
    parents = parents or []
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train GPSE genomic prediction models.",
        formatter_class=formatter_class,
        add_help=help_action is None,
        parents=parents,
    )
    if help_action is not None:
        parser.add_argument(
            "-h", "--help",
            action=help_action,
            nargs=0,
            default=False,
            help="Show this help message and exit",
        )

    if not parents:
        general = parser.add_argument_group("general")
        general.add_argument(
            "-v", "--version",
            action="store_true",
            help="Show version information and exit",
        )
        general.add_argument(
            "-l", "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set logging level (default: INFO)",
        )

    req_group = parser.add_argument_group("required arguments")
    req_group.add_argument("--geno_file", type=str, help="Path to the genotype data file (CSV format)")
    req_group.add_argument("--pheno_file", type=str, help="Path to the phenotype data file (CSV format)")
    req_group.add_argument(
        "--target_trait",
        type=str,
        help="Name of the target trait (column name in the phenotype file)",
    )

    train_group = parser.add_argument_group("training arguments")
    train_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        help=(
            "List of model names to run; runs all available models if omitted. "
            "Regression: elasticnet_reg, gbdt_reg, svr_reg, mlp_reg, knn_reg, "
            "rf_reg, xgboost_reg, adaboost_reg, lightgbm_reg, catboost_reg, "
            "kernelridge_reg, histgradientboost_reg, sgd_reg, lasso_reg. "
            "Classification: rf_clf, xgboost_clf, lightgbm_clf, catboost_clf, svm_clf, mlp_clf."
        ),
    )
    train_group.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Maximum number of Optuna optimization trials per model (default: 100)",
    )
    train_group.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel workers (threads) used by each individual model internally "
            "(e.g., n_jobs for sklearn, nthread for XGBoost). Recommended: 1 when running "
            "multiple models in parallel to avoid oversubscription (default: 1)"
        ),
    )
    train_group.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of parallel training processes via ProcessPoolExecutor (default: 1)",
    )
    train_group.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    train_group.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    train_group.add_argument(
        "--n_repeats",
        type=int,
        default=100,
        help="Number of repeated training iterations (default: 100)",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience for Optuna (default: 20)",
    )
    train_group.add_argument(
        "--results_dir",
        type=str,
        default="optimization_results_v2",
        help="Directory to save optimization and prediction results (default: optimization_results_v2)",
    )
    train_group.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    train_group.add_argument(
        "--use_default_params",
        action="store_true",
        help="Use default model parameters and skip Optuna hyper-parameter optimization",
    )
    train_group.add_argument("--save_models", action="store_true", help="Save trained models (one .pkl per fold)")
    train_group.add_argument(
        "--save_representative",
        action="store_true",
        help="Save the representative model whose performance is closest to the average",
    )
    train_group.add_argument(
        "--use_same_test_set",
        action="store_true",
        default=True,
        help="Use the identical test set across all repeats for reproducibility (default: True)",
    )

    task_group = parser.add_argument_group("task configuration")
    task_group.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type: regression or classification (default: regression)",
    )
    task_group.add_argument(
        "--n_classes",
        type=int,
        default=None,
        help="Number of classes for classification tasks (required when --task_type=classification)",
    )
    task_group.add_argument(
        "--standardize_phenotype",
        action="store_true",
        help="Apply Z-score standardization to phenotype values (regression tasks only)",
    )

    cv_group = parser.add_argument_group("cross-validation options")
    cv_group.add_argument(
        "--cv_file",
        type=str,
        default=None,
        help="Path to a pre-defined CV fold file; created automatically if it does not exist",
    )
    cv_group.add_argument(
        "--force_new_cv",
        action="store_true",
        help="Force regeneration of the CV fold file even if one already exists",
    )
    cv_group.add_argument(
        "--cv_id_column",
        type=str,
        default="ID",
        help="Name of the ID column in the phenotype file used for CV fold generation (default: ID)",
    )

    stack_group = parser.add_argument_group("stacking ensemble options")
    stack_group.add_argument(
        "--use_stacking",
        action="store_true",
        help="Enable Stacking ensemble learning to fuse predictions from multiple models",
    )
    stack_group.add_argument(
        "--top_n_models",
        type=int,
        default=5,
        help="Number of top-performing models to include in the Stacking ensemble (default: 5)",
    )
    stack_group.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds used internally by Stacking to generate meta-features (default: 5)",
    )

    pre_group = parser.add_argument_group("data preprocessing options")
    pre_group.add_argument(
        "--enable_preprocess",
        action="store_true",
        help="Enable the full preprocessing pipeline: VCF/PLINK -> matrix -> phenotype matching -> cleaning",
    )
    pre_group.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Run only the preprocessing step and exit without model training",
    )
    pre_group.add_argument(
        "--preprocess_prefix",
        type=str,
        help="Output file prefix for all intermediate and final preprocessed files",
    )
    pre_group.add_argument("--vcf_file", type=str, help="Path to the raw VCF file (preprocessing input)")
    pre_group.add_argument(
        "--bfile",
        type=str,
        help="Path prefix for PLINK binary files (.bed/.bim/.fam) without the extension",
    )
    pre_group.add_argument("--ped_file", type=str, help="Path to the PLINK PED text-format file")
    pre_group.add_argument("--map_file", type=str, help="Path to the PLINK MAP file")
    pre_group.add_argument("--plink_path", type=str, default="plink", help="Path to the PLINK executable (default: plink)")
    pre_group.add_argument("--plink_out", type=str, help="Output prefix for PLINK when converting VCF to PLINK format")
    pre_group.add_argument("--extract_file", type=str, help="File containing a list of SNP IDs to extract (one per line)")
    pre_group.add_argument("--snp_dir", type=str, help="Directory containing multiple SNP-list files for batch extraction")
    pre_group.add_argument(
        "--direct_convert",
        action="store_true",
        help="Convert the entire bfile to a matrix without SNP subset extraction",
    )
    pre_group.add_argument(
        "--matrix_file",
        type=str,
        help="Path to an existing genotype matrix file; skips matrix generation",
    )
    pre_group.add_argument(
        "--raw_pheno_file",
        type=str,
        help="Path to the raw phenotype TXT file (tab-delimited, first column = ID, second = value)",
    )
    pre_group.add_argument(
        "--skip_matrix_conversion",
        action="store_true",
        help="Skip the SNP extraction and matrix conversion step",
    )
    pre_group.add_argument(
        "--skip_phenotype_match",
        action="store_true",
        help="Skip the phenotype-genotype sample matching step",
    )
    pre_group.add_argument(
        "--skip_data_clean",
        action="store_true",
        help="Skip the data-cleaning step (special-character check and replacement)",
    )
    pre_group.add_argument(
        "--load_matrix_info",
        action="store_true",
        help="Load and display basic information about an existing matrix (dimensions, sample count, etc.)",
    )

    return parser


def _show_version() -> None:
    from gpse.utils.version import show_versions
    from gpse.utils.configuration import load_software_config

    software_conf = load_software_config()
    sw = software_conf.get("software", {})
    show_versions(
        project_name=sw.get("app_name", "gpse"),
        deps=None,
        extras={
            "Version": sw.get("version", __version__),
            "Author": sw.get("author", "unknown"),
            "Email": sw.get("email", ""),
            "URL": sw.get("url", ""),
            "Description": sw.get("description", ""),
        },
        software_conf=software_conf,
    )


def _configure_logging(log_level: str) -> None:
    import logging

    level = getattr(logging, log_level.upper(), logging.INFO)
    if hasattr(main_logger, "setLevel"):
        main_logger.setLevel(level)
        for handler in getattr(main_logger, "handlers", []):
            handler.setLevel(level)


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

    ultimate_question = "the ultimate question of life, the universe, and everything"
    if len(raw_args) == 1 and raw_args[0] == "42":
        _print_easter_egg(show_question=False)
        return 0
    if raw_args:
        joined_args = " ".join(raw_args).strip().lower().rstrip("?!. ")
        if joined_args == ultimate_question:
            _print_easter_egg(show_question=True)
            return 0

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

    main_logger.info("GPSE configuration:")
    for arg, value in vars(args).items():
        main_logger.info(f"  {arg}: {value}")

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
                processed_geno_file = f"{args.preprocess_prefix}_genotype.csv"
                processed_pheno_file = f"{args.preprocess_prefix}_phenotype.csv"
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
