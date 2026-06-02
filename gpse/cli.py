#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPSE Command-Line Interface

This module is intentionally thin: it only defines argument parsers and routes
the execution flow. All business logic lives in ``gpse.core``.
"""

import os
import sys
import argparse
import traceback

# Ensure core/ is on sys.path for the legacy flat imports used by core modules
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_core = os.path.join(_pkg_dir, "core")
if _core not in sys.path:
    sys.path.insert(0, _core)

try:
    from rich_argparse import RichHelpFormatter
except ImportError:  # pragma: no cover
    RichHelpFormatter = argparse.HelpFormatter

from genomic_prediction_v2 import GenomicPredictorV2, main_logger
from genomic_data_pipeline import GenomicDataProcessor

try:
    from gpse.utils.configuration import load_software_config
    _software_conf = load_software_config()
    __version__ = _software_conf.get("software", {}).get("version", "unknown")
except Exception:
    __version__ = "unknown"


def _show_logo() -> None:
    """Display the GPSE logo."""
    try:
        from gpse.utils.logo import show_logo
        show_logo(
            style="welcome",
            version=__version__,
            app_name="GPSE",
            description="Genomic Prediction with Stacking Ensemble",
            url="https://github.com/xsx123123/GPSE",
            use_gradient=True,
            gradient_scheme="random",
        )
    except Exception:
        print(f"\n{'=' * 50}")
        print(f"  GPSE v{__version__}")
        print(f"  Genomic Prediction with Stacking Ensemble")
        print(f"{'=' * 50}\n")


class _LogoHelpAction(argparse.Action):
    """Custom help action that prints the logo before the help text."""

    def __call__(self, parser, namespace, values, option_string=None):
        _show_logo()
        parser.print_help()
        parser.exit()


# ── Common arguments shared by all modes ──────────────────────────
_common_parent = argparse.ArgumentParser(add_help=False)
_common_parent.add_argument(
    "-v", "--version",
    action="store_true",
    help="Show version information and exit",
)
_common_parent.add_argument(
    "-l", "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set logging level (default: INFO)",
)


def main() -> int:
    """Main entry point for the GPSE CLI."""
    parser = argparse.ArgumentParser(
        prog="gpse",
        description="GPSE — Genomic Prediction with Stacking Ensemble for horticultural crops",
        formatter_class=RichHelpFormatter,
        add_help=False,
        parents=[_common_parent],
    )
    parser.add_argument(
        "-h", "--help",
        action=_LogoHelpAction,
        nargs=0,
        default=False,
        help="Show this help message and exit",
    )

    # ── Required arguments ────────────────────────────────────────
    req_group = parser.add_argument_group("required arguments")
    req_group.add_argument(
        "--geno_file", type=str,
        help="Path to the genotype data file (CSV format)",
    )
    req_group.add_argument(
        "--pheno_file", type=str,
        help="Path to the phenotype data file (CSV format)",
    )
    req_group.add_argument(
        "--target_trait", type=str,
        help="Name of the target trait (column name in the phenotype file)",
    )

    # ── Training arguments ────────────────────────────────────────
    train_group = parser.add_argument_group("training arguments")
    train_group.add_argument(
        "--models", type=str, nargs="+",
        help="List of model names to run; runs all available models if omitted",
    )
    train_group.add_argument(
        "--trials", type=int, default=100,
        help="Maximum number of Optuna optimization trials per model (default: 100)",
    )
    train_group.add_argument(
        "--threads", type=int, default=1,
        help="Number of threads per model (recommended: 1 to avoid thread contention) (default: 1)",
    )
    train_group.add_argument(
        "--parallel_jobs", type=int, default=1,
        help="Number of parallel training jobs (multi-process) (default: 1)",
    )
    train_group.add_argument(
        "--test_size", type=float, default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    train_group.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    train_group.add_argument(
        "--n_repeats", type=int, default=100,
        help="Number of repeated training iterations (default: 100)",
    )
    train_group.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience for Optuna (default: 20)",
    )
    train_group.add_argument(
        "--results_dir", type=str, default="optimization_results_v2",
        help="Directory to save optimization and prediction results (default: optimization_results_v2)",
    )
    train_group.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    train_group.add_argument(
        "--use_default_params", action="store_true",
        help="Use default model parameters and skip Optuna hyper-parameter optimization",
    )
    train_group.add_argument(
        "--save_models", action="store_true",
        help="Save trained models (one .pkl per fold)",
    )
    train_group.add_argument(
        "--save_representative", action="store_true",
        help="Save the representative model whose performance is closest to the average",
    )
    train_group.add_argument(
        "--use_same_test_set", action="store_true", default=True,
        help="Use the identical test set across all repeats for reproducibility (default: True)",
    )

    # ── Task configuration ────────────────────────────────────────
    task_group = parser.add_argument_group("task configuration")
    task_group.add_argument(
        "--task_type", type=str, default="regression",
        choices=["regression", "classification"],
        help="Task type: regression or classification (default: regression)",
    )
    task_group.add_argument(
        "--n_classes", type=int, default=None,
        help="Number of classes for classification tasks (required when --task_type=classification)",
    )
    task_group.add_argument(
        "--standardize_phenotype", action="store_true",
        help="Apply Z-score standardization to phenotype values (regression tasks only)",
    )

    # ── Cross-validation options ──────────────────────────────────
    cv_group = parser.add_argument_group("cross-validation options")
    cv_group.add_argument(
        "--cv_file", type=str, default=None,
        help="Path to a pre-defined CV fold file; created automatically if it does not exist",
    )
    cv_group.add_argument(
        "--force_new_cv", action="store_true",
        help="Force regeneration of the CV fold file even if one already exists",
    )
    cv_group.add_argument(
        "--cv_id_column", type=str, default="ID",
        help="Name of the ID column in the phenotype file used for CV fold generation (default: ID)",
    )

    # ── Stacking ensemble options ─────────────────────────────────
    stack_group = parser.add_argument_group("stacking ensemble options")
    stack_group.add_argument(
        "--use_stacking", action="store_true",
        help="Enable Stacking ensemble learning to fuse predictions from multiple models",
    )
    stack_group.add_argument(
        "--top_n_models", type=int, default=5,
        help="Number of top-performing models to include in the Stacking ensemble (default: 5)",
    )
    stack_group.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of CV folds used internally by Stacking to generate meta-features (default: 5)",
    )

    # ── Data preprocessing options ────────────────────────────────
    pre_group = parser.add_argument_group("data preprocessing options")
    pre_group.add_argument(
        "--enable_preprocess", action="store_true",
        help="Enable the full preprocessing pipeline: VCF/PLINK → matrix → phenotype matching → cleaning",
    )
    pre_group.add_argument(
        "--preprocess_only", action="store_true",
        help="Run only the preprocessing step and exit without model training",
    )
    pre_group.add_argument(
        "--preprocess_prefix", type=str,
        help="Output file prefix for all intermediate and final preprocessed files",
    )
    pre_group.add_argument(
        "--vcf_file", type=str,
        help="Path to the raw VCF file (preprocessing input)",
    )
    pre_group.add_argument(
        "--bfile", type=str,
        help="Path prefix for PLINK binary files (.bed/.bim/.fam) without the extension",
    )
    pre_group.add_argument(
        "--ped_file", type=str,
        help="Path to the PLINK PED text-format file",
    )
    pre_group.add_argument(
        "--map_file", type=str,
        help="Path to the PLINK MAP file",
    )
    pre_group.add_argument(
        "--plink_path", type=str, default="plink",
        help="Path to the PLINK executable (default: plink)",
    )
    pre_group.add_argument(
        "--plink_out", type=str,
        help="Output prefix for PLINK when converting VCF to PLINK format",
    )
    pre_group.add_argument(
        "--extract_file", type=str,
        help="File containing a list of SNP IDs to extract (one per line)",
    )
    pre_group.add_argument(
        "--snp_dir", type=str,
        help="Directory containing multiple SNP-list files for batch extraction",
    )
    pre_group.add_argument(
        "--direct_convert", action="store_true",
        help="Convert the entire bfile to a matrix without SNP subset extraction",
    )
    pre_group.add_argument(
        "--matrix_file", type=str,
        help="Path to an existing genotype matrix file; skips matrix generation",
    )
    pre_group.add_argument(
        "--raw_pheno_file", type=str,
        help="Path to the raw phenotype TXT file (tab-delimited, first column = ID, second = value)",
    )
    pre_group.add_argument(
        "--skip_matrix_conversion", action="store_true",
        help="Skip the SNP extraction and matrix conversion step",
    )
    pre_group.add_argument(
        "--skip_phenotype_match", action="store_true",
        help="Skip the phenotype–genotype sample matching step",
    )
    pre_group.add_argument(
        "--skip_data_clean", action="store_true",
        help="Skip the data-cleaning step (special-character check and replacement)",
    )
    pre_group.add_argument(
        "--load_matrix_info", action="store_true",
        help="Load and display basic information about an existing matrix (dimensions, sample count, etc.)",
    )

    args = parser.parse_args()

    # Show help when no arguments are provided
    if len(sys.argv) == 1:
        _show_logo()
        from rich.console import Console
        _console = Console()
        _console.print("\n[bold red][ERROR] No arguments provided. Please specify at least one option.[/bold red]\n")
        parser.print_help()
        return 1

    # ── Version display ───────────────────────────────────────────
    if args.version:
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
        return 0

    # ── Configure logging level ───────────────────────────────────
    if args.log_level:
        import logging
        level = getattr(logging, args.log_level.upper(), logging.INFO)
        main_logger.setLevel(level)
        for handler in main_logger.handlers:
            handler.setLevel(level)

    # ── Parameter validation ──────────────────────────────────────
    if args.task_type == "classification" and args.n_classes is None:
        main_logger.error("Classification tasks require the --n_classes argument")
        return 1

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

    # ── Data preprocessing stage ──────────────────────────────────
    processed_geno_file = args.geno_file
    processed_pheno_file = args.pheno_file

    if args.enable_preprocess or args.preprocess_only:
        main_logger.info("=" * 70)
        main_logger.info("Starting data preprocessing")
        main_logger.info("=" * 70)

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

    # ── Training stage ────────────────────────────────────────────
    if not os.path.exists(processed_geno_file):
        main_logger.error(f"Genotype file not found: {processed_geno_file}")
        return 1
    if not os.path.exists(processed_pheno_file):
        main_logger.error(f"Phenotype file not found: {processed_pheno_file}")
        return 1

    main_logger.info("=" * 70)
    main_logger.info("Starting model training")
    main_logger.info("=" * 70)
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
        n_threads=args.threads,
        max_parallel_jobs=args.parallel_jobs,
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
