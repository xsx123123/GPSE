#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI workflow helpers for ``gpse train``."""

from __future__ import annotations

import argparse

from gpse.train.predictor import main_logger
from gpse.utils.cli_display import _load_software_info

try:
    from rich.console import Console

    _console = Console()
except ImportError:  # pragma: no cover
    _console = None

__version__ = _load_software_info().get("version", "unknown")


def _log_stage(title: str) -> None:
    """Print a pretty stage separator to terminal and log to file."""
    if _console is not None:
        _console.rule(f"[bold blue]{title}[/bold blue]")
    main_logger.info(title)


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
    req_group.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["regression", "classification"],
        help="Task type: regression or classification (required)",
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
            "kernelridge_reg, gblup_reg, histgradientboost_reg, sgd_reg, lasso_reg. "
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
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    train_group.add_argument(
        "--train_folds",
        dest="n_splits",
        metavar="TRAIN_FOLDS",
        type=int,
        default=5,
        help="Number of folds for repeated train/validation evaluation (default: 5)",
    )
    train_group.add_argument(
        "--n_splits",
        dest="n_splits",
        type=int,
        help=argparse.SUPPRESS,
    )
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
        "--cv_stability_penalty",
        type=float,
        default=0.5,
        help="Optuna objective penalty λ in mean(CV) - λ·std(CV) (default: 0.5)",
    )
    train_group.add_argument(
        "--optuna_per_repeat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-optimize every repeat; default freezes repeat-1 parameters for later repeats",
    )
    train_group.add_argument(
        "--split_strategy",
        choices=["random", "structure_aware"],
        default="random",
        help="Hold-out strategy; structure_aware keeps PCA/KMeans genetic clusters intact",
    )
    train_group.add_argument(
        "--structure_clusters",
        type=int,
        default=None,
        help="Optional KMeans cluster count for --split_strategy structure_aware",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one fixed test set across repeats; use --no-use_same_test_set for per-repeat splits (default: True)",
    )

    performance_group = parser.add_argument_group("performance arguments")
    performance_group.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Threads used inside each model training task. Recommended: 1 when "
            "using --max_workers or --repeat_workers to avoid oversubscription (default: 1)"
        ),
    )
    performance_group.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Model-level parallelism: maximum number of models to train at the same time (default: 1)",
    )
    performance_group.add_argument(
        "--repeat_workers",
        type=int,
        default=1,
        help="Repeat-level parallelism: maximum number of repeats per model to run at the same time (default: 1)",
    )
    performance_group.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Target total parallelism (number of concurrent training units). When set, "
            "GPSE automatically derives --max_workers and --repeat_workers from this budget "
            "(with --n_jobs kept at 1). Useful shorthand for: --threads 100. "
            "Cannot exceed available CPU cores."
        ),
    )

    task_group = parser.add_argument_group("task configuration")
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

    feature_group = parser.add_argument_group("feature selection options")
    feature_group.add_argument(
        "--feature_selection",
        choices=[
            "none",
            "variance",
            "univariate",
            "mutual_info",
            "tree_importance",
            "lasso",
            "elasticnet",
            "pca",
            "svd",
        ],
        default="none",
        help="Fold-local SNP selection strategy; default keeps every feature",
    )
    feature_group.add_argument(
        "--select_k",
        type=int,
        default=5000,
        help=(
            "Maximum SNPs retained by --feature_selection univariate or mutual_info "
            "or tree_importance or lasso or elasticnet, or output components for pca "
            "and svd (default: 5000)"
        ),
    )
    feature_group.add_argument(
        "--select_percentile",
        type=float,
        default=None,
        help=(
            "Percent of SNPs/components to retain for eligible feature-selection methods; "
            "overrides --select_k"
        ),
    )
    feature_group.add_argument(
        "--variance_threshold",
        type=float,
        default=0.0,
        help="VarianceThreshold cutoff for --feature_selection variance (default: 0.0)",
    )
    feature_group.add_argument(
        "--genotype_imputation",
        choices=["none", "mean"],
        default="none",
        help="Fold-local genotype missing-value handling; mean treats encoded missing calls as missing",
    )
    feature_group.add_argument(
        "--missing_genotype_code",
        type=float,
        default=3.0,
        help="Encoded missing genotype value used by --genotype_imputation mean (default: 3)",
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
        help="Number of internal folds used by Stacking to generate meta-features (default: 5)",
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
