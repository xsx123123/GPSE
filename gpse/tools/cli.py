#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line interface for ``gpse tools`` utility subcommands."""

from __future__ import annotations

import argparse
import sys


def _build_split_parser(
    subparsers: argparse._SubParsersAction,
    formatter_class=argparse.HelpFormatter,
) -> None:
    """Register the ``split`` subcommand under ``gpse tools``."""
    parser = subparsers.add_parser(
        "split",
        formatter_class=formatter_class,
        help="Split genotype & phenotype data into train / test subsets",
        description=(
            "Split matched genotype and phenotype samples into training "
            "and test (prediction) subsets.  The test set can later be "
            "fed to ``gpse predict``."
        ),
    )
    parser.add_argument(
        "--geno",
        required=True,
        metavar="FILE",
        help="Genotype matrix (CSV / Parquet / Feather).",
    )
    parser.add_argument(
        "--pheno",
        required=True,
        metavar="FILE",
        help="Phenotype table (CSV / Parquet / Feather).",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        metavar="PREFIX",
        help=(
            "Output prefix.  Produces <PREFIX>_train_geno, "
            "<PREFIX>_train_pheno, <PREFIX>_test_geno, <PREFIX>_test_pheno."
        ),
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for the test set (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42).",
    )
    parser.add_argument(
        "--stratify",
        metavar="COLUMN",
        default=None,
        help="Phenotype column for stratified splitting (classification).",
    )
    parser.add_argument(
        "--id-column",
        metavar="NAME",
        default=None,
        help="Sample-ID column name (default: auto-detect 'ID' or first column).",
    )


def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    """Parse ``gpse tools`` arguments and dispatch to the chosen subcommand."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog=prog or "gpse tools",
        description="GPSE utility tools for data preparation and analysis.",
        formatter_class=formatter_class or argparse.HelpFormatter,
        add_help=help_action is None,
        parents=parents or [],
    )
    if help_action is not None:
        parser.add_argument(
            "-h", "--help",
            action=help_action,
            nargs=0,
            default=False,
            help="Show this help message and exit",
        )

    subparsers = parser.add_subparsers(
        dest="tool_command",
        metavar="{split}",
        title="tool commands",
    )
    _build_split_parser(subparsers, formatter_class=formatter_class or argparse.HelpFormatter)

    # Put tool commands above options in help output (same trick as root parser).
    group_by_title = {g.title: g for g in parser._action_groups}
    tool_group = group_by_title.get("tool commands")
    options_group = group_by_title.get("options") or group_by_title.get("optional arguments")
    if tool_group is not None and options_group is not None:
        groups = parser._action_groups
        try:
            tool_idx = groups.index(tool_group)
            opt_idx = groups.index(options_group)
        except ValueError:
            pass
        else:
            if tool_idx > opt_idx:
                groups[opt_idx], groups[tool_idx] = groups[tool_idx], groups[opt_idx]

    args = parser.parse_args(raw_args)

    if args.tool_command is None:
        parser.print_help()
        return 1

    # ---- dispatch: split ------------------------------------------------
    if args.tool_command == "split":
        from gpse.utils.log_utils import logger_init

        logger_init(log_level=getattr(args, "log_level", "INFO"))

        from gpse.tools.split import split_dataset

        try:
            summary = split_dataset(
                geno_file=args.geno,
                pheno_file=args.pheno,
                out_prefix=args.out_prefix,
                test_ratio=args.test_ratio,
                seed=args.seed,
                stratify_column=args.stratify,
                id_column=args.id_column,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        from loguru import logger

        logger.info(
            f"Split complete — train: {summary['train_samples']}, "
            f"test: {summary['test_samples']} "
            f"(ratio {summary['test_ratio']:.0%}, seed {summary['seed']})"
        )
        return 0

    parser.error(f"Unknown tool command: {args.tool_command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
