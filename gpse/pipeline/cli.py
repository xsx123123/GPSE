#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line entry point for ``gpse pipeline``.

The pipeline command chains the convert and train workflows: it runs the
full ``gpse convert`` pipeline on raw genotype/phenotype inputs, locates the
converted matrices, and then invokes ``gpse train`` on them. Options that do
not belong to the convert stage are forwarded verbatim to the train stage.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime

from gpse.convert.workflow import run_convert_workflow, validate_convert_mode
from gpse.utils.cli_display import _build_pipeline_parser
from gpse.utils.log_utils import logger_init

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    gpse_logger = logging.getLogger(__name__)

# Train options that the pipeline supplies itself (or that conflict with the
# chained workflow) and therefore must not be passed through by the user.
_BLOCKED_TRAIN_OPTIONS = (
    "--geno_file",
    "--pheno_file",
    "--enable_preprocess",
    "--preprocess_only",
    "--preprocess_prefix",
    "--vcf_file",
    "--raw_pheno_file",
)


def _find_blocked_train_options(train_extra: list[str]) -> list[str]:
    """Return forwarded options that conflict with the chained workflow."""
    blocked = []
    for token in train_extra:
        option = token.split("=", 1)[0]
        if option in _BLOCKED_TRAIN_OPTIONS:
            blocked.append(option)
    return blocked


def _resolve_convert_outputs(args: argparse.Namespace) -> tuple[str, str]:
    """Locate the convert-stage genotype/phenotype matrices for training."""
    if args.skip_match or not args.pheno:
        for ext in (".parquet", ".feather", ".csv"):
            candidate = args.out_prefix + ext
            if os.path.exists(candidate):
                return candidate, args.pheno
        raise FileNotFoundError(
            f"Convert stage finished but no genotype matrix was found at "
            f"'{args.out_prefix}.parquet/.feather/.csv'. If you used "
            f"--skip-matrix, provide the matrix via --matrix-file instead."
        )

    # The convert stage writes one matched file pair per trait using the
    # sanitized trait column name: {prefix}_{trait}_genotype.{ext} and
    # {prefix}_{trait}_phenotype.{ext}.
    trait = args.trait_name or args.target_trait
    safe_trait = re.sub(r"[^\w\-]", "_", trait)
    for ext in (".parquet", ".feather", ".csv"):
        geno_file = f"{args.out_prefix}_{safe_trait}_genotype{ext}"
        pheno_file = f"{args.out_prefix}_{safe_trait}_phenotype{ext}"
        if os.path.exists(geno_file) and os.path.exists(pheno_file):
            return geno_file, pheno_file

    produced = sorted(glob.glob(f"{args.out_prefix}_*_genotype.*"))
    hint = ""
    if produced:
        hint = " Produced trait matrices: " + ", ".join(os.path.basename(p) for p in produced)
    raise FileNotFoundError(
        f"Convert stage finished but the converted matrices for trait "
        f"'{trait}' were not found (expected '{args.out_prefix}_{safe_trait}_genotype.*'). "
        f"Check that --target_trait matches a trait column in the phenotype file.{hint}"
    )


def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    """Run the convert stage, then the train stage on the converted files."""
    parser = _build_pipeline_parser(
        formatter_class=formatter_class or argparse.HelpFormatter,
        prog=prog or "python -m gpse.pipeline.cli",
        help_action=help_action,
        parents=parents,
    )
    args, train_extra = parser.parse_known_args(argv)

    if getattr(args, "version", False):
        from gpse.train.cli import main as train_main

        return train_main(["--version"], prog="gpse pipeline")

    # Determine the convert mode first so missing-argument errors surface
    # before any logging initialisation, mirroring 'gpse convert'.
    mode = validate_convert_mode(parser, args)

    # Validate the train stage before doing any conversion work so typos in
    # train arguments fail fast instead of after a long conversion.
    if mode == "pipeline":
        if not args.target_trait:
            parser.error("the train stage requires --target_trait")
        if not args.task_type:
            parser.error("the train stage requires --task_type (regression or classification)")
        blocked = _find_blocked_train_options(train_extra)
        if blocked:
            parser.error(
                f"{', '.join(blocked)} must not be passed to 'gpse pipeline': "
                "the pipeline derives the training inputs from the convert-stage outputs"
            )
        if not args.trait_name:
            # The train stage uses a single trait; restrict the convert stage
            # to it so multi-trait phenotype files do not produce one genotype
            # matrix copy per trait. Also fails fast with a clear message when
            # --target_trait is not a column in the phenotype file.
            args.trait_name = args.target_trait

    log_level = getattr(args, "log_level", "INFO")
    log_file = None
    if mode == "pipeline" and getattr(args, "out_prefix", None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.dirname(args.out_prefix) or "."
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"gpse_pipeline_{timestamp}.log")
    logger_init(logger_name=log_file, log_level=log_level)

    # Standalone convert utilities (--check-deps, standalone --run-qc,
    # --recode-prefix) run as-is and stop before the train stage.
    if mode != "pipeline":
        return run_convert_workflow(args, mode)

    gpse_logger.info("[Pipeline] Stage 1/2: convert (raw inputs -> training-ready matrices)")
    if args.trait_name:
        gpse_logger.info(f"[Pipeline] Convert stage limited to trait '{args.trait_name}'")
    rc = run_convert_workflow(args, mode)
    if rc != 0:
        gpse_logger.error(f"[Pipeline] Convert stage failed (exit {rc}); train stage aborted")
        return rc

    try:
        geno_file, pheno_file = _resolve_convert_outputs(args)
    except FileNotFoundError as exc:
        gpse_logger.error(f"[Pipeline] {exc}")
        return 1

    gpse_logger.info("[Pipeline] Stage 2/2: train (model optimization + stacking ensemble)")
    gpse_logger.info(f"[Pipeline] Genotype file: {geno_file}")
    gpse_logger.info(f"[Pipeline] Phenotype file: {pheno_file}")

    train_argv = list(train_extra)
    if args.preserve_vcf_snp_ids:
        # The convert stage consumed this flag; the train stage needs it too so
        # the feature manifest records the VCF-id compatibility mode.
        train_argv.append("--preserve-vcf-snp-ids")
    train_argv += [
        "--geno_file", geno_file,
        "--pheno_file", pheno_file,
        "--target_trait", args.target_trait,
        "--task_type", args.task_type,
        "--log-level", log_level,
    ]

    from gpse.train.cli import main as train_main

    return train_main(
        train_argv,
        formatter_class=formatter_class,
        prog="gpse train",
        help_action=help_action,
        parents=parents,
    )


if __name__ == "__main__":
    # Allows direct execution: python -m gpse.pipeline.cli ...
    sys.exit(main())
