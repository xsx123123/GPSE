#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime workflow dispatch for ``gpse convert``."""

from __future__ import annotations

import argparse

from gpse.convert.external import (
    check_configured_external_tools,
    ensure_existing_file,
    get_convert_config,
)
from gpse.convert.processor import GenomicDataProcessor
from gpse.convert.qc import analyze_and_prune, recode_to_numeric

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    gpse_logger = logging.getLogger(__name__)


def _auto_project_config(args: argparse.Namespace) -> bool:
    return not args.no_project_config


def _get_beagle_jar_path(args: argparse.Namespace) -> str | None:
    convert_config = get_convert_config(
        args.config,
        auto_project_config=_auto_project_config(args),
    )
    return args.beagle_jar_path or convert_config.get("beagle_jar_path")


def validate_convert_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Validate mutually exclusive convert modes and return the selected mode."""
    selected = [
        bool(args.check_deps),
        bool(args.run_qc),
        bool(args.recode_prefix),
    ]
    if sum(selected) > 1:
        parser.error("Use only one of --check-deps, --run-qc, or --recode-prefix.")

    if args.check_deps:
        return "deps"
    if args.run_qc:
        if not args.input_prefix:
            parser.error("--run-qc requires --input-prefix.")
        if not args.out_prefix:
            parser.error("--run-qc requires --out-prefix.")
        return "qc"
    if args.recode_prefix:
        return "recode"

    if not args.out_prefix:
        parser.error("conversion mode requires --out-prefix.")
    return "pipeline"


def _run_pipeline(args: argparse.Namespace) -> int:
    processor = GenomicDataProcessor(
        plink_path=args.plink_path,
        config_path=args.config,
        auto_project_config=_auto_project_config(args),
    )
    return int(
        processor.process_genomic_data(
            out_prefix=args.out_prefix,
            bfile=args.bfile,
            vcf=args.vcf,
            ped_file=args.ped_file,
            map_file=args.map_file,
            extract=args.extract,
            snp_dir=args.snp_dir,
            direct=args.direct,
            plink_out=args.plink_out,
            load=args.load,
            matrix_file=args.matrix_file,
            pheno=args.pheno,
            trait_name=args.trait_name,
            standardize_phenotype=args.standardize_phenotype,
            skip_clean=args.skip_clean,
            skip_match=args.skip_match,
            skip_matrix=args.skip_matrix,
        )
    )


def _run_qc(args: argparse.Namespace) -> int:
    beagle_jar_path = _get_beagle_jar_path(args)
    params = {
        "plink_path": args.plink_path,
        "java_path": args.java_path,
        "config_path": args.config,
        "auto_project_config": _auto_project_config(args),
        "snpmaxmiss": args.snpmaxmiss,
        "samplemaxmiss": args.samplemaxmiss,
        "maf_max": args.maf,
        "r2_cutoff": args.r2_cutoff,
    }
    if beagle_jar_path:
        params["beagle_jar_path"] = beagle_jar_path

    qc_filled_prefix, pruned_prefix = analyze_and_prune(
        params,
        args.input_prefix,
        args.out_prefix,
        run_imputation=args.impute,
    )
    gpse_logger.info(f"QC filled prefix: {qc_filled_prefix}")
    gpse_logger.info(f"Pruned prefix: {pruned_prefix}")
    return 0


def _run_recode(args: argparse.Namespace) -> int:
    recode_to_numeric(args.recode_prefix)
    gpse_logger.info(f"Numeric genotype file: {args.recode_prefix}.geno")
    return 0


def _run_deps(args: argparse.Namespace) -> int:
    beagle_jar_path = _get_beagle_jar_path(args)
    results = check_configured_external_tools(
        {
            "plink": args.plink_path,
            "java": args.java_path,
        },
        config_path=args.config,
        auto_project_config=_auto_project_config(args),
    )

    has_required_failure = False
    for result in results:
        name = result["name"]
        required = result.get("required", True)
        if result["available"] and result.get("version_ok", True):
            version = result.get("version") or "unknown"
            gpse_logger.info(f"{name}: OK ({result.get('path')}, version {version})")
        elif result["available"]:
            has_required_failure = has_required_failure or required
            message = (
                f"{name}: version too low "
                f"({result.get('version')} < {result.get('min_version')})"
            )
            if required:
                gpse_logger.error(message)
            else:
                gpse_logger.warning(message)
        else:
            has_required_failure = has_required_failure or required
            status = "required" if required else "optional"
            message = f"{name}: not found ({status})"
            if required:
                gpse_logger.error(message)
            else:
                gpse_logger.warning(message)

    if beagle_jar_path:
        beagle_jar = ensure_existing_file(beagle_jar_path, name="Beagle JAR")
        gpse_logger.info(f"beagle.jar: {beagle_jar}")
    elif args.require_beagle:
        raise FileNotFoundError("Beagle JAR is required but no Beagle jar path was configured")
    else:
        gpse_logger.info("beagle.jar: not configured")

    return 1 if has_required_failure else 0


def run_convert_workflow(args: argparse.Namespace, mode: str) -> int:
    """Run the selected convert workflow mode."""
    try:
        if mode == "deps":
            return _run_deps(args)
        if mode == "qc":
            return _run_qc(args)
        if mode == "recode":
            return _run_recode(args)
        return _run_pipeline(args)
    except Exception as exc:
        gpse_logger.error(str(exc))
        return 1
