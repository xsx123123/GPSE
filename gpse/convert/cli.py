#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for the GPSE convert module.

Provides a unified entry point for genotype format conversion, QC/LD pruning,
numeric recoding, and external dependency validation.
"""

from __future__ import annotations

import argparse
import sys

from gpse.convert.processor import GenomicDataProcessor
from gpse.convert.qc import analyze_and_prune, recode_to_numeric
from gpse.convert.external import (
    check_configured_external_tools,
    ensure_existing_file,
    get_convert_config,
)

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging
    gpse_logger = logging.getLogger(__name__)


def _build_parser(
    formatter_class=argparse.HelpFormatter,
    prog: str = "python -m gpse.convert.cli",
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> argparse.ArgumentParser:
    parents = parents or []
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Test CLI for GPSE genotype conversion, QC, and dependency checks.",
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

    general = parser.add_argument_group("general")
    if not parents:
        general.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level.",
        )
    general.add_argument(
        "--config",
        help="User YAML config path. Overrides package defaults and project config files.",
    )
    general.add_argument(
        "--no-project-config",
        action="store_true",
        help="Do not auto-load gpse.yaml or gpse.local.yaml from the current directory.",
    )

    mode = parser.add_argument_group("test modes")
    mode.add_argument(
        "--check-deps",
        action="store_true",
        help="Check external dependencies and exit.",
    )
    mode.add_argument(
        "--run-qc",
        action="store_true",
        help="Run QC, optional Beagle imputation, and LD pruning.",
    )
    mode.add_argument(
        "--recode-prefix",
        help="Convert PED/MAP compound genotypes at this prefix to numeric additive coding and exit.",
    )

    io_group = parser.add_argument_group("conversion input/output")
    io_group.add_argument("--out-prefix", help="Output file prefix.")
    io_group.add_argument("--bfile", help="Input PLINK BED/BIM/FAM prefix.")
    io_group.add_argument("--vcf", help="Input VCF file path.")
    io_group.add_argument("--ped-file", help="Input PED file path.")
    io_group.add_argument("--map-file", help="Input MAP file path.")
    io_group.add_argument("--matrix-file", help="Existing genotype matrix CSV.")
    io_group.add_argument("--extract", help="SNP ID list file for PLINK --extract.")
    io_group.add_argument("--snp-dir", help="Directory containing SNP list .txt files.")
    io_group.add_argument("--direct", action="store_true", help="Convert whole bfile to matrix.")
    io_group.add_argument("--plink-out", help="PLINK output prefix used during VCF conversion.")
    io_group.add_argument("--load", action="store_true", help="Load and print matrix info.")
    io_group.add_argument("--skip-clean", action="store_true", help="Reserved compatibility flag.")
    io_group.add_argument("--skip-match", action="store_true", help="Skip phenotype/genotype matching.")
    io_group.add_argument("--skip-matrix", action="store_true", help="Skip matrix generation.")

    pheno = parser.add_argument_group("phenotype")
    pheno.add_argument("--pheno", help="Phenotype file path.")
    pheno.add_argument("--trait-name", help="Rename phenotype trait column.")
    pheno.add_argument(
        "--standardize-phenotype",
        action="store_true",
        help="Apply z-score standardization to the phenotype column.",
    )

    tools = parser.add_argument_group("external tools")
    tools.add_argument(
        "--plink-path",
        "--plink",
        dest="plink_path",
        help="PLINK executable path. Defaults to software.yaml config.",
    )
    tools.add_argument(
        "--java-path",
        help="Java executable path. Defaults to software.yaml config.",
    )
    tools.add_argument(
        "--beagle-jar-path",
        help="Beagle jar path. Defaults to convert.beagle_jar_path in user YAML.",
    )
    tools.add_argument(
        "--require-beagle",
        action="store_true",
        help="Fail dependency checks if Beagle jar is not configured.",
    )

    qc = parser.add_argument_group("QC")
    qc.add_argument(
        "--input-prefix",
        help="Input genotype prefix for --run-qc, without extension.",
    )
    qc.add_argument("--snpmaxmiss", type=float, default=0.1, help="PLINK --geno threshold.")
    qc.add_argument("--samplemaxmiss", type=float, default=0.1, help="PLINK --mind threshold.")
    qc.add_argument("--maf", type=float, default=0.05, help="PLINK --maf threshold.")
    qc.add_argument("--r2-cutoff", type=float, default=0.2, help="LD pruning r2 cutoff.")
    qc.add_argument("--impute", action="store_true", help="Use Beagle imputation before pruning.")

    return parser


def _auto_project_config(args: argparse.Namespace) -> bool:
    return not args.no_project_config


def _get_beagle_jar_path(args: argparse.Namespace) -> str | None:
    convert_config = get_convert_config(
        args.config,
        auto_project_config=_auto_project_config(args),
    )
    return args.beagle_jar_path or convert_config.get("beagle_jar_path")


def _validate_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
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


def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    parser = _build_parser(
        formatter_class=formatter_class or argparse.HelpFormatter,
        prog=prog or "python -m gpse.convert.cli",
        help_action=help_action,
        parents=parents,
    )
    args = parser.parse_args(argv)
    mode = _validate_mode(parser, args)

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


if __name__ == "__main__":
    sys.exit(main())
