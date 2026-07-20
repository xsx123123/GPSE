#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for prediction with canonical SNP-ID alignment."""

from __future__ import annotations

import argparse
import json
import sys

from gpse.predict.core import predict


def _build_parser(
    formatter_class=argparse.HelpFormatter,
    prog: str = "python -m gpse.predict.cli",
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> argparse.ArgumentParser:
    parents = parents or []
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Predict phenotypes using trained GPSE models.",
        formatter_class=formatter_class,
        add_help=help_action is None,
        parents=parents,
    )
    if help_action is not None:
        parser.add_argument(
            "-h", "--help", action=help_action, nargs=0, default=False,
            help="Show this help message and exit",
        )
    parser.add_argument("--model", required=True, help="Model artifact or GPSE results directory.")
    parser.add_argument(
        "--vcf-file", "--geno-file", dest="genotype_file", required=True,
        help="VCF (.vcf/.vcf.gz) or converted genotype matrix.",
    )
    parser.add_argument("--out", required=True, help="Prediction CSV output file.")
    parser.add_argument(
        "--report", help="Alignment report JSON (default: <out stem>.alignment.json)."
    )
    parser.add_argument(
        "--missing-value", type=float, default=3.0,
        help="Value used for model SNPs absent from input (default: 3, GPSE missing genotype code).",
    )
    parser.add_argument(
        "--min-feature-coverage", "--min_feature_coverage",
        type=float,
        default=0.0,
        help="Reject prediction when matched model-SNP coverage is below this 0-1 threshold (default: 0).",
    )
    return parser


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
        prog=prog or "python -m gpse.predict.cli",
        help_action=help_action,
        parents=parents,
    )
    args = parser.parse_args(argv)
    try:
        report = predict(
            args.model,
            args.genotype_file,
            args.out,
            missing_value=args.missing_value,
            report_file=args.report,
            min_feature_coverage=args.min_feature_coverage,
        )
        print(json.dumps(report, indent=2))
        if report["missing_model_snp_count"]:
            print(
                f"WARNING: {report['missing_model_snp_count']} model SNP(s) are absent from input; "
                f"filled with {report['missing_value_used']}.",
                file=sys.stderr,
            )
        if report["coverage_warning"]:
            print(f"WARNING: {report['coverage_warning']}", file=sys.stderr)
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
