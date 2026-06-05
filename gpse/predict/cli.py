#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line stub for future GPSE prediction workflows."""

from __future__ import annotations

import argparse


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
            "-h", "--help",
            action=help_action,
            nargs=0,
            default=False,
            help="Show this help message and exit",
        )
    parser.add_argument("--model", help="Path to a trained model artifact.")
    parser.add_argument("--geno-file", help="Genotype matrix for prediction.")
    parser.add_argument("--out", help="Prediction output file.")
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
    parser.parse_args(argv)
    parser.error("predict is not implemented yet")
    return 2
