#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line and workflow entry point for the GPSE convert module.

This module does two things:
1. Defines the command-line arguments for ``gpse convert``.
2. Starts the selected convert workflow: genotype conversion, QC/LD pruning,
   numeric recoding, or external dependency validation.
"""
from __future__ import annotations

import sys
import argparse

from gpse.convert.workflow import run_convert_workflow, validate_convert_mode
from gpse.utils.cli_display import _build_convert_parser

# define convert main function
def main(argv: list[str] | None = None,*,formatter_class=None,prog: str | None = None,help_action=None,parents: list[argparse.ArgumentParser] | None = None) -> int:
    """
    Parse convert arguments, select a mode, and start the workflow.
    """
    parser = _build_convert_parser(
        formatter_class = formatter_class or argparse.HelpFormatter,
        prog = prog or "python -m gpse.convert.cli",
        help_action = help_action,
        parents = parents,
    )
    args = parser.parse_args(argv)
    mode = validate_convert_mode(parser, args)
    return run_convert_workflow(args, mode)


if __name__ == "__main__":
    # Allows direct execution: python -m gpse.convert.cli ...
    sys.exit(main())
