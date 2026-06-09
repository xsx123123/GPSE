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

import os
import sys
import argparse
from datetime import datetime

from gpse.convert.workflow import run_convert_workflow, validate_convert_mode
from gpse.utils.cli_display import _build_convert_parser
from gpse.utils.log_utils import logger_init

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

    # Determine workflow mode first so that missing-argument errors and --help
    # are handled before initialising the logger.  This avoids spurious
    # "GPSE config loaded" messages when the user only wants usage info.
    mode = validate_convert_mode(parser, args)

    # Initialise unified GPSE logging so that convert uses the same
    # Rich-styled console format and file rotation as the train module.
    log_level = getattr(args, "log_level", "INFO")
    log_file = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(args, "out_prefix", None):
        log_dir = os.path.dirname(args.out_prefix) or "."
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"gpse_convert_{timestamp}.log")
    elif getattr(args, "recode_prefix", None):
        log_dir = os.path.dirname(args.recode_prefix) or "."
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"gpse_convert_{timestamp}.log")
    else:
        # No output prefix specified (e.g. --check-deps).
        # Skip file logging — only use console output.
        log_file = None

    logger_init(logger_name=log_file, log_level=log_level)

    return run_convert_workflow(args, mode)


if __name__ == "__main__":
    # Allows direct execution: python -m gpse.convert.cli ...
    sys.exit(main())
