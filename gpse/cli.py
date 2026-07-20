#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPSE Command-Line Interface

This module is intentionally thin: it only defines argument parsers and routes
the execution flow. Training business logic lives in ``gpse.train``.
"""

import os
import sys
import argparse

# Configure native thread pools before importing modules that may load
# numpy/scipy/sklearn. BLAS/MKL/OpenMP read these environment variables only
# when their pools are initialized, so this has to happen at the top.
_thread_pre_parser = argparse.ArgumentParser(add_help=False)
_thread_pre_parser.add_argument("--n_jobs", type=int, default=1)
_thread_pre_args, _ = _thread_pre_parser.parse_known_args()
_thread_n = str(_thread_pre_args.n_jobs)
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_env_var] = _thread_n
del _thread_pre_parser, _thread_pre_args, _thread_n, _env_var

# CLI display helpers are imported after thread setup because they can
# eventually touch package modules that import numerical dependencies.
from gpse.utils.cli_display import (
    ULTIMATE_QUESTION,
    _build_root_parser,
    _show_logo_for_command,
    print_easter_egg,
    show_gpse_logo,
)

# Shared argparse action: prints the logo before root/subcommand help.
from gpse.config.constants import _LogoHelpAction

# Prefer rich-formatted help when available; keep argparse as fallback so the
# CLI still works in minimal environments.
try:
    from rich_argparse import RichHelpFormatter
except ImportError:
    RichHelpFormatter = argparse.HelpFormatter

# Console is optional. All user-facing error paths below have plain print()
# fallbacks for environments without rich.
try:
    from rich.console import Console
    _console = Console()
except ImportError:
    _console = None

# Common root options are passed into subcommand parsers as parents. This keeps
# flags such as --version and --log-level consistent across gpse commands.
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


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the GPSE CLI.

    argv is injectable for tests and internal calls. When argv is None, the
    function reads the real command-line arguments from sys.argv.
    """
    # Normalize inputs to a list copy so downstream slicing and checks are
    # stable even when tests pass tuples or other sequences.
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Build the lightweight root router. Detailed workflow arguments are
    # defined inside gpse.convert.cli, gpse.train.cli, and gpse.predict.cli.
    root_parser = _build_root_parser(
        formatter_class=RichHelpFormatter,
        parents=[_common_parent],
        help_action=_LogoHelpAction,
    )

    # No subcommand was provided. Show the logo and root help, then fail with
    # exit code 1 because this is an invalid invocation.
    if not raw_args:
        show_gpse_logo()
        if _console is not None:
            _console.print("\n[bold red][ERROR] No command provided. Use convert, train, predict, or batch.[/bold red]\n")
        else:
            print("\n[ERROR] No command provided. Use convert, train, predict, or batch.\n")
        root_parser.print_help()
        return 1

    # Root help is handled by the root parser so the custom help action can
    # print the logo before the help text.
    if raw_args[0] in {"-h", "--help"}:
        root_parser.parse_args(raw_args)
        return 0

    # Root version delegates to the train CLI because that module owns the
    # current version display implementation.
    if raw_args[0] in {"-v", "--version"}:
        from gpse.train.cli import main as train_main

        return train_main(["--version"], prog="gpse")

    # Easter egg shortcuts are handled before subcommand dispatch, so commands
    # like `gpse 42` do not get treated as unknown subcommands.
    if len(raw_args) == 1 and raw_args[0] == "42":
        print_easter_egg(show_question=False)
        return 0

    # Also accept the full question, ignoring case and trailing punctuation.
    joined_args = " ".join(raw_args).strip().lower().rstrip("?!. ")
    if joined_args == ULTIMATE_QUESTION:
        print_easter_egg(show_question=True)
        return 0

    # Split root command from the arguments that belong to the selected
    # workflow. The workflow modules receive only their own arguments.
    command = raw_args[0]
    command_args = raw_args[1:]

    # Convert workflow: import lazily so gpse --help and gpse --version stay
    # lightweight and do not load workflow dependencies unnecessarily.
    if command == "convert":
        # Display the logo before convert/predict help.
        _show_logo_for_command(command_args)

        from gpse.convert.cli import main as convert_main

        # convert/predict do not own version rendering; route to the shared
        # version implementation while preserving the command name in prog.
        if command_args and command_args[0] in {"-v", "--version"}:
            from gpse.train.cli import main as train_main

            return train_main(["--version"], prog="gpse convert")
        return convert_main(
            command_args,
            formatter_class=RichHelpFormatter,
            prog="gpse convert",
            help_action=_LogoHelpAction,
            parents=[_common_parent],
        )

    # Train workflow.
    if command == "train":
        # Display the logo before convert/predict help.
        _show_logo_for_command(command_args)

        from gpse.train.cli import main as train_main

        return train_main(
            command_args,
            formatter_class=RichHelpFormatter,
            prog="gpse train",
            help_action=_LogoHelpAction,
            parents=[_common_parent],
        )

    # Predict workflow.
    if command == "predict":
        # Display the logo before convert/predict help.
        _show_logo_for_command(command_args)

        from gpse.predict.cli import main as predict_main

        if command_args and command_args[0] in {"-v", "--version"}:
            from gpse.train.cli import main as train_main

            return train_main(["--version"], prog="gpse predict")
        return predict_main(
            command_args,
            formatter_class=RichHelpFormatter,
            prog="gpse predict",
            help_action=_LogoHelpAction,
            parents=[_common_parent],
        )

    # Batch workflow: train many traits from one YAML config.
    if command == "batch":
        _show_logo_for_command(command_args)

        from gpse.batch.cli import main as batch_main

        if command_args and command_args[0] in {"-v", "--version"}:
            from gpse.train.cli import main as train_main

            return train_main(["--version"], prog="gpse batch")
        return batch_main(
            command_args,
            formatter_class=RichHelpFormatter,
            prog="gpse batch",
            parents=[_common_parent],
        )

    # Any other first token is not a supported workflow command.
    root_parser.error(f"Unknown command: {command}. Use train, convert, predict, or batch.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
