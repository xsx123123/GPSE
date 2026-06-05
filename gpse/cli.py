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

# ── Early thread control ──────────────────────────────────────
# MUST set env vars BEFORE importing numpy/scipy/sklearn, which
# initialize BLAS/MKL/OpenMP thread pools at import time.
# Once those pools are created, later changes to OMP_NUM_THREADS
# etc. have NO effect on already-initialized thread pools.
_thread_pre_parser = argparse.ArgumentParser(add_help=False)
_thread_pre_parser.add_argument("--n_jobs", type=int, default=1)
_thread_pre_args, _ = _thread_pre_parser.parse_known_args()
_thread_n = str(_thread_pre_args.n_jobs)
for _env_var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_env_var] = _thread_n
del _thread_pre_parser, _thread_pre_args, _thread_n, _env_var
# ── End early thread control ──────────────────────────────────

try:
    from rich_argparse import RichHelpFormatter
except ImportError:  # pragma: no cover
    RichHelpFormatter = argparse.HelpFormatter

try:
    from rich.console import Console
    _console = Console()
except ImportError:  # pragma: no cover
    _console = None

try:
    from gpse.utils.configuration import load_software_config
    _software_conf = load_software_config()
    __version__ = _software_conf.get("software", {}).get("version", "unknown")
except Exception:
    __version__ = "unknown"


def _show_logo() -> None:
    """Display the GPSE logo."""
    try:
        from gpse.utils.logo import show_logo
        show_logo(
            style="welcome",
            version=__version__,
            app_name="GPSE",
            description="Genomic Prediction with Stacking Ensemble",
            url="https://github.com/xsx123123/GPSE",
            use_gradient=True,
            gradient_scheme="random",
        )
    except Exception:
        print(f"\n{'=' * 50}")
        print(f"  GPSE v{__version__}")
        print(f"  Genomic Prediction with Stacking Ensemble")
        print(f"{'=' * 50}\n")


def _print_easter_egg(show_question: bool = False) -> None:
    """Display a compact easter egg for the ultimate answer."""
    import time

    question = "The ultimate question of life, the universe, and everything"
    text = "The answer to the ultimate question of life, the universe, and everything is 42"
    type_delay = 0.014
    answer_segments = [
        ("The answer", "bold cyan"),
        (" to the ", "white"),
        ("ultimate question", "bold magenta"),
        (" of ", "white"),
        ("life, the universe, and everything", "bold green"),
        (" is ", "white"),
        ("42", "bold yellow"),
    ]

    if _console is None:
        if show_question:
            print(f"\nGPSE 42\n{'-' * 7}\nQ: {question}\nA: 42")
        else:
            print(f"\nGPSE 42\n{'-' * 7}")
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(type_delay)
        print("\n")
        return

    from rich.align import Align
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    def append_styled_answer(body: Text, answer_text: str) -> None:
        offset = 0
        for segment, style in answer_segments:
            if offset >= len(answer_text):
                break
            visible = answer_text[offset : offset + len(segment)]
            if visible:
                body.append(visible, style=style)
            offset += len(segment)

    def render_panel(answer_text: str) -> Panel:
        body = Text()
        if show_question:
            body.append("Q: ", style="bold cyan")
            body.append(f"{question}\n\n", style="white")
        body.append("42\n", style="bold bright_cyan")
        append_styled_answer(body, answer_text)
        return Panel(
            Align.center(body),
            title="[bold cyan]GPSE 42[/bold cyan]",
            subtitle="[dim]Don't Panic[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )

    _console.print()
    answer = ""
    with Live(render_panel(answer), console=_console, refresh_per_second=30) as live:
        for char in text:
            answer += char
            live.update(render_panel(answer))
            time.sleep(type_delay)
    _console.print("\n")


class _LogoHelpAction(argparse.Action):
    """Custom help action that prints the logo before the help text."""

    def __call__(self, parser, namespace, values, option_string=None):
        _show_logo()
        parser.print_help()
        parser.exit()


# ── Common arguments shared by all modes ──────────────────────────
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


def _build_root_parser() -> argparse.ArgumentParser:
    """Build the top-level GPSE command router parser."""
    parser = argparse.ArgumentParser(
        prog="gpse",
        description="GPSE — Genomic Prediction with Stacking Ensemble for horticultural crops",
        formatter_class=RichHelpFormatter,
        add_help=False,
        parents=[_common_parent],
    )
    parser.add_argument(
        "-h", "--help",
        action=_LogoHelpAction,
        nargs=0,
        default=False,
        help="Show this help message and exit",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{train,convert,predict}",
        title="workflow commands",
        description="Run one of the GPSE workflows",
    )
    subparsers.add_parser(
        "train",
        add_help=False,
        help="Train genomic prediction models from genotype and phenotype matrices",
        description="Train genomic prediction models.",
    )
    subparsers.add_parser(
        "convert",
        add_help=False,
        help="Convert raw genotype/phenotype inputs and run optional QC",
        description="Convert raw genotype/phenotype inputs and run optional QC.",
    )
    subparsers.add_parser(
        "predict",
        add_help=False,
        help="Predict phenotypes using trained GPSE models",
        description="Predict phenotypes using trained GPSE models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the GPSE CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    root_parser = _build_root_parser()

    if not raw_args:
        _show_logo()
        if _console is not None:
            _console.print("\n[bold red][ERROR] No command provided. Use train, convert, or predict.[/bold red]\n")
        else:
            print("\n[ERROR] No command provided. Use train, convert, or predict.\n")
        root_parser.print_help()
        return 1

    if raw_args[0] in {"-h", "--help"}:
        root_parser.parse_args(raw_args)
        return 0

    ultimate_question = "the ultimate question of life, the universe, and everything"
    if len(raw_args) == 1 and raw_args[0] == "42":
        _print_easter_egg(show_question=False)
        return 0
    joined_args = " ".join(raw_args).strip().lower().rstrip("?!. ")
    if joined_args == ultimate_question:
        _print_easter_egg(show_question=True)
        return 0

    if raw_args[0] in {"-v", "--version"}:
        from gpse.train.cli import main as train_main

        return train_main(["--version"], prog="gpse")

    command = raw_args[0]
    command_args = raw_args[1:]

    if command == "train":
        from gpse.train.cli import main as train_main

        return train_main(
            command_args,
            formatter_class=RichHelpFormatter,
            prog="gpse train",
            help_action=_LogoHelpAction,
            parents=[_common_parent],
        )

    if command == "convert":
        from gpse.convert.cli import main as convert_main

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

    if command == "predict":
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

    root_parser.error(f"Unknown command: {command}. Use train, convert, or predict.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
