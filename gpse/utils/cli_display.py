#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared terminal display helpers for GPSE command-line entry points.
"""

from __future__ import annotations

import argparse
import sys
import time

ULTIMATE_QUESTION = "the ultimate question of life, the universe, and everything"
_ULTIMATE_QUESTION_DISPLAY = "The ultimate question of life, the universe, and everything"
_ULTIMATE_ANSWER_TEXT = (
    "The answer to the ultimate question of life, the universe, and everything is 42"
)


def _load_software_info() -> dict[str, str]:
    try:
        from gpse.utils.configuration import load_software_config

        software_conf = load_software_config()
        software = software_conf.get("software", {})
    except Exception:
        software = {}

    return {
        "version": software.get("version", "unknown"),
        "app_name": software.get("app_name", "GPSE"),
        "description": software.get(
            "description",
            "Genomic Prediction with Stacking Ensemble",
        ),
        "url": software.get("url", ""),
    }


def show_gpse_logo() -> None:
    """Display the GPSE startup logo with a plain-text fallback."""
    info = _load_software_info()
    try:
        from gpse.utils.logo import show_logo

        show_logo(
            style="welcome",
            version=info["version"],
            app_name=info["app_name"],
            description=info["description"],
            url=info["url"],
            use_gradient=True,
            gradient_scheme="random",
        )
    except Exception:
        print(f"\n{'=' * 50}")
        print(f"  GPSE v{info['version']}")
        print("  Genomic Prediction with Stacking Ensemble")
        print(f"{'=' * 50}\n")


def _build_root_parser(
    formatter_class: type[argparse.HelpFormatter] = argparse.HelpFormatter,
    parents: list[argparse.ArgumentParser] | None = None,
    help_action: type[argparse.Action] | None = None,
) -> argparse.ArgumentParser:
    """Build the top-level GPSE command router parser."""
    if help_action is None:
        from gpse.config.constants import _LogoHelpAction

        help_action = _LogoHelpAction

    parser = argparse.ArgumentParser(
        prog="gpse",
        description="GPSE — Genomic Prediction with Stacking Ensemble for horticultural crops",
        formatter_class=formatter_class,
        add_help=False,
        parents=parents or [],
    )
    parser.add_argument(
        "-h",
        "--help",
        action=help_action,
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


def _show_logo_for_command(command_args: list[str]) -> None:
    """Display the startup logo for normal subcommand execution."""
    skip_logo_options = {"-h", "--help", "-v", "--version"}
    if any(arg in skip_logo_options for arg in command_args):
        return
    show_gpse_logo()


def print_easter_egg(show_question: bool = False) -> None:
    """Display a compact easter egg for the ultimate answer."""
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

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    if console is None:
        if show_question:
            print(f"\nGPSE 42\n{'-' * 7}\nQ: {_ULTIMATE_QUESTION_DISPLAY}\nA: 42")
        else:
            print(f"\nGPSE 42\n{'-' * 7}")
        for char in _ULTIMATE_ANSWER_TEXT:
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
            body.append(f"{_ULTIMATE_QUESTION_DISPLAY}\n\n", style="white")
        body.append("42\n", style="bold bright_cyan")
        append_styled_answer(body, answer_text)
        return Panel(
            Align.center(body),
            title="[bold cyan]GPSE 42[/bold cyan]",
            subtitle="[dim]Don't Panic[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )

    console.print()
    answer = ""
    with Live(render_panel(answer), console=console, refresh_per_second=30) as live:
        for char in _ULTIMATE_ANSWER_TEXT:
            answer += char
            live.update(render_panel(answer))
            time.sleep(type_delay)
    console.print("\n")
