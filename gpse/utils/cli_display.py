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


def _build_root_parser(formatter_class: type[argparse.HelpFormatter] = argparse.HelpFormatter,
                       parents: list[argparse.ArgumentParser] | None = None,
                       help_action: type[argparse.Action] | None = None) -> argparse.ArgumentParser:
    """
    Build the top-level GPSE command router parser.
    """
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
        metavar="{convert,train,predict}",
        title="workflow commands",
        description="Run one of the GPSE workflows",
    )

    # Add the convert subcommand
    subparsers.add_parser(
        "convert",
        add_help=False,
        help="Convert raw genotype/phenotype inputs and run optional QC",
        description="Convert raw genotype/phenotype inputs and run optional QC.",
    )

    # Add the train subcommand
    subparsers.add_parser(
        "train",
        add_help=False,
        help="Train genomic prediction models from genotype and phenotype matrices",
        description="Train genomic prediction models.",
    )

    # Add the predict subcommand
    subparsers.add_parser(
        "predict",
        add_help=False,
        help="Predict phenotypes using trained GPSE models",
        description="Predict phenotypes using trained GPSE models.",
    )
    return parser


def _build_convert_parser(formatter_class=argparse.HelpFormatter,
                          prog: str = "python -m gpse.convert.cli",
                          help_action=None,
                          parents: list[argparse.ArgumentParser] | None = None) -> argparse.ArgumentParser:
    """
    Build the parser used by both direct and top-level convert entry points.
    """
    parents = parents or []
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Convert VCF + phenotype files into training-ready numeric matrices.\n"
            "Core usage:  gpse convert --vcf FILE --pheno FILE --out-prefix PREFIX [--direct]\n"
            "\n"
            "Standalone utilities (--check-deps, --run-qc, --recode-prefix) do not\n"
            "require the core arguments."
        ),
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

    # General
    general = parser.add_argument_group("general")
    if not parents:
        general.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level.",
        )
    general.add_argument(
        "-t", "--threads",
        type=int,
        default=10,
        help="Number of parallel threads for trait processing (default: 10).",
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

    # Required arguments (for conversion pipeline)
    required = parser.add_argument_group(
        "required arguments (conversion pipeline)",
        description="Required for the default VCF + phenotype conversion. "
                    "Not needed for --check-deps, --run-qc, or --recode-prefix.",
    )
    required.add_argument("--vcf", help="Input VCF file path. [REQUIRED]")
    required.add_argument("--pheno", help="Phenotype file path. [REQUIRED]")
    required.add_argument("--out-prefix", help="Output file prefix. [REQUIRED]")

    # Conversion options
    conv = parser.add_argument_group("conversion options")
    conv.add_argument("--bfile", help="Input PLINK BED/BIM/FAM prefix.")
    conv.add_argument("--ped-file", help="Input PED file path.")
    conv.add_argument("--map-file", help="Input MAP file path.")
    conv.add_argument("--matrix-file", help="Existing genotype matrix CSV.")
    conv.add_argument("--extract", help="SNP ID list file for PLINK --extract.")
    conv.add_argument("--snp-dir", help="Directory containing SNP list .txt files.")
    conv.add_argument("--direct", action="store_true", help="Convert whole bfile to matrix.")
    conv.add_argument("--plink-out", help="PLINK output prefix used during VCF conversion.")
    conv.add_argument("--load", action="store_true", help="Load and print matrix info.")
    conv.add_argument("--skip-clean", action="store_true", help="Reserved compatibility flag.")
    conv.add_argument("--skip-match", action="store_true", help="Skip phenotype/genotype matching.")
    conv.add_argument("--skip-matrix", action="store_true", help="Skip matrix generation.")
    conv.add_argument(
        "--allow-extra-chr",
        action="store_true",
        help="Pass --allow-extra-chr to PLINK to support non-standard chromosome names.",
    )
    conv.add_argument(
        "--out-format",
        default="parquet",
        choices=["csv", "parquet", "feather"],
        help="Output genotype matrix format (default: parquet).",
    )


    # Phenotype options
    pheno = parser.add_argument_group("phenotype options")
    pheno.add_argument("--trait-name", help="Rename phenotype trait column.")
    pheno.add_argument(
        "--standardize-phenotype",
        action="store_true",
        help="Apply z-score standardization to the phenotype column.",
    )

    # Utilities (standalone features, do not require --vcf/--pheno/--out-prefix)
    utils = parser.add_argument_group("utilities")
    utils.add_argument(
        "--check-deps",
        action="store_true",
        help="Check external dependencies and exit.",
    )
    utils.add_argument(
        "--run-qc",
        action="store_true",
        help="Run QC, optional Beagle imputation, and LD pruning.",
    )
    utils.add_argument(
        "--recode-prefix",
        help="Convert PED/MAP compound genotypes at this prefix to numeric additive coding and exit.",
    )

    # External tools
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

    # Quality control
    qc = parser.add_argument_group("quality control")
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
