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
        metavar="{convert,train,predict,batch}",
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

    # Add the batch subcommand
    subparsers.add_parser(
        "batch",
        add_help=False,
        help="Train models for multiple traits from one YAML config",
        description="Train GPSE models for multiple traits from one YAML config.",
    )

    # Put workflow commands above options in help output.
    # argparse prints _action_groups in order; parents inject options first.
    group_by_title = {group.title: group for group in parser._action_groups}
    workflow_group = group_by_title.get("workflow commands")
    options_group = group_by_title.get("options") or group_by_title.get("optional arguments")
    if workflow_group is not None and options_group is not None:
        groups = parser._action_groups
        try:
            workflow_idx = groups.index(workflow_group)
            options_idx = groups.index(options_group)
        except ValueError:
            pass
        else:
            if workflow_idx > options_idx:
                groups[options_idx], groups[workflow_idx] = (
                    groups[workflow_idx],
                    groups[options_idx],
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
            "Core usage:  gpse convert --vcf FILE --pheno FILE --out-prefix PREFIX [--run-qc]\n"
            "\n"
            "Standalone utilities (--check-deps, --run-qc, --recode-prefix) do not\n"
            "require all core arguments when used alone."
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
                    "Some utilities like --check-deps or standalone --run-qc "
                    "do not require all of these.",
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
    conv.add_argument(
        "--geno-encoding",
        default="012",
        choices=["012", "-101"],
        help="Genotype additive encoding: '012' = 0/1/2 (default), "
             "'-101' = -1/0/1 centered coding as in Azodi et al. 2019 "
             "([-1,0,1] = [aa, Aa, AA]).",
    )


    # Phenotype options
    pheno = parser.add_argument_group("phenotype options")
    pheno.add_argument("--trait-name", help="Rename phenotype trait column.")
    pheno.add_argument(
        "--standardize-phenotype",
        action="store_true",
        help="Apply z-score standardization to the phenotype column. "
             "Equivalent to --pheno-scale zscore.",
    )
    pheno.add_argument(
        "--pheno-scale",
        default=None,
        choices=["none", "zscore", "minmax"],
        help="Phenotype scaling mode: 'none' (default), 'zscore' standardization, "
             "or 'minmax' normalization to [0, 1] as in Azodi et al. 2019. "
             "Overrides --standardize-phenotype.",
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
    qc.add_argument("--ld-window", type=int, default=50, help="LD pruning window size in kb (default: 50).")
    qc.add_argument("--ld-step", type=int, default=10, help="LD pruning step size in SNPs (default: 10).")
    qc.add_argument("--impute", action="store_true", help="Use Beagle imputation before pruning.")

    return parser


def _show_logo_for_command(command_args: list[str]) -> None:
    """Display the startup logo for normal subcommand execution."""
    skip_logo_options = {"-h", "--help", "-v", "--version"}
    if any(arg in skip_logo_options for arg in command_args):
        return
    show_gpse_logo()


# Pure ASCII (FIGlet "big") — single-width only, safe in any terminal.
_ASCII_42 = r"""
   _  _        ___
  | || |      |__ \
  | || |_       ) |
  |__   _|     / /
     | |      / /_
     |_|     |____|
"""

_THINKING_LINES = (
    "Consulting Deep Thought...",
    "Enumerating life, the universe, and everything...",
    "Sampling random_seed=42 for reproducibility...",
    "Cross-validating against the Guide...",
    "Stacking ensembles of pure coincidence...",
    "Waiting 7.5 million years (compressed)...",
)

_AFTERGLOW = (
    "Don't Panic — and always carry a towel.",
    "Also the default random seed. Coincidence? Probably not.",
    "GPSE: Genomic Prediction, Stacking Ensemble... and 42.",
    "Share and Enjoy. (Results may vary by fold.)",
    "Mostly harmless. Highly heritable.",
    "The mice are still reviewing the SNP list.",
    "So long, and thanks for all the phenotypes.",
    "If it works, leave random_seed=42 alone.",
    "42 folds of destiny. K=42 is... ambitious.",
    "TOPSIS ranks this answer #1. Entropy agrees.",
)

# Two short lines so the panel never wraps awkwardly mid-sentence.
_ANSWER_LINE_1 = "The answer to the ultimate question"
_ANSWER_LINE_2 = "of life, the universe, and everything is 42"


def print_easter_egg(show_question: bool = False) -> None:
    """Display a theatrical Hitchhiker-meets-genomics easter egg."""
    import random

    type_delay = 0.014
    afterglow = random.choice(_AFTERGLOW)
    full_answer = f"{_ANSWER_LINE_1}\n{_ANSWER_LINE_2}"

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    if console is None:
        print(f"\nGPSE 42\n{'-' * 7}")
        if show_question:
            print(f"Q: {_ULTIMATE_QUESTION_DISPLAY}")
        for line in _THINKING_LINES[:3]:
            print(f"  ... {line}")
            time.sleep(0.18)
        print(_ASCII_42)
        for char in full_answer:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(type_delay)
        print(f"\n\n  {afterglow}\n")
        return

    from rich.align import Align
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text

    # Style map over the joined two-line answer.
    answer_segments = [
        ("The answer", "bold cyan"),
        (" to the ", "white"),
        ("ultimate question", "bold magenta"),
        ("\n", "white"),
        ("of ", "white"),
        ("life, the universe, and everything", "bold green"),
        (" is ", "white"),
        ("42", "bold bright_yellow"),
    ]
    raw_ascii_lines = [ln.rstrip() for ln in _ASCII_42.strip("\n").splitlines()]
    art_width = max(len(ln) for ln in raw_ascii_lines)
    # Equal-width lines (trailing spaces preserved via non-breaking pad later).
    ascii_lines = [ln.ljust(art_width) for ln in raw_ascii_lines]
    ascii_line_count = len(ascii_lines)
    # Panel content width driven by the longer of art / answer / footer.
    content_width = max(
        art_width,
        len(_ANSWER_LINE_1),
        len(_ANSWER_LINE_2),
        len(afterglow),
    )
    art_left = max(0, (content_width - art_width) // 2)

    console.print()

    # Act I — dramatic "computing"
    if show_question:
        q = Text()
        q.append("Q: ", style="bold cyan")
        q.append(_ULTIMATE_QUESTION_DISPLAY, style="italic white")
        console.print(Align.center(q))
        console.print()

    with Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=len(_THINKING_LINES))
        for line in _THINKING_LINES:
            progress.update(task, description=line)
            time.sleep(0.28 + random.uniform(0.0, 0.12))
            progress.advance(task)

    # Act II — big ASCII 42 with typewriter answer
    def center_line(text: str, style: str | None = None) -> Text:
        left = max(0, (content_width - len(text)) // 2)
        t = Text(" " * left)
        t.append(text, style=style)
        # Trailing pad keeps Panel width stable while typing.
        t.append(" " * max(0, content_width - left - len(text)))
        return t

    def styled_answer(answer_text: str) -> Text:
        """Render typed answer; line 1 and line 2 are each centered."""
        if "\n" in answer_text:
            line1, line2 = answer_text.split("\n", 1)
        else:
            line1, line2 = answer_text, None

        def paint(line: str, global_offset: int) -> Text:
            body = Text()
            # Walk segments that fall inside [global_offset, global_offset+len(line))
            pos = 0  # position in full_answer
            for segment, style in answer_segments:
                seg_end = pos + len(segment)
                # Intersection with this line's range
                start = max(pos, global_offset)
                end = min(seg_end, global_offset + len(line))
                if start < end:
                    body.append(line[start - global_offset : end - global_offset], style=style)
                pos = seg_end
            left = max(0, (content_width - len(line)) // 2)
            out = Text(" " * left)
            out.append_text(body)
            out.append(" " * max(0, content_width - left - len(line)))
            return out

        result = paint(line1, 0)
        if line2 is not None:
            result.append("\n")
            result.append_text(paint(line2, len(line1) + 1))
        return result

    def render_panel(answer_text: str, reveal_lines: int = 0, show_footer: bool = False) -> Align:
        # Indent whole art block (not per-line center) so the "4" stays diagonal.
        visible = ascii_lines[: max(1, reveal_lines)]
        art_block = "\n".join((" " * art_left) + ln for ln in visible)
        art = Text(art_block, style="bold bright_yellow")
        parts: list = [art]
        if reveal_lines >= ascii_line_count:
            parts.append(Text(""))
            parts.append(styled_answer(answer_text))
            if show_footer and len(answer_text) >= len(full_answer):
                parts.append(Text(""))
                parts.append(center_line(afterglow, "dim italic"))
        panel = Panel(
            Group(*parts),
            title="[bold cyan]GPSE 42[/bold cyan]",
            subtitle="[dim]Don't Panic · Share and Enjoy[/dim]",
            border_style="bright_cyan",
            padding=(1, 3),
            expand=False,
        )
        return Align.center(panel)

    # Reveal ASCII line by line, then type the answer
    with Live(render_panel("", reveal_lines=0), console=console, refresh_per_second=30) as live:
        for i in range(1, ascii_line_count + 1):
            live.update(render_panel("", reveal_lines=i))
            time.sleep(0.05)
        answer = ""
        for char in full_answer:
            answer += char
            live.update(render_panel(answer, reveal_lines=ascii_line_count))
            time.sleep(type_delay)
        live.update(render_panel(answer, reveal_lines=ascii_line_count, show_footer=True))
        time.sleep(0.4)

    console.print()
