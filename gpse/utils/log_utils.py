#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loguru-based logger initialization with Rich-styled console output.
"""

import datetime
import sys
from pathlib import Path


try:
    from loguru import logger
    _LOGURU_AVAILABLE = True
except ImportError:
    _LOGURU_AVAILABLE = False
    logger = None


def _ensure_loguru() -> None:
    if not _LOGURU_AVAILABLE:
        raise ImportError("loguru is required for logging. Install it with: pip install loguru")


# ── Rich helpers for pretty console output ──────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_CONSOLE = Console(stderr=True) if _RICH_AVAILABLE else None

_LEVEL_STYLES = {
    "DEBUG":    ("dim",              "🔍"),
    "INFO":     ("bold green",       "●"),
    "SUCCESS":  ("bold green",       "✓"),
    "WARNING":  ("bold yellow",      "⚠"),
    "ERROR":    ("bold red",         "✗"),
    "CRITICAL": ("bold red reverse", "💥"),
}


def _make_rich_sink(more_info: bool = False):
    """Return a loguru sink that prints via Rich."""
    console = _CONSOLE or Console(stderr=True)

    def sink(message):
        record = message.record
        level = record["level"].name
        time = record["time"].strftime("%H:%M:%S")
        msg = str(message).strip()  # 去掉首尾空白和换行，消除空行

        style, icon = _LEVEL_STYLES.get(level, ("white", "•"))

        text = Text()
        text.append("GPSE ", style="bold cyan")
        text.append(f"{time} ", style="dim")
        text.append(f"{icon} ", style=style)
        text.append(f"{level:<8}", style=style)
        text.append(" │ ", style="dim")

        if more_info:
            text.append(f"{record['name']}:{record['line']}", style="cyan")
            text.append(" │ ", style="dim")

        text.append(msg)
        console.print(text, end="\n", soft_wrap=True)

    return sink


def _print_run_header(sw: dict, logger_name: str, output_dir: str) -> None:
    """Print a compact software-metadata header via Rich."""
    if not _RICH_AVAILABLE:
        return

    author = sw.get("author", "unknown")
    version = sw.get("version", "unknown")
    email = sw.get("email", "")

    _CONSOLE.print("─" * 60, style="dim")
    _CONSOLE.print(
        f"[bold cyan]GPSE[/bold cyan] {version}  "
        f"[dim]│[/dim]  Author: [bold]{author}[/bold]  "
        f"[dim]│[/dim]  Email: [bold]{email}[/bold]"
    )
    _CONSOLE.print(f"[dim]Log    : {logger_name}[/dim]")
    _CONSOLE.print(f"[dim]Output : {output_dir}[/dim]")
    _CONSOLE.print("─" * 60, style="dim")


# ── Public API ─────────────────────────────────────────────────────────────

def logger_init(
    logger_name: str = None,
    log_level: str = "INFO",
    more_info: bool = False,
) -> "logger":
    """
    Configure Loguru logger with Rich-styled console output.
    """
    _ensure_loguru()

    from .configuration import load_default_config
    default = load_default_config()
    cfg_logs = default.get("logs", {})

    log_level = log_level or cfg_logs.get("log_level", "INFO")
    more_info = more_info if more_info is not None else cfg_logs.get("more_info", False)

    logger.remove()

    # Console handler — Rich styled (or plain fallback)
    if _RICH_AVAILABLE:
        logger.add(
            _make_rich_sink(more_info=more_info),
            level=log_level,
            format="{message}",   # ← 让 Rich sink 完全接管格式
        )
    else:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=True)

    # File handler — plain text, no colours, enqueue for multi-process safety
    if logger_name:
        file_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
            + (" | {name}:{line}" if more_info else "")
        )
        Path(logger_name).parent.mkdir(parents=True, exist_ok=True)
        logger.add(logger_name, format=file_fmt, level=log_level, colorize=False, enqueue=True)

    return logger


def logger_generator(
    output_dir: str,
    log_level: str = "INFO",
    more_info: bool = False,
) -> tuple:
    """
    Create a logger, print software metadata via Rich, and return ``(logger, output_dir)``.
    """
    _ensure_loguru()

    from .configuration import load_software_config, load_default_config

    software = load_software_config()
    default = load_default_config()

    sw = software.get("software", {})
    label = default.get("logs", {}).get("Label", "GPSE")

    times = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_name = f"{output_dir}/{label}_{times}.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Pretty console header
    _print_run_header(sw, logger_name, output_dir)

    # Init logger
    logger = logger_init(logger_name, log_level=log_level, more_info=more_info)

    # Also persist metadata to the log file
    logger.info(f"GPSE Author : {sw.get('author', 'unknown')}")
    logger.info(f"GPSE Version : {sw.get('version', 'unknown')}")
    logger.info(f"GPSE Email : {sw.get('email', '')}")
    logger.info(f"Logger initialized, log file : {logger_name}")
    logger.info(f"GPSE Analysis Result : {output_dir}")
    logger.debug(f"Software Full config : {software}")
    logger.debug(f"Default Full config : {default}")

    return logger, output_dir
