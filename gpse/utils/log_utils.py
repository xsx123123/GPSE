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
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_CONSOLE = Console(stderr=True) if _RICH_AVAILABLE else None

_LEVEL_STYLES = {
    "DEBUG": ("dim", ""),
    "INFO": ("green", ""),
    "SUCCESS": ("bold green", "✓"),
    "WARNING": ("bold yellow", "⚠"),
    "ERROR": ("bold red", "✗"),
    "CRITICAL": ("bold red", "💥"),
}


def _make_rich_sink(more_info: bool = False):
    """Return a loguru sink that prints via Rich in a clean, modern format."""
    console = _CONSOLE or Console(stderr=True)

    def sink(message):
        record = message.record
        level = record["level"].name
        time = record["time"].strftime("%H:%M:%S")
        msg = str(message).strip()

        style, icon = _LEVEL_STYLES.get(level, ("white", ""))

        text = Text()
        text.append(f"[{time}] ", style="dim")
        level_str = f"{icon} {level}" if icon else level
        text.append(f"{level_str: <8} ", style=style)

        if more_info:
            text.append(f"[{record['name']}:{record['line']}] ", style="cyan dim")

        text.append(Text.from_ansi(msg))
        console.print(text, end="\n", soft_wrap=True)

    return sink


def _print_run_header(
    sw: dict,
    logger_name: str,
    output_dir: str,
    project_configs: list | None = None,
) -> None:
    """Print a compact software-metadata header via Rich."""
    if not _RICH_AVAILABLE:
        return

    author = sw.get("author", "unknown")
    version = sw.get("version", "unknown")

    header_text = Text()
    header_text.append(
        f"GPSE v{version} | Genomic Prediction with Stacking Ensemble\n",
        style="bold cyan",
    )
    header_text.append(f"Author: {author}\n\n", style="dim")
    header_text.append(f"Log    : {logger_name}\n", style="white")
    header_text.append(f"Output : {output_dir}\n", style="white")

    cfg_str = ", ".join(project_configs) if project_configs else "(no project config found)"
    header_text.append(f"Config : {cfg_str}", style="white")

    _CONSOLE.print(Panel(header_text, border_style="dim", expand=False))


def setup_subprocess_logging(
    model_name: str,
    repeat_idx: int,
    log_dir: Path,
    log_level: str = "DEBUG",
):
    """
    Configure per-worker logging for subprocess tasks.

    Each subprocess writes to its own file under ``log_dir/workers/`` to avoid
    multi-process file-write races on the main log.  Console output is also
    enabled so the user still sees real-time progress.

    Parameters
    ----------
    model_name : str
        Model name (used in the log filename).
    repeat_idx : int
        Repeat index (used in the log filename).
    log_dir : Path
        Base logs directory (e.g. ``results/logs``).
    log_level : str
        Logging level for the worker file.

    Returns
    -------
    loguru.Logger
        The configured global loguru logger (same object, new handlers).
    """
    workers_dir = Path(log_dir) / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    log_file = workers_dir / f"{model_name}_repeat_{repeat_idx + 1:03d}.log"

    # In spawn-context child processes the global logger is fresh (only the
    # default stderr handler #0 exists).  Remove it and set up our own sinks.
    logger.remove()

    # Console — plain format (Rich may not be fully initialised in children)
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>[{extra[worker]}]</cyan> | "
        "<level>{message}</level>"
    )
    try:
        from rich.console import Console
        _rich = True
    except ImportError:
        _rich = False

    if _rich:
        logger.add(
            _make_rich_sink(more_info=False),
            level=log_level,
            format="{message}",
        )
    else:
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=True)

    # Per-worker file — DEBUG level, enqueue for safety
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "[{extra[worker]}] | {message}"
    )
    logger.add(
        str(log_file),
        format=file_fmt,
        level=log_level,
        colorize=False,
        enqueue=True,
    )

    # Bind a default ``worker`` extra field so messages don't KeyError
    logger.configure(extra={"worker": f"{model_name}_R{repeat_idx + 1}"})

    return logger


def collect_subprocess_logs(log_dir: Path, main_logger) -> None:
    """
    Read all per-worker log files and append a digest to the main log.

    Called by the main process after all subprocess tasks have finished so
    that the main log contains a complete record of every repeat.

    Parameters
    ----------
    log_dir : Path
        Base logs directory containing the ``workers/`` subfolder.
    main_logger
        The main-process loguru logger.
    """
    workers_dir = Path(log_dir) / "workers"
    if not workers_dir.exists():
        return

    worker_logs = sorted(workers_dir.glob("*.log"))
    if not worker_logs:
        return

    main_logger.info(f"Collecting {len(worker_logs)} subprocess log(s) from {workers_dir}")

    for log_file in worker_logs:
        try:
            content = log_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            main_logger.info(f"{'─' * 40}")
            main_logger.info(f"▶ Subprocess log: {log_file.name}")
            main_logger.info(f"{'─' * 40}")
            for line in content.splitlines():
                main_logger.info(f"  {line}")
        except Exception as e:
            main_logger.warning(f"Failed to read subprocess log {log_file.name}: {e}")

    main_logger.info(f"{'─' * 40}")
    main_logger.info(f"Subprocess log collection complete ({len(worker_logs)} files)")


def log_config_panel(config_dict: dict, title: str = "GPSE Configuration"):
    """
    Print a Rich panel for configuration and write plain text to the log file.
    """
    if not _RICH_AVAILABLE:
        for k, v in config_dict.items():
            logger.info(f"{title} - {k}: {v}")
        return

    text = Text()
    for section, params in config_dict.items():
        text.append(f"\n[ {section} ]\n", style="bold cyan")
        if isinstance(params, dict):
            for k, v in params.items():
                text.append(f"  • {k: <20}: ", style="dim")
                text.append(f"{v}\n", style="white")
        else:
            text.append(f"  {params}\n", style="white")

    _CONSOLE.print(
        Panel(text, title=f"[bold]{title}[/bold]", border_style="cyan", expand=False)
    )

    logger.debug(f"=== {title} ===")
    for section, params in config_dict.items():
        if isinstance(params, dict):
            for k, v in params.items():
                logger.debug(f"[{section}] {k}: {v}")
        else:
            logger.debug(f"[{section}] {params}")


def get_loading_status(msg: str):
    """
    Return a Rich status context manager, or a logging fallback.
    """
    if _RICH_AVAILABLE:
        return _CONSOLE.status(f"[bold green]{msg}[/bold green]", spinner="dots")

    class DummyStatus:
        def __enter__(self):
            logger.info(msg)

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return DummyStatus()


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

    from .configuration import load_default_config, get_loaded_project_configs
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
            format="{message}",   # Let the Rich sink handle all formatting
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

    # Print project config info as the first log messages
    project_configs = get_loaded_project_configs()
    if project_configs:
        for cfg_path in project_configs:
            logger.info(f"GPSE config loaded: {cfg_path}")
    else:
        logger.info("GPSE config: none found (using package defaults)")

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

    from .configuration import (
        load_software_config,
        load_default_config,
        get_loaded_project_configs,
    )

    software = load_software_config()
    default = load_default_config()

    sw = software.get("software", {})
    label = default.get("logs", {}).get("Label", "GPSE")

    times = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_name = f"{output_dir}/{label}_{times}.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Detect project-level config files
    project_configs = get_loaded_project_configs()

    # Pretty console header
    _print_run_header(sw, logger_name, output_dir, project_configs=project_configs)

    # Init logger (also prints project config info)
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
