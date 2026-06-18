#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loguru-based logger initialization with RichHandler console output.
"""

import datetime
import io
import logging
import sys
import warnings
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


def _capture_warnings() -> None:
    """Redirect Python :mod:`warnings` output to the Loguru logger."""
    if not _LOGURU_AVAILABLE:
        return

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(
            f"{category.__name__}: {message} ({filename}:{lineno})"
        )

    warnings.showwarning = _showwarning


class _InterceptHandler(logging.Handler):
    """Send records from the standard :mod:`logging` module to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _capture_std_logging() -> None:
    """Redirect standard-library logging (e.g., Optuna) to the Loguru logger."""
    if not _LOGURU_AVAILABLE:
        return

    # Replace the root logger configuration so all std-logging records flow
    # through Loguru. Existing handlers are cleared to avoid duplicate output.
    logging.basicConfig(
        handlers=[_InterceptHandler()],
        level=logging.DEBUG,
        force=True,
    )


try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_CONSOLE = Console(stderr=True) if _RICH_AVAILABLE else None
_ANALYSIS_LOGGER = None
_ANALYSIS_LOG_FILE_PATH = None


def _add_console_handler(
    log_level: str,
    more_info: bool = False,
    style: str = "default",
) -> None:
    if not _RICH_AVAILABLE:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=True, enqueue=True)
        return

    if style == "minimal":
        handler = RichHandler(
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        fmt = "{message}"
    elif style == "detailed" or more_info:
        handler = RichHandler(
            show_time=True,
            omit_repeated_times=False,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]",
        )
        fmt = "{name}:{function}:{line} - {message}"
    elif style == "plain":
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=True, enqueue=True)
        return
    else:
        handler = RichHandler(
            show_time=True,
            omit_repeated_times=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]",
        )
        fmt = "{message}"

    logger.add(handler, format=fmt, level=log_level, enqueue=True)

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

    _add_console_handler(log_level, style="default")

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


def log_header(title: str) -> None:
    logger.info("")
    logger.info(f"[bold cyan]{'=' * 10} {title.upper()} {'=' * 10}[/bold cyan]")


def log_section(title: str) -> None:
    logger.info(f"\n[bold blue]─── {title} ───[/bold blue]")


def log_success(msg: str) -> None:
    logger.info(f"[bold green]✔ {msg}[/bold green]")


def log_warning(msg: str) -> None:
    logger.warning(f"[bold yellow]⚠ {msg}[/bold yellow]")


def log_error(msg: str) -> None:
    logger.error(f"[bold red]✘ {msg}[/bold red]")


def log_info(msg: str) -> None:
    logger.info(msg)


def log_step(step: int, total: int, msg: str) -> None:
    logger.info(f"[bold magenta][Step {step}/{total}][/bold magenta] {msg}")


def log_config(config_dict: dict, title: str = "Configuration") -> None:
    if not _RICH_AVAILABLE:
        for k, v in config_dict.items():
            logger.info(f"{title} - {k}: {v}")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta", box=None)
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold")

    for k, v in config_dict.items():
        table.add_row(str(k), str(v))

    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    console.print(table)
    logger.info("\n" + console.file.getvalue())


def log_config_panel(config_dict: dict, title: str = "GPSE Configuration") -> None:
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
    style: str = "default",
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
    style = style or cfg_logs.get("style", "default")

    logger.remove()

    _add_console_handler(log_level, more_info=more_info, style=style)
    _capture_warnings()
    _capture_std_logging()

    # File handler — plain text, no colours, enqueue for multi-process safety
    if logger_name:
        file_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        Path(logger_name).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            logger_name,
            format=file_fmt,
            level="DEBUG",
            colorize=False,
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

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
    style: str = "default",
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
    logger = logger_init(
        logger_name,
        log_level=log_level,
        more_info=more_info,
        style=style,
    )

    # Also persist metadata to the log file
    logger.info(f"GPSE Author : {sw.get('author', 'unknown')}")
    logger.info(f"GPSE Version : {sw.get('version', 'unknown')}")
    logger.info(f"GPSE Email : {sw.get('email', '')}")
    logger.info(f"Logger initialized, log file : {logger_name}")
    logger.info(f"GPSE Analysis Result : {output_dir}")
    logger.debug(f"Software Full config : {software}")
    logger.debug(f"Default Full config : {default}")

    return logger, output_dir


def setup_analysis_logging(
    log_dir="logs",
    log_file_prefix="analysis",
    max_file_size="100 MB",
    console_level="INFO",
    file_level="DEBUG",
    style="default",
):
    """
    Setup logging for analysis scripts with RichHandler console output.
    """
    _ensure_loguru()

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_dir_path / f"{log_file_prefix}_{timestamp}.log"

    logger.remove()
    logger.add(
        log_file_path,
        rotation=max_file_size,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        level=file_level,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    _add_console_handler(console_level, style=style)
    _capture_warnings()
    _capture_std_logging()

    logger.info(
        f"[bold green]{log_file_prefix.capitalize()} Script Initialized[/bold green] "
        f"(Style: {style})"
    )
    logger.info(f"Log file: {log_file_path}")

    return logger, log_file_path


def get_logger():
    return logger


def initialize_analysis_logger(**kwargs):
    global _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH

    if _ANALYSIS_LOGGER is None:
        _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH = setup_analysis_logging(**kwargs)

    return _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH


def get_analysis_logger():
    global _ANALYSIS_LOGGER

    if _ANALYSIS_LOGGER is None:
        _ANALYSIS_LOGGER, _ = initialize_analysis_logger()

    return _ANALYSIS_LOGGER


def get_analysis_log_file_path():
    return _ANALYSIS_LOG_FILE_PATH
