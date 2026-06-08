#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for resolving and invoking external tools in GPSE convert.

Provides configuration-driven discovery of third-party executables (PLINK,
Java, Beagle, etc.), version checks, and safe command execution.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from gpse.utils.configuration import load_software_config
from gpse.utils.dependency_checker import check_all_external_tools, check_external_tool


def _load_external_tools(
    config_path: Optional[str] = None,
    *,
    auto_project_config: bool = False,
) -> list[dict]:
    """Load external tool definitions from gpse/config/software.yaml."""
    return load_software_config(
        config_path,
        auto_project_config=auto_project_config,
    ).get("external_tools", [])


def get_external_tool_config(
    name: str,
    *,
    config_path: Optional[str] = None,
    auto_project_config: bool = False,
) -> dict:
    """Return a configured external tool by name, or a minimal fallback."""
    for tool in _load_external_tools(
        config_path,
        auto_project_config=auto_project_config,
    ):
        if tool.get("name") == name:
            return dict(tool)
    return {"name": name, "cmd": name, "version_flag": "--version", "required": True}


def check_configured_external_tools(
    command_overrides: Optional[dict[str, str]] = None,
    *,
    config_path: Optional[str] = None,
    auto_project_config: bool = False,
) -> list[dict]:
    """Check every external tool defined in software.yaml."""
    command_overrides = command_overrides or {}
    tools = []
    for tool in _load_external_tools(
        config_path,
        auto_project_config=auto_project_config,
    ):
        configured = dict(tool)
        override = command_overrides.get(configured.get("name"))
        if override:
            configured["cmd"] = override
        tools.append(configured)
    return check_all_external_tools(tools)


def get_convert_config(
    config_path: Optional[str] = None,
    *,
    auto_project_config: bool = False,
) -> dict:
    """Load the optional 'convert' section from merged GPSE YAML config."""
    convert_config = load_software_config(
        config_path,
        auto_project_config=auto_project_config,
    ).get("convert") or {}
    if not isinstance(convert_config, dict):
        raise TypeError("'convert' config section must be a mapping")
    return convert_config


def resolve_external_tool(
    tool_path: str,
    *,
    name: Optional[str] = None,
    version_flag: str = "--version",
    min_version: Optional[str] = None,
) -> str:
    """Resolve an executable path and optionally validate its version."""
    if not tool_path:
        raise FileNotFoundError("External tool path is empty")

    result = check_external_tool(
        name=name or Path(tool_path).name,
        cmd=tool_path,
        version_flag=version_flag,
        min_version=min_version,
    )
    if not result["available"]:
        raise FileNotFoundError(f"External tool not found: {tool_path}")
    if not result.get("version_ok", True):
        raise RuntimeError(
            f"External tool version too low: {result['name']} "
            f"{result.get('version')} < required {result.get('min_version')}"
        )
    return result["path"] or tool_path


def resolve_configured_tool(
    name: str,
    command_override: Optional[str] = None,
    *,
    config_path: Optional[str] = None,
    auto_project_config: bool = False,
) -> str:
    """Resolve a tool using software.yaml defaults, with an optional command override."""
    tool = get_external_tool_config(
        name,
        config_path=config_path,
        auto_project_config=auto_project_config,
    )
    return resolve_external_tool(
        command_override or tool.get("cmd") or name,
        name=tool.get("name", name),
        version_flag=tool.get("version_flag", "--version"),
        min_version=tool.get("min_version"),
    )


def ensure_existing_file(file_path: str, *, name: str = "file") -> str:
    """Return *file_path* if it points to an existing regular file."""
    if not file_path:
        raise FileNotFoundError(f"{name} path is empty")
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {file_path}")
    return str(path)


def run_command(
    cmd_list: Sequence[object],
    *,
    log_file: Optional[str] = None,
    logger=None,
) -> None:
    """Run an external command without invoking a shell."""
    cmd_args = [
        os.fspath(arg) if isinstance(arg, (Path, os.PathLike)) else str(arg)
        for arg in cmd_list
    ]
    cmd_text = shlex.join(cmd_args)
    if logger is not None:
        logger.info(f"Executing: {cmd_text}")

    if log_file:
        with open(log_file, "a") as f:
            subprocess.run(cmd_args, stdout=f, stderr=f, check=True)
    else:
        # When a logger is available but no dedicated log file is requested,
        # capture the subprocess output and stream it through the logger so
        # that external tool messages (e.g. PLINK) are tidy and unified with
        # GPSE's own log format instead of being dumped raw to the terminal.
        if logger is not None:
            result = subprocess.run(
                cmd_args, capture_output=True, text=True, check=True
            )
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    if line.strip():
                        logger.info(f"[stdout] {line}")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    if line.strip():
                        logger.warning(f"[stderr] {line}")
        else:
            subprocess.run(cmd_args, check=True)
