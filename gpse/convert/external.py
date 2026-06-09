#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for resolving and invoking external tools in GPSE convert.

Provides configuration-driven discovery of third-party executables (PLINK,
Java, Beagle, etc.), version checks, and safe command execution.
"""

from __future__ import annotations

import os
import re
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


def _compress_stdout(lines: list[str]) -> list[str]:
    """Compress high-frequency progress spam from external tools (e.g. PLINK).

    Rules:
      1. ``--vcf: Nk variants complete.``  — keep every 10 k plus the first.
      2. ``0%1%2%...99% done``            — strip the percentage run, keep the prefix.
      3. ``0%1%2%...99%done``             — same, but without the space before *done*.
      4. Standalone ``N%`` fragments       — drop entirely.
      5. Lines with many inline percentages — strip the percentage run, keep the prefix.
    """
    out: list[str] = []
    vcf_seen = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 1. VCF milestone filter
        m = re.match(r"^--vcf:\s+(\d+)k variants complete\.$", line)
        if m:
            vcf_seen += 1
            n = int(m.group(1))
            if n == 1 or n % 10 == 0:
                out.append(line)
            continue

        # 2. Strip inline progress-bar completion (e.g. "0%1%2%...99% done.")
        #    Keep the meaningful prefix before the progress run.
        if re.search(r"(?:\d+%)+\s*done\.?$", line):
            cleaned = re.sub(r"\s*(?:\d+%)+\s*done\.?$", "", line).strip()
            if cleaned:
                out.append(cleaned)
            continue

        # 3. Drop \r-split progress fragments (e.g. standalone "1%", "2%" ...).
        if re.fullmatch(r"\d+%", line):
            continue

        # 4. Strip many inline percentages (PLINK progress bars without "done").
        #    Keep the meaningful prefix.
        if re.search(r"(?:\d+%){5,}", line):
            cleaned = re.sub(r"\s*(?:\d+%){5,}.*", "", line).strip()
            if cleaned:
                out.append(cleaned)
            continue

        out.append(line)

    # If any VCF lines were filtered, append a summary so the user still knows
    # the total scale of the job.
    if vcf_seen:
        kept = sum(1 for l in out if l.startswith("--vcf:"))
        if kept < vcf_seen:
            out.append(f"(VCF progress: {vcf_seen} milestones total, {kept} shown)")

    return out


def _is_plink_cmd(cmd_args: list[str]) -> bool:
    """Return True if the command looks like a PLINK invocation."""
    if not cmd_args:
        return False
    return "plink" in os.path.basename(cmd_args[0]).lower()


def _plink_chr_error_hint(cmd_args: list[str], logger) -> None:
    """When a PLINK command fails, check its log for chromosome-name errors
    and suggest ``--allow-extra-chr`` if applicable."""
    if not _is_plink_cmd(cmd_args) or "--allow-extra-chr" in cmd_args:
        return

    # Look for PLINK's auto-generated log next to the --out prefix.
    try:
        out_idx = cmd_args.index("--out")
        log_path = cmd_args[out_idx + 1] + ".log"
    except (ValueError, IndexError):
        return

    if not os.path.isfile(log_path):
        return

    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # PLINK prints this exact phrase when it encounters non-standard chroms.
    if "Allow-extra-chr" in log_text or "allow-extra-chr" in log_text:
        logger.error(
            "PLINK rejected non-standard chromosome names in this VCF. "
            "Re-run with --allow-extra-chr to accept them."
        )


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
        try:
            with open(log_file, "a") as f:
                subprocess.run(cmd_args, stdout=f, stderr=f, check=True)
        except subprocess.CalledProcessError as exc:
            if logger is not None:
                _plink_chr_error_hint(cmd_args, logger)
            raise
    else:
        # When a logger is available but no dedicated log file is requested,
        # capture the subprocess output and stream it through the logger so
        # that external tool messages (e.g. PLINK) are tidy and unified with
        # GPSE's own log format instead of being dumped raw to the terminal.
        if logger is not None:
            try:
                result = subprocess.run(
                    cmd_args, capture_output=True, check=True
                )
            except subprocess.CalledProcessError as exc:
                # Print whatever output PLINK produced before the error.
                if exc.stdout:
                    stdout_text = exc.stdout.decode("utf-8", errors="replace").replace("\r", "")
                    for line in _compress_stdout(re.split(r"[\r\n]+", stdout_text.strip())):
                        logger.info(f"[stdout] {line}")
                if exc.stderr:
                    stderr_text = exc.stderr.decode("utf-8", errors="replace").replace("\r", "")
                    for line in _compress_stdout(re.split(r"[\r\n]+", stderr_text.strip())):
                        logger.warning(f"[stderr] {line}")
                _plink_chr_error_hint(cmd_args, logger)
                raise
            if result.stdout:
                stdout_text = result.stdout.decode('utf-8', errors='replace').replace('\r', '')
                raw_lines = re.split(r'[\r\n]+', stdout_text.strip())
                for line in _compress_stdout(raw_lines):
                    logger.info(f"[stdout] {line}")
            if result.stderr:
                stderr_text = result.stderr.decode('utf-8', errors='replace').replace('\r', '')
                raw_err_lines = re.split(r'[\r\n]+', stderr_text.strip())
                for line in _compress_stdout(raw_err_lines):
                    logger.warning(f"[stderr] {line}")
        else:
            subprocess.run(cmd_args, check=True)
