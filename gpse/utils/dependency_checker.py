#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpse External Dependency Checker
================================
Detects third-party command-line tools (e.g. PLINK) required by the
analysis pipeline and reports their availability / version.

Typical usage:
    from gpse.utils.dependency_checker import check_all_external_tools
    results = check_all_external_tools(tools_config)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", "command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"
    except Exception as exc:
        return 1, "", str(exc)


def _extract_version(text: str) -> str | None:
    """Try to extract a version string from tool output."""
    # Common patterns: v1.2.3, 1.2.3, PLINK v1.90b6.21, etc.
    patterns = [
        r"v?(\d+\.\d+[^\s]*)",            # v1.90b6.21  or  1.90b6.21
        r"version\s*:?\s*v?(\d+\.\d+[^\s]*)",  # version: 1.2.3
        r"(\d+\.\d+\.\d+)",                # 1.2.3
        r"(\d+\.\d+)",                     # 1.90
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _version_tuple(ver: str | None) -> tuple[int, ...]:
    """Convert a version string like '1.90b6' → (1, 90, 6) for comparison.

    Trailing zero segments are stripped so that '1.9.0' and '1.9'
    are treated as equivalent — this is important for tools like PLINK
    where '1.90' and '1.9.0' refer to the same release line.
    """
    if ver is None:
        return (0,)
    # Strip leading 'v', split by dots / dashes / plus, keep only numeric parts
    cleaned = ver.lstrip("vV")
    parts = re.split(r"[.\-+]", cleaned)
    result: list[int] = []
    for p in parts:
        num = re.match(r"(\d+)", p)
        if num:
            result.append(int(num.group(1)))
    # Strip trailing zeros so 1.9.0 → 1.9
    while result and result[-1] == 0:
        result.pop()
    return tuple(result) if result else (0,)


def _meets_min_version(current: str | None, minimum: str | None) -> bool:
    """Return True if *current* >= *minimum*."""
    if minimum is None:
        return True
    if current is None:
        return False
    return _version_tuple(current) >= _version_tuple(minimum)


def check_external_tool(
    name: str,
    cmd: str = None,
    version_flag: str = "--version",
    min_version: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """
    Check a single external tool.

    Parameters
    ----------
    name : str
        Human-readable tool name.
    cmd : str, optional
        Command to invoke. Defaults to *name*.
    version_flag : str
        Flag passed to the command to obtain version info.
    min_version : str, optional
        Minimum acceptable version.

    Returns
    -------
    dict
        {
            "name": str,
            "available": bool,
            "path": str | None,
            "version": str | None,
            "min_version": str | None,
            "version_ok": bool,
            "raw_output": str,
        }
    """
    cmd = cmd or name
    result: dict[str, Any] = {
        "name": name,
        "available": False,
        "path": None,
        "version": None,
        "min_version": min_version,
        "version_ok": True,
        "raw_output": "",
    }

    # 1. Locate executable
    exe = shutil.which(cmd)
    if exe is None:
        result["raw_output"] = f"'{cmd}' not found in PATH"
        result["version_ok"] = False
        return result

    result["path"] = exe
    result["available"] = True

    # 2. Query version
    rc, stdout, stderr = _run_cmd([cmd, version_flag])
    raw = (stdout + "\n" + stderr).strip()
    result["raw_output"] = raw

    version = _extract_version(raw)
    result["version"] = version

    # 3. Check minimum version
    if min_version is not None:
        result["version_ok"] = _meets_min_version(version, min_version)

    return result


def _log_single_result(result: dict[str, Any], logger) -> None:
    """Emit a single tool-check result through a loguru-style logger."""
    name = result["name"]
    ver = result.get("version") or "unknown"
    if not result["available"]:
        logger.error(f"External tool missing — {name}: not found in PATH")
    elif not result.get("version_ok", True):
        logger.warning(
            f"External tool version too low — {name} {ver} "
            f"(required >= {result['min_version']})"
        )
    else:
        logger.info(f"External tool OK — {name} {ver}")


def check_all_external_tools(
    tools_config: list[dict[str, Any]] | None,
    logger=None,
) -> list[dict[str, Any]]:
    """
    Batch-check every tool defined in *tools_config*.

    Parameters
    ----------
    tools_config : list[dict] | None
        Usually loaded from ``software.yaml`` under the
        ``external_tools`` key.
    logger : loguru.Logger | None
        If given, each tool result is automatically logged at the
        appropriate level (info / warning / error).

    Returns
    -------
    list[dict]
        One result dict per tool, in the same order.
    """
    if not tools_config:
        return []
    results = []
    for tool in tools_config:
        result = check_external_tool(**tool)
        results.append(result)
        if logger is not None:
            _log_single_result(result, logger)
    return results


def format_tool_status(tool: dict[str, Any], use_rich: bool = False) -> str:
    """
    Return a short status string for a tool.

    Parameters
    ----------
    tool : dict
        Result dict from :func:`check_external_tool`.
    use_rich : bool
        If *True*, wrap the string with Rich colour markup.
        If *False* (default), return plain ASCII text suitable
        for direct printing or logging.
    """
    name = tool["name"]
    ver = tool.get("version") or "unknown"

    if not tool["available"]:
        msg = f"✗ {name} — not found"
        return f"[red]{msg}[/red]" if use_rich else msg

    if tool.get("min_version") and not tool.get("version_ok", True):
        msg = f"⚠ {name} {ver} < required {tool['min_version']}"
        return f"[yellow]{msg}[/yellow]" if use_rich else msg

    msg = f"✓ {name} {ver}"
    return f"[green]{msg}[/green]" if use_rich else msg


def assert_required_tools(results: list[dict[str, Any]]) -> None:
    """
    Raise RuntimeError if any tool marked ``required: true`` is missing
    or below its minimum version.
    """
    failures = []
    for r in results:
        if not r.get("available"):
            failures.append(f"{r['name']}: not found")
        elif not r.get("version_ok", True):
            failures.append(
                f"{r['name']}: version {r.get('version')} < required {r.get('min_version')}"
            )
    if failures:
        raise RuntimeError(
            "Missing or incompatible external dependencies:\n  • "
            + "\n  • ".join(failures)
        )


if __name__ == "__main__":
    # Quick smoke-test — use the project's loguru logger for unified output
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from gpse.utils.log_utils import logger_init

    _logger = logger_init(log_level="INFO")

    _demo_tools = [
        {"name": "plink", "cmd": "plink", "version_flag": "--version", "min_version": "1.9"},
        {"name": "python", "cmd": "python3", "version_flag": "--version"},
    ]
    check_all_external_tools(_demo_tools, logger=_logger)
