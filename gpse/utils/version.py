#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import platform
import importlib.metadata as importlib_metadata
from pathlib import Path
from rich.table import Table as RichTable

# TOML parser: tomllib (py311+) or tomli (backport)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Load custom functions from the configuration module
# Compatible with both package import and direct execution
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from gpse.utils.configuration import load_software_config
    from gpse.utils.print_utils import print_info_table, print_section_panel, print_in_columns, console
    from gpse.utils.dependency_checker import check_all_external_tools
else:
    from .configuration import load_software_config
    from .print_utils import print_info_table, print_section_panel, print_in_columns, console
    from .dependency_checker import check_all_external_tools


def _get_project_deps():
    """Read production dependencies from pyproject.toml (Poetry section)."""
    # Locate pyproject.toml relative to this file: ../../..
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return []
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    return [k for k in deps.keys() if k != "python"]

def _get_sys_info():
    return {
        "Python": sys.version.split("|")[0].strip(),
        "Executable": sys.executable,
        "Machine": platform.platform(),
    }

def _get_deps_info(deps):
    def get_ver(pkg):
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return "[red]Not Found[/red]"
    return {pkg: get_ver(pkg) for pkg in deps}

def _get_external_tools_info(software_conf: dict | None = None):
    """Check external tools defined in software.yaml and return a status dict."""
    if software_conf is None:
        software_conf = load_software_config()
    tools_cfg = software_conf.get("external_tools", [])
    if not tools_cfg:
        return {}
    results = check_all_external_tools(tools_cfg)
    return {
        r["name"]: (
            f"[green]{r.get('version', 'OK')}[/green]"
            if r["available"] and r.get("version_ok", True)
            else (
                f"[yellow]{r.get('version', 'low')} < {r.get('min_version')}[/yellow]"
                if r["available"]
                else "[red]Not Found[/red]"
            )
        )
        for r in results
    }

def show_versions(project_name=None, deps=None, extras=None, software_conf=None):
    """
    Print debug version and system information with rich beauty.
    When *deps* is None, dependencies are auto-read from pyproject.toml.
    External tools are auto-checked from software.yaml.
    """
    if deps is None:
        deps = _get_project_deps()
    # 1. Python Dependencies Table
    deps_info = _get_deps_info(deps)
    deps_table = print_info_table(
        title=f"[bold blue]{project_name}[/bold blue] Deps",
        data=deps_info,
        column_names=["Package", "Version"],
        header_style="bold magenta",
        return_table=True
    )

    # 2. Software Metadata Table
    right_tables = []
    metadata_tables = right_tables
    if extras:
        metadata_tables.append(print_info_table(
            title="[bold green]Software Info[/bold green]",
            data=extras,
            column_names=["Attribute", "Value"],
            header_style="bold yellow",
            border_style="green",
            return_table=True
        ))

    # 3. Runtime Environment Table
    runtime_info = _get_sys_info()
    ext_tools_info = _get_external_tools_info(software_conf)
    for name, status in ext_tools_info.items():
        runtime_info[f"Tool: {name}"] = status

    metadata_tables.append(print_info_table(
        title="[bold cyan]Runtime Environment[/bold cyan]",
        data=runtime_info,
        column_names=["Component", "Details"],
        header_style="bold white",
        border_style="cyan",
        return_table=True
    ))

    # Uniform width for the right-hand metadata tables.
    if len(metadata_tables) > 1:
        widths = [console.measure(t).maximum for t in metadata_tables]
        max_width = max(widths)
        for t in metadata_tables:
            t.width = max_width

    # Stack Software Info and Runtime Environment vertically.
    right_column = RichTable(show_header=False, box=None, padding=0, expand=False)
    for rt in right_tables:
        right_column.add_row(rt)

    # Print Python dependencies on the left and runtime metadata on the right.
    left_column = RichTable(show_header=False, box=None, padding=0, expand=False)
    left_column.add_row(deps_table)

    print_in_columns([left_column, right_column])

if __name__ == "__main__":
    # get software configuration from the config file
    _software_conf = load_software_config()
    sw = _software_conf.get("software", {})

    # Show versions with full metadata (deps + external tools auto-read)
    show_versions(
        project_name=sw.get("app_name", "gpse"),
        deps=None,
        extras={
            "Version": sw.get("version", "unknown"),
            "Author": sw.get("author", "unknown"),
            "Email": sw.get("email", "unknown"),
            "URL": sw.get("url", ""),
            "Description": sw.get("description", "")
        },
        software_conf=_software_conf,
    )
