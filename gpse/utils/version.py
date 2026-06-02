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
else:
    from .configuration import load_software_config
    from .print_utils import print_info_table, print_section_panel, print_in_columns, console


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

def show_versions(project_name=None, deps=None, extras=None):
    """
    Print debug version and system information with rich beauty.
    When *deps* is None, dependencies are auto-read from pyproject.toml.
    """
    if deps is None:
        deps = _get_project_deps()
    tables = []
    
    # 1. Dependencies Table
    deps_info = _get_deps_info(deps)
    tables.append(print_info_table(
        title=f"[bold blue]{project_name}[/bold blue] Deps",
        data=deps_info,
        column_names=["Package", "Version"],
        header_style="bold magenta",
        return_table=True
    ))

    # 2. Software Metadata Table
    right_tables = []
    if extras:
        right_tables.append(print_info_table(
            title="[bold green]Software Info[/bold green]",
            data=extras,
            column_names=["Attribute", "Value"],
            header_style="bold yellow",
            border_style="green",
            return_table=True
        ))

    # 3. System Environment Table
    right_tables.append(print_info_table(
        title="[bold cyan]System Environment[/bold cyan]",
        data=_get_sys_info(),
        column_names=["Component", "Details"],
        header_style="bold white",
        border_style="cyan",
        return_table=True
    ))

    # Uniform width for the right-hand stacked tables
    if len(right_tables) > 1:
        widths = [console.measure(t).maximum for t in right_tables]
        max_width = max(widths)
        for t in right_tables:
            t.width = max_width

    # Stack Software Info + System Environment vertically as the right column
    right_column = RichTable(show_header=False, box=None, padding=0, expand=False)
    for rt in right_tables:
        right_column.add_row(rt)

    # Print Deps on the left, stacked info on the right
    print_in_columns([tables[0], right_column])

if __name__ == "__main__":
    # get software configuration from the config file
    _software_conf = load_software_config()
    sw = _software_conf.get("software", {})
    
    # Show versions with full metadata (deps auto-read from pyproject.toml)
    show_versions(
        project_name=sw.get("app_name", "gpse"),
        deps=None,
        extras={
            "Version": sw.get("version", "unknown"),
            "Author": sw.get("author", "unknown"),
            "Email": sw.get("email", "unknown"),
            "URL": sw.get("url", ""),
            "Description": sw.get("description", "")
        }
    )
