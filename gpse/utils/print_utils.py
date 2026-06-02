#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpse Print Utilities
====================
This module provides reusable rich-based printing functions for consistent
and beautiful terminal output.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Dict, List, Optional, Any

console = Console()

from rich.columns import Columns

def print_info_table(
    title: str,
    data: Dict[str, Any],
    column_names: List[str] = ["Key", "Value"],
    header_style: str = "bold cyan",
    border_style: str = "dim",
    box_type: box.Box = box.ROUNDED,
    return_table: bool = False
):
    """
    Print a dictionary as a beautiful rich table or return the Table object.
    """
    table = Table(
        title=title,
        header_style=header_style,
        border_style=border_style,
        box=box_type,
        show_header=True
    )
    
    for col in column_names:
        table.add_column(col)
        
    for key, value in data.items():
        table.add_row(str(key), str(value))
        
    if return_table:
        return table
    console.print(table)

def print_in_columns(renderables: List[Any], align: str = "center"):
    """
    Print multiple renderables (like tables) side-by-side.
    """
    columns = Columns(renderables, align=align, expand=False, equal=False)
    console.print(columns)

def print_section_panel(
    content: Any,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    border_style: str = "blue",
    padding: tuple = (0, 1)
):
    """
    Print content within a rich panel.
    """
    panel = Panel(
        content,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        padding=padding,
        expand=False
    )
    console.print(panel)
