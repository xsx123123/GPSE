"""MCP (Model Context Protocol) server integration for GPSE."""

__all__ = ["main"]


def main() -> None:
    """Start the GPSE MCP server over stdio (lazy import to keep it optional)."""
    from gpse.mcp.server import main as _main

    _main()
