#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loading utilities for gpse.

All YAML configs are stored inside the installed package under
gpse/config/ and accessed via importlib.resources so that
paths remain valid regardless of the current working directory.
"""

from importlib import resources
from typing import Any

import yaml

# Use the project's unified logger (loguru) without circular imports.
# loguru is a third-party package and does not depend on gpse modules.
try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("gpse.config")


def _load_yaml(config_name: str) -> dict[str, Any]:
    """Load a YAML file from gpse.config by basename (no extension)."""
    config_path = resources.files("gpse.config") / f"{config_name}.yaml"
    logger.debug(f"Loading config '{config_name}' from {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error(f"Failed to load config '{config_name}': {exc}")
        raise
    logger.debug(f"Config '{config_name}' loaded successfully ({len(data)} top-level keys)")
    return data


def load_software_config() -> dict[str, Any]:
    """Load software metadata from software.yaml."""
    cfg = _load_yaml("software")
    sw = cfg.get("software", {})
    logger.debug(
        f"Software config: app={sw.get('app_name')}, ver={sw.get('version')}, "
        f"tools={len(cfg.get('external_tools', []))}"
    )
    return cfg


def load_default_config() -> dict[str, Any]:
    """Load default analysis parameters from default.yaml."""
    cfg = _load_yaml("default")
    logger.debug(
        f"Default config: log_level={cfg.get('logs', {}).get('log_level')}, "
        f"label={cfg.get('logs', {}).get('Label')}"
    )
    return cfg
