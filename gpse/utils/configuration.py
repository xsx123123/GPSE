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

# Use stdlib logging here instead of importing loguru at the module level.
# Loguru's default handler emits DEBUG lines to stderr immediately,
# which means DEBUG messages appear *before* logger_init() has a chance
# to configure the real handlers.  Stdlib logging defaults to WARNING,
# so debug/info are silent until an application explicitly lowers the
# level or installs an intercept handler.
import logging

logger = logging.getLogger("gpse.config")


def _load_yaml(config_name: str) -> dict[str, Any]:
    """Load a YAML file from gpse.config by basename (no extension)."""
    config_path = resources.files("gpse.config") / f"{config_name}.yaml"
    logger.debug("Loading config '%s' from %s", config_name, config_path)
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to load config '%s': %s", config_name, exc)
        raise
    logger.debug("Config '%s' loaded successfully (%d top-level keys)", config_name, len(data))
    return data


def load_software_config() -> dict[str, Any]:
    """Load software metadata from software.yaml."""
    cfg = _load_yaml("software")
    sw = cfg.get("software", {})
    logger.debug(
        "Software config: app=%s, ver=%s, tools=%d",
        sw.get("app_name"),
        sw.get("version"),
        len(cfg.get("external_tools", [])),
    )
    return cfg


def load_default_config() -> dict[str, Any]:
    """Load default analysis parameters from default.yaml."""
    cfg = _load_yaml("default")
    logger.debug(
        "Default config: log_level=%s, label=%s",
        cfg.get("logs", {}).get("log_level"),
        cfg.get("logs", {}).get("Label"),
    )
    return cfg
