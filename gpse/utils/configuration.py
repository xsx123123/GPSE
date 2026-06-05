#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loading utilities for gpse.

All YAML configs are stored inside the installed package under
gpse/config/ and accessed via importlib.resources so that
paths remain valid regardless of the current working directory.
"""

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import logging
import yaml

logger = logging.getLogger("gpse.config")

PROJECT_CONFIG_FILES = ("gpse.yaml", "gpse.local.yaml")


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


def _load_yaml_path(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file from an explicit filesystem path."""
    path = Path(config_path).expanduser()
    logger.debug("Loading user config from %s", path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to load user config '%s': %s", path, exc)
        raise
    if not isinstance(data, dict):
        raise TypeError(f"User config must be a YAML mapping: {path}")
    return data


def _merge_named_list(base: list[Any], override: list[Any]) -> list[Any]:
    """Merge list items by their 'name' key when every item is a mapping."""
    if not all(isinstance(item, dict) and "name" in item for item in base + override):
        return deepcopy(override)

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in base:
        name = item["name"]
        merged[name] = deepcopy(item)
        order.append(name)
    for item in override:
        name = item["name"]
        if name in merged:
            merged[name] = _deep_merge(merged[name], item)
        else:
            merged[name] = deepcopy(item)
            order.append(name)
    return [merged[name] for name in order]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        elif (
            key in result
            and isinstance(result[key], list)
            and isinstance(value, list)
        ):
            result[key] = _merge_named_list(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _project_config_paths(search_dir: str | Path | None = None) -> list[Path]:
    """Return existing project-level config files in merge order."""
    root = Path(search_dir or Path.cwd())
    return [root / name for name in PROJECT_CONFIG_FILES if (root / name).is_file()]


def _load_config_with_overrides(
    config_name: str,
    user_config_path: str | Path | None = None,
    *,
    auto_project_config: bool = False,
    search_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load package config and merge optional project/user YAML overrides."""
    config = _load_yaml(config_name)

    override_paths: list[Path] = []
    if auto_project_config:
        override_paths.extend(_project_config_paths(search_dir))
    if user_config_path:
        override_paths.append(Path(user_config_path).expanduser())

    for path in override_paths:
        if not path.is_file():
            raise FileNotFoundError(f"User config not found: {path}")
        config = _deep_merge(config, _load_yaml_path(path))

    return config


def load_software_config(
    user_config_path: str | Path | None = None,
    *,
    auto_project_config: bool = False,
    search_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load software metadata from software.yaml."""
    cfg = _load_config_with_overrides(
        "software",
        user_config_path,
        auto_project_config=auto_project_config,
        search_dir=search_dir,
    )
    sw = cfg.get("software", {})
    logger.debug(
        "Software config: app=%s, ver=%s, tools=%d",
        sw.get("app_name"),
        sw.get("version"),
        len(cfg.get("external_tools", [])),
    )
    return cfg


def load_default_config(
    user_config_path: str | Path | None = None,
    *,
    auto_project_config: bool = False,
    search_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load default analysis parameters from default.yaml."""
    cfg = _load_config_with_overrides(
        "default",
        user_config_path,
        auto_project_config=auto_project_config,
        search_dir=search_dir,
    )
    logger.debug(
        "Default config: log_level=%s, label=%s",
        cfg.get("logs", {}).get("log_level"),
        cfg.get("logs", {}).get("Label"),
    )
    return cfg


def load_topsis_config(
    user_config_path: str | Path | None = None,
    *,
    auto_project_config: bool = False,
    search_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load TOPSIS evaluation parameters from topsis.yaml."""
    cfg = _load_config_with_overrides(
        "topsis",
        user_config_path,
        auto_project_config=auto_project_config,
        search_dir=search_dir,
    )
    logger.debug("TOPSIS config: tasks=%d", len(cfg.get("tasks", {})))
    return cfg
