#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Training Runner
=====================
Loads a YAML batch config and runs ``gpse train`` once per trait.

Config schema::

    defaults:
      <any gpse train option>: value   # inherited by every trait
      results_root: /path/to/output    # per-trait output = <results_root>/<name>
    traits:
      - name: FT                       # required; becomes --target_trait
        task_type: classification      # any train option may be overridden
        n_classes: 3
        models: [rf_clf, xgboost_clf]
        results_dir: /custom/output    # overrides results_root
        enabled: false                 # optional; skip this trait
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import yaml
from loguru import logger as main_logger

# Keys consumed by the batch runner itself; everything else must be a
# ``gpse train`` long option and is passed through.
RESERVED_KEYS = {"name", "enabled", "results_root", "target_trait"}


def _train_option_actions() -> dict[str, argparse.Action]:
    """Map every ``gpse train`` long option name to its argparse action."""
    from gpse.train.workflow import _build_parser

    parser = _build_parser()
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                actions[option[2:]] = action
    return actions


def load_batch_config(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the YAML batch config; return (defaults, traits)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Batch config not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Batch config must be a YAML mapping: {config_path}")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping of gpse train options")

    traits = data.get("traits")
    if not isinstance(traits, list) or not traits:
        raise ValueError("'traits' must be a non-empty list")
    for index, trait in enumerate(traits, start=1):
        if not isinstance(trait, dict):
            raise ValueError(f"traits entry #{index} must be a mapping")
        name = trait.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"traits entry #{index} is missing a string 'name'")
    return defaults, traits


def build_trait_argv(
    defaults: dict[str, Any],
    trait: dict[str, Any],
    actions: dict[str, argparse.Action] | None = None,
) -> list[str]:
    """Merge defaults with one trait entry and build a ``gpse train`` argv."""
    actions = actions or _train_option_actions()
    name = trait["name"]

    if "target_trait" in defaults or "target_trait" in trait:
        raise ValueError(
            f"trait '{name}': 'target_trait' is derived from 'name'; do not set it directly"
        )

    merged = {**defaults, **{k: v for k, v in trait.items() if k not in RESERVED_KEYS}}
    results_root = trait.get("results_root", defaults.get("results_root"))
    merged.pop("results_root", None)

    if not merged.get("results_dir"):
        if not results_root:
            raise ValueError(
                f"trait '{name}': define 'results_dir' or set 'results_root' in defaults"
            )
        merged["results_dir"] = str(Path(str(results_root)) / name)
    merged["target_trait"] = name

    argv: list[str] = []
    for key, value in merged.items():
        if value is None:
            continue
        action = actions.get(key)
        if action is None:
            raise ValueError(
                f"trait '{name}': unknown option '{key}' (not a 'gpse train' flag)"
            )
        flag = f"--{key}"
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if bool(value) else f"--no-{key}")
        elif isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(flag)
        elif action.nargs in ("+", "*"):
            values = value if isinstance(value, (list, tuple)) else [value]
            argv.append(flag)
            argv.extend(str(item) for item in values)
        else:
            argv.append(flag)
            argv.append(str(value))
    return argv


def run_batch(config_path: str | Path, dry_run: bool = False) -> int:
    """Run ``gpse train`` for every enabled trait; return 0 when all succeed."""
    defaults, traits = load_batch_config(config_path)
    actions = _train_option_actions()

    runnable = [trait for trait in traits if trait.get("enabled", True)]
    skipped = [trait["name"] for trait in traits if not trait.get("enabled", True)]
    if skipped:
        main_logger.info(f"Skipping disabled traits: {skipped}")
    if not runnable:
        main_logger.error("No enabled traits to run")
        return 1

    main_logger.info(f"Batch training: {len(runnable)} trait(s) from {config_path}")
    outcomes: list[tuple[str, str]] = []
    for index, trait in enumerate(runnable, start=1):
        name = trait["name"]
        try:
            argv = build_trait_argv(defaults, trait, actions)
        except ValueError as exc:
            main_logger.error(str(exc))
            outcomes.append((name, "config-error"))
            continue

        if dry_run:
            print(f"gpse train {' '.join(argv)}")
            outcomes.append((name, "dry-run"))
            continue

        main_logger.info("=" * 70)
        main_logger.info(f"[{index}/{len(runnable)}] Training trait: {name}")
        main_logger.info("=" * 70)
        from gpse.train.cli import main as train_main

        try:
            exit_code = train_main(argv, prog="gpse train")
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
        except Exception:
            main_logger.error(f"Trait {name} failed with an unexpected error:")
            main_logger.error(traceback.format_exc())
            exit_code = 1

        if exit_code == 130:
            main_logger.warning("Batch interrupted by user (Ctrl+C).")
            outcomes.append((name, "interrupted"))
            _log_summary(outcomes)
            return 130
        outcomes.append((name, "ok" if exit_code == 0 else f"failed({exit_code})"))

    _log_summary(outcomes)
    return 0 if all(status in {"ok", "dry-run"} for _, status in outcomes) else 1


def _log_summary(outcomes: list[tuple[str, str]]) -> None:
    main_logger.info("=" * 70)
    main_logger.info("Batch summary:")
    for name, status in outcomes:
        main_logger.info(f"  {name}: {status}")
    main_logger.info("=" * 70)
