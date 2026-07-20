#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line interface for ``gpse batch``."""

from __future__ import annotations

import argparse
import sys

_EPILOG = """\
YAML config schema:
  defaults:
    <any gpse train option>: value  # inherited by every trait
    results_root: /path/to/output   # per-trait output = <results_root>/<name>
  traits:
    - name: FT                      # required; becomes --target_trait
      task_type: classification     # any train option may be overridden
      n_classes: 3
      models: [rf_clf, xgboost_clf]
      results_dir: /custom/output   # optional; overrides results_root
      enabled: false                # optional; skip this trait
"""


def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    """Run the GPSE batch training CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=prog or "python -m gpse.batch.cli",
        description="Train GPSE models for multiple traits from one YAML config.",
        formatter_class=formatter_class or argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
        parents=parents or [],
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML batch config file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the generated 'gpse train' commands without running them",
    )
    args = parser.parse_args(raw_args)

    from gpse.batch.runner import run_batch

    try:
        return run_batch(args.config, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
