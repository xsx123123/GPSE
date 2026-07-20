"""Portable, read-only result bundles generated after model training finishes."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULT_SCHEMA_VERSION = 1


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_columns(task_type: str) -> tuple[str, list[str]]:
    if task_type == "classification":
        return "accuracy", ["accuracy", "accuracy_std", "f1", "auc", "pr_auc"]
    return "pearson", ["pearson", "pearson_std", "spearman", "mse", "rmse", "mae"]


def build_result_rows(
    all_model_results: dict[str, dict[str, Any]], task_type: str, split_strategy: str
) -> pd.DataFrame:
    """Create one normalized report row per model from completed summaries."""
    primary_metric, metric_columns = _metric_columns(task_type)
    rows: list[dict[str, Any]] = []
    for model_name, result in all_model_results.items():
        selection = result.get("training_selection", {})
        row: dict[str, Any] = {
            "split_strategy": split_strategy,
            "model": model_name,
            "status": "failed" if result.get("error") else "completed",
            "is_gblup_baseline": model_name == "gblup_reg",
            "primary_metric": primary_metric,
            "cv_metric": selection.get("cv_metric"),
            "cv_mean": selection.get("cv_mean"),
            "cv_std": selection.get("cv_std"),
            "selected_repeat_idx": selection.get("selected_repeat_idx"),
            "selection_source": selection.get("source"),
            "selected_models": ",".join(result.get("selected_models") or []),
            "best_params": json.dumps(selection.get("best_params", {}), sort_keys=True),
            "deployment_ensemble_path": result.get("deployment_ensemble", {}).get("path"),
            "representative_model_path": result.get("representative_model", {}).get("model_path"),
            "error": result.get("error"),
        }
        if task_type == "classification":
            row.update(
                {
                    "accuracy": result.get("avg_ensemble_accuracy", result.get("avg_test_accuracy")),
                    "accuracy_std": result.get(
                        "std_ensemble_accuracy", result.get("std_test_accuracy")
                    ),
                    "f1": result.get("avg_test_f1"),
                    "auc": result.get("avg_test_auc"),
                    "pr_auc": result.get("avg_ensemble_pr_auc", result.get("avg_test_pr_auc")),
                }
            )
        else:
            row.update(
                {
                    "pearson": result.get("avg_ensemble_pearson", result.get("avg_test_pearson")),
                    "pearson_std": result.get(
                        "std_ensemble_pearson", result.get("std_test_pearson")
                    ),
                    "spearman": result.get(
                        "avg_ensemble_spearman", result.get("avg_test_spearman")
                    ),
                    "mse": result.get("avg_ensemble_mse", result.get("avg_test_mse")),
                    "rmse": result.get("avg_ensemble_rmse"),
                    "mae": result.get("avg_ensemble_mae"),
                    "metric_scale": "original" if result.get("phenotype_standardized") else "raw",
                }
            )
        rows.append(row)

    columns = [
        "rank",
        "split_strategy",
        "model",
        "status",
        "is_gblup_baseline",
        "primary_metric",
        *metric_columns,
        "cv_metric",
        "cv_mean",
        "cv_std",
        "selected_repeat_idx",
        "selection_source",
        "selected_models",
        "best_params",
        "deployment_ensemble_path",
        "representative_model_path",
        "error",
    ]
    if task_type == "regression":
        columns.insert(11, "metric_scale")
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    sortable = frame[frame["status"] == "completed"].copy()
    sortable[primary_metric] = pd.to_numeric(sortable[primary_metric], errors="coerce")
    sortable = sortable.sort_values(primary_metric, ascending=False, na_position="last")
    sortable["rank"] = np.arange(1, len(sortable) + 1)
    failed = frame[frame["status"] != "completed"].copy()
    failed["rank"] = pd.NA
    return pd.concat([sortable, failed], ignore_index=True)


def _markdown_report(summary: dict[str, Any], leaderboard: pd.DataFrame) -> str:
    run = summary["run"]
    table_columns = [
        column
        for column in ["rank", "model", "status", "primary_metric", "pearson", "accuracy", "cv_mean", "cv_std"]
        if column in leaderboard.columns
    ]
    markdown_rows = leaderboard[table_columns].fillna("").infer_objects(copy=False).astype(str)
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(row) + " |" for row in markdown_rows.itertuples(index=False, name=None)
    )
    lines = [
        "# GPSE Training Result Report",
        "",
        "## Run",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Task: `{run['task_type']}`",
        f"- Trait: `{run['target_trait']}`",
        f"- Split strategy: `{run['split_strategy']}`",
        f"- Samples / SNPs: `{run['sample_count']}` / `{run['feature_count']}`",
        f"- Completed / failed models: `{run['completed_models']}` / `{run['failed_models']}`",
        "",
        "## Preprocessing",
        "",
        f"- Feature selection: `{json.dumps(run['feature_selection'], sort_keys=True)}`",
        f"- Genotype imputation: `{json.dumps(run['genotype_imputation'], sort_keys=True)}`",
        "",
        "## Leaderboard",
        "",
        *table_lines,
        "",
        "## Audit",
        "",
        "- Hold-out metrics are reporting-only; model/parameter/repeat selection is train-side CV only.",
        "- See `run_summary.json` for machine-readable configuration, artifacts, and split metadata.",
        "",
    ]
    return "\n".join(lines)


def _html_report(summary: dict[str, Any], leaderboard: pd.DataFrame) -> str:
    run = summary["run"]
    table = leaderboard.to_html(index=False, na_rep="", escape=True, classes="leaderboard")
    metadata = html.escape(json.dumps(run, indent=2, sort_keys=True, default=_json_default))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>GPSE Training Result Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 2rem; color: #18212f; }}
h1, h2 {{ color: #0f4c5c; }}
.card {{ background: #f5f8fa; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th {{ background: #0f4c5c; color: white; position: sticky; top: 0; }}
th, td {{ border: 1px solid #d9e2e8; padding: 0.45rem; text-align: left; vertical-align: top; }}
tr:nth-child(even) {{ background: #f7fafc; }}
pre {{ white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
<h1>GPSE Training Result Report</h1>
<div class=\"card\"><strong>Trait:</strong> {html.escape(str(run['target_trait']))}<br>
<strong>Task:</strong> {html.escape(str(run['task_type']))}<br>
<strong>Split:</strong> {html.escape(str(run['split_strategy']))}<br>
<strong>Generated:</strong> {html.escape(summary['generated_at'])}</div>
<h2>Leaderboard</h2>
{table}
<h2>Run and Audit Metadata</h2>
<div class=\"card\"><pre>{metadata}</pre></div>
</body>
</html>
"""


def write_result_bundle(
    results_dir: str | Path,
    all_model_results: dict[str, dict[str, Any]],
    *,
    task_type: str,
    target_trait: str,
    split_strategy: str,
    sample_count: int,
    feature_count: int,
    feature_selection: dict[str, Any],
    genotype_imputation: dict[str, Any],
) -> dict[str, Path]:
    """Write read-only CSV, JSON, Markdown, and HTML summaries for one training run."""
    results_dir = Path(results_dir)
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = build_result_rows(all_model_results, task_type, split_strategy)
    leaderboard_path = reports_dir / "model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    completed = int((leaderboard["status"] == "completed").sum())
    split_manifest = _read_json(results_dir / "split_manifest.json")
    stacking_selection = _read_json(results_dir / "stacking_selected_models.json")
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "task_type": task_type,
            "target_trait": target_trait,
            "split_strategy": split_strategy,
            "sample_count": int(sample_count),
            "feature_count": int(feature_count),
            "completed_models": completed,
            "failed_models": int(len(leaderboard) - completed),
            "feature_selection": feature_selection,
            "genotype_imputation": genotype_imputation,
            "split_manifest": split_manifest,
            "stacking_selection": stacking_selection,
        },
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    json_path = reports_dir / "run_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    markdown_path = reports_dir / "training_report.md"
    markdown_path.write_text(_markdown_report(summary, leaderboard), encoding="utf-8")
    html_path = reports_dir / "training_report.html"
    html_path.write_text(_html_report(summary, leaderboard), encoding="utf-8")
    return {
        "leaderboard": leaderboard_path,
        "summary": json_path,
        "markdown": markdown_path,
        "html": html_path,
    }
