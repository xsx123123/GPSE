#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Pipeline Module
====================
Runs all or specified models and optionally performs Stacking ensemble.
"""

import os
import json
import traceback
from concurrent.futures import as_completed

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from loguru import logger as main_logger

from gpse.config import ModelConstants
from gpse.utils.genomic_utils import create_comparison_table, call_topsis_evaluator
from gpse.utils.paralle import graceful_process_pool
from gpse.train.stacking import StackingEnsemble
from gpse.train._repeat_training import create_holdout_indices
from gpse.train._results import write_result_bundle
from gpse.utils.log_utils import shorten_path


def _init_model_worker_threads(n_threads: int) -> None:
    for env_var in ModelConstants.thread_env_vars:
        os.environ[env_var] = str(n_threads)


def _run_model_task(predictor, model_name, X, y, cv_pheno_data, use_same_test_set, test_indices):
    return predictor.run_model_multiple_repeats(
        model_name, X, y, cv_pheno_data, use_same_test_set, test_indices
    )


def _save_split_manifest(
    results_dir, X, train_indices, test_indices, task_type, random_seed, phenotype_scaler=None,
    split_metadata=None,
):
    manifest = {
        "split_type": "stratified_holdout" if task_type == "classification" else "random_holdout",
        "random_seed": random_seed,
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "train_ids_file": "train_ids.txt",
        "test_ids_file": "test_ids.txt",
        "phenotype_scaler": phenotype_scaler,
        **(split_metadata or {}),
    }
    (results_dir / "train_ids.txt").write_text(
        "\n".join(map(str, X.index[train_indices])) + "\n", encoding="utf-8"
    )
    (results_dir / "test_ids.txt").write_text(
        "\n".join(map(str, X.index[test_indices])) + "\n", encoding="utf-8"
    )
    with open(results_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_holdout_reports(results_dir, all_model_results, task_type, split_strategy):
    """Persist final hold-out metrics after all model/stacking choices are frozen."""
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, result in all_model_results.items():
        row = {
            "split_strategy": split_strategy,
            "model": model_name,
            "is_gblup_baseline": model_name == "gblup_reg",
        }
        if task_type == "classification":
            row.update(
                {
                    "accuracy": result.get("avg_ensemble_accuracy", result.get("avg_test_accuracy", 0.0)),
                    "accuracy_std": result.get("std_ensemble_accuracy", result.get("std_test_accuracy", 0.0)),
                    "f1": result.get("avg_test_f1", 0.0),
                    "auc": result.get("avg_test_auc", 0.0),
                    "pr_auc": result.get("avg_ensemble_pr_auc", result.get("avg_test_pr_auc", 0.0)),
                }
            )
        else:
            row.update(
                {
                    "pearson": result.get("avg_ensemble_pearson", result.get("avg_test_pearson", 0.0)),
                    "pearson_std": result.get("std_ensemble_pearson", result.get("std_test_pearson", 0.0)),
                    "spearman": result.get("avg_ensemble_spearman", result.get("avg_test_spearman", 0.0)),
                    "mse": result.get("avg_ensemble_mse", result.get("avg_test_mse", 0.0)),
                    "rmse": result.get("avg_ensemble_rmse", 0.0),
                    "mae": result.get("avg_ensemble_mae", 0.0),
                    "metric_scale": "original" if result.get("phenotype_standardized") else "raw",
                }
            )
        rows.append(row)

    current = pd.DataFrame(rows)
    csv_path = reports_dir / "model_comparison_holdout.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["split_strategy"] != split_strategy]
        current = pd.concat([existing, current], ignore_index=True, sort=False)
    current.to_csv(csv_path, index=False)
    current.to_csv(reports_dir / "holdout_metrics.csv", index=False)
    with open(reports_dir / "holdout_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(current.to_dict(orient="records"), handle, indent=2)
    main_logger.info(f"Final hold-out reports saved to {shorten_path(reports_dir)}")


def _write_topsis_final_predictions(results_dir, all_model_results, X, task_type):
    """Persist hold-out predictions from the CV-TOPSIS top-ranked model.

    The best model is the rank-1 entry of ``model_comparison_cv_topsis_simple.csv``
    (falling back to CV mean/std ranking). Predictions come from that model's
    selected repeat: per-fold test predictions are averaged (regression) or
    majority-voted (classification). Values are converted back to the original
    phenotype scale when standardization was applied.
    """
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    topsis_simple_path = results_dir / "model_comparison_cv_topsis_simple.csv"
    topsis_score = None
    if topsis_simple_path.exists():
        ranking = pd.read_csv(topsis_simple_path).sort_values("TOPSIS_Rank")
        best_row = ranking.iloc[0]
        best_model = best_row["Model"]
        topsis_score = float(best_row["TOPSIS_Score"])
    else:
        candidates = [
            (
                name,
                result["training_selection"]["cv_mean"],
                result["training_selection"]["cv_std"],
            )
            for name, result in all_model_results.items()
            if "training_selection" in result
        ]
        if not candidates:
            main_logger.warning(
                "No training-selection CV results available; "
                "skipping TOPSIS-based final predictions"
            )
            return None
        candidates.sort(key=lambda item: (-item[1], item[2]))
        best_model = candidates[0][0]

    selection = all_model_results[best_model].get("training_selection", {})
    repeat_idx = int(selection.get("selected_repeat_idx", 0))
    predictions_file = (
        results_dir / best_model / f"repeat_{repeat_idx + 1}" / "all_predictions.json"
    )
    if not predictions_file.exists():
        main_logger.warning(
            f"Missing {predictions_file}; skipping TOPSIS-based final predictions"
        )
        return None
    with open(predictions_file, encoding="utf-8") as handle:
        all_predictions = json.load(handle)
    test_entries = all_predictions.get("test") or []
    if not test_entries:
        main_logger.warning(
            f"No test predictions in {predictions_file}; "
            "skipping TOPSIS-based final predictions"
        )
        return None

    indices = np.asarray(test_entries[0]["indices"], dtype=int)
    observed = np.asarray(test_entries[0]["true_values"], dtype=float)
    predicted_stack = np.asarray(
        [entry["predicted_values"] for entry in test_entries], dtype=float
    )
    if task_type == "classification":
        predicted = pd.DataFrame(predicted_stack).mode().iloc[0].to_numpy()
    else:
        predicted = predicted_stack.mean(axis=0)

    split_manifest_path = results_dir / "split_manifest.json"
    if split_manifest_path.exists():
        with open(split_manifest_path, encoding="utf-8") as handle:
            scaler = json.load(handle).get("phenotype_scaler") or {}
        if scaler.get("applied"):
            observed = observed * scaler["std"] + scaler["mean"]
            predicted = predicted * scaler["std"] + scaler["mean"]

    output_path = reports_dir / "final_predictions.csv"
    pd.DataFrame(
        {
            "ID": X.index[indices].astype(str),
            "observed": observed,
            "predicted": predicted,
        }
    ).to_csv(output_path, index=False)
    score_note = f", TOPSIS_Score={topsis_score:.6f}" if topsis_score is not None else ""
    main_logger.info(
        f"Final TOPSIS-based predictions saved to {output_path} "
        f"(model={best_model}{score_note}, repeat={repeat_idx + 1}, "
        f"fold_predictions={len(test_entries)})"
    )
    return output_path


def run_all_models(
    self,
    geno_file: str,
    pheno_file: str,
    target_trait: str,
    models: Optional[List[str]] = None,
    use_stacking: bool = False,
    top_n_models: int = 5,
    cv_folds: int = 5,
    use_same_test_set: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all or specified models, optionally with Stacking ensemble.

    Parameters
    ----------
    geno_file : str
        Genotype data file path.
    pheno_file : str
        Phenotype data file path.
    target_trait : str
        Target trait name.
    models : Optional[List[str]]
        List of model names; None means all available models.
    use_stacking : bool
        Whether to use Stacking ensemble.
    top_n_models : int
        Number of top models for Stacking.
    cv_folds : int
        CV folds for Stacking ensemble.
    use_same_test_set : bool
        Whether to use the same test set across all models and repeats.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing all model results.
    """
    X, y, pheno_data = self.load_data(geno_file, pheno_file, target_trait)

    fixed_test_indices = None
    if use_same_test_set:
        train_indices, fixed_test_indices, split_metadata = create_holdout_indices(
            y,
            self.test_size,
            self.random_seed,
            self.task_type,
            X=X,
            split_strategy=self.split_strategy,
            structure_clusters=self.structure_clusters,
            return_metadata=True,
        )
        phenotype_scaler = None
        if self.task_type == "regression" and self.standardize_phenotype:
            _, phenotype_scaler = self._standardize_phenotype(y.iloc[train_indices])
        _save_split_manifest(
            self.results_dir,
            X,
            train_indices,
            fixed_test_indices,
            self.task_type,
            self.random_seed,
            phenotype_scaler,
            split_metadata,
        )
        main_logger.info("Saved fixed hold-out split manifest; test data is reserved for final evaluation")

    cv_pheno_data = None
    if self.cv_file is not None:
        main_logger.info("Loading user-supplied cross-validation folds...")
        cv_pheno_data = self.prepare_cv_folds(pheno_data, target_trait)
        main_logger.info("External cross-validation folds ready")
    else:
        main_logger.info(
            "CV folds will be generated and persisted from each post-hold-out training set"
        )

    if models is None:
        models = self.available_models

    main_logger.info(f"Will run the following models: {', '.join(models)}")

    all_model_results = {}
    model_workers = min(self.max_parallel_jobs, len(models))

    if model_workers > 1:
        main_logger.info(f"Using {model_workers} parallel model workers")
        with graceful_process_pool(
            max_workers=model_workers,
            initializer=_init_model_worker_threads,
            initargs=(self.n_threads,),
            logger=main_logger,
        ) as executor:
            futures = {
                executor.submit(
                    _run_model_task,
                    self,
                    model_name,
                    X,
                    y,
                    cv_pheno_data,
                    use_same_test_set,
                    fixed_test_indices,
                ): (idx, model_name)
                for idx, model_name in enumerate(models, start=1)
            }
            for future in as_completed(futures):
                idx, model_name = futures[future]
                try:
                    all_model_results[model_name] = future.result()
                    main_logger.info(f"[{idx}/{len(models)}] Completed model: {model_name}")
                except Exception as e:
                    main_logger.error(f"Model {model_name} execution failed: {str(e)}")
                    main_logger.error(traceback.format_exc())
    else:
        for i, model_name in enumerate(models, start=1):
            main_logger.info(
                f"[{i}/{len(models)}] Starting training for model: {model_name}"
            )
            try:
                model_summary = self.run_model_multiple_repeats(
                    model_name, X, y, cv_pheno_data, use_same_test_set, fixed_test_indices
                )
                all_model_results[model_name] = model_summary
            except Exception as e:
                main_logger.error(f"Model {model_name} execution failed: {str(e)}")
                main_logger.error(traceback.format_exc())

    main_logger.info("=" * 50)
    main_logger.info("Model performance comparison before ensemble:")
    if not all_model_results:
        raise RuntimeError(
            "All models failed to train; see the per-model errors above. "
            "No comparison, ensemble, or reports can be produced."
        )
    create_comparison_table(all_model_results, self.results_dir, main_logger)

    selected_models_for_stacking = None
    if len(all_model_results) >= 2:
        primary_label = "CV Accuracy" if self.task_type == "classification" else "CV Pearson"
        cv_rows = [
            {
                "Model": name,
                primary_label: result["training_selection"]["cv_mean"],
                "CV Std": result["training_selection"]["cv_std"],
            }
            for name, result in all_model_results.items()
            if "training_selection" in result
        ]
        cv_comparison = pd.DataFrame(cv_rows)
        if not cv_comparison.empty:
            cv_path = self.results_dir / "model_comparison_cv.csv"
            topsis_path = self.results_dir / "model_comparison_cv_topsis.csv"
            topsis_simple_path = self.results_dir / "model_comparison_cv_topsis_simple.csv"
            cv_comparison.to_csv(cv_path, index=False)
            try:
                call_topsis_evaluator(
                    comparison_csv=str(cv_path),
                    output_csv=str(topsis_path),
                    criteria=[primary_label, "CV Std"],
                    criteria_types=["max", "min"],
                    manual_weights="0.8,0.2",
                    min_transform="neglog",
                    simple_output=str(topsis_simple_path),
                    logger=main_logger,
                )
                topsis_ranked_models = pd.read_csv(topsis_simple_path)["Model"].tolist()
            except Exception as exc:
                main_logger.warning(
                    f"Training-only TOPSIS failed; falling back to CV mean/std ranking: {exc}"
                )
                cv_comparison = cv_comparison.sort_values(
                    [primary_label, "CV Std"], ascending=[False, True]
                )
                topsis_ranked_models = cv_comparison["Model"].tolist()
            if use_stacking:
                selected_models_for_stacking = topsis_ranked_models[:top_n_models]
                with open(self.results_dir / "stacking_selected_models.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "selection_source": "train_only_cv",
                            "criteria": [primary_label, "CV Std"],
                            "selection_method": "TOPSIS (fallback: CV mean/std rank)",
                            "selected_models": selected_models_for_stacking,
                        },
                        handle,
                        indent=2,
                    )
                main_logger.info(
                    "Selected stacking base models using training-only CV metrics: "
                    f"{selected_models_for_stacking}"
                )

    if use_stacking and len(all_model_results) >= 2:
        try:
            main_logger.info(f"{'=' * 70}")
            main_logger.info(f"Starting Stacking ensemble")

            if not selected_models_for_stacking:
                raise ValueError("No training-side CV results are available for stacking selection")
            selected_models = selected_models_for_stacking
            if fixed_test_indices is None:
                train_indices, fixed_test_indices = create_holdout_indices(
                    y,
                    self.test_size,
                    self.random_seed,
                    self.task_type,
                    X=X,
                    split_strategy=self.split_strategy,
                    structure_clusters=self.structure_clusters,
                )
            else:
                train_indices = np.setdiff1d(np.arange(len(X)), fixed_test_indices, assume_unique=True)
            X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X.iloc[fixed_test_indices], y.iloc[fixed_test_indices]
            if self.task_type == "regression" and self.standardize_phenotype:
                y_train, scaler = self._standardize_phenotype(y_train)
                if scaler["applied"]:
                    y_test = (y_test - scaler["mean"]) / scaler["std"]
            selected_params = {
                name: all_model_results[name]["training_selection"]["best_params"]
                for name in selected_models
            }

            stacking_model = StackingEnsemble(
                base_models_dir=str(self.results_dir),
                model_factory=lambda name: self.create_model(name, selected_params[name]),
                top_n_models=top_n_models,
                cv_folds=cv_folds,
                random_seed=self.random_seed,
                n_threads=self.n_threads,
                use_default_params=self.use_default_params,
                task_type=self.task_type,
                feature_selection_config=self.feature_selection_config,
                genotype_imputation_config=self.genotype_imputation_config,
            )

            stacking_results = stacking_model.fit(
                X_train, y_train, X_test, y_test, model_names=selected_models
            )

            if self.task_type == "classification":
                all_model_results["stacking_ensemble"] = {
                    "model_name": "stacking_ensemble",
                    "avg_test_accuracy": stacking_results["test_metrics"]["accuracy"],
                    "std_test_accuracy": 0.0,
                    "avg_test_f1": stacking_results["test_metrics"]["f1"],
                    "std_test_f1": 0.0,
                    "avg_test_auc": stacking_results["test_metrics"].get("auc", 0.0),
                    "std_test_auc": 0.0,
                    "selected_models": selected_models,
                }
            else:
                all_model_results["stacking_ensemble"] = {
                    "model_name": "stacking_ensemble",
                    "avg_test_pearson": stacking_results["test_metrics"]["pearson"],
                    "std_test_pearson": 0.0,
                    "selected_models": selected_models,
                }

            main_logger.info("=" * 50)
            main_logger.info("Model performance comparison after ensemble:")
            create_comparison_table(all_model_results, self.results_dir, main_logger)

        except Exception as e:
            main_logger.error(f"Stacking ensemble execution failed: {str(e)}")
            main_logger.error(traceback.format_exc())

    _write_holdout_reports(
        self.results_dir, all_model_results, self.task_type, self.split_strategy
    )
    _write_topsis_final_predictions(
        self.results_dir, all_model_results, X, self.task_type
    )
    report_paths = write_result_bundle(
        self.results_dir,
        all_model_results,
        task_type=self.task_type,
        target_trait=target_trait,
        split_strategy=self.split_strategy,
        sample_count=len(X),
        feature_count=X.shape[1],
        feature_selection=self.feature_selection_config.as_dict(),
        genotype_imputation=self.genotype_imputation_config.as_dict(),
    )
    main_logger.info(
        "Result bundle saved: "
        + ", ".join(shorten_path(path) for path in report_paths.values())
    )
    return all_model_results
