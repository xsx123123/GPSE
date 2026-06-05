#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Pipeline Module
====================
Runs all or specified models and optionally performs Stacking ensemble.
"""

import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split

from loguru import logger as main_logger

from gpse.utils.genomic_utils import create_comparison_table, call_topsis_evaluator
from gpse.train.stacking import StackingEnsemble


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

    cv_pheno_data = self.prepare_cv_folds(pheno_data, target_trait)

    if models is None:
        models = self.available_models

    main_logger.info(f"Will run the following models: {', '.join(models)}")

    all_model_results = {}

    for model_name in models:
        try:
            model_summary = self.run_model_multiple_repeats(
                model_name, X, y, cv_pheno_data, use_same_test_set
            )
            all_model_results[model_name] = model_summary
        except Exception as e:
            main_logger.error(f"Model {model_name} execution failed: {str(e)}")
            main_logger.error(traceback.format_exc())

    main_logger.info("=" * 50)
    main_logger.info("Model performance comparison before ensemble:")
    create_comparison_table(all_model_results, self.results_dir, main_logger)

    selected_models_for_stacking = None
    if use_stacking and len(all_model_results) >= 2:
        comparison_path = self.results_dir / "model_comparison.csv"
        if comparison_path.exists():
            out_base = comparison_path.stem + "_topsis"
            topsis_out = comparison_path.with_name(f"{out_base}.csv")
            topsis_simple = comparison_path.with_name(f"{out_base}_simple.csv")

            try:
                criteria, criteria_types, manual_weights = self.get_topsis_configuration()

                call_topsis_evaluator(
                    comparison_csv=str(comparison_path),
                    output_csv=str(topsis_out),
                    criteria=criteria,
                    criteria_types=criteria_types,
                    manual_weights=manual_weights,
                    min_transform="neglog",
                    simple_output=str(topsis_simple),
                    logger=main_logger,
                )

                topsis_df = pd.read_csv(topsis_simple)
                selected_models_for_stacking = topsis_df.head(top_n_models)["Model"].tolist()
                main_logger.info(
                    f"Top {top_n_models} models selected by TOPSIS ranking: {selected_models_for_stacking}"
                )

            except Exception as e:
                main_logger.error(f"TOPSIS evaluation failed: {str(e)}")
                selected_models_for_stacking = None

    if use_stacking and len(all_model_results) >= 2:
        try:
            main_logger.info(f"\n{'=' * 70}")
            main_logger.info(f"Starting Stacking ensemble")
            main_logger.info(f"{'=' * 70}")

            if selected_models_for_stacking:
                selected_models = selected_models_for_stacking
                main_logger.info(
                    f"Using top {len(selected_models)} models by TOPSIS for ensemble: {', '.join(selected_models)}"
                )
            else:
                if self.task_type == "classification":
                    model_performances = [
                        (
                            name,
                            results.get(
                                "avg_ensemble_accuracy",
                                results.get("avg_test_accuracy", 0.0),
                            ),
                        )
                        for name, results in all_model_results.items()
                    ]
                else:
                    model_performances = [
                        (
                            name,
                            results.get(
                                "avg_ensemble_pearson",
                                results.get("avg_test_pearson", 0.0),
                            ),
                        )
                        for name, results in all_model_results.items()
                    ]
                model_performances.sort(key=lambda x: x[1], reverse=True)
                selected_models = [name for name, _ in model_performances[:top_n_models]]
                main_logger.info(
                    f"Using top {len(selected_models)} models by test performance for ensemble: {', '.join(selected_models)}"
                )

            if use_same_test_set:
                _, test_indices = train_test_split(
                    range(len(X)), test_size=self.test_size, random_state=self.random_seed
                )
                train_indices = np.array([i for i in range(len(X)) if i not in test_indices])
                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_seed
                )

            stacking_model = StackingEnsemble(
                base_models_dir=str(self.results_dir),
                top_n_models=top_n_models,
                cv_folds=cv_folds,
                random_seed=self.random_seed,
                n_threads=self.n_threads,
                use_default_params=self.use_default_params,
                task_type=self.task_type,
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

    return all_model_results
