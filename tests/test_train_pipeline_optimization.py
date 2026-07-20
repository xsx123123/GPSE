import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from gpse.train._cv_manager import _build_structure_groups, generate_cv_folds_from_file
from gpse.train._repeat_training import create_holdout_indices, select_repeat_by_train_cv
from gpse.train._pipeline import _write_holdout_reports
from gpse.train._ensemble import _compute_ensemble_predictions
from gpse.train._results import write_result_bundle
from gpse.train.stacking import StackingEnsemble
from gpse.models.regression_model_optimizer import RegressionModelOptimizer
from gpse.utils.genomic_utils import calculate_repeat_statistics
from gpse.tasks.classification import GenomicClassifier
from gpse.train._feature_selection import (
    fit_genotype_imputer,
    make_model_artifact,
    fit_feature_selector,
    transform_genotypes,
    transform_features,
    unpack_model_artifact,
    validate_genotype_imputation_config,
    validate_feature_selection_config,
)
from gpse.train._model_pipeline import build_training_pipeline
from gpse.train import cli as train_cli
from gpse.train.predictor import GenomicPredictorV2


class _ClassifierPredictor:
    task_type = "classification"
    n_splits = 4
    n_repeats = 1
    random_seed = 17
    split_strategy = "random"
    structure_clusters = None


class _MemorizingRegressor(RegressorMixin, BaseEstimator):
    """Returns a target only for feature IDs observed during fit."""

    def fit(self, X, y):
        self.targets_ = {float(row[0]): float(target) for row, target in zip(X, y)}
        return self

    def predict(self, X):
        return np.asarray([self.targets_.get(float(row[0]), 0.0) for row in X])


class _TreeEnsemblePredictor:
    task_type = "regression"
    n_splits = 2
    n_threads = 1
    save_models = False
    feature_selection_config = validate_feature_selection_config("none", 10, 0.0)
    genotype_imputation_config = validate_genotype_imputation_config("mean", 3.0)

    @staticmethod
    def create_model(_model_name, _params):
        return RandomForestRegressor(n_estimators=10, random_state=17)


def test_classification_holdout_is_stratified():
    y = pd.Series([0] * 30 + [1] * 10)
    train_indices, test_indices = create_holdout_indices(
        y, test_size=0.25, random_seed=17, task_type="classification"
    )

    train_counts = y.iloc[train_indices].value_counts().to_dict()
    test_counts = y.iloc[test_indices].value_counts().to_dict()
    assert set(train_counts) == set(test_counts) == {0, 1}
    assert abs(train_counts[1] / len(train_indices) - y.mean()) < 0.05
    assert abs(test_counts[1] / len(test_indices) - y.mean()) < 0.05


def test_structure_aware_holdout_is_reproducible_and_keeps_clusters_intact():
    rng = np.random.RandomState(17)
    X = pd.DataFrame(
        np.vstack([
            rng.normal(loc=-5, scale=0.1, size=(12, 4)),
            rng.normal(loc=5, scale=0.1, size=(12, 4)),
            rng.normal(loc=15, scale=0.1, size=(12, 4)),
        ])
    )
    y = pd.Series(np.arange(len(X), dtype=float))
    first = create_holdout_indices(
        y, 0.25, 17, "regression", X=X,
        split_strategy="structure_aware", structure_clusters=3, return_metadata=True,
    )
    second = create_holdout_indices(
        y, 0.25, 17, "regression", X=X,
        split_strategy="structure_aware", structure_clusters=3, return_metadata=True,
    )
    train_indices, test_indices, metadata = first

    assert np.array_equal(train_indices, second[0])
    assert np.array_equal(test_indices, second[1])
    assert metadata["split_strategy"] == "structure_aware"
    assert metadata["n_clusters"] == 3
    assert set(metadata["train_clusters"]).isdisjoint(metadata["test_clusters"])
    assert set(train_indices).isdisjoint(test_indices)
    assert len(train_indices) + len(test_indices) == len(X)


def test_structure_aware_cv_keeps_clusters_out_of_validation_training_overlap():
    rng = np.random.RandomState(17)
    X = pd.DataFrame(
        np.vstack([
            rng.normal(loc=-5, scale=0.1, size=(9, 3)),
            rng.normal(loc=5, scale=0.1, size=(9, 3)),
            rng.normal(loc=15, scale=0.1, size=(9, 3)),
        ])
    )
    y = pd.Series([0, 1, 0] * 9)
    predictor = _ClassifierPredictor()
    predictor.n_splits = 3
    predictor.split_strategy = "structure_aware"
    predictor.structure_clusters = 3
    folds = generate_cv_folds_from_file(predictor, X, y, None, repeat_idx=0)
    groups = _build_structure_groups(X, 3, 17, 3)

    for train_indices, validation_indices in folds:
        assert set(groups[train_indices]).isdisjoint(set(groups[validation_indices]))


def test_classification_cv_is_stratified_even_with_full_sample_cv_file():
    predictor = _ClassifierPredictor()
    X = pd.DataFrame({"marker": range(24)})
    y = pd.Series([0] * 12 + [1] * 12)
    cv_data = pd.DataFrame({"cv0": [0] * len(X)}, index=X.index)

    folds = generate_cv_folds_from_file(predictor, X, y, cv_data, repeat_idx=0)

    assert len(folds) == 4
    for _, validation_indices in folds:
        assert y.iloc[validation_indices].value_counts().to_dict() == {0: 3, 1: 3}


def test_stacking_meta_features_are_out_of_fold(tmp_path):
    X = pd.DataFrame({"sample_id": np.arange(20, dtype=float)})
    y = pd.Series(np.arange(20, dtype=float))
    stacking = StackingEnsemble(
        str(tmp_path), model_factory=lambda _: _MemorizingRegressor(), cv_folds=5
    )
    stacking.selected_models = ["memorizer"]

    meta_train, _ = stacking.create_meta_features(X, y)

    assert np.all(meta_train == 0.0)
    full_pipeline = stacking.base_models["memorizer"]
    assert np.array_equal(full_pipeline.predict(X), y.to_numpy())


def test_gblup_baseline_uses_fixed_linear_kernel_ridge():
    optimizer = RegressionModelOptimizer(random_seed=17, n_threads=1)

    assert "gblup_reg" in optimizer.model_configs
    params = optimizer.get_default_params("gblup_reg")
    model = optimizer.create_model("gblup_reg", params)

    assert params == {"alpha": 1.0, "kernel": "linear"}
    assert model.kernel == "linear"


def test_linear_svr_uses_extended_convergence_budget():
    optimizer = RegressionModelOptimizer(random_seed=17, n_threads=1)

    default_params = optimizer.get_default_params("svr_reg")
    optimized_params = optimizer.get_param_func("svr_reg")(
        optuna.trial.FixedTrial(
            {
                "loss": "epsilon_insensitive",
                "C": 1.0,
                "tol": 1e-4,
            }
        )
    )

    assert default_params["max_iter"] == 10_000
    assert optimized_params["max_iter"] == 10_000
    assert optimizer.create_model("svr_reg", default_params).max_iter == 10_000


def test_holdout_reports_accumulate_split_strategies_and_mark_gblup(tmp_path):
    results = {
        "gblup_reg": {"avg_ensemble_pearson": 0.3, "std_ensemble_pearson": 0.01},
        "rf_reg": {"avg_ensemble_pearson": 0.5, "std_ensemble_pearson": 0.02},
    }
    _write_holdout_reports(tmp_path, results, "regression", "random")
    _write_holdout_reports(tmp_path, results, "regression", "structure_aware")

    report = pd.read_csv(tmp_path / "reports" / "model_comparison_holdout.csv")
    assert set(report["split_strategy"]) == {"random", "structure_aware"}
    assert report.loc[report["model"] == "gblup_reg", "is_gblup_baseline"].all()


def test_repeat_summary_preserves_original_scale_ensemble_errors():
    summary = calculate_repeat_statistics(
        [
            {
                "model_name": "rf_reg",
                "avg_training_time": 1.0,
                "avg_test_pearson": 0.1,
                "avg_val_pearson": 0.2,
                "avg_train_pearson": 0.3,
                "avg_test_spearman": 0.1,
                "avg_test_mse": 99.0,
                "ensemble_metrics": {
                    "pearson": 0.4,
                    "spearman": 0.5,
                    "mse": 4.0,
                    "rmse": 2.0,
                    "mae": 1.5,
                },
            }
        ],
        "regression",
    )

    assert summary["avg_ensemble_mse"] == 4.0
    assert summary["avg_ensemble_rmse"] == 2.0
    assert summary["avg_ensemble_mae"] == 1.5


def test_classification_metrics_include_pr_auc_and_holdout_checks_class_counts(tmp_path):
    classifier = GenomicClassifier(n_classes=2, results_dir=str(tmp_path), random_seed=17)
    metrics = classifier.calculate_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]),
    )
    assert metrics["pr_auc"] == 1.0

    with pytest.raises(ValueError, match="at least 5 samples"):
        create_holdout_indices(
            pd.Series([0, 0, 0, 1]),
            test_size=0.25,
            random_seed=17,
            task_type="classification",
            min_class_count=5,
        )


def test_deployment_repeat_selection_uses_train_cv_not_holdout_metric():
    repeat_results = [
        {
            "repeat_idx": 0,
            "avg_val_pearson": 0.20,
            "std_val_pearson": 0.01,
            "ensemble_metrics": {"pearson": 0.99},
            "params": {"alpha": 1},
        },
        {
            "repeat_idx": 1,
            "avg_val_pearson": 0.70,
            "std_val_pearson": 0.02,
            "ensemble_metrics": {"pearson": -0.99},
            "params": {"alpha": 2},
        },
    ]

    selected_idx, selected_repeat, selection = select_repeat_by_train_cv(
        repeat_results, "regression"
    )

    assert selected_idx == 1
    assert selected_repeat["params"] == {"alpha": 2}
    assert selection == {
        "metric": "avg_val_pearson",
        "cv_mean": 0.70,
        "cv_std": 0.02,
    }


def test_univariate_feature_selection_fits_only_training_partition():
    X_train = pd.DataFrame(
        {
            "train_signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "validation_only_noise": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    config = validate_feature_selection_config("univariate", select_k=1, variance_threshold=0.0)

    selector, selected_names = fit_feature_selector(X_train, y_train, config, "regression")
    transformed_validation = transform_features(
        selector,
        pd.DataFrame(
            {
                "train_signal": [10.0, 11.0],
                "validation_only_noise": [10_000.0, -10_000.0],
            }
        ),
    )

    assert selected_names == ["train_signal"]
    assert list(transformed_validation.columns) == ["train_signal"]


def test_model_artifact_preserves_selector_and_legacy_unpacking():
    config = validate_feature_selection_config("variance", select_k=10, variance_threshold=0.0)
    X = pd.DataFrame({"variable": [0.0, 1.0, 2.0], "constant": [1.0, 1.0, 1.0]})
    selector, selected_names = fit_feature_selector(X, pd.Series([0.0, 1.0, 2.0]), config, "regression")
    artifact = make_model_artifact("model", "scaler", selector, config, selected_names)

    assert unpack_model_artifact(artifact) == ("model", "scaler", selector)
    assert unpack_model_artifact(("legacy-model", "legacy-scaler")) == (
        "legacy-model",
        "legacy-scaler",
        None,
    )
    assert artifact["feature_scaling"] == "standard"


def test_mean_genotype_imputer_uses_training_partition_statistics_only():
    config = validate_genotype_imputation_config("mean", missing_genotype_code=3.0)
    X_train = pd.DataFrame({"snp_a": [0.0, 2.0, 3.0], "snp_b": [np.nan, 1.0, 1.0]})
    imputer = fit_genotype_imputer(X_train, config)
    X_validation = pd.DataFrame({"snp_a": [3.0], "snp_b": [np.nan]})

    transformed = transform_genotypes(imputer, X_validation, config)

    assert transformed.loc[0, "snp_a"] == 1.0
    assert transformed.loc[0, "snp_b"] == 1.0


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_mutual_information_feature_selection_keeps_informative_snp(task_type):
    rng = np.random.RandomState(17)
    signal = np.tile([0.0, 1.0], 30)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(signal))})
    y = signal + rng.normal(scale=0.01, size=len(signal)) if task_type == "regression" else signal.astype(int)

    selector, selected_features = fit_feature_selector(
        X,
        pd.Series(y),
        validate_feature_selection_config("mutual_info", select_k=1, variance_threshold=0.0),
        task_type,
    )

    assert selector is not None
    assert selected_features == ["signal"]


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_tree_importance_feature_selection_keeps_informative_snp(task_type):
    rng = np.random.RandomState(23)
    signal = np.tile([0.0, 1.0], 30)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(signal))})
    y = signal + rng.normal(scale=0.01, size=len(signal)) if task_type == "regression" else signal.astype(int)

    selector, selected_features = fit_feature_selector(
        X,
        pd.Series(y),
        validate_feature_selection_config("tree_importance", select_k=1, variance_threshold=0.0),
        task_type,
    )

    assert selector is not None
    assert selected_features == ["signal"]


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_lasso_feature_selection_keeps_informative_snp(task_type):
    rng = np.random.RandomState(29)
    signal = np.tile([0.0, 1.0], 30)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(signal))})
    y = signal + rng.normal(scale=0.01, size=len(signal)) if task_type == "regression" else signal.astype(int)

    selector, selected_features = fit_feature_selector(
        X,
        pd.Series(y),
        validate_feature_selection_config("lasso", select_k=1, variance_threshold=0.0),
        task_type,
    )

    assert selector is not None
    assert selected_features == ["signal"]


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_elasticnet_feature_selection_keeps_informative_snp(task_type):
    rng = np.random.RandomState(31)
    signal = np.tile([0.0, 1.0], 30)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(signal))})
    y = signal + rng.normal(scale=0.01, size=len(signal)) if task_type == "regression" else signal.astype(int)

    selector, selected_features = fit_feature_selector(
        X,
        pd.Series(y),
        validate_feature_selection_config("elasticnet", select_k=1, variance_threshold=0.0),
        task_type,
    )

    assert selector is not None
    assert selected_features == ["signal"]


@pytest.mark.parametrize(
    ("method", "expected_prefix"),
    [("pca", "pca"), ("svd", "truncatedsvd")],
)
def test_dimensionality_reducers_create_deterministic_components(method, expected_prefix):
    rng = np.random.RandomState(37)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"snp_{index}" for index in range(5)])
    config = validate_feature_selection_config(method, select_k=3, variance_threshold=0.0)

    selector, component_names = fit_feature_selector(X, pd.Series(rng.normal(size=len(X))), config, "regression")
    transformed = transform_features(selector, X)

    assert component_names == transformed.columns.tolist()
    assert component_names == [f"{expected_prefix}{index}" for index in range(3)]
    assert transformed.shape == (len(X), 3)


def test_select_percentile_overrides_select_k_and_is_persisted_in_config():
    rng = np.random.RandomState(41)
    signal = rng.normal(size=80)
    X = pd.DataFrame(
        {"signal": signal, **{f"noise_{index}": rng.normal(size=len(signal)) for index in range(9)}}
    )
    config = validate_feature_selection_config(
        "univariate",
        select_k=9,
        variance_threshold=0.0,
        select_percentile=20,
    )

    _, selected_features = fit_feature_selector(
        X, pd.Series((signal > 0).astype(int)), config, "classification"
    )

    assert len(selected_features) == 2
    assert "signal" in selected_features
    assert config.as_dict()["select_percentile"] == 20


@pytest.mark.parametrize("percentile", [0, -1, 100.1])
def test_select_percentile_must_be_in_open_closed_percentage_range(percentile):
    with pytest.raises(ValueError, match="select_percentile"):
        validate_feature_selection_config(
            "univariate",
            select_k=5,
            variance_threshold=0.0,
            select_percentile=percentile,
        )


def test_preprocess_training_defers_phenotype_standardization_to_holdout(tmp_path, monkeypatch):
    captured = {}

    class FakeProcessor:
        def __init__(self, **_kwargs):
            pass

        def process_genomic_data(self, **kwargs):
            captured["preprocess"] = kwargs
            prefix = Path(kwargs["out_prefix"])
            prefix.parent.mkdir(parents=True, exist_ok=True)
            (prefix.parent / f"{prefix.name}_genotype.csv").touch()
            (prefix.parent / f"{prefix.name}_phenotype.csv").touch()
            return 0

    class FakePredictor:
        def __init__(self, **kwargs):
            captured["predictor"] = kwargs

        def run_all_models(self, **kwargs):
            captured["run"] = kwargs
            return {}

    monkeypatch.setattr(train_cli, "GenomicDataProcessor", FakeProcessor)
    monkeypatch.setattr(train_cli, "GenomicPredictorV2", FakePredictor)
    monkeypatch.setattr(train_cli, "validate_parallelism", lambda *args, **_kwargs: args[:3])

    prefix = tmp_path / "prepared"
    result = train_cli.main(
        [
            "--enable_preprocess",
            "--preprocess_prefix",
            str(prefix),
            "--raw_pheno_file",
            str(tmp_path / "raw_pheno.tsv"),
            "--target_trait",
            "Trait",
            "--task_type",
            "regression",
            "--standardize_phenotype",
        ]
    )

    assert result == 0
    assert captured["preprocess"]["standardize_phenotype"] is False
    assert captured["predictor"]["standardize_phenotype"] is True


def test_preprocess_only_rejects_full_dataset_phenotype_standardization(tmp_path):
    result = train_cli.main(
        [
            "--preprocess_only",
            "--preprocess_prefix",
            str(tmp_path / "prepared"),
            "--task_type",
            "regression",
            "--standardize_phenotype",
        ]
    )

    assert result == 1


def test_repeat_standardizes_phenotype_from_holdout_training_labels_only(tmp_path):
    predictor = GenomicPredictorV2(
        results_dir=str(tmp_path),
        n_splits=2,
        n_repeats=1,
        n_threads=1,
        max_parallel_jobs=1,
        repeat_workers=1,
        use_default_params=True,
        standardize_phenotype=True,
    )
    X = pd.DataFrame(np.arange(36, dtype=float).reshape(12, 3), columns=["a", "b", "c"])
    y = pd.Series([float(index) for index in range(10)] + [100.0, 200.0])
    test_indices = np.array([10, 11])

    repeat = predictor.train_and_evaluate_model_for_repeat(
        "gblup_reg", X, y, repeat_idx=0, test_indices=test_indices, cv_pheno_data=None
    )

    assert repeat["phenotype_scaler"]["mean"] == pytest.approx(y.iloc[:10].mean())
    assert repeat["phenotype_scaler"]["mean"] != pytest.approx(y.mean())


def test_fold_ensemble_retrains_imputed_tree_models_without_scaler(tmp_path):
    predictor = _TreeEnsemblePredictor()
    X_train = pd.DataFrame(
        {
            "snp_a": [0.0, 1.0, 2.0, 3.0, 0.0, 2.0],
            "snp_b": [0.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        }
    )
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0, 0.5, 2.5])
    X_test = pd.DataFrame({"snp_a": [3.0, 1.0], "snp_b": [2.0, 0.0]})
    y_test = pd.Series([2.5, 0.5])
    folds = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]

    metrics = _compute_ensemble_predictions(
        predictor,
        "rf_reg",
        {},
        folds,
        X_train,
        X_train,
        y_train,
        X_test,
        y_test,
        tmp_path,
        logging.getLogger(__name__),
    )

    assert np.isfinite(metrics["mse"])
    assert metrics["mse"] > 0


def test_training_pipeline_fits_preprocessing_only_on_its_training_data():
    X_train = pd.DataFrame({"signal": [0.0, 1.0, 2.0], "missing": [3.0, 1.0, 1.0]})
    pipeline = build_training_pipeline(
        _MemorizingRegressor(),
        task_type="regression",
        feature_selection_config=validate_feature_selection_config("none", 2, 0.0),
        genotype_imputation_config=validate_genotype_imputation_config("mean", 3.0),
    ).fit(X_train, pd.Series([0.0, 1.0, 2.0]))

    transformed = pipeline.named_steps["preprocess"].transform(
        pd.DataFrame({"signal": [10.0], "missing": [3.0]})
    )

    assert transformed.shape == (1, 2)
    assert pipeline.named_steps["preprocess"].imputer_.statistics_[1] == 1.0


def test_tree_models_use_scaler_passthrough_but_ridge_is_scaled():
    X = pd.DataFrame({"snp_a": [0.0, 1.0, 2.0, 1.0], "snp_b": [2.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0.0, 1.0, 2.0, 1.0])
    tree_pipeline = build_training_pipeline(
        RandomForestRegressor(n_estimators=5, random_state=17),
        task_type="regression",
        model_name="rf_reg",
    ).fit(X, y)
    ridge_pipeline = build_training_pipeline(
        Ridge(), task_type="regression", model_name="kernelridge_reg"
    ).fit(X, y)

    assert tree_pipeline.named_steps["preprocess"].scaler_ is None
    assert isinstance(ridge_pipeline.named_steps["preprocess"].scaler_, StandardScaler)


def test_result_bundle_writes_ranked_machine_and_human_reports(tmp_path):
    (tmp_path / "split_manifest.json").write_text(
        '{"n_train": 8, "n_test": 2}\n', encoding="utf-8"
    )
    paths = write_result_bundle(
        tmp_path,
        {
            "rf_reg": {
                "training_selection": {
                    "best_params": {"n_estimators": 100},
                    "cv_metric": "avg_val_pearson",
                    "cv_mean": 0.45,
                    "cv_std": 0.03,
                    "selected_repeat_idx": 0,
                    "source": "train_only_cv",
                },
                "avg_ensemble_pearson": 0.40,
                "std_ensemble_pearson": 0.02,
                "avg_ensemble_spearman": 0.35,
                "avg_ensemble_mse": 2.0,
                "avg_ensemble_rmse": 1.41,
                "avg_ensemble_mae": 1.2,
                "phenotype_standardized": True,
            },
            "failed_reg": {"error": "simulated failure"},
        },
        task_type="regression",
        target_trait="height",
        split_strategy="random",
        sample_count=10,
        feature_count=100,
        feature_selection={"method": "univariate", "select_k": 20},
        genotype_imputation={"method": "mean", "missing_genotype_code": 3.0},
    )

    leaderboard = pd.read_csv(paths["leaderboard"])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert leaderboard.loc[0, "model"] == "rf_reg"
    assert leaderboard.loc[0, "rank"] == 1
    assert (tmp_path / "reports" / "training_report.md").exists()
    assert "GPSE Training Result Report" in paths["html"].read_text(encoding="utf-8")
    assert summary["run"]["task_type"] == "regression"
