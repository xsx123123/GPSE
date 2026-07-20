import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from gpse.convert.genotype_matrix import convert_to_matrix
from gpse.predict.core import predict
from gpse.train._feature_selection import make_model_artifact, validate_feature_selection_config
from gpse.train._model_pipeline import build_training_pipeline
from gpse.utils.snp_ids import canonical_snp_id
from gpse.utils.genomic_utils import prepare_cv_data


def test_convert_to_matrix_uses_canonical_ids_and_manifest(tmp_path):
    prefix = tmp_path / "demo"
    prefix.with_suffix(".map").write_text(
        "1 old_a 0 100\n2 old_b 0 200\n", encoding="utf-8"
    )
    prefix.with_suffix(".ped").write_text(
        "f1 s1 0 0 1 1 00 11\nf2 s2 0 0 1 1 01 00\n", encoding="utf-8"
    )

    output = tmp_path / "demo.csv"
    convert_to_matrix(str(prefix), str(output), out_format="csv")

    matrix = pd.read_csv(output)
    assert list(matrix.columns) == ["ID", "chr1_99_100", "chr2_199_200"]
    manifest = json.loads((tmp_path / "demo.features.json").read_text(encoding="utf-8"))
    assert manifest["feature_names"] == ["chr1_99_100", "chr2_199_200"]


def test_predict_aligns_matrix_and_reports_missing_features(tmp_path):
    feature_names = ["chr1_99_100", "chr2_199_200"]
    train = pd.DataFrame([[0, 1], [1, 2], [2, 0]], columns=feature_names)
    scaler = StandardScaler().fit(train)
    model = LinearRegression().fit(scaler.transform(train), [0.0, 1.0, 2.0])
    model_path = tmp_path / "model.pkl"
    joblib.dump((model, scaler), model_path)
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"feature_names": feature_names}), encoding="utf-8"
    )

    input_path = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["sample"], "chr1_99_100": [1]}).to_csv(input_path, index=False)
    output_path = tmp_path / "predictions.csv"
    report = predict(model_path, input_path, output_path)

    assert output_path.exists()
    assert report["missing_model_snps"] == ["chr2_199_200"]
    assert report["extra_input_snps"] == []
    assert report["matched_model_snp_count"] == 1
    assert report["feature_coverage"] == 0.5
    assert report["coverage_warning"]


def test_predict_rejects_input_below_minimum_feature_coverage(tmp_path):
    feature_names = ["chr1_99_100", "chr2_199_200"]
    training = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=feature_names)
    scaler = StandardScaler().fit(training)
    model = LinearRegression().fit(scaler.transform(training), [0.0, 1.0])
    model_path = tmp_path / "model.pkl"
    joblib.dump((model, scaler), model_path)
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"feature_names": feature_names}), encoding="utf-8"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["sample"], "chr1_99_100": [1.0]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="coverage is below"):
        predict(
            model_path,
            input_path,
            tmp_path / "predictions.csv",
            min_feature_coverage=0.75,
        )


def test_predict_loads_deployment_ensemble(tmp_path):
    feature_names = ["chr1_99_100"]
    training = pd.DataFrame([[0.0], [1.0], [2.0]], columns=feature_names)
    ensemble_dir = tmp_path / "deployment_ensemble"
    ensemble_dir.mkdir()
    for member_idx, offset in enumerate([0.0, 1.0], start=1):
        scaler = StandardScaler().fit(training)
        model = LinearRegression().fit(scaler.transform(training), training.iloc[:, 0] + offset)
        joblib.dump((model, scaler), ensemble_dir / f"member_{member_idx}.pkl")
    (ensemble_dir / "info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"feature_names": feature_names}), encoding="utf-8"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["sample"], "chr1_99_100": [1.0]}).to_csv(input_path, index=False)

    output_path = tmp_path / "predictions.csv"
    predict(ensemble_dir, input_path, output_path)

    assert pd.read_csv(output_path)["prediction"].iloc[0] == 1.5


def test_predict_uses_complete_pipeline_artifact(tmp_path):
    feature_names = ["chr1_99_100", "chr2_199_200"]
    training = pd.DataFrame([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], columns=feature_names)
    target = pd.Series([0.0, 1.0, 2.0])
    pipeline = build_training_pipeline(
        LinearRegression(), task_type="regression", feature_selection_config=validate_feature_selection_config("variance", 2, 0.0)
    ).fit(training, target)
    preprocessor = pipeline.named_steps["preprocess"]
    artifact = make_model_artifact(
        pipeline.named_steps["model"],
        preprocessor.scaler_,
        preprocessor.selector_,
        validate_feature_selection_config("variance", 2, 0.0),
        preprocessor.selected_features_,
        imputer=preprocessor.imputer_,
    )
    model_path = tmp_path / "model.pkl"
    joblib.dump(artifact, model_path)
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"feature_names": feature_names}), encoding="utf-8"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["sample"], "chr1_99_100": [1.0], "chr2_199_200": [1.0]}).to_csv(
        input_path, index=False
    )

    predict(model_path, input_path, tmp_path / "predictions.csv")

    assert artifact["artifact_version"] == 3
    assert pd.read_csv(tmp_path / "predictions.csv")["prediction"].iloc[0] == 1.0


def test_canonical_snp_id_uses_zero_based_half_open_coordinates():
    assert canonical_snp_id("1", 100, "A") == "chr1_99_100"
    assert canonical_snp_id("chrX", 7, "AT") == "chrX_6_8"


def test_cv_assignment_is_reproducible_with_seed_42(tmp_path):
    phenotype = pd.DataFrame({"trait": range(12)}, index=[f"s{i}" for i in range(12)])
    first = prepare_cv_data(phenotype.copy(), str(tmp_path / "first.csv"), 3, 4, seed=42)
    second = prepare_cv_data(phenotype.copy(), str(tmp_path / "second.csv"), 3, 4, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()
