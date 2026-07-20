"""Prediction workflow with canonical SNP-ID alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gpse.utils.feature_manifest import find_feature_manifest, read_feature_manifest
from gpse.train._feature_selection import (
    GenotypeImputationConfig,
    get_artifact_imputer,
    transform_features,
    transform_genotypes,
    unpack_model_artifact,
)
from gpse.utils.snp_ids import canonical_snp_id


def _find_result_file(model_path: Path, filename: str) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a model file, deployment ensemble directory, or results directory."""
    path = Path(model)
    if path.is_file():
        return path
    if path.is_dir():
        if (path / "info.json").exists() and list(path.glob("member_*.pkl")):
            return path
        direct_candidates = [
            path / "deployment_ensemble",
            path / "ensemble_stacking" / "stacking_ensemble_model.pkl",
            path / "representative_model" / "model.pkl",
            path / "model.pkl",
        ]
        for candidate in direct_candidates:
            if candidate.is_dir() and (candidate / "info.json").exists():
                return candidate
            if candidate.exists():
                return candidate
        deployment_ensembles = sorted(path.glob("*/deployment_ensemble"))
        representative_models = sorted(path.glob("*/representative_model/model.pkl"))
        stacking_models = sorted(path.glob("*/ensemble_stacking/stacking_ensemble_model.pkl"))
        candidates = deployment_ensembles + stacking_models + representative_models
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "Model directory contains multiple model artifacts; pass a specific model.pkl path"
            )
    raise FileNotFoundError(f"Could not find a supported model artifact under: {model}")


def load_genotype_matrix(path: str | Path) -> pd.DataFrame:
    """Load a converted genotype matrix with sample IDs as its index."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    elif path.suffix.lower() == ".feather":
        data = pd.read_feather(path)
    else:
        data = pd.read_csv(path)
    id_column = "ID" if "ID" in data.columns else data.columns[0]
    return data.set_index(id_column)


def load_vcf_matrix(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load VCF genotypes using canonical IDs and additive genotype coding."""
    try:
        from cyvcf2 import VCF
    except ImportError as exc:  # pragma: no cover - dependency is project-required
        raise ImportError("VCF prediction requires cyvcf2 to be installed") from exc

    reader = VCF(str(path))
    sample_ids = list(reader.samples)
    columns: list[str] = []
    values: list[np.ndarray] = []
    seen: set[str] = set()
    for variant in reader:
        feature_id = canonical_snp_id(variant.CHROM, variant.POS, variant.REF)
        if feature_id in seen:
            raise ValueError(f"VCF contains duplicate canonical SNP ID: {feature_id}")
        seen.add(feature_id)
        columns.append(feature_id)
        genotype_values = []
        for genotype in variant.genotypes:
            first, second = genotype[:2]
            genotype_values.append(3 if first < 0 or second < 0 else int(first + second))
        values.append(np.asarray(genotype_values, dtype=float))
    reader.close()
    if not columns:
        raise ValueError(f"VCF contains no variants: {path}")
    matrix = pd.DataFrame(np.asarray(values).T, index=sample_ids, columns=columns)
    matrix.index.name = "ID"
    return matrix, columns


def align_features(
    genotype: pd.DataFrame,
    model_features: list[str],
    *,
    missing_value: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align input features to model order and summarize feature differences."""
    input_features = [str(column) for column in genotype.columns]
    model_features = [str(feature) for feature in model_features]
    missing = [feature for feature in model_features if feature not in genotype.columns]
    extra = [feature for feature in input_features if feature not in set(model_features)]
    aligned = genotype.reindex(columns=model_features, fill_value=missing_value)
    report = {
        "model_feature_count": len(model_features),
        "input_feature_count": len(input_features),
        "matched_model_snp_count": len(model_features) - len(missing),
        "feature_coverage": (
            (len(model_features) - len(missing)) / len(model_features)
            if model_features
            else 0.0
        ),
        "missing_model_snps": missing,
        "missing_model_snp_count": len(missing),
        "extra_input_snps": extra,
        "extra_input_snp_count": len(extra),
        "missing_value_used": missing_value,
    }
    return aligned, report


def _load_model_and_features(model: str | Path) -> tuple[Path, Any, list[str], Path]:
    model_path = resolve_model_path(model)
    if model_path.is_dir():
        member_paths = sorted(model_path.glob("member_*.pkl"))
        if not member_paths:
            raise FileNotFoundError(f"No deployment ensemble members found in {model_path}")
        artifact = [joblib.load(member_path) for member_path in member_paths]
    else:
        artifact = joblib.load(model_path)
    manifest_path = find_feature_manifest(model_path)
    if manifest_path is not None:
        features = read_feature_manifest(manifest_path)
    elif isinstance(artifact, tuple) and len(artifact) == 2 and hasattr(artifact[1], "feature_names_in_"):
        features = [str(feature) for feature in artifact[1].feature_names_in_]
    else:
        raise FileNotFoundError(
            f"No feature_manifest.json found for {model_path}; cannot safely align a VCF to this legacy model"
        )
    return model_path, artifact, features, manifest_path or model_path


def predict(
    model: str | Path,
    genotype_file: str | Path,
    output_file: str | Path,
    *,
    missing_value: float = 3.0,
    report_file: str | Path | None = None,
    min_feature_coverage: float = 0.0,
) -> dict[str, Any]:
    """Predict from a matrix or VCF after canonical SNP-ID alignment."""
    if not 0.0 <= min_feature_coverage <= 1.0:
        raise ValueError("min_feature_coverage must be between 0 and 1")
    model_path, artifact, model_features, manifest_path = _load_model_and_features(model)
    genotype_path = Path(genotype_file)
    if genotype_path.suffix.lower() == ".vcf" or genotype_path.name.lower().endswith(".vcf.gz"):
        genotype, input_features = load_vcf_matrix(genotype_path)
    else:
        genotype = load_genotype_matrix(genotype_path)
        input_features = [str(column) for column in genotype.columns]

    artifacts = artifact if isinstance(artifact, list) else [artifact]
    imputation_metadata = [
        member.get("genotype_imputation", {}) if isinstance(member, dict) else {}
        for member in artifacts
    ]
    model_missing_value = next(
        (
            metadata.get("missing_genotype_code", missing_value)
            for metadata in imputation_metadata
            if metadata.get("method") == "mean"
        ),
        missing_value,
    )
    aligned, alignment = align_features(
        genotype, model_features, missing_value=model_missing_value
    )
    if alignment["feature_coverage"] < min_feature_coverage:
        raise ValueError(
            "Input SNP coverage is below the configured minimum: "
            f"{alignment['feature_coverage']:.2%} matched "
            f"({alignment['matched_model_snp_count']}/{alignment['model_feature_count']}), "
            f"minimum={min_feature_coverage:.2%}"
        )

    def transform_artifact_input(member):
        _, scaler, selector = unpack_model_artifact(member)
        imputation = member.get("genotype_imputation", {}) if isinstance(member, dict) else {}
        config = GenotypeImputationConfig(
            imputation.get("method", "none"),
            imputation.get("missing_genotype_code", missing_value),
        )
        imputed = transform_genotypes(get_artifact_imputer(member), aligned, config)
        return scaler.transform(transform_features(selector, imputed))

    label_encoder_path = _find_result_file(model_path, "label_encoder.pkl")
    if isinstance(artifact, list) and artifact and all(
        isinstance(member, dict) and "pipeline" in member for member in artifact
    ):
        pipelines = [member["pipeline"] for member in artifact]
        task_type = artifact[0].get(
            "task_type", "classification" if label_encoder_path is not None else "regression"
        )
        if task_type == "classification":
            probabilities = []
            for pipeline in pipelines:
                if hasattr(pipeline, "predict_proba"):
                    probabilities.append(pipeline.predict_proba(aligned))
                else:
                    labels = pipeline.predict(aligned).astype(int)
                    n_classes = (
                        len(joblib.load(label_encoder_path).classes_)
                        if label_encoder_path is not None
                        else int(np.max(labels)) + 1
                    )
                    one_hot = np.zeros((len(labels), n_classes))
                    one_hot[np.arange(len(labels)), labels] = 1.0
                    probabilities.append(one_hot)
            predictions = np.argmax(np.mean(probabilities, axis=0), axis=1)
        else:
            predictions = np.mean([pipeline.predict(aligned) for pipeline in pipelines], axis=0)
    elif isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        predictions = pipeline.predict(aligned)
        task_type = artifact.get(
            "task_type", "classification" if label_encoder_path is not None else "regression"
        )
    elif isinstance(artifact, list) and artifact:
        unpacked = [unpack_model_artifact(member) for member in artifact]
        estimators = [member[0] for member in unpacked]
        scalers = [member[1] for member in unpacked]
        selectors = [member[2] for member in unpacked]
        transformed = [transform_artifact_input(member) for member in artifact]
        task_type = ("classification" if label_encoder_path is not None else "regression")
        if task_type == "classification":
            probabilities = [
                estimator.predict_proba(matrix)
                for estimator, matrix in zip(estimators, transformed)
            ]
            predictions = np.argmax(np.mean(probabilities, axis=0), axis=1)
        else:
            predictions = np.mean(
                [estimator.predict(matrix) for estimator, matrix in zip(estimators, transformed)],
                axis=0,
            )
    elif isinstance(artifact, (tuple, dict)):
        estimator, scaler, selector = unpack_model_artifact(artifact)
        predictions = estimator.predict(transform_artifact_input(artifact))
        task_type = ("classification" if label_encoder_path is not None else "regression")
    elif hasattr(artifact, "predict"):
        result = artifact.predict(aligned)
        predictions = result[0] if isinstance(result, tuple) else result
        task_type = getattr(artifact, "task_type", "regression")
    else:
        raise TypeError(f"Unsupported model artifact: {model_path}")

    predictions = np.asarray(predictions)
    if task_type == "classification" and label_encoder_path is not None:
        label_encoder = joblib.load(label_encoder_path)
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    phenotype_scaler_path = _find_result_file(model_path, "phenotype_scaler.json")
    if task_type != "classification" and phenotype_scaler_path is not None:
        scaler_info = json.loads(phenotype_scaler_path.read_text(encoding="utf-8"))
        if scaler_info.get("applied"):
            predictions = predictions * scaler_info["std"] + scaler_info["mean"]

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": genotype.index.astype(str), "prediction": predictions}).to_csv(output, index=False)
    report = {
        "model": str(model_path),
        "feature_manifest": str(manifest_path),
        "genotype_file": str(genotype_path),
        "output_file": str(output),
        "input_variant_count": len(input_features),
        "sample_count": len(genotype),
        "task_type": task_type,
        "minimum_feature_coverage": min_feature_coverage,
        "coverage_warning": (
            "Input covers fewer than 80% of model SNPs; predictions may be unreliable. "
            "Use --min-feature-coverage to enforce a stricter threshold."
            if alignment["feature_coverage"] < 0.8
            else None
        ),
        "selected_feature_counts": (
            [len(member.get("selected_features", [])) for member in artifact]
            if isinstance(artifact, list) and all(isinstance(member, dict) for member in artifact)
            else [len(artifact.get("selected_features", []))]
            if isinstance(artifact, dict)
            else []
        ),
        "genotype_imputation": imputation_metadata,
        **alignment,
    }
    report_path = Path(report_file) if report_file else output.with_suffix(".alignment.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
