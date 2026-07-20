"""Tests for the Azodi et al. 2019 conversion mode (-101 encoding, min-max scaling)."""

import json

import pandas as pd
import pytest

from gpse.convert.genotype_matrix import convert_to_matrix
from gpse.convert.phenotype import (
    minmax_normalize_phenotype,
    standardize_phenotype,
)
from gpse.convert.processor import GenomicDataProcessor
from gpse.utils.cli_display import _build_convert_parser


def _write_ped_map(prefix):
    prefix.with_suffix(".map").write_text(
        "1 old_a 0 100\n2 old_b 0 200\n", encoding="utf-8"
    )
    prefix.with_suffix(".ped").write_text(
        "f1 s1 0 0 1 1 00 11\n"
        "f2 s2 0 0 1 1 01 00\n"
        "f3 s3 0 0 1 1 11 10\n",
        encoding="utf-8",
    )


def test_convert_to_matrix_centered_encoding(tmp_path):
    prefix = tmp_path / "demo"
    _write_ped_map(prefix)

    output = tmp_path / "demo.csv"
    convert_to_matrix(str(prefix), str(output), out_format="csv", geno_encoding="-101")

    matrix = pd.read_csv(output, index_col=0)
    assert matrix.loc["s1"].tolist() == [-1, 1]
    assert matrix.loc["s2"].tolist() == [0, -1]
    assert matrix.loc["s3"].tolist() == [1, 0]


def test_convert_to_matrix_default_encoding_unchanged(tmp_path):
    prefix = tmp_path / "demo"
    _write_ped_map(prefix)

    output = tmp_path / "demo.csv"
    convert_to_matrix(str(prefix), str(output), out_format="csv")

    matrix = pd.read_csv(output, index_col=0)
    assert matrix.loc["s1"].tolist() == [0, 2]
    assert matrix.loc["s2"].tolist() == [1, 0]
    assert matrix.loc["s3"].tolist() == [2, 1]


def test_convert_to_matrix_rejects_unknown_encoding(tmp_path):
    prefix = tmp_path / "demo"
    _write_ped_map(prefix)

    with pytest.raises(ValueError, match="geno_encoding"):
        convert_to_matrix(str(prefix), str(tmp_path / "demo.csv"),
                          out_format="csv", geno_encoding="abc")


def test_minmax_normalize_phenotype_scales_to_zero_one():
    df = pd.DataFrame({"ID": ["a", "b", "c"], "trait": [10.0, 20.0, 40.0]})

    out, scaler = minmax_normalize_phenotype(df, "trait")

    assert out["trait"].tolist() == [0.0, 1.0 / 3.0, 1.0]
    assert scaler["applied"] is True
    assert scaler["method"] == "minmax"
    assert scaler["min"] == 10.0
    assert scaler["max"] == 40.0
    # Original DataFrame must not be modified.
    assert df["trait"].tolist() == [10.0, 20.0, 40.0]


def test_minmax_normalize_phenotype_constant_column_skipped():
    df = pd.DataFrame({"ID": ["a", "b"], "trait": [5.0, 5.0]})

    out, scaler = minmax_normalize_phenotype(df, "trait")

    assert out["trait"].tolist() == [5.0, 5.0]
    assert scaler["applied"] is False


def test_standardize_phenotype_reports_zscore_method():
    df = pd.DataFrame({"ID": ["a", "b", "c"], "trait": [1.0, 2.0, 3.0]})

    _, scaler = standardize_phenotype(df, "trait")

    assert scaler["method"] == "zscore"
    assert scaler["applied"] is True


def test_process_single_trait_minmax_writes_scaler(tmp_path):
    pheno_file = tmp_path / "pheno.tsv"
    pheno_file.write_text("ID\ttrait\ns1\t10\ns2\t20\ns3\t40\n", encoding="utf-8")
    geno_df = pd.DataFrame(
        {"chr1_99_100": [0, 1, 2]},
        index=["s1", "s2", "s3"],
    )

    processor = GenomicDataProcessor()
    out_prefix = str(tmp_path / "out")
    result = processor._process_single_trait(
        "trait", str(pheno_file), geno_df, set(geno_df.index),
        out_prefix, None, "minmax", "csv",
    )

    assert result is not None
    pheno_out = pd.read_csv(tmp_path / "out_trait_phenotype.csv")
    assert pheno_out["trait"].min() == 0.0
    assert pheno_out["trait"].max() == 1.0

    scaler = json.loads((tmp_path / "out_trait_scaler.json").read_text(encoding="utf-8"))
    assert scaler["method"] == "minmax"
    assert scaler["applied"] is True


def test_process_single_trait_default_no_scaling(tmp_path):
    pheno_file = tmp_path / "pheno.tsv"
    pheno_file.write_text("ID\ttrait\ns1\t10\ns2\t20\n", encoding="utf-8")
    geno_df = pd.DataFrame({"chr1_99_100": [0, 1]}, index=["s1", "s2"])

    processor = GenomicDataProcessor()
    out_prefix = str(tmp_path / "out")
    processor._process_single_trait(
        "trait", str(pheno_file), geno_df, set(geno_df.index),
        out_prefix, None, "none", "csv",
    )

    pheno_out = pd.read_csv(tmp_path / "out_trait_phenotype.csv")
    assert pheno_out["trait"].tolist() == [10, 20]
    assert not (tmp_path / "out_trait_scaler.json").exists()


def test_cli_parser_accepts_readme_mode_arguments():
    parser = _build_convert_parser()
    args = parser.parse_args([
        "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out",
        "--geno-encoding", "-101", "--pheno-scale", "minmax",
    ])
    assert args.geno_encoding == "-101"
    assert args.pheno_scale == "minmax"


def test_cli_parser_defaults_keep_legacy_behavior():
    parser = _build_convert_parser()
    args = parser.parse_args([
        "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out",
    ])
    assert args.geno_encoding == "012"
    assert args.pheno_scale is None
    assert args.standardize_phenotype is False
