"""Tests for gpse convert with PLINK --bfile input (and other genotype-source alternatives)."""

import argparse

import pytest

from gpse.convert.workflow import validate_convert_mode
from gpse.utils.cli_display import _build_convert_parser


def _make_args(**kwargs):
    """Build an argparse.Namespace with sensible defaults."""
    defaults = {
        "check_deps": False,
        "run_qc": False,
        "recode_prefix": None,
        "vcf": None,
        "bfile": None,
        "ped_file": None,
        "map_file": None,
        "matrix_file": None,
        "pheno": "pheno.tsv",
        "out_prefix": "out",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_validate_convert_mode_accepts_bfile_without_vcf():
    parser = argparse.ArgumentParser()
    args = _make_args(bfile="qc/gwas")
    mode = validate_convert_mode(parser, args)
    assert mode == "pipeline"


def test_validate_convert_mode_accepts_ped_file_without_vcf():
    parser = argparse.ArgumentParser()
    args = _make_args(ped_file="data.gped", map_file="data.map")
    mode = validate_convert_mode(parser, args)
    assert mode == "pipeline"


def test_validate_convert_mode_accepts_matrix_file_without_vcf():
    parser = argparse.ArgumentParser()
    args = _make_args(matrix_file="geno.csv")
    mode = validate_convert_mode(parser, args)
    assert mode == "pipeline"


def test_validate_convert_mode_still_accepts_vcf():
    parser = argparse.ArgumentParser()
    args = _make_args(vcf="data.vcf")
    mode = validate_convert_mode(parser, args)
    assert mode == "pipeline"


def test_validate_convert_mode_rejects_missing_genotype_source():
    parser = argparse.ArgumentParser()
    args = _make_args()
    with pytest.raises(SystemExit):
        validate_convert_mode(parser, args)


def test_validate_convert_mode_rejects_multiple_genotype_sources():
    parser = argparse.ArgumentParser()
    args = _make_args(vcf="data.vcf", bfile="qc/gwas")
    with pytest.raises(SystemExit):
        validate_convert_mode(parser, args)


def test_cli_parser_accepts_bfile_alternative():
    parser = _build_convert_parser()
    args = parser.parse_args([
        "--bfile", "qc/gwas",
        "--pheno", "pheno.tsv",
        "--out-prefix", "out",
        "--trait-name", "trait1",
    ])
    assert args.bfile == "qc/gwas"
    assert args.vcf is None
    assert args.pheno == "pheno.tsv"


def test_cli_parser_accepts_matrix_file_alternative():
    parser = _build_convert_parser()
    args = parser.parse_args([
        "--matrix-file", "geno.csv",
        "--pheno", "pheno.tsv",
        "--out-prefix", "out",
    ])
    assert args.matrix_file == "geno.csv"
    assert args.vcf is None
