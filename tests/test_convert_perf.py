"""Tests for convert-stage performance work: vectorized encoding, in-memory
matrix handoff, and per-trait genotype deduplication."""

import os

import pandas as pd
import pytest

from gpse.convert import processor as processor_module
from gpse.convert.genotype_matrix import convert_to_matrix, vcf_numeric_to_matrix
from gpse.convert.processor import GenomicDataProcessor
from gpse.utils.snp_ids import canonical_ids_from_map_file, canonical_snp_id


def _write_ped_map(tmp_path, name="data"):
    map_path = tmp_path / f"{name}.map"
    map_path.write_text("".join(f"1\tsnp{i}\t0\t{100 + i}\n" for i in range(4)))
    ped_path = tmp_path / f"{name}.ped"
    ped_path.write_text(
        "F1 s1 0 0 0 0 00 01 11 33\n"
        "F2 s2 0 0 0 0 11 10 00 01\n"
    )
    return str(tmp_path / name)


def _write_pheno(tmp_path, traits=("yield", "height")):
    pheno_path = tmp_path / "pheno.tsv"
    header = "ID\t" + "\t".join(traits)
    rows = "\n".join(
        f"{sample}\t" + "\t".join(str(10 + i + j) for j in range(len(traits)))
        for i, sample in enumerate(["s1", "s2"])
    )
    pheno_path.write_text(header + "\n" + rows + "\n")
    return str(pheno_path)


def test_convert_to_matrix_vectorized_encoding_012(tmp_path):
    prefix = _write_ped_map(tmp_path)
    out_file = str(tmp_path / "m.parquet")

    result, df = convert_to_matrix(prefix, out_file, collect_df=True)

    assert result == out_file
    assert list(df.columns) == canonical_ids_from_map_file(str(tmp_path / "data.map"))
    assert df.loc["s1"].tolist() == ["0", "1", "2", "3"]
    assert df.loc["s2"].tolist() == ["2", "1", "0", "1"]

    on_disk = pd.read_parquet(out_file).set_index("ID")
    # Binary formats store genotype dosages as numeric (int8) columns.
    pd.testing.assert_frame_equal(on_disk, df.astype("int8"))


def test_convert_to_matrix_vectorized_encoding_centered(tmp_path):
    prefix = _write_ped_map(tmp_path)
    out_file = str(tmp_path / "m.csv")

    _, df = convert_to_matrix(prefix, out_file, out_format="csv",
                              geno_encoding="-101", collect_df=True)

    assert df.loc["s1"].tolist() == ["-1", "0", "1", "3"]
    assert df.loc["s2"].tolist() == ["1", "0", "-1", "0"]


def test_convert_to_matrix_skip_returns_none_df(tmp_path):
    prefix = _write_ped_map(tmp_path)
    out_file = str(tmp_path / "m.parquet")

    convert_to_matrix(prefix, out_file)
    result, df = convert_to_matrix(prefix, out_file, collect_df=True)

    assert result == out_file
    assert df is None


def test_vcf_numeric_to_matrix_collects_df(tmp_path):
    vcf_path = tmp_path / "numeric.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\t.\tA\tG\t.\t.\t.\tGT\t0\t1\n"
        "1\t101\trs2\tC\tT\t.\t.\t.\tGT\t2\t.\n"
    )
    out_file = str(tmp_path / "m.parquet")

    _, df = vcf_numeric_to_matrix(str(vcf_path), out_file, collect_df=True)

    assert list(df.columns) == [
        canonical_snp_id("1", "100", "A"),
        canonical_snp_id("1", "101", "C"),
    ]
    assert df.loc["s1"].tolist() == ["0", "2"]
    assert df.loc["s2"].tolist() == ["1", "3"]

    on_disk = pd.read_parquet(out_file).set_index("ID")
    # Binary formats store genotype dosages as numeric (int8) columns.
    pd.testing.assert_frame_equal(on_disk, df.astype("int8"))


def _run_ped_map_workflow(tmp_path, threads=1):
    _write_ped_map(tmp_path)
    pheno = _write_pheno(tmp_path)
    out_prefix = str(tmp_path / "out" / "run")
    processor = GenomicDataProcessor()
    rc = processor.process_genomic_data(
        out_prefix=out_prefix,
        ped_file=str(tmp_path / "data.ped"),
        map_file=str(tmp_path / "data.map"),
        pheno=pheno,
        threads=threads,
        out_format="parquet",
    )
    return rc, out_prefix, processor


def test_matching_stage_uses_in_memory_matrix(tmp_path, monkeypatch):
    def fail_read(file_path):  # pragma: no cover - must not be reached
        raise AssertionError("stage 3 must not re-read the matrix from disk")

    monkeypatch.setattr(processor_module, "_read_geno_matrix", fail_read)

    rc, out_prefix, _ = _run_ped_map_workflow(tmp_path)

    assert rc == 0
    assert os.path.exists(f"{out_prefix}_yield_genotype.parquet")
    assert os.path.exists(f"{out_prefix}_yield_phenotype.parquet")


def test_multi_trait_genotype_files_are_hardlinked(tmp_path):
    rc, out_prefix, _ = _run_ped_map_workflow(tmp_path, threads=2)

    assert rc == 0
    geno_yield = f"{out_prefix}_yield_genotype.parquet"
    geno_height = f"{out_prefix}_height_genotype.parquet"
    assert os.stat(geno_yield).st_ino == os.stat(geno_height).st_ino
    pd.testing.assert_frame_equal(
        pd.read_parquet(geno_yield), pd.read_parquet(geno_height)
    )
