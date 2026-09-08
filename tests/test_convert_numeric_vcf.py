"""Tests for direct extraction of pre-encoded numeric (0/1/2) VCF genotypes."""

import gzip
import json
import os

import pytest

from gpse.convert.genotype_matrix import (
    vcf_genotypes_are_numeric,
    vcf_numeric_to_matrix,
)
from gpse.convert.processor import GenomicDataProcessor


class ListLogger:
    """Minimal logger stub that records messages for assertions."""

    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": []}

    def _log(self, level, message):
        self.messages[level].append(str(message))

    def info(self, message):
        self._log("info", message)

    def warning(self, message):
        self._log("warning", message)

    def error(self, message):
        self._log("error", message)

    def contains(self, level, needle):
        return any(needle in m for m in self.messages[level])


NUMERIC_VCF = """\
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Numeric genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3
1\t100\trs1\tA\tG\t.\t.\t.\tGT\t0\t1\t2
1\t200\trs2\tC\tT\t.\t.\t.\tGT\t1\t.\t0
2\t300\t.\tG\tA\t.\t.\t.\tGT\t2\t2\t1
"""

ALLELE_VCF = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3
1\t100\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1
1\t200\trs2\tC\tT\t.\t.\t.\tGT\t0|1\t./.\t0/0
"""

PHENO_TSV = "ID\ttrait1\ns1\t1.5\ns2\t2.5\ns3\t3.5\n"


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_detects_numeric_genotypes(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    assert vcf_genotypes_are_numeric(vcf) is True


def test_detects_numeric_genotypes_gzipped(tmp_path):
    gz_path = tmp_path / "numeric.vcf.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(NUMERIC_VCF)
    assert vcf_genotypes_are_numeric(str(gz_path)) is True


def test_rejects_allele_style_genotypes(tmp_path):
    vcf = _write(tmp_path / "alleles.vcf", ALLELE_VCF)
    assert vcf_genotypes_are_numeric(vcf) is False


def test_rejects_out_of_range_genotypes(tmp_path):
    vcf = _write(tmp_path / "tetraploid.vcf", NUMERIC_VCF.replace("GT\t0\t1\t2", "GT\t0\t3\t4"))
    assert vcf_genotypes_are_numeric(vcf) is False


def test_numeric_vcf_to_csv_matrix(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    out_file = str(tmp_path / "geno.csv")
    logger = ListLogger()

    result = vcf_numeric_to_matrix(vcf, out_file, out_format="csv", logger=logger)

    assert result == out_file
    with open(out_file, encoding="utf-8") as handle:
        lines = handle.read().strip().split("\n")
    header = lines[0].split(",")
    assert header[0] == "ID"
    assert header[1:] == ["chr1_99_100", "chr1_199_200", "chr2_299_300"]
    rows = {line.split(",")[0]: line.split(",")[1:] for line in lines[1:]}
    assert rows["s1"] == ["0", "1", "2"]
    assert rows["s2"] == ["1", "3", "2"]  # '.' -> '3' (missing)
    assert rows["s3"] == ["2", "0", "1"]

    manifest = json.loads((tmp_path / "geno.features.json").read_text(encoding="utf-8"))
    assert manifest["feature_id_mode"] == "canonical"
    assert logger.contains("info", "skipping PLINK conversion")


def test_numeric_vcf_to_matrix_centered_encoding(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    out_file = str(tmp_path / "geno.csv")

    vcf_numeric_to_matrix(vcf, out_file, out_format="csv", geno_encoding="-101")

    with open(out_file, encoding="utf-8") as handle:
        lines = handle.read().strip().split("\n")
    rows = {line.split(",")[0]: line.split(",")[1:] for line in lines[1:]}
    assert rows["s1"] == ["-1", "0", "1"]
    assert rows["s2"] == ["0", "3", "1"]
    assert rows["s3"] == ["1", "-1", "0"]


def test_numeric_vcf_to_matrix_with_snp_extract(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    extract = _write(tmp_path / "snps.txt", "rs1\nchr2_299_300\n")
    out_file = str(tmp_path / "geno.csv")

    vcf_numeric_to_matrix(vcf, out_file, out_format="csv", extract_file=extract)

    with open(out_file, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header[1:] == ["chr1_99_100", "chr2_299_300"]


def test_numeric_vcf_to_matrix_preserve_vcf_ids(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    out_file = str(tmp_path / "geno.csv")

    vcf_numeric_to_matrix(vcf, out_file, out_format="csv", preserve_vcf_snp_ids=True)

    with open(out_file, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    # '.' IDs fall back to canonical coordinates
    assert header[1:] == ["rs1", "rs2", "chr2_299_300"]


def test_pipeline_skips_plink_for_numeric_vcf(tmp_path):
    """End-to-end: numeric VCF goes straight to the matrix without PLINK."""
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    pheno = _write(tmp_path / "pheno.tsv", PHENO_TSV)
    out_prefix = str(tmp_path / "out" / "run")
    logger = ListLogger()

    processor = GenomicDataProcessor(
        logger=logger,
        plink_path="/nonexistent/plink",  # would fail if PLINK were invoked
        auto_project_config=False,
    )
    rc = processor.process_genomic_data(
        out_prefix=out_prefix,
        vcf=vcf,
        pheno=pheno,
        out_format="csv",
    )

    assert rc == 0
    assert logger.contains("info", "Pre-encoded numeric genotypes (0/1/2) detected")
    assert logger.contains("info", "skipping PLINK conversion")
    assert os.path.exists(out_prefix + ".csv")
    geno_out = out_prefix + "_trait1_genotype.csv"
    pheno_out = out_prefix + "_trait1_phenotype.csv"
    assert os.path.exists(geno_out)
    assert os.path.exists(pheno_out)
    with open(geno_out, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header[1:] == ["chr1_99_100", "chr1_199_200", "chr2_299_300"]


def test_pipeline_allele_vcf_still_uses_plink(tmp_path):
    """Allele-style VCF input keeps the PLINK conversion path (and logs it)."""
    vcf = _write(tmp_path / "alleles.vcf", ALLELE_VCF)
    pheno = _write(tmp_path / "pheno.tsv", PHENO_TSV)
    out_prefix = str(tmp_path / "out" / "run")
    logger = ListLogger()

    processor = GenomicDataProcessor(
        logger=logger,
        plink_path="/nonexistent/plink",
        auto_project_config=False,
    )
    rc = processor.process_genomic_data(
        out_prefix=out_prefix,
        vcf=vcf,
        pheno=pheno,
        out_format="csv",
    )

    # PLINK is not available, so the allele-style path must fail here —
    # proving the numeric fast-path was NOT taken.
    assert rc == 1
    assert not logger.contains("info", "Pre-encoded numeric genotypes")


def test_pipeline_numeric_vcf_with_qc_warns_and_skips_qc(tmp_path):
    vcf = _write(tmp_path / "numeric.vcf", NUMERIC_VCF)
    pheno = _write(tmp_path / "pheno.tsv", PHENO_TSV)
    out_prefix = str(tmp_path / "out" / "run")
    logger = ListLogger()

    processor = GenomicDataProcessor(
        logger=logger,
        plink_path="/nonexistent/plink",
        auto_project_config=False,
    )
    rc = processor.process_genomic_data(
        out_prefix=out_prefix,
        vcf=vcf,
        pheno=pheno,
        out_format="csv",
        run_qc=True,
    )

    assert rc == 0
    assert logger.contains("warning", "QC/LD pruning")
    assert os.path.exists(out_prefix + ".csv")
