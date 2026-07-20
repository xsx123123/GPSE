#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genotype format conversion and matrix construction utilities.

Pure functions for converting between VCF, PLINK binary, PED/MAP, and
numeric CSV matrix formats.  No class state required — the caller
(typically ``GenomicDataProcessor``) passes in the logger and tool paths.
"""

import os
import glob
from datetime import datetime

from gpse.convert.external import resolve_configured_tool, run_command, ensure_log_dir
from gpse.utils.feature_manifest import write_feature_manifest
from gpse.utils.snp_ids import canonical_ids_from_map_file

try:
    from gpse.utils.log_utils import logger as _default_logger
except Exception:  # pragma: no cover
    import logging
    _default_logger = logging.getLogger(__name__)

# Additive genotype encoding map
GENO_DICT = {
    '00': '0',  # Homozygous reference
    '01': '1',  # Heterozygous
    '10': '1',  # Heterozygous
    '11': '2',  # Homozygous alternate
}


def _resolve_plink(plink_path, config_path=None, auto_project_config=False):
    """Resolve the PLINK executable from config or command override."""
    return resolve_configured_tool(
        "plink",
        command_override=plink_path,
        config_path=config_path,
        auto_project_config=auto_project_config,
    )


def _timestamp_plink_log(out_prefix, logger=None):
    """Move PLINK's auto-generated ``<prefix>.log`` into the ``log/`` sub-directory.

    If a log with the same destination name already exists, a timestamp is
    appended to avoid collisions.
    """
    log_path = f"{out_prefix}.log"
    if not os.path.exists(log_path):
        return
    log_dir = ensure_log_dir(out_prefix)
    base_name = os.path.basename(out_prefix)
    dest = os.path.join(log_dir, f"{base_name}.log")
    if os.path.exists(dest):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(log_dir, f"{base_name}_{timestamp}.log")
    os.rename(log_path, dest)
    if logger is not None:
        logger.info(f"PLINK log moved to: {dest}")


# ---------------------------------------------------------------------------
# VCF → PLINK BED
# ---------------------------------------------------------------------------

def vcf_to_plink(
    vcf_file,
    out_prefix,
    *,
    plink_path="plink",
    config_path=None,
    auto_project_config=False,
    allow_extra_chr=False,
    logger=None,
):
    """Convert a VCF file to PLINK binary format (BED/BIM/FAM)."""
    log = logger or _default_logger
    log.info(f"Converting VCF file {vcf_file} to PLINK binary format...")

    # Skip when output already exists.
    if all(os.path.exists(f"{out_prefix}{ext}") for ext in (".bed", ".bim", ".fam")):
        log.info(f"PLINK binary files already exist: {out_prefix}")
        log.info("Skipping conversion step...")
        return out_prefix

    plink = _resolve_plink(plink_path, config_path, auto_project_config)
    cmd = [plink, "--vcf", vcf_file, "--make-bed", "--out", out_prefix, "--double-id"]
    if allow_extra_chr:
        cmd.extend(["--allow-extra-chr"])
    run_command(cmd, logger=log)
    _timestamp_plink_log(out_prefix, logger=log)

    log.info(f"VCF conversion completed: {out_prefix}.bed, {out_prefix}.bim, {out_prefix}.fam")
    return out_prefix


# ---------------------------------------------------------------------------
# BED → PED/MAP  (with SNP extraction)
# ---------------------------------------------------------------------------

def extract_snps(
    bfile,
    extract_file,
    out_prefix,
    *,
    plink_path="plink",
    config_path=None,
    auto_project_config=False,
    allow_extra_chr=False,
    logger=None,
):
    """Extract selected SNPs from a PLINK binary dataset to PED/MAP."""
    log = logger or _default_logger
    log.info(f"Extracting SNPs from {bfile}...")

    if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
        log.info(f"PLINK PED/MAP files already exist: {out_prefix}")
        log.info("Skipping extraction step...")
        return out_prefix

    plink = _resolve_plink(plink_path, config_path, auto_project_config)
    cmd = [
        plink,
        "--bfile", bfile,
        "--out", out_prefix,
        "--extract", extract_file,
        "--recode", "compound-genotypes", "01",
        "--output-missing-genotype", "3",
    ]
    if allow_extra_chr:
        cmd.extend(["--allow-extra-chr"])
    run_command(cmd, logger=log)
    _timestamp_plink_log(out_prefix, logger=log)

    log.info(f"SNP extraction completed: {out_prefix}.ped, {out_prefix}.map")
    return out_prefix


# ---------------------------------------------------------------------------
# BED → PED/MAP  (full, no filtering)
# ---------------------------------------------------------------------------

def convert_bfile_to_ped(
    bfile,
    out_prefix,
    *,
    plink_path="plink",
    config_path=None,
    auto_project_config=False,
    allow_extra_chr=False,
    logger=None,
):
    """Convert PLINK binary files directly to PED/MAP without SNP filtering."""
    log = logger or _default_logger
    log.info(f"Converting PLINK binary dataset {bfile} to PED/MAP format...")

    if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
        log.info(f"PLINK PED/MAP files already exist: {out_prefix}")
        log.info("Skipping conversion step...")
        return out_prefix

    plink = _resolve_plink(plink_path, config_path, auto_project_config)
    cmd = [
        plink,
        "--bfile", bfile,
        "--out", out_prefix,
        "--recode", "compound-genotypes", "01",
        "--output-missing-genotype", "3",
    ]
    if allow_extra_chr:
        cmd.extend(["--allow-extra-chr"])
    run_command(cmd, logger=log)
    _timestamp_plink_log(out_prefix, logger=log)

    log.info(f"Conversion completed: {out_prefix}.ped, {out_prefix}.map")
    return out_prefix


# ---------------------------------------------------------------------------
# PED/MAP → numeric CSV matrix
# ---------------------------------------------------------------------------

def convert_to_matrix(fileprefix, out_file=None, *, out_format="parquet", logger=None):
    """Convert PLINK PED/MAP genotype data to a numeric CSV or binary matrix.

    Encoding: ``00→0, 01→1, 10→1, 11→2``, missing → ``3``.
    """
    log = logger or _default_logger
    log.info(f"Converting {fileprefix}.ped and {fileprefix}.map to matrix format...")

    # Detect pyarrow
    out_format = out_format.lower()
    if out_format in ('parquet', 'feather'):
        try:
            import pyarrow
        except ImportError:
            log.warning(
                f"Output format '{out_format}' requires the 'pyarrow' package, which is not installed. "
                "Falling back to 'csv'. Please run 'pip install pyarrow' to enable highly optimized binary formats."
            )
            out_format = 'csv'

    ped_path = fileprefix + '.ped'
    map_path = fileprefix + '.map'

    # Try alternate paths when the prefix contains a directory component.
    if not os.path.exists(ped_path) and '/' in fileprefix:
        base_name = os.path.basename(fileprefix)
        dir_name = os.path.dirname(fileprefix)
        alt_ped = os.path.join(dir_name, base_name + '.ped')
        alt_map = os.path.join(dir_name, base_name + '.map')
        if os.path.exists(alt_ped) and os.path.exists(alt_map):
            ped_path = alt_ped
            map_path = alt_map

    if out_file is None:
        ext = '.parquet' if out_format == 'parquet' else '.feather' if out_format == 'feather' else '.csv'
        out_file = fileprefix + ext

    manifest_file = os.path.splitext(out_file)[0] + ".features.json"
    if os.path.exists(out_file) and os.path.exists(manifest_file):
        log.info(f"Matrix file already exists: {out_file}")
        log.info("Skipping conversion step...")
        return out_file
    if os.path.exists(out_file):
        log.warning("Existing matrix has no feature manifest; regenerating it with canonical SNP IDs.")

    if not os.path.exists(ped_path) or not os.path.exists(map_path):
        raise FileNotFoundError(f"Input file not found: {ped_path} or {map_path}")

    # Use stable UCSC-style IDs from chromosome and base-pair coordinates.
    snpid_list = canonical_ids_from_map_file(map_path)

    # Read sample IDs and genotypes from .ped.
    sample_genotypes = []
    with open(ped_path) as ped_file:
        for row in ped_file:
            row = row.strip().split()
            sample_id = row[1]

            # Normalize sample IDs produced from VCF conversion.
            if sample_id.endswith('_'):
                sample_id = sample_id.rstrip('_')

            geno = row[6:]  # Genotypes start at column 7.
            encoded_geno = [GENO_DICT.get(g, '3') for g in geno]
            sample_genotypes.append((sample_id, encoded_geno))

    # Write matrix based on format.
    if out_format in ('parquet', 'feather'):
        import pandas as pd
        data = {sample_id: genotypes for sample_id, genotypes in sample_genotypes}
        df = pd.DataFrame.from_dict(data, orient='index', columns=snpid_list)
        df.index.name = 'ID'
        df_reset = df.reset_index()
        if out_format == 'parquet':
            df_reset.to_parquet(out_file, index=False)
        else:
            df_reset.to_feather(out_file)
    else:
        # Write CSV matrix.
        with open(out_file, 'w') as csv_file:
            csv_file.write("ID," + ",".join(snpid_list) + '\n')
            for sample_id, genotypes in sample_genotypes:
                csv_file.write(sample_id + "," + ",".join(genotypes) + '\n')

    manifest_path = write_feature_manifest(
        os.path.dirname(out_file) or ".",
        snpid_list,
        source_file=out_file,
        filename=os.path.basename(manifest_file),
    )
    log.info(f"Feature manifest written: {manifest_path}")
    log.info(f"Matrix conversion completed: {out_file}")
    return out_file



# ---------------------------------------------------------------------------
# Batch processing: SNP directory
# ---------------------------------------------------------------------------

def process_snp_dir(
    bfile,
    snp_dir,
    out_dir,
    *,
    plink_path="plink",
    config_path=None,
    auto_project_config=False,
    allow_extra_chr=False,
    logger=None,
):
    """Process all SNP list files (*.txt) in a directory."""
    log = logger or _default_logger
    os.makedirs(out_dir, exist_ok=True)

    snp_files = glob.glob(os.path.join(snp_dir, "*.txt"))
    if not snp_files:
        log.warning(f"No .txt files found in {snp_dir}")
        return

    log.info(f"Found {len(snp_files)} SNP list file(s). Starting batch processing...")

    for snp_file in snp_files:
        phenotype = os.path.basename(snp_file).replace('.txt', '')
        log.info("")
        log.info(f"Processing phenotype: {phenotype}")

        out_prefix = os.path.join(out_dir, phenotype)
        try:
            extract_snps(
                bfile, snp_file, out_prefix,
                plink_path=plink_path,
                config_path=config_path,
                auto_project_config=auto_project_config,
                allow_extra_chr=allow_extra_chr,
                logger=log,
            )
            matrix_file = convert_to_matrix(out_prefix, logger=log)
            log.info(f"Phenotype {phenotype} completed. CSV matrix: {matrix_file}")
        except Exception as e:
            log.error(f"Error while processing phenotype {phenotype}: {str(e)}")
