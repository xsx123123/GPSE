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
import json
from datetime import datetime

from gpse.convert.external import resolve_configured_tool, run_command, ensure_log_dir
from gpse.utils.feature_manifest import write_feature_manifest
from gpse.utils.snp_ids import (
    canonical_ids_from_map_file,
    vcf_ids_from_map_file,
    canonical_snp_id,
    ensure_unique_feature_ids,
)

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

# Centered additive encoding, Azodi et al. 2019 style: [-1, 0, 1] = [aa, Aa, AA]
GENO_DICT_CENTERED = {
    '00': '-1',  # Homozygous reference (aa)
    '01': '0',   # Heterozygous (Aa)
    '10': '0',   # Heterozygous (Aa)
    '11': '1',   # Homozygous alternate (AA)
}

# Supported genotype encodings for convert_to_matrix().
GENO_ENCODINGS = {
    '012': GENO_DICT,
    '-101': GENO_DICT_CENTERED,
}


def _encode_tokens(token_rows, geno_dict):
    """Vectorized re-encoding of a 2-D genotype-token array.

    Maps each distinct token once (via ``np.unique`` inverse indexing) instead
    of running a Python callable per cell, which is several times faster than
    ``np.vectorize`` on large sample x SNP matrices.
    """
    import numpy as np

    token_array = np.asarray(token_rows, dtype='U2')
    if token_array.size == 0:
        return token_array
    uniq, inverse = np.unique(token_array, return_inverse=True)
    mapped = np.array([geno_dict.get(token, '3') for token in uniq], dtype='U2')
    return mapped[inverse].reshape(token_array.shape)


def _build_matrix_frame(sample_ids, encoded_array, snpid_list):
    """Build the samples x SNPs genotype DataFrame as a single block."""
    import pandas as pd

    encoded_array = encoded_array.reshape(len(sample_ids), len(snpid_list))
    df = pd.DataFrame(encoded_array, index=sample_ids, columns=snpid_list)
    df.index.name = 'ID'
    return df


def _write_matrix(df, out_file, out_format):
    """Write the genotype matrix frame in the requested format."""
    if out_format in ('parquet', 'feather'):
        import numpy as np

        # Genotype dosages are small integers (-1/0/1/2/3); store them as a
        # numeric dtype so binary formats do not round-trip as object columns.
        df = df.astype(np.int8)
        df_reset = df.reset_index()
        if out_format == 'parquet':
            df_reset.to_parquet(out_file, index=False)
        else:
            df_reset.to_feather(out_file)
    else:
        df.to_csv(out_file)


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
# Conversion provenance (stale-output protection)
# ---------------------------------------------------------------------------
#
# Every conversion step records a small sidecar manifest
# ``<out_prefix>.prov.json`` fingerprinting its input files (abspath, size,
# mtime).  On reruns, existing outputs are only reused when the recorded
# provenance matches the current inputs; a mismatch means the files are
# stale leftovers (e.g. an earlier run aborted or converted different
# inputs) and they are regenerated.  Outputs without a sidecar manifest
# keep the legacy behavior and are reused as-is.

def _file_fingerprint(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _provenance_path(out_prefix):
    return f"{out_prefix}.prov.json"


def _write_provenance(out_prefix, sources, logger=None):
    manifest = {}
    for key, path in sources.items():
        try:
            manifest[key] = _file_fingerprint(path)
        except OSError:
            manifest[key] = None
    with open(_provenance_path(out_prefix), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _provenance_matches(out_prefix, sources):
    """True = outputs match current inputs, False = stale, None = unknown."""
    manifest_file = _provenance_path(out_prefix)
    if not os.path.exists(manifest_file):
        return None
    try:
        with open(manifest_file, encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    for key, path in sources.items():
        try:
            current = _file_fingerprint(path)
        except OSError:
            return False
        if recorded.get(key) != current:
            return False
    return True


def _remove_outputs(paths, logger=None):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _reuse_or_clean(out_prefix, outputs, sources, describe, logger):
    """Skip when outputs exist and provenance matches; clean stale outputs.

    Returns True when the existing outputs can be reused.
    """
    if not all(os.path.exists(p) for p in outputs):
        return False
    state = _provenance_matches(out_prefix, sources)
    if state is False:
        logger.warning(
            f"Existing {describe} were produced from different inputs; regenerating."
        )
        _remove_outputs(outputs + [_provenance_path(out_prefix)])
        return False
    logger.info(f"{describe} already exist: {out_prefix}")
    logger.info("Skipping conversion step...")
    return True


def _run_command_cleaning_outputs_on_failure(cmd, outputs, logger):
    """Run an external command; remove partial outputs when it fails."""
    try:
        run_command(cmd, logger=logger)
    except Exception:
        _remove_outputs(outputs, logger=logger)
        raise


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

    # Skip when output already exists (only if provenance matches the inputs).
    outputs = [f"{out_prefix}{ext}" for ext in (".bed", ".bim", ".fam")]
    sources = {"vcf": vcf_file}
    if _reuse_or_clean(out_prefix, outputs, sources, "PLINK binary files", log):
        return out_prefix

    plink = _resolve_plink(plink_path, config_path, auto_project_config)
    cmd = [plink, "--vcf", vcf_file, "--make-bed", "--out", out_prefix, "--double-id"]
    if allow_extra_chr:
        cmd.extend(["--allow-extra-chr"])
    _run_command_cleaning_outputs_on_failure(cmd, outputs, log)
    _timestamp_plink_log(out_prefix, logger=log)
    _write_provenance(out_prefix, sources)

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

    outputs = [f"{out_prefix}.ped", f"{out_prefix}.map"]
    sources = {
        "bed": f"{bfile}.bed", "bim": f"{bfile}.bim", "fam": f"{bfile}.fam",
        "extract": extract_file,
    }
    if _reuse_or_clean(out_prefix, outputs, sources, "PLINK PED/MAP files", log):
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
    _run_command_cleaning_outputs_on_failure(cmd, outputs, log)
    _timestamp_plink_log(out_prefix, logger=log)
    _write_provenance(out_prefix, sources)

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

    outputs = [f"{out_prefix}.ped", f"{out_prefix}.map"]
    sources = {"bed": f"{bfile}.bed", "bim": f"{bfile}.bim", "fam": f"{bfile}.fam"}
    if _reuse_or_clean(out_prefix, outputs, sources, "PLINK PED/MAP files", log):
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
    _run_command_cleaning_outputs_on_failure(cmd, outputs, log)
    _timestamp_plink_log(out_prefix, logger=log)
    _write_provenance(out_prefix, sources)

    log.info(f"Conversion completed: {out_prefix}.ped, {out_prefix}.map")
    return out_prefix


# ---------------------------------------------------------------------------
# PED/MAP → numeric CSV matrix
# ---------------------------------------------------------------------------

def convert_to_matrix(
    fileprefix,
    out_file=None,
    *,
    out_format="parquet",
    geno_encoding="012",
    preserve_vcf_snp_ids=False,
    collect_df=False,
    logger=None,
):
    """Convert PLINK PED/MAP genotype data to a numeric CSV or binary matrix.

    Encoding (``geno_encoding="012"``): ``00→0, 01→1, 10→1, 11→2``,
    missing → ``3``.  With ``geno_encoding="-101"`` (Azodi et al. 2019
    style): ``00→-1, 01→0, 10→0, 11→1``, missing → ``3``.

    With ``collect_df=True`` the return value is ``(out_file, df)`` where
    ``df`` is the in-memory matrix (``None`` when conversion was skipped
    because the output already exists), so callers can avoid re-reading
    the file they just wrote.
    """
    log = logger or _default_logger
    if geno_encoding not in GENO_ENCODINGS:
        raise ValueError(
            f"Unknown geno_encoding '{geno_encoding}'. "
            f"Supported: {sorted(GENO_ENCODINGS)}"
        )
    geno_dict = GENO_ENCODINGS[geno_encoding]
    log.info(f"Converting {fileprefix}.ped and {fileprefix}.map to matrix format...")
    log.info(f"Genotype encoding: {geno_encoding}")

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
        try:
            with open(manifest_file, encoding="utf-8") as manifest_handle:
                existing_mode = json.load(manifest_handle).get("feature_id_mode", "canonical")
        except (OSError, json.JSONDecodeError):
            existing_mode = None
        requested_mode = "vcf" if preserve_vcf_snp_ids else "canonical"
        if existing_mode == requested_mode:
            matrix_prefix = os.path.splitext(out_file)[0]
            prov_state = _provenance_matches(
                matrix_prefix, {"ped": ped_path, "map": map_path}
            )
            if prov_state is False:
                log.warning(
                    "Existing matrix was built from different PED/MAP inputs; regenerating it."
                )
                _remove_outputs(
                    [out_file, manifest_file, _provenance_path(matrix_prefix)]
                )
            else:
                log.info(f"Matrix file already exists: {out_file}")
                log.info("Skipping conversion step...")
                return (out_file, None) if collect_df else out_file
        log.warning(
            f"Existing matrix uses SNP ID mode '{existing_mode or 'unknown'}'; "
            f"requested '{requested_mode}'. Regenerating the matrix."
        )
    if os.path.exists(out_file) and not os.path.exists(manifest_file):
        log.warning("Existing matrix has no feature manifest; regenerating it.")

    if not os.path.exists(ped_path) or not os.path.exists(map_path):
        raise FileNotFoundError(f"Input file not found: {ped_path} or {map_path}")

    if preserve_vcf_snp_ids:
        log.warning(
            "Compatibility mode enabled: preserving VCF default SNP IDs from MAP/BIM. "
            "Use --preserve-vcf-snp-ids during prediction too."
        )
        snpid_list = vcf_ids_from_map_file(map_path)
        feature_id_mode = "vcf"
    else:
        snpid_list = canonical_ids_from_map_file(map_path)
        feature_id_mode = "canonical"

    # Read sample IDs and genotypes from .ped using vectorized encoding.
    sample_ids = []
    raw_rows = []
    with open(ped_path) as ped_file:
        for row in ped_file:
            parts = row.split()
            sample_id = parts[1]
            if sample_id.endswith('_'):
                sample_id = sample_id.rstrip('_')
            sample_ids.append(sample_id)
            raw_rows.append(parts[6:])

    encoded_array = _encode_tokens(raw_rows, geno_dict)
    df = _build_matrix_frame(sample_ids, encoded_array, snpid_list)
    _write_matrix(df, out_file, out_format)

    manifest_path = write_feature_manifest(
        os.path.dirname(out_file) or ".",
        snpid_list,
        source_file=out_file,
        filename=os.path.basename(manifest_file),
        feature_id_mode=feature_id_mode,
    )
    _write_provenance(
        os.path.splitext(out_file)[0], {"ped": ped_path, "map": map_path}
    )
    log.info(f"Feature manifest written: {manifest_path}")
    log.info(f"Matrix conversion completed: {out_file}")
    return (out_file, df) if collect_df else out_file



# ---------------------------------------------------------------------------
# VCF with pre-encoded numeric genotypes (0/1/2) → direct matrix extraction
# ---------------------------------------------------------------------------

# Genotype tokens that indicate an already-recoded numeric VCF.
_NUMERIC_GT_VALUES = {"0", "1", "2", "."}

# Missing-genotype token used by the numeric matrices (same as PED conversion).
_NUMERIC_MISSING = "3"

# Re-encoding of already-numeric dosages for the supported encodings.
_NUMERIC_GT_ENCODINGS = {
    "012": {"0": "0", "1": "1", "2": "2"},
    "-101": {"0": "-1", "1": "0", "2": "1"},
}


def _open_vcf_text(vcf_file):
    """Open a plain or bgzip/gzip-compressed VCF as a text stream."""
    if vcf_file.endswith((".gz", ".bgz")):
        import gzip
        return gzip.open(vcf_file, "rt", encoding="utf-8", errors="replace")
    return open(vcf_file, encoding="utf-8", errors="replace")


def vcf_genotypes_are_numeric(vcf_file, max_records=4000, logger=None):
    """Return True when the VCF already stores genotypes as 0/1/2 dosages.

    Scans up to ``max_records`` variant records.  A VCF qualifies when every
    observed GT subfield is one of ``0``, ``1``, ``2`` or ``.`` — i.e. the
    file was already recoded to additive dosages and no allele-style
    conversion (``0/0`` → ``0`` etc.) is needed.  Any genotype containing
    ``/`` or ``|`` allele separators, or a value outside the 0/1/2 range,
    disqualifies the file.
    """
    log = logger or _default_logger
    records_seen = 0
    genotypes_seen = 0
    try:
        with _open_vcf_text(vcf_file) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    return False
                for sample_col in fields[9:]:
                    gt = sample_col.split(":", 1)[0]
                    if "/" in gt or "|" in gt or gt not in _NUMERIC_GT_VALUES:
                        return False
                    genotypes_seen += 1
                records_seen += 1
                if records_seen >= max_records:
                    break
    except OSError as exc:
        log.warning(f"Could not scan VCF for numeric genotypes ({exc}); assuming allele-style genotypes")
        return False
    return records_seen > 0 and genotypes_seen > 0


def vcf_numeric_to_matrix(
    vcf_file,
    out_file=None,
    *,
    out_format="parquet",
    geno_encoding="012",
    preserve_vcf_snp_ids=False,
    extract_file=None,
    collect_df=False,
    logger=None,
):
    """Extract genotypes directly from a numeric (0/1/2) VCF into a matrix.

    Used when :func:`vcf_genotypes_are_numeric` detects that the VCF already
    stores additive dosages, so the PLINK VCF→BED→PED round-trip is skipped
    entirely.  Missing genotypes (``.``) are encoded as ``3``, matching the
    PED-based conversion path.  ``extract_file`` optionally restricts the
    output to the SNP IDs listed in that file (matched against both the VCF
    ID column and canonical ``chr<chrom>_<start>_<end>`` coordinates).
    """
    log = logger or _default_logger
    if geno_encoding not in _NUMERIC_GT_ENCODINGS:
        raise ValueError(
            f"Unknown geno_encoding '{geno_encoding}'. "
            f"Supported: {sorted(_NUMERIC_GT_ENCODINGS)}"
        )
    gt_map = _NUMERIC_GT_ENCODINGS[geno_encoding]

    log.info(
        f"VCF {vcf_file} already contains numeric 0/1/2 genotypes; "
        "skipping PLINK conversion and extracting genotypes directly."
    )

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

    if out_file is None:
        base = vcf_file
        for suffix in ('.vcf.gz', '.vcf.bgz', '.vcf'):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        ext = '.parquet' if out_format == 'parquet' else '.feather' if out_format == 'feather' else '.csv'
        out_file = base + ext

    manifest_file = os.path.splitext(out_file)[0] + ".features.json"
    if os.path.exists(out_file) and os.path.exists(manifest_file):
        try:
            with open(manifest_file, encoding="utf-8") as manifest_handle:
                existing_mode = json.load(manifest_handle).get("feature_id_mode", "canonical")
        except (OSError, json.JSONDecodeError):
            existing_mode = None
        requested_mode = "vcf" if preserve_vcf_snp_ids else "canonical"
        if existing_mode == requested_mode:
            log.info(f"Matrix file already exists: {out_file}")
            log.info("Skipping conversion step...")
            return (out_file, None) if collect_df else out_file
        log.warning(
            f"Existing matrix uses SNP ID mode '{existing_mode or 'unknown'}'; "
            f"requested '{requested_mode}'. Regenerating the matrix."
        )
    if os.path.exists(out_file) and not os.path.exists(manifest_file):
        log.warning("Existing matrix has no feature manifest; regenerating it.")

    # Optional SNP filter list.
    extract_ids = None
    if extract_file:
        with open(extract_file, encoding="utf-8") as extract_handle:
            extract_ids = {line.strip() for line in extract_handle if line.strip()}
        log.info(f"Restricting extraction to {len(extract_ids)} SNP(s) from {extract_file}")

    sample_ids = None
    snpid_list = []
    columns = []  # per-variant list of encoded genotypes in sample order
    with _open_vcf_text(vcf_file) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                sample_ids = [s.rstrip('_') if s.endswith('_') else s for s in fields[9:]]
                continue
            if sample_ids is None:
                raise ValueError(f"VCF header (#CHROM line) not found in {vcf_file}")
            if len(fields) < 10:
                continue
            chrom, pos, variant_id, ref = fields[0], fields[1], fields[2].strip(), fields[3]
            canonical_id = canonical_snp_id(chrom, pos, ref)
            if extract_ids is not None and variant_id not in extract_ids and canonical_id not in extract_ids:
                continue
            if preserve_vcf_snp_ids and variant_id and variant_id != ".":
                snpid_list.append(variant_id)
            else:
                snpid_list.append(canonical_id)
            columns.append([
                gt_map.get(sample_col.split(":", 1)[0], _NUMERIC_MISSING)
                for sample_col in fields[9:]
            ])

    if sample_ids is None:
        raise ValueError(f"VCF header (#CHROM line) not found in {vcf_file}")
    if not columns:
        raise ValueError(
            f"No variants extracted from {vcf_file}"
            + (f" using SNP list {extract_file}" if extract_file else "")
        )
    snpid_list = ensure_unique_feature_ids(snpid_list, source=vcf_file)

    # Transpose per-variant columns to per-sample rows with numpy instead of
    # Python zip(), then build the frame as a single block.
    import numpy as np

    encoded_array = np.asarray(columns, dtype='U2').T
    df = _build_matrix_frame(sample_ids, encoded_array, snpid_list)
    _write_matrix(df, out_file, out_format)

    feature_id_mode = "vcf" if preserve_vcf_snp_ids else "canonical"
    manifest_path = write_feature_manifest(
        os.path.dirname(out_file) or ".",
        snpid_list,
        source_file=out_file,
        filename=os.path.basename(manifest_file),
        feature_id_mode=feature_id_mode,
    )
    log.info(f"Feature manifest written: {manifest_path}")
    log.info(
        f"Direct numeric VCF extraction completed: {out_file} "
        f"({len(sample_ids)} samples x {len(snpid_list)} SNPs)"
    )
    return (out_file, df) if collect_df else out_file


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
    geno_encoding="012",
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
            matrix_file = convert_to_matrix(out_prefix, geno_encoding=geno_encoding, logger=log)
            log.info(f"Phenotype {phenotype} completed. CSV matrix: {matrix_file}")
        except Exception as e:
            log.error(f"Error while processing phenotype {phenotype}: {str(e)}")
