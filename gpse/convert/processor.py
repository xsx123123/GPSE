#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated genomic data conversion processor for GPSE.

This module provides :class:`GenomicDataProcessor`, a thin orchestrator
that coordinates genotype conversion, phenotype matching, and data
validation by delegating to specialised sub-modules:

    - :mod:`gpse.convert.genotype_matrix` — VCF/PLINK/PED → numeric CSV
    - :mod:`gpse.convert.phenotype`      — phenotype conversion, matching, standardization
    - :mod:`gpse.convert.validators`     — column-name sanitization, matrix loading
"""

import os
import re
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import cyvcf2
except ImportError:  # pragma: no cover
    cyvcf2 = None

from gpse.convert.genotype_matrix import (
    vcf_to_plink as _vcf_to_plink,
    extract_snps as _extract_snps,
    convert_bfile_to_ped as _convert_bfile_to_ped,
    convert_to_matrix as _convert_to_matrix,
    process_snp_dir as _process_snp_dir,
    GENO_DICT,
)
from gpse.convert.phenotype import (
    convert_phenotype as _convert_phenotype,
    match_genotype_phenotype as _match_genotype_phenotype,
    standardize_phenotype as _standardize_phenotype,
    save_scaler_params,
)
from gpse.convert.validators import (
    check_special_chars as _check_special_chars,
    clean_column_names as _clean_column_names,
    process_file as _process_file,
    load_matrix as _load_matrix,
    validate_trait_names as _validate_trait_names,
    SPECIAL_CHARS,
    TRAIT_INVALID_CHARS,
)

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging
    gpse_logger = logging.getLogger(__name__)

def _write_large_csv(df, output_path, *, chunk_rows=100, logger=None):
    """Write a large DataFrame to CSV in chunks with progress logging.

    pandas ``to_csv`` is single-threaded and can be extremely slow for very
    wide DataFrames (e.g. 800 samples × 113 000 SNPs).  Writing in chunks
    keeps memory bounded and lets us emit progress messages so the user
    doesn't think the process has hung.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write (index is included).
    output_path : str
        Destination CSV path.
    chunk_rows : int
        Number of rows per chunk.
    logger
        Logger for progress messages.
    """
    total_rows = len(df)
    if total_rows <= chunk_rows:
        df.to_csv(output_path)
        return

    # Write header + first chunk
    first_chunk = chunk_rows
    df.iloc[:first_chunk].to_csv(output_path)
    if logger:
        pct = min(100, int(first_chunk / total_rows * 100))
        logger.info(f"  CSV write progress: {first_chunk}/{total_rows} rows ({pct}%)")

    # Append remaining chunks
    for start in range(first_chunk, total_rows, chunk_rows):
        end = min(start + chunk_rows, total_rows)
        df.iloc[start:end].to_csv(output_path, mode='a', header=False)
        if logger and (end - start == chunk_rows or end == total_rows):
            pct = min(100, int(end / total_rows * 100))
            logger.info(f"  CSV write progress: {end}/{total_rows} rows ({pct}%)")


class GenomicDataProcessor:
    """
    Thin orchestrator for genomic data conversion.

    Delegates all heavy lifting to :mod:`gpse.convert.genotype_matrix`,
    :mod:`gpse.convert.phenotype`, and :mod:`gpse.convert.validators`.
    """

    def __init__(
        self,
        logger=None,
        plink_path="plink",
        config_path=None,
        auto_project_config=False,
        allow_extra_chr=False,
    ):
        """
        Initialize a genomic data processor.

        Parameters
        ----------
        logger
            Logger instance supplied by the caller. If omitted, the shared
            GPSE loguru logger from ``gpse.utils.log_utils`` is used.
        plink_path
            PLINK executable path or command name.
        config_path
            Optional user YAML config path.
        auto_project_config
            Whether to auto-load ``gpse.yaml``/``gpse.local.yaml`` from the
            current working directory.
        allow_extra_chr
            Pass ``--allow-extra-chr`` to PLINK so that non-standard
            chromosome names (e.g. scaffold IDs) are accepted.
        """
        self.logger = logger or gpse_logger
        self.plink_path = plink_path
        self.config_path = config_path
        self.auto_project_config = auto_project_config
        self.allow_extra_chr = allow_extra_chr

    # -- Shared tool-path kwargs used when delegating to genotype_matrix --

    def _plink_kwargs(self):
        return {
            "plink_path": self.plink_path,
            "config_path": self.config_path,
            "auto_project_config": self.auto_project_config,
            "allow_extra_chr": self.allow_extra_chr,
            "logger": self.logger,
        }

    # =================== 1. SNP extraction and matrix conversion ===================

    def vcf_to_plink(self, vcf_file, out_prefix):
        """Convert a VCF file to PLINK binary format."""
        return _vcf_to_plink(vcf_file, out_prefix, **self._plink_kwargs())

    def extract_snps(self, bfile, extract_file, out_prefix):
        """Extract selected SNPs from a PLINK binary dataset."""
        return _extract_snps(bfile, extract_file, out_prefix, **self._plink_kwargs())

    def convert_bfile_to_ped(self, bfile, out_prefix):
        """Convert PLINK binary files directly to PED/MAP without SNP filtering."""
        return _convert_bfile_to_ped(bfile, out_prefix, **self._plink_kwargs())

    def convert_to_matrix(self, fileprefix, out_file=None):
        """Convert PLINK PED/MAP genotype data to a numeric CSV matrix."""
        return _convert_to_matrix(fileprefix, out_file, logger=self.logger)

    def process_snp_dir(self, bfile, snp_dir, out_dir):
        """Process all SNP list files in a directory."""
        return _process_snp_dir(bfile, snp_dir, out_dir, **self._plink_kwargs())

    # =================== 2. Phenotype/genotype matching ===================

    def convert_phenotype(self, pheno_file, out_file=None, trait_name=None, trait_col=None):
        """Convert a phenotype file to CSV and remove missing phenotype values."""
        return _convert_phenotype(
            pheno_file, out_file,
            trait_name=trait_name,
            trait_col=trait_col,
            clean_column_names_fn=_clean_column_names,
            logger=self.logger,
        )

    def match_genotype_phenotype(self, pheno_df, geno_file, out_prefix):
        """Match genotype and phenotype samples and preserve a shared order."""
        return _match_genotype_phenotype(pheno_df, geno_file, out_prefix, logger=self.logger)

    # =================== 3. Data validation and cleanup ===================

    def check_special_chars(self, column_names):
        """Check whether column names contain unsupported characters."""
        return _check_special_chars(column_names)

    def clean_column_names(self, column_names):
        """Replace unsupported characters in column names with underscores."""
        return _clean_column_names(column_names)

    def _validate_vcf_phenotype_overlap(self, vcf_file, pheno_file):
        """Defensive check: ensure VCF sample IDs overlap with phenotype sample IDs.

        Uses ``cyvcf2`` to read VCF headers and pandas to read the phenotype
        file.  Logs an error and returns ``False`` when the intersection is
        smaller than the smaller of the two sets, allowing the caller to abort
        gracefully without a traceback.
        """
        if cyvcf2 is None:
            self.logger.warning(
                "cyvcf2 not installed; skipping VCF/phenotype sample overlap check"
            )
            return True

        import os
        import pandas as pd

        # Silence htslib C-level stderr warnings (e.g. header sanity).
        _stderr_fd = os.dup(2)
        with open(os.devnull, "w") as _devnull:
            os.dup2(_devnull.fileno(), 2)
            try:
                vcf = cyvcf2.VCF(vcf_file)
            finally:
                os.dup2(_stderr_fd, 2)
                os.close(_stderr_fd)
        vcf_samples = set(vcf.samples)

        try:
            pheno_df = pd.read_csv(pheno_file, sep='\t')
            if pheno_df.shape[1] < 2:
                pheno_df = pd.read_csv(pheno_file, sep=',')
        except Exception:
            pheno_df = pd.read_csv(pheno_file, sep=None, engine='python')

        pheno_samples = set(pheno_df.iloc[:, 0].astype(str))

        common_samples = vcf_samples & pheno_samples
        min_expected = min(len(vcf_samples), len(pheno_samples))

        self.logger.info(
            f"VCF/phenotype overlap check — VCF: {len(vcf_samples)}, "
            f"Phenotype: {len(pheno_samples)}, Shared: {len(common_samples)}"
        )

        if len(common_samples) < min_expected:
            vcf_examples = sorted(vcf_samples)[:5]
            pheno_examples = sorted(pheno_samples)[:5]
            only_in_vcf = sorted(vcf_samples - pheno_samples)[:5]
            only_in_pheno = sorted(pheno_samples - vcf_samples)[:5]
            _rb = "\033[1;31m"
            _rs = "\033[0m"
            self.logger.error(
                f"Insufficient sample overlap between VCF and phenotype: "
                f"{_rb}only {len(common_samples)} shared samples, but minimum required "
                f"is {min_expected} (the smaller of VCF={len(vcf_samples)} and "
                f"phenotype={len(pheno_samples)}){_rs}. "
                f"This usually means sample IDs do not match between the two files. "
                f"VCF examples: {vcf_examples}; "
                f"Phenotype examples: {pheno_examples}; "
                f"Only in VCF: {only_in_vcf}; "
                f"Only in phenotype: {only_in_pheno}"
            )
            return False

        return True

    def process_file(self, file_path, output_path):
        """Sanitize column names in a CSV file and write the result."""
        return _process_file(file_path, output_path, logger=self.logger)

    def load_matrix(self, matrix_file):
        """Load a matrix file and log basic summary information."""
        return _load_matrix(matrix_file, logger=self.logger)

    def validate_trait_names(self, trait_names):
        """Validate trait names; raises ``ValueError`` if any are invalid."""
        return _validate_trait_names(trait_names, logger=self.logger)

    def standardize_phenotype(self, pheno_df, trait_col):
        """Apply z-score standardization to a phenotype column."""
        return _standardize_phenotype(pheno_df, trait_col, logger=self.logger)


    def _process_single_trait(self, trait, pheno_file, geno_df, geno_samples,
                               out_prefix, user_trait, standardize_phenotype):
        """Process a single trait: match phenotype with genotype and write CSVs."""
        safe_trait = re.sub(r'[^\w\-]', '_', trait)
        pheno_raw_file = f"{out_prefix}_{safe_trait}_phenotype_raw.csv"

        pheno_df = self.convert_phenotype(
            pheno_file,
            out_file=pheno_raw_file,
            trait_col=trait,
            trait_name=user_trait if user_trait else None,
        )

        pheno_samples = set(pheno_df['ID'])
        common_samples = sorted(list(geno_samples.intersection(pheno_samples)))

        self.logger.info(f"Phenotype samples: {len(pheno_samples)}")
        self.logger.info(f"Shared samples: {len(common_samples)}")

        if len(common_samples) == 0:
            self.logger.warning(f"No shared samples for trait '{trait}', skipping")
            return None

        pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
        pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
        self.logger.info("Filtering genotype matrix to shared samples...")
        geno_filtered = geno_df.loc[common_samples]

        self.logger.info("Cleaning column names...")
        geno_filtered.columns = self.clean_column_names(geno_filtered.columns)
        pheno_filtered.columns = self.clean_column_names(pheno_filtered.columns)

        actual_trait_col = pheno_filtered.columns[1]

        trait_scaler = None
        if standardize_phenotype:
            self.logger.info("Standardizing phenotype data...")
            pheno_filtered, trait_scaler = self.standardize_phenotype(pheno_filtered, actual_trait_col)
            save_scaler_params(
                trait_scaler,
                f"{out_prefix}_{safe_trait}_scaler.json",
                logger=self.logger,
            )

        final_pheno_file = f"{out_prefix}_{safe_trait}_phenotype.csv"
        final_geno_file = f"{out_prefix}_{safe_trait}_genotype.csv"

        pheno_filtered.to_csv(final_pheno_file, index=False)
        self.logger.info(
            f"Writing genotype matrix ({geno_filtered.shape[0]} samples x "
            f"{geno_filtered.shape[1]} SNPs) to {final_geno_file} ..."
        )
        _write_large_csv(geno_filtered, final_geno_file, logger=self.logger)
        self.logger.info("Genotype matrix written successfully.")

        self.logger.info(f"Trait '{trait}' completed:")
        self.logger.info(f"  Phenotype: {final_pheno_file}")
        self.logger.info(f"  Genotype:  {final_geno_file}")
        self.logger.info(f"  Samples:   {len(common_samples)}")
        self.logger.info(f"  SNPs:      {len(geno_filtered.columns)}")
        if trait_scaler and trait_scaler.get('applied'):
            self.logger.info(
                f"  Standardized: mean={trait_scaler['mean']:.4f}, "
                f"std={trait_scaler['std']:.4f}"
            )

        return final_pheno_file, final_geno_file

    # =================== Main workflow ===================

    def process_genomic_data(self, **kwargs):
        """
        Run the complete genomic data processing workflow.

        Parameters
        ----------
        kwargs
            Processing parameters.
        """
        import pandas as pd

        out_prefix = kwargs.get('out_prefix')
        output_dir = os.path.dirname(out_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            # ── Step 0: Early trait name validation ──
            if kwargs.get('pheno') and not kwargs.get('skip_match'):
                pheno_file = kwargs['pheno']
                if not os.path.exists(pheno_file):
                    raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")

                try:
                    _pheno_check = pd.read_csv(pheno_file, sep='\t')
                    if _pheno_check.shape[1] < 2:
                        _pheno_check = pd.read_csv(pheno_file, sep=',')
                except Exception:
                    _pheno_check = pd.read_csv(pheno_file, sep=None, engine='python')

                if _pheno_check.shape[1] < 2:
                    raise ValueError(f"Phenotype file {pheno_file} must contain at least two columns")

                trait_cols = _pheno_check.columns[1:].tolist()

                user_trait = kwargs.get('trait_name')
                if user_trait:
                    if user_trait not in trait_cols:
                        raise ValueError(
                            f"--trait-name '{user_trait}' not found in phenotype file. "
                            f"Available traits: {trait_cols}"
                        )
                    trait_cols = [user_trait]

                self.validate_trait_names(trait_cols)

            # ── Step 0b: Defensive VCF/phenotype sample overlap check ──
            if kwargs.get('vcf') and kwargs.get('pheno') and not kwargs.get('skip_match'):
                if not self._validate_vcf_phenotype_overlap(kwargs['vcf'], kwargs['pheno']):
                    self.logger.error("Aborting due to insufficient sample overlap.")
                    return 1

            # ── Step 1: SNP extraction / matrix conversion ──
            geno_matrix_file = None

            if kwargs.get('matrix_file'):
                matrix_file = kwargs['matrix_file']
                if os.path.exists(matrix_file):
                    self.logger.info(f"Using existing genotype matrix file: {matrix_file}")
                    geno_matrix_file = matrix_file
                    if kwargs.get('load'):
                        self.load_matrix(geno_matrix_file)
                else:
                    raise FileNotFoundError(f"Matrix file not found: {matrix_file}")

            elif not kwargs.get('skip_matrix'):
                # Use existing PED/MAP files when provided.
                if kwargs.get('ped_file') and kwargs.get('map_file'):
                    ped_file = kwargs['ped_file']
                    map_file = kwargs['map_file']
                    self.logger.info(f"Using existing PED/MAP files: {ped_file}, {map_file}")

                    if not os.path.exists(ped_file) or not os.path.exists(map_file):
                        raise FileNotFoundError(f"PED/MAP file not found: {ped_file} or {map_file}")

                    ped_prefix = os.path.splitext(ped_file)[0]
                    temp_prefix = out_prefix + "_temp"

                    if ped_prefix != os.path.splitext(map_file)[0]:
                        self.logger.info("PED and MAP prefixes differ; creating temporary links")
                        if os.path.exists(temp_prefix + ".ped"):
                            os.remove(temp_prefix + ".ped")
                        if os.path.exists(temp_prefix + ".map"):
                            os.remove(temp_prefix + ".map")
                        os.symlink(os.path.abspath(ped_file), temp_prefix + ".ped")
                        os.symlink(os.path.abspath(map_file), temp_prefix + ".map")
                        ped_prefix = temp_prefix

                    geno_matrix_file = self.convert_to_matrix(ped_prefix, out_prefix + ".csv")
                    if kwargs.get('load') and geno_matrix_file:
                        self.load_matrix(geno_matrix_file)

                # Convert VCF input to PLINK format first.
                elif kwargs.get('vcf'):
                    vcf_file = kwargs['vcf']
                    if not kwargs.get('bfile'):
                        if kwargs.get('plink_out'):
                            plink_prefix = kwargs['plink_out']
                        else:
                            plink_prefix = os.path.join(os.path.dirname(out_prefix), "plink_data")

                        kwargs['bfile'] = self.vcf_to_plink(vcf_file, plink_prefix)
                        self.logger.info(f"Using converted PLINK prefix: {kwargs['bfile']}")

                # Continue when a PLINK binary prefix is available.
                if kwargs.get('bfile'):
                    bfile = kwargs['bfile']
                    if kwargs.get('snp_dir'):
                        self.process_snp_dir(bfile, kwargs['snp_dir'], os.path.dirname(out_prefix))
                        snp_files = glob.glob(os.path.join(kwargs['snp_dir'], "*.txt"))
                        if snp_files:
                            first_phenotype = os.path.basename(snp_files[0]).replace('.txt', '')
                            geno_matrix_file = os.path.join(os.path.dirname(out_prefix), first_phenotype + ".csv")
                    elif kwargs.get('extract'):
                        out_prefix_temp = self.extract_snps(bfile, kwargs['extract'], out_prefix)
                        geno_matrix_file = self.convert_to_matrix(out_prefix_temp)
                        if kwargs.get('load') and geno_matrix_file:
                            self.load_matrix(geno_matrix_file)
                    elif kwargs.get('direct') or kwargs.get('pheno'):
                        self.logger.info("Converting the full bfile to a genotype matrix...")
                        out_prefix_temp = self.convert_bfile_to_ped(bfile, out_prefix)
                        geno_matrix_file = self.convert_to_matrix(out_prefix_temp)
                        if kwargs.get('load') and geno_matrix_file:
                            self.load_matrix(geno_matrix_file)
                    else:
                        self.logger.warning("No SNP list, SNP directory, or --direct option provided; only VCF-to-PLINK conversion was completed")
            else:
                self.logger.info("Skipping SNP extraction and matrix conversion")
                if kwargs.get('load'):
                    matrix_file = out_prefix + ".csv"
                    if os.path.exists(matrix_file):
                        self.load_matrix(matrix_file)
                        geno_matrix_file = matrix_file
                    else:
                        self.logger.warning(f"Matrix file not found: {matrix_file}")
                elif kwargs.get('extract') or kwargs.get('direct'):
                    geno_matrix_file = out_prefix + ".csv"
                    if not os.path.exists(geno_matrix_file):
                        raise FileNotFoundError(f"Genotype matrix file not found: {geno_matrix_file}")

            # ── Step 2: phenotype/genotype matching, cleanup, standardization ──
            final_pheno_file = None
            final_geno_file = None
            scaler_params = None

            if not kwargs.get('skip_match') and kwargs.get('pheno') and geno_matrix_file:
                self.logger.info("\nStarting integrated data processing: matching, cleanup, and standardization")

                try:
                    pheno_raw = pd.read_csv(kwargs['pheno'], sep='\t')
                    if pheno_raw.shape[1] < 2:
                        pheno_raw = pd.read_csv(kwargs['pheno'], sep=',')
                except Exception:
                    pheno_raw = pd.read_csv(kwargs['pheno'], sep=None, engine='python')

                if pheno_raw.shape[1] < 2:
                    raise ValueError(f"Phenotype file {kwargs['pheno']} must contain at least two columns")

                id_col = pheno_raw.columns[0]
                trait_cols = pheno_raw.columns[1:].tolist()

                user_trait = kwargs.get('trait_name')
                if user_trait:
                    if user_trait in trait_cols:
                        trait_cols = [user_trait]
                    else:
                        self.logger.warning(
                            f"--trait-name '{user_trait}' not found in phenotype file; "
                            f"available traits: {trait_cols}"
                        )

                self.logger.info(f"Discovered {len(trait_cols)} trait(s): {', '.join(trait_cols)}")

                geno_df = pd.read_csv(geno_matrix_file, index_col=0)
                geno_samples = set(geno_df.index)
                self.logger.info(f"Genotype file contains {len(geno_samples)} samples")

                threads = kwargs.get('threads', 1)
                if threads > 1:
                    self.logger.info(f"Processing {len(trait_cols)} trait(s) with {threads} threads...")
                    with ThreadPoolExecutor(max_workers=threads) as executor:
                        futures = {
                            executor.submit(
                                self._process_single_trait,
                                trait,
                                kwargs['pheno'],
                                geno_df,
                                geno_samples,
                                out_prefix,
                                user_trait,
                                kwargs.get('standardize_phenotype', False),
                            ): trait
                            for trait in trait_cols
                        }
                        for future in futures:
                            trait = futures[future]
                            try:
                                future.result()
                            except Exception as exc:
                                self.logger.error(f"Trait '{trait}' failed: {exc}")
                else:
                    for trait in trait_cols:
                        self.logger.info(f"\n--- Processing trait: {trait} ---")
                        try:
                            self._process_single_trait(
                                trait,
                                kwargs['pheno'],
                                geno_df,
                                geno_samples,
                                out_prefix,
                                user_trait,
                                kwargs.get('standardize_phenotype', False),
                            )
                        except Exception as exc:
                            self.logger.error(f"Trait '{trait}' failed: {exc}")

                self.logger.info("\nAll traits processed")

            else:
                if kwargs.get('skip_match'):
                    self.logger.info("Skipping phenotype/genotype matching")
                elif not kwargs.get('pheno'):
                    self.logger.warning("No phenotype file provided; skipping phenotype/genotype matching")
                elif not geno_matrix_file:
                    self.logger.warning("No genotype matrix available; skipping phenotype/genotype matching")

            self.logger.info("\nProcessing workflow completed")

        except ValueError as e:
            self.logger.error(str(e))
            return 1
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error: {str(e)}")
            return 1
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1

        return 0


# ── Backward-compatible standalone functions ────────────────────────────

def vcf_to_plink(vcf_file, out_prefix, plink_path="plink"):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.vcf_to_plink(vcf_file, out_prefix)


def extract_snps(bfile, extract_file, out_prefix, plink_path="plink"):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.extract_snps(bfile, extract_file, out_prefix)


def convert_bfile_to_ped(bfile, out_prefix, plink_path="plink"):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.convert_bfile_to_ped(bfile, out_prefix)


def convert_to_matrix(fileprefix, out_file=None):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.convert_to_matrix(fileprefix, out_file)


def process_snp_dir(bfile, snp_dir, out_dir, plink_path="plink"):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.process_snp_dir(bfile, snp_dir, out_dir)


def convert_phenotype(pheno_file, out_file=None, trait_name=None, trait_col=None):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.convert_phenotype(pheno_file, out_file, trait_name=trait_name, trait_col=trait_col)


def match_genotype_phenotype(pheno_df, geno_file, out_prefix):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.match_genotype_phenotype(pheno_df, geno_file, out_prefix)


def check_special_chars(column_names):
    """Backward-compatible wrapper."""
    return _check_special_chars(column_names)


def clean_column_names(column_names):
    """Backward-compatible wrapper."""
    return _clean_column_names(column_names)


def process_file(file_path, output_path):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.process_file(file_path, output_path)


def load_matrix(matrix_file):
    """Backward-compatible wrapper."""
    return _load_matrix(matrix_file)


def validate_trait_names(trait_names):
    """Backward-compatible wrapper."""
    return _validate_trait_names(trait_names)
