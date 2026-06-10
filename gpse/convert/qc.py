#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genotype quality control and recoding utilities for GPSE.

This module wraps PLINK and Beagle to perform format conversion, missingness
filtering, optional imputation, LD pruning, and conversion of compound
genotypes to additive numeric coding.

Main entry points:
    format_converter       — Normalize VCF/PED/BED inputs to PLINK BED.
    filter_genotype        — Filter samples and variants with PLINK.
    impute_genotype_beagle — Run Beagle genotype imputation.
    analyze_and_prune      — Run QC, missingness statistics, and LD pruning.
    recode_to_numeric      — Convert PLINK compound genotypes to additive coding.
"""

import os
from typing import Dict, Tuple, Optional, List

try:
    from gpse.convert.external import ensure_existing_file, resolve_configured_tool, run_command
except ImportError:  # pragma: no cover - allows direct script execution
    from external import ensure_existing_file, resolve_configured_tool, run_command

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging
    gpse_logger = logging.getLogger("GenotypeProcessor")

logger = gpse_logger


def _check_tool_path(
    tool_path: str,
    *,
    config_path: Optional[str] = None,
    auto_project_config: bool = False,
) -> str:
    """
    Resolve an external PLINK executable path.
    """
    return resolve_configured_tool(
        'plink',
        command_override=tool_path,
        config_path=config_path,
        auto_project_config=auto_project_config,
    )


def _config_context(user_params: Dict) -> Dict[str, object]:
    """Extract optional user configuration context from user parameters."""
    return {
        'config_path': user_params.get('config_path'),
        'auto_project_config': bool(user_params.get('auto_project_config', False)),
    }


def _run_command(cmd_list: List[str], log_file: Optional[str] = None) -> None:
    """
    Run an external command without invoking a shell.
    """
    try:
        run_command(cmd_list, log_file=log_file, logger=logger)
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise


def recode_to_numeric(fileprefix: str) -> None:
    """
    Convert PLINK compound genotypes to additive numeric coding.
    
    Args:
        fileprefix: Input prefix for .ped and .map files.
    """
    ped_path = fileprefix + '.ped'
    map_path = fileprefix + '.map'
    geno_path = fileprefix + '.geno'
    
    # Genotype encoding map
    # 00 -> 0 (Homozygous Major)
    # 01 -> 1 (Heterozygous)
    # 10 -> 1 (Heterozygous)
    # 11 -> 2 (Homozygous Minor)
    code_map = {'00': '0', '01': '1', '10': '1', '11': '2'}

    logger.info(f"Recoding {ped_path} to numeric format...")

    try:
        # Read SNP IDs for the header.
        with open(map_path, 'r') as map_file:
            snpid_list = [line.strip().split()[1] for line in map_file]
        
        with open(geno_path, 'w+') as geno_file:
            geno_file.write('ID,' + ','.join(snpid_list) + '\n')
            
            with open(ped_path, 'r') as ped_file:
                for line in ped_file:
                    row = line.strip().split()
                    sample_id = row[1]
                    genotypes = row[6:]
                    
                    # Fast mapping. Missing values should have been handled upstream.
                    numeric_genos = [code_map.get(g, 'NaN') for g in genotypes]
                    
                    line_out = sample_id + ',' + ','.join(numeric_genos) + '\n'
                    geno_file.write(line_out)
                    
        logger.info(f"Recoding finished. Saved to {geno_path}")

    except Exception as e:
        logger.error(f"Error in recode_to_numeric: {e}")
        raise


def format_converter(user_params: Dict, input_prefix: str, output_prefix: str) -> str:
    """
    Detect input format and normalize it to PLINK BED format.
    
    Args:
        user_params: User parameter dictionary.
        input_prefix: Input file prefix.
        output_prefix: Output file prefix.

    Returns:
        PLINK input flag after conversion, usually ``--bfile``.
    """
    config_ctx = _config_context(user_params)
    plink_path = _check_tool_path(user_params.get('plink_path', 'plink'), **config_ctx)
    allow_extra_chr = user_params.get('allow_extra_chr', False)
    
    # Infer input format from available files.
    if os.path.exists(input_prefix + '.vcf') or os.path.exists(input_prefix + '.vcf.gz'):
        logger.info("Detected VCF format. Converting to BED...")
        vcf_file = input_prefix + '.vcf' if os.path.exists(input_prefix + '.vcf') else input_prefix + '.vcf.gz'
        cmd = [plink_path, '--vcf', vcf_file, '--make-bed', '--out', output_prefix]
        # Preserve VCF IDs and allow non-standard chromosome names.
        cmd.extend(['--const-fid'])
        if allow_extra_chr:
            cmd.append('--allow-extra-chr')
        _run_command(cmd, output_prefix + '.log')
        return '--bfile'
        
    elif os.path.exists(input_prefix + '.ped') and os.path.exists(input_prefix + '.map'):
        logger.info("Detected PED/MAP format. Converting to BED...")
        cmd = [plink_path, '--file', input_prefix, '--make-bed', '--out', output_prefix]
        if allow_extra_chr:
            cmd.append('--allow-extra-chr')
        _run_command(cmd, output_prefix + '.log')
        return '--bfile'
        
    elif os.path.exists(input_prefix + '.bed') and os.path.exists(input_prefix + '.bim') and os.path.exists(input_prefix + '.fam'):
        logger.info("Detected BED/BIM/FAM format.")
        return '--bfile'
    
    else:
        logger.warning(f"Unknown format for {input_prefix}. Assuming user provided parameters are correct for PLINK.")
        return user_params.get('fileformat', '--bfile')


def filter_genotype(user_params: Dict, input_prefix: str, output_prefix: str, input_flag: str = '--bfile') -> None:
    """
    Filter samples and SNPs according to user-provided ID lists.
    """
    extract_snpid = user_params.get('extract_snpid_path')
    exclude_snpid = user_params.get('exclude_snpid_path')
    keep_sampleid = user_params.get('keep_sampleid_path')
    remove_sampleid = user_params.get('remove_sampleid_path')
    config_ctx = _config_context(user_params)
    plink_path = _check_tool_path(user_params.get('plink_path', 'plink'), **config_ctx)
    allow_extra_chr = user_params.get('allow_extra_chr', False)
    
    log_file = output_prefix + '_preprocessed.log'

    # Base command
    cmd = [plink_path, input_flag, input_prefix, '--out', output_prefix]
    
    # Add optional filtering arguments.
    if extract_snpid:
        cmd.extend(['--extract', extract_snpid])
    if exclude_snpid:
        cmd.extend(['--exclude', exclude_snpid])
    if keep_sampleid:
        cmd.extend(['--keep', keep_sampleid])
    if remove_sampleid:
        cmd.extend(['--remove', remove_sampleid])
    
    # Recode compound genotypes for downstream numeric conversion.
    cmd.extend(['--recode', 'compound-genotypes', '01', '--output-missing-genotype', '3'])
    if allow_extra_chr:
        cmd.append('--allow-extra-chr')
    
    _run_command(cmd, log_file)


def impute_genotype_beagle(user_params: Dict, input_prefix: str, output_prefix: str) -> None:
    """
    Run Beagle genotype imputation.
    
    Args:
        user_params: Must include ``beagle_jar_path``.
        input_prefix: Input PLINK binary prefix.
        output_prefix: Output prefix.
    """
    beagle_jar = ensure_existing_file(user_params.get('beagle_jar_path'), name='Beagle JAR')

    config_ctx = _config_context(user_params)
    plink_path = _check_tool_path(user_params.get('plink_path', 'plink'), **config_ctx)
    java_path = resolve_configured_tool(
        'java',
        command_override=user_params.get('java_path'),
        **config_ctx,
    )
    allow_extra_chr = user_params.get('allow_extra_chr', False)
    
    # 0. Pre-filter: Beagle fails if any variant has a missing allele (REF/ALT is '0' in bim)
    # We must exclude these variants before converting to VCF.
    bim_path = input_prefix + '.bim'
    exclude_list = []
    if os.path.exists(bim_path):
        with open(bim_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    # PLINK uses '0' to represent a missing allele definition.
                    if parts[4] == '0' or parts[5] == '0':
                        exclude_list.append(parts[1])
    
    clean_input_prefix = input_prefix
    if exclude_list:
        exclude_file = output_prefix + '_bad_alleles_for_beagle.txt'
        with open(exclude_file, 'w') as f:
            f.write('\n'.join(exclude_list))
        
        logger.warning(
            f"Found {len(exclude_list)} variants with missing alleles (0 in bim). "
            f"Beagle requires valid REF/ALT; excluding these variants first..."
        )
        clean_input_prefix = output_prefix + '_clean_for_beagle'
        cmd_clean = [
            plink_path, '--bfile', input_prefix, 
            '--exclude', exclude_file, 
            '--make-bed', '--out', clean_input_prefix
        ]
        if allow_extra_chr:
            cmd_clean.append('--allow-extra-chr')
        _run_command(cmd_clean, output_prefix + '.log')

    # 1. PLINK BED -> VCF
    logger.info("Converting BED to VCF for Beagle...")
    vcf_temp = clean_input_prefix + '_temp_for_beagle'
    cmd_to_vcf = [plink_path, '--bfile', clean_input_prefix, '--recode', 'vcf', '--out', vcf_temp]
    if allow_extra_chr:
        cmd_to_vcf.append('--allow-extra-chr')
    _run_command(cmd_to_vcf, output_prefix + '.log')
    
    # 2. Run Beagle
    # Beagle output is usually prefix.vcf.gz
    logger.info("Running Beagle imputation...")
    out_beagle = output_prefix + '_imputed'
    # Beagle 5.x syntax: gt=input.vcf out=output_prefix
    cmd_beagle = [java_path, '-jar', beagle_jar, f'gt={vcf_temp}.vcf', f'out={out_beagle}']
    n_threads = user_params.get('threads')
    if n_threads is not None and n_threads > 0:
        cmd_beagle.append(f'nthreads={n_threads}')
        logger.info(f"Beagle imputation using {n_threads} thread(s)")
    _run_command(cmd_beagle, output_prefix + '.log')

    # 3. VCF (Imputed) -> PLINK BED
    logger.info("Converting Imputed VCF back to BED...")
    imputed_vcf = out_beagle + '.vcf.gz'
    if not os.path.exists(imputed_vcf):
         # Fall back to a plain VCF suffix.
         imputed_vcf = out_beagle + '.vcf'
    if not os.path.exists(imputed_vcf):
         raise FileNotFoundError(f"Beagle output VCF not found: {out_beagle}.vcf.gz or {out_beagle}.vcf")
         
    cmd_to_bed = [plink_path, '--vcf', imputed_vcf, '--make-bed', '--out', output_prefix, '--const-fid']
    if allow_extra_chr:
        cmd_to_bed.append('--allow-extra-chr')
    _run_command(cmd_to_bed, output_prefix + '.log')

    # Clean temporary files.
    try:
        if os.path.exists(f"{vcf_temp}.vcf"): os.remove(f"{vcf_temp}.vcf")
        if os.path.exists(f"{vcf_temp}.log"): os.remove(f"{vcf_temp}.log")
        if os.path.exists(f"{vcf_temp}.nosex"): os.remove(f"{vcf_temp}.nosex")
        if clean_input_prefix != input_prefix:
            for ext in ('.bed', '.bim', '.fam', '.log'):
                p = clean_input_prefix + ext
                if os.path.exists(p): os.remove(p)
    except OSError:
        pass


def analyze_and_prune(user_params: Dict, input_prefix: str, output_prefix: str, 
                     run_imputation: bool = False) -> Tuple[str, str]:
    """
    Run genotype QC filters and LD pruning.
    
    Args:
        user_params: User parameter dictionary.
        input_prefix: Input file prefix.
        output_prefix: Output file prefix.
        run_imputation: Whether to run Beagle imputation.

    Returns:
        Tuple of ``(qc_filled_prefix, pruned_prefix)``.
    """
    # 0. Normalize input format to PLINK BED.
    raw_prefix = output_prefix + "_raw"
    current_flag = format_converter(user_params, input_prefix, raw_prefix)
    if current_flag != '--bfile':
        raise ValueError(f"QC pipeline requires PLINK bfile input after conversion, got: {current_flag}")
    
    if all(os.path.exists(raw_prefix + suffix) for suffix in ('.bed', '.bim', '.fam')):
        bed_prefix = raw_prefix
    else:
        bed_prefix = input_prefix

    snp_miss = user_params['snpmaxmiss']
    sample_miss = user_params['samplemaxmiss']
    maf = user_params['maf_max']
    r2 = user_params['r2_cutoff']
    config_ctx = _config_context(user_params)
    plink_path = _check_tool_path(user_params.get('plink_path', 'plink'), **config_ctx)
    allow_extra_chr = user_params.get('allow_extra_chr', False)

    # 1. Handle Imputation / Missingness Filling FIRST if requested
    # This avoids "All people removed" errors when initial missingness is high.
    qc_filled_prefix = output_prefix + '_qc_filled'
    
    if run_imputation and user_params.get('beagle_jar_path'):
        logger.info("Running Beagle imputation (pre-filtering)...")
        impute_genotype_beagle(user_params, bed_prefix, qc_filled_prefix)
    elif run_imputation:
        logger.warning("Beagle imputation requested but beagle_jar_path is not configured. Falling back to PLINK filling.")
        logger.info("Running simple PLINK filling...")
        cmd_fill = [
            plink_path, 
            '--bfile', bed_prefix, 
            '--out', qc_filled_prefix,
            '--make-bed', 
            '--fill-missing-a2'
        ]
        if allow_extra_chr:
            cmd_fill.append('--allow-extra-chr')
        _run_command(cmd_fill, output_prefix + '_qc.log')
    else:
        # No imputation requested, we'll use bed_prefix for the next step.
        qc_filled_prefix = bed_prefix

    # 2. Basic QC statistics and filtering (Strict).
    # If imputed, this will filter low MAF variants.
    # If not imputed, this will filter based on missingness and MAF.
    qc_prefix = output_prefix + '_qc'
    logger.info("Running QC filtering (missingness, MAF)...")
    cmd_qc = [
        plink_path, 
        '--bfile', qc_filled_prefix,
        '--out', qc_prefix,
        '--make-bed',
        '--geno', str(snp_miss),
        '--mind', str(sample_miss),
        '--maf', str(maf),
        '--freq', 
        '--missing'
    ]
    if allow_extra_chr:
        cmd_qc.append('--allow-extra-chr')
    _run_command(cmd_qc, output_prefix + '_qc.log')

    # Update qc_filled_prefix to point to the filtered result for LD pruning.
    qc_final_prefix = qc_prefix

    # 3. LD pruning.
    # First calculate the list of SNPs to keep (.prune.in).
    logger.info("Calculating LD for pruning...")
    cmd_indep = [
        plink_path,
        '--bfile', qc_final_prefix,
        '--out', qc_final_prefix, # writes .prune.in
        '--indep-pairwise', '50', '10', str(r2)
    ]
    if allow_extra_chr:
        cmd_indep.append('--allow-extra-chr')
    _run_command(cmd_indep, output_prefix + '_qc.log')

    # Then extract pruned SNPs.
    pruned_prefix = output_prefix + '_pruned'
    logger.info("Extracting pruned SNPs...")
    cmd_extract = [
        plink_path,
        '--bfile', qc_final_prefix,
        '--out', pruned_prefix,
        '--extract', qc_final_prefix + '.prune.in',
        '--make-bed'
    ]
    if allow_extra_chr:
        cmd_extract.append('--allow-extra-chr')
    _run_command(cmd_extract, output_prefix + '_qc.log')

    return qc_final_prefix, pruned_prefix
