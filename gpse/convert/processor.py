#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated genomic data conversion processor for GPSE.

Handles end-to-end preparation of genotype and phenotype data for downstream
machine learning, including:

    1. SNP extraction and matrix conversion
       - Convert VCF to PLINK binary format
       - Extract selected SNPs and build numeric CSV matrices

    2. Phenotype / genotype matching
       - Normalize phenotype tables
       - Keep only samples shared between genotype and phenotype data
       - Write matched files in consistent sample order

    3. Data validation and cleanup
       - Detect feature names with unsupported characters
       - Sanitize column names for downstream ML tools
"""

import os
import argparse
import pandas as pd
import glob
import re
import sys

try:
    from gpse.convert.external import resolve_configured_tool, run_command
except ImportError:  # pragma: no cover - allows direct script execution
    from external import resolve_configured_tool, run_command

try:
    from gpse.utils.log_utils import logger as gpse_logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging
    gpse_logger = logging.getLogger(__name__)

# Additive genotype encoding map
GENO_DICT = {
    '00': '0',  # Homozygous reference
    '01': '1',  # Heterozygous
    '10': '1',  # Heterozygous
    '11': '2'   # Homozygous alternate
}

# Characters unsupported by LightGBM feature names
SPECIAL_CHARS = [
    ':', '|', '[', ']', '{', '}', '"', '\\', ',', ' '
]


class GenomicDataProcessor:
    """
    Genomic data conversion processor.
    """
    
    def __init__(
        self,
        logger=None,
        plink_path="plink",
        config_path=None,
        auto_project_config=False,
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
        """
        self.logger = logger or gpse_logger
        
        self.plink_path = plink_path
        self.config_path = config_path
        self.auto_project_config = auto_project_config
    
    # =================== 1. SNP extraction and matrix conversion ===================
    
    def vcf_to_plink(self, vcf_file, out_prefix):
        """
        Convert a VCF file to PLINK binary format.
        """
        self.logger.info(f"Converting VCF file {vcf_file} to PLINK binary format...")
        
        # Skip conversion when all PLINK binary files already exist.
        if os.path.exists(f"{out_prefix}.bed") and os.path.exists(f"{out_prefix}.bim") and os.path.exists(f"{out_prefix}.fam"):
            self.logger.info(f"PLINK binary files already exist: {out_prefix}.bed, {out_prefix}.bim, {out_prefix}.fam")
            self.logger.info("Skipping conversion step...")
            return out_prefix
        
        # Resolve PLINK from config or command override.
        plink_path = resolve_configured_tool(
            "plink",
            command_override=self.plink_path,
            config_path=self.config_path,
            auto_project_config=self.auto_project_config,
        )
        
        # Build and run PLINK command.
        cmd = [plink_path, "--vcf", vcf_file, "--make-bed", "--out", out_prefix, "--double-id"]
        
        run_command(cmd, logger=self.logger)
        
        self.logger.info(f"VCF conversion completed: {out_prefix}.bed, {out_prefix}.bim, {out_prefix}.fam")
        return out_prefix
    
    def extract_snps(self, bfile, extract_file, out_prefix):
        """Extract selected SNPs from a PLINK binary dataset."""
        self.logger.info(f"Extracting SNPs from {bfile}...")
        
        # Skip extraction when PED/MAP outputs already exist.
        if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
            self.logger.info(f"PLINK PED/MAP files already exist: {out_prefix}.ped, {out_prefix}.map")
            self.logger.info("Skipping extraction step...")
            return out_prefix
        
        # Resolve PLINK from config or command override.
        plink_path = resolve_configured_tool(
            "plink",
            command_override=self.plink_path,
            config_path=self.config_path,
            auto_project_config=self.auto_project_config,
        )
        
        # Build and run PLINK command.
        cmd = [
            plink_path,
            "--bfile", bfile,
            "--out", out_prefix,
            "--extract", extract_file,
            "--recode", "compound-genotypes", "01",
            "--output-missing-genotype", "3",
        ]
        
        run_command(cmd, logger=self.logger)
        
        self.logger.info(f"SNP extraction completed: {out_prefix}.ped, {out_prefix}.map")
        return out_prefix
    
    def convert_bfile_to_ped(self, bfile, out_prefix):
        """Convert PLINK binary files directly to PED/MAP without SNP filtering."""
        self.logger.info(f"Converting PLINK binary dataset {bfile} to PED/MAP format...")
        
        # Skip conversion when PED/MAP outputs already exist.
        if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
            self.logger.info(f"PLINK PED/MAP files already exist: {out_prefix}.ped, {out_prefix}.map")
            self.logger.info("Skipping conversion step...")
            return out_prefix
        
        # Resolve PLINK from config or command override.
        plink_path = resolve_configured_tool(
            "plink",
            command_override=self.plink_path,
            config_path=self.config_path,
            auto_project_config=self.auto_project_config,
        )
        
        # Build and run PLINK command.
        cmd = [
            plink_path,
            "--bfile", bfile,
            "--out", out_prefix,
            "--recode", "compound-genotypes", "01",
            "--output-missing-genotype", "3",
        ]
        
        run_command(cmd, logger=self.logger)
        
        self.logger.info(f"Conversion completed: {out_prefix}.ped, {out_prefix}.map")
        return out_prefix
    
    def convert_to_matrix(self, fileprefix, out_file=None):
        """Convert PLINK PED/MAP genotype data to a numeric CSV matrix."""
        self.logger.info(f"Converting {fileprefix}.ped and {fileprefix}.map to matrix format...")
        
        ped_path = fileprefix + '.ped'
        map_path = fileprefix + '.map'
        
        # If PED/MAP paths are explicit, try resolving them in the same directory.
        if not os.path.exists(ped_path) and '/' in fileprefix:
            # Try alternate paths using the provided prefix directory.
            base_name = os.path.basename(fileprefix)
            dir_name = os.path.dirname(fileprefix)
            alt_ped_path = os.path.join(dir_name, base_name + '.ped')
            alt_map_path = os.path.join(dir_name, base_name + '.map')
            if os.path.exists(alt_ped_path) and os.path.exists(alt_map_path):
                ped_path = alt_ped_path
                map_path = alt_map_path
        
        if out_file is None:
            out_file = fileprefix + '.csv'
        
        # Skip conversion when output already exists.
        if os.path.exists(out_file):
            self.logger.info(f"Matrix file already exists: {out_file}")
            self.logger.info("Skipping conversion step...")
            return out_file
        
        # Validate input files.
        if not os.path.exists(ped_path) or not os.path.exists(map_path):
            raise FileNotFoundError(f"Input file not found: {ped_path} or {map_path}")
        
        # Read SNP IDs.
        snpid_list = []
        with open(map_path) as map_file:
            for row in map_file:
                row = row.strip().split()
                snpid_list.append(row[1])  # Column 2 is SNP ID.
        
        # Read sample IDs and genotype data.
        sample_genotypes = []
        with open(ped_path) as ped_file:
            for row in ped_file:
                row = row.strip().split()
                sample_id = row[1]  # Column 2 is sample ID.
                
                # Normalize sample IDs produced from VCF conversion.
                if sample_id.endswith('_'):
                    sample_id = sample_id.rstrip('_')
                
                geno = row[6:]  # Genotypes start at column 7.
                
                # Convert compound genotypes to numeric additive coding.
                encoded_geno = []
                for genotype in geno:
                    # 00->0, 01->1, 10->1, 11->2
                    diploid = GENO_DICT.get(genotype, '3')  # Missing value defaults to 3.
                    encoded_geno.append(diploid)
                
                sample_genotypes.append((sample_id, encoded_geno))
        
        # Write CSV matrix.
        with open(out_file, 'w') as csv_file:
            header = "ID," + ",".join(snpid_list)
            csv_file.write(header + '\n')
            
            for sample_id, genotypes in sample_genotypes:
                line = sample_id + "," + ",".join(genotypes)
                csv_file.write(line + '\n')
        
        self.logger.info(f"Matrix conversion completed: {out_file}")
        return out_file
    
    def process_snp_dir(self, bfile, snp_dir, out_dir):
        """Process all SNP list files in a directory."""
        # Ensure output directory exists.
        os.makedirs(out_dir, exist_ok=True)
        
        # Collect all text files in the SNP directory.
        snp_files = glob.glob(os.path.join(snp_dir, "*.txt"))
        
        if not snp_files:
            self.logger.warning(f"No .txt files found in {snp_dir}")
            return
        
        self.logger.info(f"Found {len(snp_files)} SNP list file(s). Starting batch processing...")
        
        for snp_file in snp_files:
            # Use the file stem as the phenotype name.
            phenotype = os.path.basename(snp_file).replace('.txt', '')
            self.logger.info(f"\nProcessing phenotype: {phenotype}")
            
            # Create an output prefix for each phenotype.
            out_prefix = os.path.join(out_dir, phenotype)
            
            try:
                self.extract_snps(bfile, snp_file, out_prefix)
                
                matrix_file = self.convert_to_matrix(out_prefix)
                
                self.logger.info(f"Phenotype {phenotype} completed. CSV matrix: {matrix_file}")
            except Exception as e:
                self.logger.error(f"Error while processing phenotype {phenotype}: {str(e)}")
    
    # =================== 2. Phenotype/genotype matching ===================
    
    def convert_phenotype(self, pheno_file, out_file=None, trait_name=None, trait_col=None):
        """Convert a phenotype file to CSV and remove missing phenotype values.

        Parameters
        ----------
        pheno_file : str
            Path to the input phenotype file.
        out_file : str, optional
            Path for the output CSV. If omitted, defaults to a path derived from
            *pheno_file* (with a ``_converted`` suffix when the input is already a
            ``.csv`` so the original file is never overwritten).
        trait_name : str, optional
            Rename the selected trait column to this value in the output.
        trait_col : str, optional
            Name of the trait column to extract. When omitted the second column
            is used.
        """
        self.logger.info(f"Processing phenotype file: {pheno_file}")

        # Read with pandas, assuming the first row contains headers.
        try:
            # Try tab-separated input first.
            df = pd.read_csv(pheno_file, sep='\t')
            if df.shape[1] < 2:
                # Fall back to comma-separated input if needed.
                df = pd.read_csv(pheno_file, sep=',')
        except Exception as e:
            self.logger.warning(f"pandas failed to read phenotype file; falling back to raw parsing: {str(e)}")
            # Compatibility fallback for simple tab-separated files.
            pheno_data = []
            with open(pheno_file, 'r') as f:
                header = next(f).strip().split('\t')  # Skip header.
                for line in f:
                    line = line.strip().split('\t')
                    if len(line) >= 2:
                        pheno_data.append([line[0], line[1]])
            df = pd.DataFrame(pheno_data, columns=['ID', 'Phenotype'])

        # Ensure at least ID and trait columns are available.
        if df.shape[1] < 2:
            raise ValueError(f"Phenotype file {pheno_file} must contain at least two columns")

        # Determine ID and trait columns.
        original_id_col = df.columns[0]
        if trait_col is not None:
            if trait_col not in df.columns:
                raise ValueError(
                    f"Trait column '{trait_col}' not found in {pheno_file}. "
                    f"Available columns: {list(df.columns)}"
                )
            original_trait_col = trait_col
        else:
            original_trait_col = df.columns[1]

        self.logger.info(f"Original columns: ID={original_id_col}, Trait={original_trait_col}")

        # Normalize the first column to ID.
        rename_dict = {original_id_col: 'ID'}

        # Normalize the trait column name.
        target_col_name = original_trait_col
        if trait_name:
            rename_dict[original_trait_col] = trait_name
            target_col_name = trait_name
            self.logger.info(f"Renaming phenotype column '{original_trait_col}' to '{trait_name}'")
        else:
            cleaned_col = self.clean_column_names([original_trait_col])[0]
            if cleaned_col != original_trait_col:
                rename_dict[original_trait_col] = cleaned_col
                target_col_name = cleaned_col

        df = df.rename(columns=rename_dict)

        # Keep only ID and target trait columns.
        df = df[['ID', target_col_name]]

        # Drop missing phenotype values.
        df = df.dropna(subset=[target_col_name])

        # Remove string-form NA values as well.
        if df[target_col_name].dtype == object:
             df = df[df[target_col_name] != 'NA']

        self.logger.info(f"Remaining samples after dropping missing phenotypes: {len(df)}")

        # Save as CSV.
        if out_file is None:
            base, ext = os.path.splitext(pheno_file)
            if ext.lower() == '.csv':
                out_file = f"{base}_converted.csv"
            else:
                out_file = f"{base}.csv"

        df.to_csv(out_file, index=False)
        self.logger.info(f"Phenotype CSV saved to: {out_file}")

        return df
    
    def match_genotype_phenotype(self, pheno_df, geno_file, out_prefix):
        """Match genotype and phenotype samples and preserve a shared order."""
        self.logger.info(f"Reading genotype file: {geno_file}")
        
        geno_df = pd.read_csv(geno_file, index_col=0)
        
        geno_samples = set(geno_df.index)
        self.logger.info(f"Genotype file contains {len(geno_samples)} samples")
        
        pheno_samples = set(pheno_df['ID'])
        self.logger.info(f"Phenotype file contains {len(pheno_samples)} samples")
        
        common_samples = sorted(list(geno_samples.intersection(pheno_samples)))
        self.logger.info(f"Found {len(common_samples)} shared samples")
        
        if len(common_samples) == 0:
            raise ValueError("No shared sample IDs found. Check sample ID formatting.")
        
        # Filter and sort phenotype data by shared sample IDs.
        pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
        pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
        
        # Filter and sort genotype data by shared sample IDs.
        geno_filtered = geno_df.loc[common_samples]
        
        # Save matched files.
        pheno_out_file = f"{out_prefix}_phenotype.csv"
        geno_out_file = f"{out_prefix}_genotype.csv"
        
        pheno_filtered.to_csv(pheno_out_file, index=False)
        geno_filtered.to_csv(geno_out_file)
        
        self.logger.info(f"Matched phenotype file saved to: {pheno_out_file}")
        self.logger.info(f"Matched genotype file saved to: {geno_out_file}")
        
        return pheno_out_file, geno_out_file
    
    # =================== 3. Data validation and cleanup ===================
    
    def check_special_chars(self, column_names):
        """Check whether column names contain unsupported characters."""
        problematic_columns = {}
        
        for col in column_names:
            chars_found = []
            for char in SPECIAL_CHARS:
                if char in col:
                    chars_found.append(char)
            
            if chars_found:
                problematic_columns[col] = chars_found
        
        return problematic_columns
    
    def clean_column_names(self, column_names):
        """Replace unsupported characters in column names with underscores."""
        cleaned_names = []
        for col in column_names:
            cleaned_col = col
            for char in SPECIAL_CHARS:
                cleaned_col = cleaned_col.replace(char, '_')
            cleaned_names.append(cleaned_col)
        
        return cleaned_names
    
    def process_file(self, file_path, output_path):
        """Sanitize column names in a CSV file and write the result."""
        self.logger.info(f"Processing file: {file_path}")
        
        df = pd.read_csv(file_path)
        
        problematic_columns = self.check_special_chars(df.columns)
        
        if problematic_columns:
            self.logger.warning(f"Found {len(problematic_columns)} column name(s) with unsupported characters")
            for col, chars in problematic_columns.items():
                self.logger.warning(f"Column '{col}' contains unsupported characters: {', '.join(chars)}")
            
            cleaned_columns = self.clean_column_names(df.columns)
            column_mapping = dict(zip(df.columns, cleaned_columns))
            
            df = df.rename(columns=column_mapping)
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Sanitized data saved to: {output_path}")
            
            return True
        else:
            self.logger.info("No unsupported characters found in column names")
            df.to_csv(output_path, index=False)
            return False
    
    def load_matrix(self, matrix_file):
        """Load a matrix file and log basic summary information."""
        self.logger.info(f"Loading matrix file: {matrix_file}")
        
        matrix_df = pd.read_csv(matrix_file, index_col=0)
        
        self.logger.info(f"Matrix shape: {matrix_df.shape}")
        self.logger.info(f"Sample count: {len(matrix_df.index)}")
        self.logger.info(f"SNP count: {len(matrix_df.columns)}")
        
        missing_values = matrix_df.isnull().sum().sum()
        self.logger.info(f"Missing value count: {missing_values}")
        
        return matrix_df
    
    def standardize_phenotype(self, pheno_df, trait_col):
        """
        Apply z-score standardization to a phenotype column.

        Parameters
        ----------
        pheno_df
            Phenotype DataFrame.
        trait_col
            Phenotype column name.

        Returns
        -------
        tuple
            Standardized DataFrame and scaler parameter dictionary.
        """
        import json
        
        y = pheno_df[trait_col]
        mean_val = float(y.mean())
        std_val = float(y.std())
        
        # Avoid division by zero.
        if std_val < 1e-10:
            self.logger.warning("Phenotype standard deviation is near zero; skipping standardization")
            return pheno_df, {'mean': mean_val, 'std': 1.0, 'applied': False}
        
        # Standardize a copy to avoid mutating the caller's DataFrame.
        pheno_standardized = pheno_df.copy()
        pheno_standardized[trait_col] = (y - mean_val) / std_val
        
        scaler_params = {
            'mean': mean_val,
            'std': std_val,
            'applied': True,
            'trait': trait_col
        }
        
        self.logger.info(f"Phenotype standardization completed: mean={mean_val:.4f}, std={std_val:.4f}")
        
        return pheno_standardized, scaler_params
    
    def process_genomic_data(self, **kwargs):
        """
        Run the complete genomic data processing workflow.

        Parameters
        ----------
        kwargs
            Processing parameters.
        """
        # Create output directory if needed.
        out_prefix = kwargs.get('out_prefix')
        output_dir = os.path.dirname(out_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: SNP extraction/matrix conversion or reuse an existing matrix.
            geno_matrix_file = None
            
            # Use an existing matrix file when provided.
            if kwargs.get('matrix_file'):
                matrix_file = kwargs['matrix_file']
                if os.path.exists(matrix_file):
                    self.logger.info(f"Using existing genotype matrix file: {matrix_file}")
                    geno_matrix_file = matrix_file
                    
                    if kwargs.get('load'):
                        self.load_matrix(geno_matrix_file)
                else:
                    raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
            
            # Otherwise run SNP extraction and matrix conversion.
            elif not kwargs.get('skip_matrix'):
                # Use existing PED/MAP files when provided.
                if kwargs.get('ped_file') and kwargs.get('map_file'):
                    ped_file = kwargs['ped_file']
                    map_file = kwargs['map_file']
                    self.logger.info(f"Using existing PED/MAP files: {ped_file}, {map_file}")
                    
                    if not os.path.exists(ped_file) or not os.path.exists(map_file):
                        raise FileNotFoundError(f"PED/MAP file not found: {ped_file} or {map_file}")
                    
                    ped_prefix = os.path.splitext(ped_file)[0]
                    
                    # Create a temporary shared prefix when PED/MAP prefixes differ.
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
                        # Continue with the first processed SNP list for downstream steps.
                        snp_files = glob.glob(os.path.join(kwargs['snp_dir'], "*.txt"))
                        if snp_files:
                            first_phenotype = os.path.basename(snp_files[0]).replace('.txt', '')
                            geno_matrix_file = os.path.join(os.path.dirname(out_prefix), first_phenotype + ".csv")
                    elif kwargs.get('extract'):
                        out_prefix_temp = self.extract_snps(bfile, kwargs['extract'], out_prefix)
                        
                        geno_matrix_file = self.convert_to_matrix(out_prefix_temp)
                        
                        if kwargs.get('load') and geno_matrix_file:
                            self.load_matrix(geno_matrix_file)
                    elif kwargs.get('direct'):
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
                    # Assume the genotype matrix already exists.
                    geno_matrix_file = out_prefix + ".csv"
                    if not os.path.exists(geno_matrix_file):
                        raise FileNotFoundError(f"Genotype matrix file not found: {geno_matrix_file}")
            
            # Step 2: phenotype/genotype matching, cleanup, and optional standardization.
            final_pheno_file = None
            final_geno_file = None
            scaler_params = None

            if not kwargs.get('skip_match') and kwargs.get('pheno') and geno_matrix_file:
                self.logger.info("\nStarting integrated data processing: matching, cleanup, and standardization")

                # Discover trait columns from the phenotype file.
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

                # If user passed --trait-name, keep only that trait.
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

                for trait in trait_cols:
                    self.logger.info(f"\n--- Processing trait: {trait} ---")

                    safe_trait = re.sub(r'[^\w\-]', '_', trait)
                    pheno_raw_file = f"{out_prefix}_{safe_trait}_phenotype_raw.csv"

                    # Convert phenotype for this trait, explicitly writing to out-prefix.
                    pheno_df = self.convert_phenotype(
                        kwargs['pheno'],
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
                        continue

                    pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
                    pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
                    geno_filtered = geno_df.loc[common_samples]

                    geno_cleaned_cols = self.clean_column_names(geno_filtered.columns)
                    geno_filtered.columns = geno_cleaned_cols

                    pheno_cols = pheno_filtered.columns.tolist()
                    pheno_cleaned_cols = self.clean_column_names(pheno_cols)
                    pheno_filtered.columns = pheno_cleaned_cols

                    actual_trait_col = pheno_filtered.columns[1]

                    trait_scaler = None
                    if kwargs.get('standardize_phenotype', False):
                        self.logger.info("Standardizing phenotype data...")
                        pheno_filtered, trait_scaler = self.standardize_phenotype(pheno_filtered, actual_trait_col)

                        import json
                        scaler_file = f"{out_prefix}_{safe_trait}_scaler.json"
                        with open(scaler_file, 'w') as f:
                            json.dump(trait_scaler, f, indent=2)
                        self.logger.info(f"Phenotype scaler parameters saved to: {scaler_file}")

                    final_pheno_file = f"{out_prefix}_{safe_trait}_phenotype.csv"
                    final_geno_file = f"{out_prefix}_{safe_trait}_genotype.csv"

                    pheno_filtered.to_csv(final_pheno_file, index=False)
                    geno_filtered.to_csv(final_geno_file)

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

                self.logger.info("\nAll traits processed")

            else:
                if kwargs.get('skip_match'):
                    self.logger.info("Skipping phenotype/genotype matching")
                elif not kwargs.get('pheno'):
                    self.logger.warning("No phenotype file provided; skipping phenotype/genotype matching")
                elif not geno_matrix_file:
                    self.logger.warning("No genotype matrix available; skipping phenotype/genotype matching")
            
            self.logger.info("\nProcessing workflow completed")
            
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
        
        return 0


# Backward-compatible standalone functions
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
    return processor.convert_phenotype(pheno_file, out_file, trait_name, trait_col)

def match_genotype_phenotype(pheno_df, geno_file, out_prefix):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.match_genotype_phenotype(pheno_df, geno_file, out_prefix)

def check_special_chars(column_names):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.check_special_chars(column_names)

def clean_column_names(column_names):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.clean_column_names(column_names)

def process_file(file_path, output_path):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.process_file(file_path, output_path)

def load_matrix(matrix_file):
    """Backward-compatible wrapper."""
    processor = GenomicDataProcessor()
    return processor.load_matrix(matrix_file)


def main():
    parser = argparse.ArgumentParser(description='Integrated genomic data conversion utility')
    parser.add_argument('--bfile', help='Input PLINK binary prefix without extension')
    parser.add_argument('--vcf', help='Input VCF file path')
    parser.add_argument('--ped-file', help='Input PLINK PED file path')
    parser.add_argument('--map-file', help='Input PLINK MAP file path')
    parser.add_argument('--extract', help='File containing SNP IDs to extract')
    parser.add_argument('--snp-dir', help='Directory containing SNP list files')
    parser.add_argument('--plink', default='plink', help='PLINK executable path')
    parser.add_argument('--direct', action='store_true', help='Convert the full bfile to a matrix without SNP filtering')
    parser.add_argument('--plink-out', help='PLINK binary output prefix used during VCF conversion')
    parser.add_argument('--load', action='store_true', help='Load and report matrix information')
    parser.add_argument('--matrix-file', help='Existing genotype matrix file; skips matrix generation')
    parser.add_argument('--pheno', help='Phenotype file path')
    parser.add_argument('--out-prefix', required=True, help='Output file prefix')
    parser.add_argument('--skip-clean', action='store_true', help='Skip data cleanup')
    parser.add_argument('--skip-match', action='store_true', help='Skip phenotype/genotype matching')
    parser.add_argument('--skip-matrix', action='store_true', help='Skip SNP extraction and matrix conversion')
    
    args = parser.parse_args()
    
    processor = GenomicDataProcessor(plink_path=args.plink)
    
    kwargs = vars(args)
    
    return processor.process_genomic_data(**kwargs)


if __name__ == '__main__':
    sys.exit(main())
