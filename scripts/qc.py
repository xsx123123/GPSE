#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genotype Data Processing Module
================================

此模块用于处理基因型数据，包括格式转换、质量控制(QC)、LD剪枝、
基于Beagle的缺失值插补以及编码转换。

主要功能：
1. format_converter: 统一将 VCF/PED 等格式转换为 PLINK BED 格式。
2. filter_genotype: 调用 PLINK 进行样本和位点的过滤。
3. impute_genotype: 调用 Beagle 进行基因型插补。
4. analyze_and_prune: 执行 QC、统计缺失率并进行 LD 剪枝。
5. recode_to_numeric: 将 PLINK 01 编码转换为加性编码 (0, 1, 2)。

注意：
- 需要系统环境中安装 PLINK (1.9版本推荐) 和 Java (用于 Beagle)。
"""

import os
import subprocess
import logging
import shutil
from typing import Dict, Tuple, Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GenotypeProcessor")


def _check_tool_path(tool_path: str) -> str:
    """
    验证外部工具路径是否存在，如果不仅是命令名，检查绝对路径。
    """
    # 如果是命令名 (如 'plink')，使用 shutil.which 查找
    if os.path.sep not in tool_path:
        tool_loc = shutil.which(tool_path)
        if tool_loc:
            return tool_loc
    
    # 如果是路径，检查是否存在
    if os.path.exists(tool_path):
        return tool_path
        
    raise FileNotFoundError(f"External tool not found at: {tool_path}")


def _run_command(cmd_list: List[str], log_file: Optional[str] = None) -> None:
    """
    执行 Shell 命令并记录日志。
    """
    cmd_str = " ".join(cmd_list)
    logger.info(f"Executing: {cmd_str}")
    
    try:
        if log_file:
            with open(log_file, 'a') as f:
                process = subprocess.Popen(cmd_str, shell=True, stdout=f, stderr=f)
        else:
            process = subprocess.Popen(cmd_str, shell=True)
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}: {cmd_str}")
            raise subprocess.CalledProcessError(process.returncode, cmd_str)
            
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise


def recode_to_numeric(fileprefix: str) -> None:
    """
    将 PLINK 的 compound-genotypes (01 编码) 转换为加性编码 (0, 1, 2)。
    
    Args:
        fileprefix: 文件前缀 (寻找 .ped 和 .map)
    """
    ped_path = fileprefix + '.ped'
    map_path = fileprefix + '.map'
    geno_path = fileprefix + '.geno'
    
    # 编码映射字典
    # 00 -> 0 (Homozygous Major)
    # 01 -> 1 (Heterozygous)
    # 10 -> 1 (Heterozygous)
    # 11 -> 2 (Homozygous Minor)
    code_map = {'00': '0', '01': '1', '10': '1', '11': '2'}

    logger.info(f"Recoding {ped_path} to numeric format...")

    try:
        # 读取 SNP ID 作为 header
        with open(map_path, 'r') as map_file:
            # 假设 map 文件第二列是 SNP ID
            snpid_list = [line.strip().split('\t')[1] for line in map_file]
        
        with open(geno_path, 'w+') as geno_file:
            # 写入 Header
            geno_file.write(','.join(snpid_list) + '\n')
            
            with open(ped_path, 'r') as ped_file:
                for line in ped_file:
                    row = line.strip().split()
                    sample_id = row[1]
                    genotypes = row[6:]
                    
                    # 快速映射
                    # 注意：如果有缺失值，这里需要额外处理，假设此时输入已经是处理过的无缺失数据
                    numeric_genos = [code_map.get(g, 'NaN') for g in genotypes]
                    
                    # 构造输出行：SampleID,0,1,2...
                    line_out = sample_id + ',' + ','.join(numeric_genos) + '\n'
                    geno_file.write(line_out)
                    
        logger.info(f"Recoding finished. Saved to {geno_path}")

    except Exception as e:
        logger.error(f"Error in recode_to_numeric: {e}")
        raise


def format_converter(user_params: Dict, input_prefix: str, output_prefix: str) -> str:
    """
    检测输入格式并统一转换为 PLINK BED 格式。
    
    Args:
        user_params: 参数字典
        input_prefix: 输入文件前缀
        output_prefix: 输出文件前缀

    Returns:
        转换后的 PLINK 格式标志 (如 --bfile)
    """
    plink_path = _check_tool_path(user_params['plink_path'])
    
    # 检测文件是否存在以推断格式
    if os.path.exists(input_prefix + '.vcf') or os.path.exists(input_prefix + '.vcf.gz'):
        logger.info("Detected VCF format. Converting to BED...")
        vcf_file = input_prefix + '.vcf' if os.path.exists(input_prefix + '.vcf') else input_prefix + '.vcf.gz'
        cmd = [plink_path, '--vcf', vcf_file, '--make-bed', '--out', output_prefix]
        # VCF ID 处理：保留 ID
        cmd.extend(['--const-fid', '--allow-extra-chr']) 
        _run_command(cmd, output_prefix + '.log')
        return '--bfile'
        
    elif os.path.exists(input_prefix + '.ped') and os.path.exists(input_prefix + '.map'):
        logger.info("Detected PED/MAP format. Converting to BED...")
        cmd = [plink_path, '--file', input_prefix, '--make-bed', '--out', output_prefix]
        _run_command(cmd, output_prefix + '.log')
        return '--bfile'
        
    elif os.path.exists(input_prefix + '.bed') and os.path.exists(input_prefix + '.bim') and os.path.exists(input_prefix + '.fam'):
        logger.info("Detected BED/BIM/FAM format.")
        # 如果已经是二进制格式，不需要转换，直接复制或链接过去，或者直接返回
        # 为了流程统一，这里不做拷贝，直接告知后续步骤使用 --bfile 和原始路径
        return '--bfile'
    
    else:
        logger.warning(f"Unknown format for {input_prefix}. Assuming user provided parameters are correct for PLINK.")
        # 默认返回用户指定的格式参数
        return user_params.get('fileformat', '--bfile')


def filter_genotype(user_params: Dict, input_prefix: str, output_prefix: str, input_flag: str = '--bfile') -> None:
    """
    根据 ID 列表过滤样本和 SNP。
    """
    extract_snpid = user_params.get('extract_snpid_path')
    exclude_snpid = user_params.get('exclude_snpid_path')
    keep_sampleid = user_params.get('keep_sampleid_path')
    remove_sampleid = user_params.get('remove_sampleid_path')
    plink_path = _check_tool_path(user_params['plink_path'])
    
    log_file = output_prefix + '_preprocessed.log'

    # 基础命令
    cmd = [plink_path, input_flag, input_prefix, '--out', output_prefix]
    
    # 动态添加过滤参数
    if extract_snpid:
        cmd.extend(['--extract', extract_snpid])
    if exclude_snpid:
        cmd.extend(['--exclude', exclude_snpid])
    if keep_sampleid:
        cmd.extend(['--keep', keep_sampleid])
    if remove_sampleid:
        cmd.extend(['--remove', remove_sampleid])
    
    # 输出格式控制: recode compound-genotypes 01 (用于后续 recode_to_numeric)
    cmd.extend(['--recode', 'compound-genotypes', '01', '--output-missing-genotype', '3'])
    
    _run_command(cmd, log_file)


def impute_genotype_beagle(user_params: Dict, input_prefix: str, output_prefix: str) -> None:
    """
    使用 Beagle 进行基因型插补。
    
    Args:
        user_params: 需包含 'beagle_jar_path' (Beagle jar包路径)
        input_prefix: 输入 PLINK 二进制文件前缀
        output_prefix: 输出前缀
    """
    beagle_jar = user_params.get('beagle_jar_path')
    if not beagle_jar or not os.path.exists(beagle_jar):
        logger.warning("Beagle JAR not found or not specified. Skipping imputation.")
        return

    plink_path = _check_tool_path(user_params['plink_path'])
    
    # 1. PLINK BED -> VCF
    logger.info("Converting BED to VCF for Beagle...")
    vcf_temp = input_prefix + '_temp_for_beagle'
    cmd_to_vcf = [plink_path, '--bfile', input_prefix, '--recode', 'vcf', '--out', vcf_temp]
    _run_command(cmd_to_vcf, output_prefix + '.log')
    
    # 2. Run Beagle
    # Beagle output is usually prefix.vcf.gz
    logger.info("Running Beagle imputation...")
    out_beagle = output_prefix + '_imputed'
    # Beagle 5.x 语法: gt=input.vcf out=output_prefix
    # Beagle 4.x 语法可能不同，此处假定为 Beagle 5.0+
    cmd_beagle = ['java', '-jar', beagle_jar, f'gt={vcf_temp}.vcf', f'out={out_beagle}']
    # 注意：Beagle 使用 Java，不通过 PLINK 调用
    try:
        subprocess.check_call(cmd_beagle)
    except subprocess.CalledProcessError as e:
        logger.error(f"Beagle failed: {e}")
        raise

    # 3. VCF (Imputed) -> PLINK BED
    # Beagle 输出通常是 .vcf.gz
    logger.info("Converting Imputed VCF back to BED...")
    imputed_vcf = out_beagle + '.vcf.gz'
    if not os.path.exists(imputed_vcf):
         # 尝试非 gz 后缀
         imputed_vcf = out_beagle + '.vcf'
         
    cmd_to_bed = [plink_path, '--vcf', imputed_vcf, '--make-bed', '--out', output_prefix]
    _run_command(cmd_to_bed, output_prefix + '.log')

    # 清理中间文件
    try:
        os.remove(f"{vcf_temp}.vcf")
        os.remove(f"{vcf_temp}.log")
        os.remove(f"{vcf_temp}.nosex")
    except OSError:
        pass


def analyze_and_prune(user_params: Dict, input_prefix: str, output_prefix: str, 
                     run_imputation: bool = False) -> Tuple[str, str]:
    """
    执行 QC (MAF, Missingness) 和 LD Pruning。
    
    Args:
        user_params: 参数字典
        input_prefix: 输入文件前缀
        output_prefix: 输出文件前缀
        run_imputation: 是否运行 Beagle 插补 (需配置 beagle_jar_path)

    Returns:
        (qc_filled_prefix, pruned_prefix): 处理后的文件前缀
    """
    # 0. 格式标准化检测 (处理 VCF/PED 等输入)
    current_flag = format_converter(user_params, input_prefix, output_prefix + "_raw")
    if current_flag == '--bfile' and input_prefix != (output_prefix + "_raw"):
        # 如果进行了转换
        bed_prefix = output_prefix + "_raw"

    snp_miss = user_params['snpmaxmiss']
    sample_miss = user_params['samplemaxmiss']
    maf = user_params['maf_max'] # 最小等位基因频率
    r2 = user_params['r2_cutoff']
    plink_path = _check_tool_path(user_params['plink_path'])
    
    qc_prefix = output_prefix + '_qc'
    
    # 1. 基本 QC统计 (Missing, Freq) + 过滤
    logger.info("Running QC filtering...")
    cmd_qc = [
        plink_path, 
        '--bfile', bed_prefix,
        '--out', qc_prefix,
        '--make-bed',
        '--geno', str(snp_miss),
        '--mind', str(sample_miss),
        '--maf', str(maf),
        '--freq', 
        '--missing'
    ]
    _run_command(cmd_qc, output_prefix + '_qc.log')

    # 2. 插补 (可选) 或 简单填充
    qc_filled_prefix = qc_prefix + '_filled'
    
    if run_imputation and 'beagle_jar_path' in user_params:
        logger.info("Running Beagle imputation option...")
        impute_genotype_beagle(user_params, qc_prefix, qc_filled_prefix)
    else:
        # 使用 PLINK 简单填充 (fill-missing-a2: 填入参考等位基因)
        logger.info("Running simple PLINK filling...")
        cmd_fill = [
            plink_path, 
            '--bfile', qc_prefix, 
            '--out', qc_filled_prefix,
            '--make-bed', 
            '--fill-missing-a2'
        ]
        _run_command(cmd_fill, output_prefix + '_qc.log')

    # 3. LD Pruning (去连锁不平衡)
    # 第一步：计算要保留的 SNP 列表 (.prune.in)
    logger.info("Calculating LD for pruning...")
    cmd_indep = [
        plink_path,
        '--bfile', qc_filled_prefix,
        '--out', qc_filled_prefix, # 输出 .prune.in
        '--indep-pairwise', '50', '10', str(r2)
    ]
    _run_command(cmd_indep, output_prefix + '_qc.log')

    # 第二步：根据列表提取 SNP
    pruned_prefix = output_prefix + '_pruned'
    logger.info("Extracting pruned SNPs...")
    cmd_extract = [
        plink_path,
        '--bfile', qc_filled_prefix,
        '--out', pruned_prefix,
        '--extract', qc_filled_prefix + '.prune.in',
        '--make-bed'
    ]
    _run_command(cmd_extract, output_prefix + '_qc.log')

    return qc_filled_prefix, pruned_prefix