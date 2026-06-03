#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基因组预测数据处理一体化工具

功能流程：
1. SNP提取与矩阵转换
   - 将VCF格式转换为PLINK二进制格式
   - 提取指定SNP位点并转换为CSV矩阵

2. 表型基因型匹配
   - 将表型TXT文件转换为CSV格式
   - 提取基因型和表型文件中共有的样品ID
   - 生成排序一致的表型和基因型文件

3. 数据检查与清理
   - 检查基因型和表型数据中的特征名称是否包含特殊字符
   - 清理并替换特殊字符
   - 生成可直接用于机器学习的数据文件

可作为工具模块调用或独立命令行工具使用
"""

import os
import argparse
import subprocess
import pandas as pd
import glob
import re
import logging
import sys

# 定义基因型编码字典（加性模型）
GENO_DICT = {
    '00': '0',  # 纯合子参考等位基因
    '01': '1',  # 杂合子
    '10': '1',  # 杂合子
    '11': '2'   # 纯合子替代等位基因
}

# LightGBM不支持的特殊字符列表
SPECIAL_CHARS = [
    ':', '|', '[', ']', '{', '}', '"', '\\', ',', ' '
]


class GenomicDataProcessor:
    """基因组数据处理器"""
    
    def __init__(self, logger=None, plink_path="plink"):
        """
        初始化基因组数据处理器
        
        参数:
            logger: 日志记录器，如果不提供则创建默认记录器
            plink_path: PLINK可执行文件路径
        """
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.plink_path = plink_path
    
    #=================== 1. SNP提取与矩阵转换 ===================
    
    def vcf_to_plink(self, vcf_file, out_prefix):
        """将VCF格式转换为PLINK二进制格式"""
        self.logger.info(f"正在将VCF文件 {vcf_file} 转换为PLINK二进制格式...")
        
        # 检查输出文件是否已存在
        if os.path.exists(f"{out_prefix}.bed") and os.path.exists(f"{out_prefix}.bim") and os.path.exists(f"{out_prefix}.fam"):
            self.logger.info(f"PLINK二进制文件已存在: {out_prefix}.bed, {out_prefix}.bim, {out_prefix}.fam")
            self.logger.info("跳过转换步骤...")
            return out_prefix
        
        # 检查PLINK是否可用
        plink_result = os.popen(f"which {self.plink_path}").read().strip()
        if not os.path.exists(plink_result):
            raise IOError(f"找不到PLINK: {self.plink_path}")
        
        # 构建PLINK命令
        cmd = f"{self.plink_path} --vcf {vcf_file} --make-bed --out {out_prefix} --double-id"
        
        # 执行PLINK命令
        self.logger.info(f"执行命令: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("PLINK命令执行失败，请检查日志文件")
        
        self.logger.info(f"VCF转换完成，输出文件: {out_prefix}.bed, {out_prefix}.bim, {out_prefix}.fam")
        return out_prefix
    
    def extract_snps(self, bfile, extract_file, out_prefix):
        """使用PLINK提取指定的SNP位点"""
        self.logger.info(f"正在从{bfile}中提取SNP位点...")
        
        # 检查输出文件是否已存在
        if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
            self.logger.info(f"PLINK PED/MAP文件已存在: {out_prefix}.ped, {out_prefix}.map")
            self.logger.info("跳过提取步骤...")
            return out_prefix
        
        # 检查PLINK是否可用
        plink_result = os.popen(f"which {self.plink_path}").read().strip()
        if not os.path.exists(plink_result):
            raise IOError(f"找不到PLINK: {self.plink_path}")
        
        # 构建PLINK命令
        cmd = (f"{self.plink_path} --bfile {bfile} --out {out_prefix} "
               f"--extract {extract_file} "
               f"--recode compound-genotypes 01 --output-missing-genotype 3")
        
        # 执行PLINK命令
        self.logger.info(f"执行命令: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("PLINK命令执行失败，请检查日志文件")
        
        self.logger.info(f"SNP提取完成，输出文件: {out_prefix}.ped, {out_prefix}.map")
        return out_prefix
    
    def convert_bfile_to_ped(self, bfile, out_prefix):
        """将PLINK二进制文件直接转换为PED/MAP格式，不进行SNP提取"""
        self.logger.info(f"正在将PLINK二进制文件 {bfile} 转换为PED/MAP格式...")
        
        # 检查输出文件是否已存在
        if os.path.exists(f"{out_prefix}.ped") and os.path.exists(f"{out_prefix}.map"):
            self.logger.info(f"PLINK PED/MAP文件已存在: {out_prefix}.ped, {out_prefix}.map")
            self.logger.info("跳过转换步骤...")
            return out_prefix
        
        # 检查PLINK是否可用
        plink_result = os.popen(f"which {self.plink_path}").read().strip()
        if not os.path.exists(plink_result):
            raise IOError(f"找不到PLINK: {self.plink_path}")
        
        # 构建PLINK命令
        cmd = (f"{self.plink_path} --bfile {bfile} --out {out_prefix} "
               f"--recode compound-genotypes 01 --output-missing-genotype 3")
        
        # 执行PLINK命令
        self.logger.info(f"执行命令: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("PLINK命令执行失败，请检查日志文件")
        
        self.logger.info(f"转换完成，输出文件: {out_prefix}.ped, {out_prefix}.map")
        return out_prefix
    
    def convert_to_matrix(self, fileprefix, out_file=None):
        """将PLINK格式的基因型数据转换为CSV格式的数值矩阵"""
        self.logger.info(f"正在将{fileprefix}.ped和{fileprefix}.map转换为矩阵格式...")
        
        ped_path = fileprefix + '.ped'
        map_path = fileprefix + '.map'
        
        # 如果PED/MAP文件具有完整路径，使用这些路径
        if not os.path.exists(ped_path) and '/' in fileprefix:
            # 尝试查找其他可能的文件位置
            base_name = os.path.basename(fileprefix)
            dir_name = os.path.dirname(fileprefix)
            alt_ped_path = os.path.join(dir_name, base_name + '.ped')
            alt_map_path = os.path.join(dir_name, base_name + '.map')
            if os.path.exists(alt_ped_path) and os.path.exists(alt_map_path):
                ped_path = alt_ped_path
                map_path = alt_map_path
        
        if out_file is None:
            out_file = fileprefix + '.csv'
        
        # 检查输出文件是否已存在
        if os.path.exists(out_file):
            self.logger.info(f"矩阵文件已存在: {out_file}")
            self.logger.info("跳过转换步骤...")
            return out_file
        
        # 检查输入文件是否存在
        if not os.path.exists(ped_path) or not os.path.exists(map_path):
            raise FileNotFoundError(f"找不到输入文件: {ped_path}或{map_path}")
        
        # 读取SNP ID
        snpid_list = []
        with open(map_path) as map_file:
            for row in map_file:
                row = row.strip().split('\t')
                snpid_list.append(row[1])  # 第2列是SNP ID
        
        # 读取样本ID和基因型数据
        sample_genotypes = []
        with open(ped_path) as ped_file:
            for row in ped_file:
                row = row.strip().split()
                sample_id = row[1]  # 第2列是样本ID
                
                # 清理ID：如果以_结尾，去掉它（解决VCF中ID带下划线的问题）
                if sample_id.endswith('_'):
                    sample_id = sample_id.rstrip('_')
                
                geno = row[6:]  # 从第7列开始是基因型数据
                
                # 将复合基因型转换为数值编码
                encoded_geno = []
                for genotype in geno:
                    # 使用编码字典将基因型转换为数值
                    # 00->0, 01->1, 10->1, 11->2
                    diploid = GENO_DICT.get(genotype, '3')  # 默认缺失值为3
                    encoded_geno.append(diploid)
                
                sample_genotypes.append((sample_id, encoded_geno))
        
        # 写入CSV格式的矩阵文件
        with open(out_file, 'w') as csv_file:
            # 写入带有ID的表头
            header = "ID," + ",".join(snpid_list)
            csv_file.write(header + '\n')
            
            # 写入样本ID和基因型数据
            for sample_id, genotypes in sample_genotypes:
                line = sample_id + "," + ",".join(genotypes)
                csv_file.write(line + '\n')
        
        self.logger.info(f"转换完成，输出CSV文件: {out_file}")
        return out_file
    
    def process_snp_dir(self, bfile, snp_dir, out_dir):
        """批量处理指定目录中的SNP列表文件"""
        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)
        
        # 获取目录中所有txt文件
        snp_files = glob.glob(os.path.join(snp_dir, "*.txt"))
        
        if not snp_files:
            self.logger.warning(f"警告: 在{snp_dir}中没有找到.txt文件")
            return
        
        self.logger.info(f"找到{len(snp_files)}个SNP列表文件，开始批量处理...")
        
        for snp_file in snp_files:
            # 获取文件名（不含路径和扩展名）作为表型名
            phenotype = os.path.basename(snp_file).replace('.txt', '')
            self.logger.info(f"\n处理表型: {phenotype}")
            
            # 为每个表型创建输出前缀
            out_prefix = os.path.join(out_dir, phenotype)
            
            try:
                # 提取SNP位点
                self.extract_snps(bfile, snp_file, out_prefix)
                
                # 转换为矩阵格式
                matrix_file = self.convert_to_matrix(out_prefix)
                
                self.logger.info(f"表型 {phenotype} 处理完成，CSV矩阵文件: {matrix_file}")
            except Exception as e:
                self.logger.error(f"处理表型 {phenotype} 时出错: {str(e)}")
    
    #=================== 2. 表型基因型匹配 ===================
    
    def convert_phenotype(self, pheno_file, out_file=None, trait_name=None):
        """将表型TXT文件转换为CSV格式，并去除NA缺失值"""
        self.logger.info(f"正在处理表型文件: {pheno_file}")
        
        # 使用pandas读取文件，假设第一行为表头
        try:
            # 尝试使用制表符分隔读取
            df = pd.read_csv(pheno_file, sep='\t')
            if df.shape[1] < 2:
                # 如果列数不足，尝试逗号分隔
                df = pd.read_csv(pheno_file, sep=',')
        except Exception as e:
            self.logger.warning(f"使用pandas读取失败，尝试原始方法: {str(e)}")
            # 回退到原始读取方法（不建议，仅作兼容）
            pheno_data = []
            with open(pheno_file, 'r') as f:
                header = next(f).strip().split('\t') # 跳过表头
                for line in f:
                    line = line.strip().split('\t')
                    if len(line) >= 2:
                        pheno_data.append([line[0], line[1]])
            df = pd.DataFrame(pheno_data, columns=['ID', 'Phenotype'])

        # 确保至少有两列
        if df.shape[1] < 2:
            raise ValueError(f"表型文件 {pheno_file} 列数不足2列")

        # 获取原始列名
        original_id_col = df.columns[0]
        original_trait_col = df.columns[1]
        self.logger.info(f"原始文件列名: ID={original_id_col}, Trait={original_trait_col}")
        
        # 重命名列
        # 第一列统一为ID
        rename_dict = {original_id_col: 'ID'}
        
        # 第二列处理：如果指定了trait_name，则强制重命名；否则使用文件内的名称
        target_col_name = original_trait_col
        if trait_name:
            rename_dict[original_trait_col] = trait_name
            target_col_name = trait_name
            self.logger.info(f"将表型列 '{original_trait_col}' 重命名为 '{trait_name}' (以匹配目标性状)")
        else:
            # 如果未指定，使用原始列名，但需要清理特殊字符
            cleaned_col = self.clean_column_names([original_trait_col])[0]
            if cleaned_col != original_trait_col:
                rename_dict[original_trait_col] = cleaned_col
                target_col_name = cleaned_col
        
        df = df.rename(columns=rename_dict)
        
        # 只保留ID和目标性状列
        df = df[['ID', target_col_name]]
        
        # 去除NA值
        # 注意：pandas读取时通常会自动将NA/NaN识别为np.nan
        df = df.dropna(subset=[target_col_name])
        
        # 确保没有字符串类型的'NA'
        if df[target_col_name].dtype == object:
             df = df[df[target_col_name] != 'NA']
             
        self.logger.info(f"去除NA值后剩余 {len(df)} 个样品")
        
        # 保存为CSV
        if out_file is None:
            out_file = os.path.splitext(pheno_file)[0] + '.csv'
        
        df.to_csv(out_file, index=False)
        self.logger.info(f"表型数据已保存为CSV: {out_file}")
        
        return df
    
    def match_genotype_phenotype(self, pheno_df, geno_file, out_prefix):
        """匹配基因型和表型文件，提取共有样品，并保持排序一致"""
        self.logger.info(f"正在读取基因型文件: {geno_file}")
        
        # 读取基因型文件
        geno_df = pd.read_csv(geno_file, index_col=0)
        
        # 获取基因型文件的样品ID
        geno_samples = set(geno_df.index)
        self.logger.info(f"基因型文件包含 {len(geno_samples)} 个样品")
        
        # 获取表型文件的样品ID
        pheno_samples = set(pheno_df['ID'])
        self.logger.info(f"表型文件包含 {len(pheno_samples)} 个样品")
        
        # 找出共有的样品ID
        common_samples = sorted(list(geno_samples.intersection(pheno_samples)))
        self.logger.info(f"共有 {len(common_samples)} 个样品同时存在于基因型和表型文件中")
        
        if len(common_samples) == 0:
            raise ValueError("没有找到共有的样品ID，请检查样品ID格式是否一致")
        
        # 按共有样品ID筛选并排序表型数据
        pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
        pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
        
        # 按共有样品ID筛选并排序基因型数据
        geno_filtered = geno_df.loc[common_samples]
        
        # 保存匹配后的文件
        pheno_out_file = f"{out_prefix}_phenotype.csv"
        geno_out_file = f"{out_prefix}_genotype.csv"
        
        pheno_filtered.to_csv(pheno_out_file, index=False)
        geno_filtered.to_csv(geno_out_file)
        
        self.logger.info(f"匹配后的表型文件已保存: {pheno_out_file}")
        self.logger.info(f"匹配后的基因型文件已保存: {geno_out_file}")
        
        return pheno_out_file, geno_out_file
    
    #=================== 3. 数据检查与清理 ===================
    
    def check_special_chars(self, column_names):
        """检查列名中是否包含特殊字符"""
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
        """清理列名中的特殊字符"""
        cleaned_names = []
        for col in column_names:
            # 替换所有特殊字符为下划线
            cleaned_col = col
            for char in SPECIAL_CHARS:
                cleaned_col = cleaned_col.replace(char, '_')
            cleaned_names.append(cleaned_col)
        
        return cleaned_names
    
    def process_file(self, file_path, output_path):
        """处理数据文件，检查并清理特殊字符"""
        self.logger.info(f"正在处理文件: {file_path}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 检查列名中的特殊字符
        problematic_columns = self.check_special_chars(df.columns)
        
        if problematic_columns:
            self.logger.warning(f"发现 {len(problematic_columns)} 个包含特殊字符的列名")
            for col, chars in problematic_columns.items():
                self.logger.warning(f"列名 '{col}' 包含特殊字符: {', '.join(chars)}")
            
            # 清理列名
            cleaned_columns = self.clean_column_names(df.columns)
            column_mapping = dict(zip(df.columns, cleaned_columns))
            
            # 重命名列
            df = df.rename(columns=column_mapping)
            
            # 保存清理后的数据
            df.to_csv(output_path, index=False)
            self.logger.info(f"已将清理后的数据保存到: {output_path}")
            
            return True
        else:
            self.logger.info("未发现包含特殊字符的列名")
            # 保存与源文件相同的数据
            df.to_csv(output_path, index=False)
            return False
    
    def load_matrix(self, matrix_file):
        """加载并显示矩阵信息"""
        self.logger.info(f"加载矩阵文件: {matrix_file}")
        
        # 读取矩阵文件
        matrix_df = pd.read_csv(matrix_file, index_col=0)
        
        # 显示基本信息
        self.logger.info(f"矩阵维度: {matrix_df.shape}")
        self.logger.info(f"样本数量: {len(matrix_df.index)}")
        self.logger.info(f"SNP数量: {len(matrix_df.columns)}")
        
        # 查看缺失值情况
        missing_values = matrix_df.isnull().sum().sum()
        self.logger.info(f"缺失值数量: {missing_values}")
        
        return matrix_df
    
    def standardize_phenotype(self, pheno_df, trait_col):
        """
        对表型数据进行Z-score标准化
        
        参数:
            pheno_df: 表型DataFrame
            trait_col: 表型列名
            
        返回:
            标准化后的DataFrame和标准化参数字典
        """
        import json
        
        y = pheno_df[trait_col]
        mean_val = float(y.mean())
        std_val = float(y.std())
        
        # 避免除以零
        if std_val < 1e-10:
            self.logger.warning("表型数据标准差接近零，跳过标准化")
            return pheno_df, {'mean': mean_val, 'std': 1.0, 'applied': False}
        
        # 创建副本进行标准化
        pheno_standardized = pheno_df.copy()
        pheno_standardized[trait_col] = (y - mean_val) / std_val
        
        scaler_params = {
            'mean': mean_val,
            'std': std_val,
            'applied': True,
            'trait': trait_col
        }
        
        self.logger.info(f"表型标准化完成 - 原始均值: {mean_val:.4f}, 原始标准差: {std_val:.4f}")
        
        return pheno_standardized, scaler_params
    
    def process_genomic_data(self, **kwargs):
        """
        完整的基因组数据处理流程
        
        参数:
            kwargs: 包含处理参数的字典
        """
        # 创建输出目录
        out_prefix = kwargs.get('out_prefix')
        output_dir = os.path.dirname(out_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 第1步: SNP提取与矩阵转换或使用已有矩阵
            geno_matrix_file = None
            
            # 如果指定了已有的矩阵文件，直接使用它
            if kwargs.get('matrix_file'):
                matrix_file = kwargs['matrix_file']
                if os.path.exists(matrix_file):
                    self.logger.info(f"使用已有的基因型矩阵文件: {matrix_file}")
                    geno_matrix_file = matrix_file
                    
                    # 如果需要加载矩阵信息
                    if kwargs.get('load'):
                        self.load_matrix(geno_matrix_file)
                else:
                    raise FileNotFoundError(f"找不到指定的矩阵文件: {matrix_file}")
            
            # 否则进行SNP提取与矩阵转换
            elif not kwargs.get('skip_matrix'):
                # 如果提供了PED/MAP文件，直接使用它们
                if kwargs.get('ped_file') and kwargs.get('map_file'):
                    ped_file = kwargs['ped_file']
                    map_file = kwargs['map_file']
                    self.logger.info(f"使用已有的PED/MAP文件: {ped_file}, {map_file}")
                    
                    # 检查文件是否存在
                    if not os.path.exists(ped_file) or not os.path.exists(map_file):
                        raise FileNotFoundError(f"找不到PED/MAP文件: {ped_file}或{map_file}")
                    
                    # 获取不带扩展名的文件前缀
                    ped_prefix = os.path.splitext(ped_file)[0]
                    
                    # 创建一个临时的文件前缀，确保PED和MAP文件使用相同的前缀
                    temp_prefix = out_prefix + "_temp"
                    
                    # 如果PED和MAP文件不在同一位置或不使用相同前缀，创建符号链接
                    if ped_prefix != os.path.splitext(map_file)[0]:
                        self.logger.info(f"PED和MAP文件前缀不一致，创建临时链接")
                        if os.path.exists(temp_prefix + ".ped"):
                            os.remove(temp_prefix + ".ped")
                        if os.path.exists(temp_prefix + ".map"):
                            os.remove(temp_prefix + ".map")
                        os.symlink(os.path.abspath(ped_file), temp_prefix + ".ped")
                        os.symlink(os.path.abspath(map_file), temp_prefix + ".map")
                        ped_prefix = temp_prefix
                    
                    # 转换为矩阵格式
                    geno_matrix_file = self.convert_to_matrix(ped_prefix, out_prefix + ".csv")
                    
                    # 如果需要加载矩阵信息
                    if kwargs.get('load') and geno_matrix_file:
                        self.load_matrix(geno_matrix_file)
                
                # 如果提供了VCF文件，先转换为PLINK格式
                elif kwargs.get('vcf'):
                    vcf_file = kwargs['vcf']
                    if not kwargs.get('bfile'):
                        # 如果提供了plink-out参数，使用它作为PLINK输出前缀
                        if kwargs.get('plink_out'):
                            plink_prefix = kwargs['plink_out']
                        else:
                            # 否则使用默认前缀
                            plink_prefix = os.path.join(os.path.dirname(out_prefix), "plink_data")
                        
                        # 转换VCF到PLINK格式（只转换一次）
                        kwargs['bfile'] = self.vcf_to_plink(vcf_file, plink_prefix)
                        self.logger.info(f"使用转换后的PLINK文件: {kwargs['bfile']}")
                
                # 确保有PLINK二进制文件
                if kwargs.get('bfile'):
                    bfile = kwargs['bfile']
                    # 如果提供了SNP目录，批量处理
                    if kwargs.get('snp_dir'):
                        self.process_snp_dir(bfile, kwargs['snp_dir'], os.path.dirname(out_prefix))
                        # 假设我们对第一个处理的文件继续进行下一步
                        snp_files = glob.glob(os.path.join(kwargs['snp_dir'], "*.txt"))
                        if snp_files:
                            first_phenotype = os.path.basename(snp_files[0]).replace('.txt', '')
                            geno_matrix_file = os.path.join(os.path.dirname(out_prefix), first_phenotype + ".csv")
                    # 如果只提供了单个SNP文件
                    elif kwargs.get('extract'):
                        # 提取SNP位点
                        out_prefix_temp = self.extract_snps(bfile, kwargs['extract'], out_prefix)
                        
                        # 转换为矩阵格式
                        geno_matrix_file = self.convert_to_matrix(out_prefix_temp)
                        
                        # 如果需要加载矩阵信息
                        if kwargs.get('load') and geno_matrix_file:
                            self.load_matrix(geno_matrix_file)
                    # 直接将整个bfile转换为矩阵
                    elif kwargs.get('direct'):
                        self.logger.info("直接将整个bfile转换为矩阵...")
                        
                        # 将bfile转换为PED/MAP格式
                        out_prefix_temp = self.convert_bfile_to_ped(bfile, out_prefix)
                        
                        # 转换为矩阵格式
                        geno_matrix_file = self.convert_to_matrix(out_prefix_temp)
                        
                        # 如果需要加载矩阵信息
                        if kwargs.get('load') and geno_matrix_file:
                            self.load_matrix(geno_matrix_file)
                    else:
                        self.logger.warning("警告: 未提供SNP列表文件或目录，也未指定--direct参数，仅完成VCF到PLINK的转换")
            else:
                self.logger.info("跳过SNP提取和矩阵转换步骤")
                # 尝试加载已有的矩阵文件
                if kwargs.get('load'):
                    matrix_file = out_prefix + ".csv"
                    if os.path.exists(matrix_file):
                        self.load_matrix(matrix_file)
                        geno_matrix_file = matrix_file
                    else:
                        self.logger.warning(f"找不到矩阵文件: {matrix_file}")
                elif kwargs.get('extract') or kwargs.get('direct'):
                    # 假设基因型矩阵文件已存在
                    geno_matrix_file = out_prefix + ".csv"
                    if not os.path.exists(geno_matrix_file):
                        raise FileNotFoundError(f"找不到基因型矩阵文件: {geno_matrix_file}")
            
            # 第2步: 表型基因型匹配、清理、标准化一体化处理
            final_pheno_file = None
            final_geno_file = None
            scaler_params = None
            
            if not kwargs.get('skip_match') and kwargs.get('pheno') and geno_matrix_file:
                self.logger.info("\n开始一体化数据处理（匹配 + 清理 + 标准化）...")
                
                # 1. 转换表型文件并去除NA值
                pheno_df = self.convert_phenotype(kwargs['pheno'], trait_name=kwargs.get('trait_name'))
                
                # 2. 读取基因型矩阵
                geno_df = pd.read_csv(geno_matrix_file, index_col=0)
                
                # 3. 找出共有样品
                geno_samples = set(geno_df.index)
                pheno_samples = set(pheno_df['ID'])
                common_samples = sorted(list(geno_samples.intersection(pheno_samples)))
                
                self.logger.info(f"基因型样品数: {len(geno_samples)}, 表型样品数: {len(pheno_samples)}")
                self.logger.info(f"共有样品数: {len(common_samples)}")
                
                if len(common_samples) == 0:
                    raise ValueError("没有找到共有的样品ID，请检查样品ID格式是否一致")
                
                # 4. 筛选并排序
                pheno_filtered = pheno_df[pheno_df['ID'].isin(common_samples)]
                pheno_filtered = pheno_filtered.set_index('ID').loc[common_samples].reset_index()
                geno_filtered = geno_df.loc[common_samples]
                
                # 5. 清理列名中的特殊字符
                geno_cleaned_cols = self.clean_column_names(geno_filtered.columns)
                geno_filtered.columns = geno_cleaned_cols
                
                pheno_cols = pheno_filtered.columns.tolist()
                pheno_cleaned_cols = self.clean_column_names(pheno_cols)
                pheno_filtered.columns = pheno_cleaned_cols
                
                # 获取表型列名（第二列）
                trait_col = pheno_filtered.columns[1]
                
                # 6. 表型标准化（如果启用）
                if kwargs.get('standardize_phenotype', False):
                    self.logger.info("正在对表型数据进行标准化...")
                    pheno_filtered, scaler_params = self.standardize_phenotype(pheno_filtered, trait_col)
                    
                    # 保存标准化参数
                    import json
                    scaler_file = f"{out_prefix}_phenotype_scaler.json"
                    with open(scaler_file, 'w') as f:
                        json.dump(scaler_params, f, indent=2)
                    self.logger.info(f"标准化参数已保存到: {scaler_file}")
                
                # 7. 保存最终文件（只有一套）
                final_pheno_file = f"{out_prefix}_phenotype.csv"
                final_geno_file = f"{out_prefix}_genotype.csv"
                
                pheno_filtered.to_csv(final_pheno_file, index=False)
                geno_filtered.to_csv(final_geno_file)
                
                self.logger.info(f"\n最终表型文件: {final_pheno_file}")
                self.logger.info(f"最终基因型文件: {final_geno_file}")
                self.logger.info(f"样品数量: {len(common_samples)}")
                self.logger.info(f"SNP数量: {len(geno_filtered.columns)}")
                if scaler_params and scaler_params.get('applied'):
                    self.logger.info(f"表型已标准化: 均值={scaler_params['mean']:.4f}, 标准差={scaler_params['std']:.4f}")
                
            else:
                if kwargs.get('skip_match'):
                    self.logger.info("跳过表型基因型匹配步骤")
                elif not kwargs.get('pheno'):
                    self.logger.warning("未提供表型文件，跳过表型基因型匹配步骤")
                elif not geno_matrix_file:
                    self.logger.warning("没有可用的基因型矩阵文件，跳过表型基因型匹配步骤")
            
            self.logger.info("\n处理流程完成!")
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
        
        return 0


# 向后兼容的独立函数
def vcf_to_plink(vcf_file, out_prefix, plink_path="plink"):
    """向后兼容函数"""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.vcf_to_plink(vcf_file, out_prefix)

def extract_snps(bfile, extract_file, out_prefix, plink_path="plink"):
    """向后兼容函数"""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.extract_snps(bfile, extract_file, out_prefix)

def convert_bfile_to_ped(bfile, out_prefix, plink_path="plink"):
    """向后兼容函数"""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.convert_bfile_to_ped(bfile, out_prefix)

def convert_to_matrix(fileprefix, out_file=None):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.convert_to_matrix(fileprefix, out_file)

def process_snp_dir(bfile, snp_dir, out_dir, plink_path="plink"):
    """向后兼容函数"""
    processor = GenomicDataProcessor(plink_path=plink_path)
    return processor.process_snp_dir(bfile, snp_dir, out_dir)

def convert_phenotype(pheno_file, out_file=None, trait_name=None):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.convert_phenotype(pheno_file, out_file, trait_name)

def match_genotype_phenotype(pheno_df, geno_file, out_prefix):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.match_genotype_phenotype(pheno_df, geno_file, out_prefix)

def check_special_chars(column_names):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.check_special_chars(column_names)

def clean_column_names(column_names):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.clean_column_names(column_names)

def process_file(file_path, output_path):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.process_file(file_path, output_path)

def load_matrix(matrix_file):
    """向后兼容函数"""
    processor = GenomicDataProcessor()
    return processor.load_matrix(matrix_file)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基因组预测数据处理一体化工具')
    # SNP提取参数
    parser.add_argument('--bfile', help='输入 PLINK 二进制格式文件 (BED/BIM/FAM) 的前缀 (不含扩展名)')
    parser.add_argument('--vcf', help='VCF格式文件路径')
    parser.add_argument('--ped-file', help='PLINK PED文件路径')
    parser.add_argument('--map-file', help='PLINK MAP文件路径')
    parser.add_argument('--extract', help='包含要提取的SNP ID的文件')
    parser.add_argument('--snp-dir', help='包含SNP列表文件的目录')
    parser.add_argument('--plink', default='plink', help='PLINK可执行文件路径')
    parser.add_argument('--direct', action='store_true', help='直接将整个bfile转换为矩阵，不进行SNP提取')
    parser.add_argument('--plink-out', help='PLINK二进制文件输出前缀（VCF转换时使用）')
    parser.add_argument('--load', action='store_true', help='加载并显示矩阵信息')
    parser.add_argument('--matrix-file', help='指定已有的基因型矩阵文件路径，跳过矩阵生成步骤')
    # 表型文件参数
    parser.add_argument('--pheno', help='表型TXT文件路径')
    # 输出参数
    parser.add_argument('--out-prefix', required=True, help='输出文件前缀')
    parser.add_argument('--skip-clean', action='store_true', help='跳过数据清理步骤')
    parser.add_argument('--skip-match', action='store_true', help='跳过表型基因型匹配步骤')
    parser.add_argument('--skip-matrix', action='store_true', help='跳过SNP提取和矩阵转换步骤')
    
    args = parser.parse_args()
    
    # 创建数据处理器并执行处理
    processor = GenomicDataProcessor(plink_path=args.plink)
    
    # 将参数转换为字典
    kwargs = vars(args)
    
    return processor.process_genomic_data(**kwargs)


if __name__ == '__main__':
    sys.exit(main())