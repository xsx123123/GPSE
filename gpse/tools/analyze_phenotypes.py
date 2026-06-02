#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import pandas as pd
import numpy as np

def analyze_phenotype_file(file_path):
    """
    分析单个表型文件，判断性状类型
    """
    try:
        # 读取文件，假设是制表符分隔
        df = pd.read_csv(file_path, sep='\t')
        
        # 获取文件名作为性状名
        trait_name = os.path.basename(file_path).replace('.txt', '')
        
        # 假设第二列是表型值（第一列通常是ID）
        if len(df.columns) < 2:
            return trait_name, "Error: Less than 2 columns", {}, []
            
        pheno_col = df.iloc[:, 1]
        
        # 去除NA值
        pheno_col = pheno_col.replace('NA', np.nan).dropna()
        
        if len(pheno_col) == 0:
            return trait_name, "Empty (All NA)", {}, []

        # 尝试转换为数值
        try:
            pheno_numeric = pd.to_numeric(pheno_col)
            is_numeric = True
        except ValueError:
            is_numeric = False
            pheno_numeric = pheno_col

        # 统计唯一值
        unique_values = pheno_numeric.unique()
        n_unique = len(unique_values)
        total_count = len(pheno_numeric)
        
        stats = {
            'n_samples': total_count,
            'n_unique': n_unique,
            'is_numeric': is_numeric,
            'min': pheno_numeric.min() if is_numeric else None,
            'max': pheno_numeric.max() if is_numeric else None
        }

        # 判断逻辑
        trait_type = "Unknown"
        reason = ""

        if not is_numeric:
            trait_type = "Classification"
            reason = "Non-numeric values detected"
        else:
            # 检查是否有小数
            is_integer = np.all(np.mod(pheno_numeric, 1) == 0)
            
            if not is_integer:
                trait_type = "Continuous"
                reason = "Contains decimal values"
            else:
                # 整数情况，根据唯一值数量判断
                # 阈值设为 15，如果只有少于15种不同的整数值，通常是等级评分
                if n_unique <= 15:
                    trait_type = "Classification"
                    reason = f"Integer values with few unique levels ({n_unique} <= 15)"
                else:
                    trait_type = "Continuous"
                    reason = f"Integer values with many unique levels ({n_unique} > 15)"

        return trait_name, trait_type, stats, reason

    except Exception as e:
        return os.path.basename(file_path), f"Error: {str(e)}", {}, []

def main():
    pheno_dir = "melon_phenotyp_split"
    
    # 创建输出文件夹
    continuous_dir = "melon_phenotyp_continuous"
    classification_dir = "melon_phenotyp_classification"
    
    os.makedirs(continuous_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)
    
    # 获取所有txt文件
    files = glob.glob(os.path.join(pheno_dir, "*.txt"))
    files.sort()
    
    print(f"{'Trait':<30} {'Type':<15} {'Samples':<10} {'Unique':<10} {'Reason'}")
    print("-" * 90)
    
    summary = []
    continuous_count = 0
    classification_count = 0
    
    for f in files:
        name, p_type, stats, reason = analyze_phenotype_file(f)
        n_samples = stats.get('n_samples', 0)
        n_unique = stats.get('n_unique', 0)
        
        print(f"{name:<30} {p_type:<15} {n_samples:<10} {n_unique:<10} {reason}")
        
        # 根据类型复制文件到对应文件夹
        if p_type == 'Continuous':
            dest_path = os.path.join(continuous_dir, os.path.basename(f))
            shutil.copy2(f, dest_path)
            continuous_count += 1
        elif p_type == 'Classification':
            dest_path = os.path.join(classification_dir, os.path.basename(f))
            shutil.copy2(f, dest_path)
            classification_count += 1
        
        summary.append({
            'Trait': name,
            'Type': p_type,
            'File': f
        })
    
    print("-" * 90)
    print(f"\n文件分类完成:")
    print(f"  - 连续性状: {continuous_count} 个文件 → {continuous_dir}/")
    print(f"  - 分类性状: {classification_count} 个文件 → {classification_dir}/")
    
    print("\n建议的分析策略:")
    for item in summary:
        if item['Type'] == 'Continuous':
            print(f"  - {item['Trait']}: 使用回归模型 (Regression)")
        elif item['Type'] == 'Classification':
             print(f"  - {item['Trait']}: 使用分类模型 (Classification)")

if __name__ == "__main__":
    main()
