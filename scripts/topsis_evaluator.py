#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于熵权法的TOPSIS模型综合评价

可作为工具模块调用或独立命令行工具使用
"""

import pandas as pd
import numpy as np
import argparse
import os


class TOPSISEvaluator:
    """TOPSIS综合评价器"""
    
    def __init__(self, logger=None):
        """
        初始化TOPSIS评价器
        
        参数:
            logger: 日志记录器，如果不提供则使用print输出
        """
        self.logger = logger
    
    def _log(self, message):
        """统一的日志输出方法"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def entropy_weight_method(self, data):
        """熵权法计算权重"""
        # 归一化
        P = data / data.sum(axis=0)
        P = P.replace(0, 1e-12)
        n = data.shape[0]
        k = 1.0 / np.log(n)
        E = -k * (P * np.log(P)).sum(axis=0)
        d = 1 - E
        w = d / d.sum()
        return w
    
    def topsis(self, data, weights, criteria_types, min_transform='reciprocal'):
        """TOPSIS综合评价方法"""
        data_proc = data.copy()
        for i, ctype in enumerate(criteria_types):
            if ctype == 'min':
                col = data_proc.iloc[:, i]
                if min_transform == 'reciprocal':
                    data_proc.iloc[:, i] = 1.0 / (col + 1e-12)
                elif min_transform == 'neglog':
                    # 稳定性越小越好，用 -log(x) 抑制极小值差异的放大
                    data_proc.iloc[:, i] = -np.log(col + 1e-12)
                elif min_transform == 'minmax_inv':
                    # 线性反向到 [0,1]，作为一个更温和的替代
                    mx, mn = col.max(), col.min()
                    data_proc.iloc[:, i] = (mx - col) / (mx - mn + 1e-12)
                else:
                    raise ValueError(f'Unsupported min_transform: {min_transform}')
        # 归一化
        norm = np.sqrt((data_proc ** 2).sum(axis=0))
        data_norm = data_proc / norm
        # 加权
        data_weighted = data_norm * weights
        # 理想/负理想解
        ideal = data_weighted.max(axis=0)
        nadir = data_weighted.min(axis=0)
        # 距离
        D_pos = np.sqrt(((data_weighted - ideal) ** 2).sum(axis=1))
        D_neg = np.sqrt(((data_weighted - nadir) ** 2).sum(axis=1))
        # 得分
        scores = D_neg / (D_pos + D_neg)
        return scores, data_proc, data_norm
    
    def evaluate(self, input_file, output_file, criteria=None, criteria_types=None, 
                 simple_output=None, manual_weights='0.8,0.2', min_transform='reciprocal',
                 use_entropy_weights=False):
        """
        执行TOPSIS评价
        
        参数:
            input_file: 输入CSV文件路径
            output_file: 输出CSV文件路径
            criteria: 评价指标名列表
            criteria_types: 指标类型列表（max/min）
            simple_output: 精简版输出文件路径
            manual_weights: 手动权重字符串
            min_transform: min型指标正向化方式
            use_entropy_weights: 是否使用熵权法自动赋权
            
        返回:
            处理后的DataFrame
        """
        # 设置默认值
        if criteria is None:
            criteria = ['Test Pearson', 'Test Pearson (std)']
        if criteria_types is None:
            criteria_types = ['max', 'min']
        
        # 读取数据
        df = pd.read_csv(input_file)
        
        # 验证参数
        if len(criteria) != len(criteria_types):
            raise ValueError("指标数量和类型数量必须一致")
        
        # 只保留有效行
        data = df[criteria].copy()
        data = data.loc[~((data == 0) | (data.isna())).all(axis=1)]
        df = df.loc[data.index].reset_index(drop=True)
        data = data.reset_index(drop=True)
        
        # 权重设置
        if use_entropy_weights:
            # 使用熵权法自动赋权
            weights = self.entropy_weight_method(data)
            self._log(f"使用熵权法自动赋权: {dict(zip(criteria, weights.round(4)))}")
        else:
            # 使用手动设置权重（默认8:2）
            weights = [float(w.strip()) for w in manual_weights.split(',')]
            if len(weights) != len(criteria):
                raise ValueError("权重数量与指标数量必须一致")
            # 归一化权重
            weights = np.array(weights) / sum(weights)
            self._log(f"使用手动设置权重: {dict(zip(criteria, weights.round(4)))}")
        
        # TOPSIS综合评价，并返回正向化和归一化结果
        scores, data_proc, data_norm = self.topsis(data, weights, criteria_types, min_transform)
        df['TOPSIS_Score'] = scores
        
        # 只保留有效分数
        df = df[np.isfinite(df['TOPSIS_Score'])].reset_index(drop=True)
        
        # 按分数排序
        df = df.sort_values('TOPSIS_Score', ascending=False).reset_index(drop=True)
        df['TOPSIS_Rank'] = np.arange(1, len(df) + 1)
        
        # 添加中间转化结果
        for i, c in enumerate(criteria):
            df[f'{c}_positive'] = data_proc.iloc[:, i]
            df[f'{c}_norm'] = data_norm.iloc[:, i]
            df[f'{c}_weight'] = weights[i]
        
        # 只输出需要的列（完整版）
        output_cols = ['Model'] + criteria
        output_cols += [f'{c}_positive' for c in criteria]
        output_cols += [f'{c}_norm' for c in criteria]
        output_cols += [f'{c}_weight' for c in criteria]
        output_cols += ['TOPSIS_Score', 'TOPSIS_Rank']
        
        df_out = df[output_cols]
        df_out.to_csv(output_file, index=False)
        self._log(f"已保存带TOPSIS综合评价的结果到: {output_file}")
        self._log(str(df[['Model', 'TOPSIS_Score', 'TOPSIS_Rank']].sort_values('TOPSIS_Rank')))
        
        # 生成精简版结果
        if simple_output:
            simple_cols = ['Model'] + criteria + ['TOPSIS_Score', 'TOPSIS_Rank']
            df_simple = df[simple_cols]
            df_simple.to_csv(simple_output, index=False)
            self._log(f"已保存精简版TOPSIS结果到: {simple_output}")
        
        return df


def entropy_weight_method(data):
    """熵权法计算权重（向后兼容函数）"""
    evaluator = TOPSISEvaluator()
    return evaluator.entropy_weight_method(data)


def topsis(data, weights, criteria_types, min_transform='reciprocal'):
    """TOPSIS综合评价（向后兼容函数）"""
    evaluator = TOPSISEvaluator()
    return evaluator.topsis(data, weights, criteria_types, min_transform)


def main():
    parser = argparse.ArgumentParser(description="基于熵权法的TOPSIS模型综合评价")
    parser.add_argument('--input', type=str, required=True, help='输入csv文件（如model_comparison.csv）')
    parser.add_argument('--output', type=str, required=True, help='输出csv文件')
    parser.add_argument('--criteria', type=str, default='Test Pearson,Test Pearson (std)', help='用逗号分隔的评价指标名')
    parser.add_argument('--criteria_types', type=str, default='max,min', help='用逗号分隔的指标类型（max/min）')
    parser.add_argument('--simple_output', type=str, default=None, help='精简版输出文件名（只保留原始指标和TOPSIS结果）')
    parser.add_argument('--manual_weights', type=str, default='0.8,0.2', help='手动设置权重，用逗号分隔（默认0.8,0.2表示精度:稳定性=8:2）')
    parser.add_argument('--min_transform', type=str, default='reciprocal',
                        choices=['reciprocal', 'neglog', 'minmax_inv'],
                        help='min型指标的正向化方式（默认 reciprocal，可选 neglog、minmax_inv）')
    parser.add_argument('--use_entropy_weights', action='store_true',
                        help='使用熵权法自动赋权，否则使用手动权重')
    args = parser.parse_args()

    # 解析参数
    criteria = [c.strip() for c in args.criteria.split(',')]
    criteria_types = [c.strip() for c in args.criteria_types.split(',')]

    # 创建TOPSIS评价器并执行评价
    evaluator = TOPSISEvaluator()
    evaluator.evaluate(
        input_file=args.input,
        output_file=args.output,
        criteria=criteria,
        criteria_types=criteria_types,
        simple_output=args.simple_output,
        manual_weights=args.manual_weights,
        min_transform=args.min_transform,
        use_entropy_weights=args.use_entropy_weights
    )

if __name__ == '__main__':
    main()
