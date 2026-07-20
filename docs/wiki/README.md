# GPSE Wiki

> GPSE (Genomic Prediction with Stacking Ensemble) —— 面向园艺作物基因组选择的机器学习流水线。

本 Wiki 描述当前版本（v0.0.1, Alpha）的软件定位、子命令功能与 Python API。

## 目录

| 章节 | 内容 |
|------|------|
| [01. 软件总览](01-overview.md) | 软件作用、核心特性、整体架构、依赖 |
| [02. `gpse convert`](02-cli-convert.md) | 数据转换子命令：pipeline / qc / recode / deps 四种模式 |
| [03. `gpse train`](03-cli-train.md) | 训练子命令：超参优化、重复交叉验证、Stacking 集成 |
| [04. `gpse predict`](04-cli-predict.md) | 预测子命令：特征对齐、覆盖度校验、表型预测 |
| [05. 配置文件](05-configuration.md) | `gpse.yaml` / 包内置配置 / TOPSIS 配置 |
| [06. API 参考](06-api-reference.md) | 各模块公开类与函数签名 |

## 快速开始

```bash
# 1. 检查依赖
gpse convert --check-deps

# 2. 数据转换（VCF → 训练矩阵）
gpse convert --vcf data.vcf.gz --pheno pheno.csv --out-prefix out/run

# 3. 训练（回归，Optuna 优化 + Stacking）
gpse train --task_type regression \
  --geno_file out/run_geno.csv --pheno_file out/run_pheno.csv \
  --target_trait yield --models xgboost_reg lightgbm_reg \
  --trials 100 --n_repeats 100 --use_stacking --threads 16

# 4. 预测新样本
gpse predict --model optimization_results_v2/deployment_ensemble \
  --vcf-file new_samples.vcf.gz --out predictions.csv
```
