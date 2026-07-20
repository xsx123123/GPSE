# 01. 软件总览

## 软件作用

GPSE (Genomic Prediction with Stacking Ensemble) 是一个基因组选择（Genomic Selection）机器学习流水线，面向园艺作物育种场景。它从原始 VCF/PLINK 基因型数据出发，完成：

1. **数据转换**：VCF → PLINK → PED/MAP → 加性数值矩阵 → 表型匹配；
2. **质量控制**：缺失率/MAF 过滤、Beagle 插补（可选）、LD pruning；
3. **模型训练**：Optuna 超参优化 + 重复交叉验证 + TOPSIS 多指标排名；
4. **集成学习**：严格 OOF（out-of-fold）Stacking，防止数据泄漏；
5. **预测部署**：canonical SNP ID 特征对齐，用训练产物预测新样本表型。

## 核心特性

- **14+ 回归算法 + 6 分类算法**：elasticnet、GBDT、SVR、MLP、KNN、RF、XGBoost、AdaBoost、LightGBM、CatBoost、KernelRidge、GBLUP 基线等；
- **双任务模式**：`--task_type regression | classification`；
- **可复现 / 防泄漏**：固定 hold-out + train-only CV + 严格 OOF stacking（详见 `train_pipeline.md`）；
- **跨用户模型复用**：canonical SNP ID（`chr<chrom>_<chromStart>_<chromEnd>`）与 `feature_manifest.json`；
- **并行训练**：`--threads` 自动推导 `n_jobs / max_workers / repeat_workers`；
- **TOPSIS 综合评价**：熵权法/手动权重，兼顾精度与稳定性。

## 整体架构

```
gpse/
├── cli.py            # CLI 入口（argparse + rich-argparse），路由到子命令
├── convert/          # gpse convert：数据转换 / QC / recode / deps
├── train/            # gpse train：训练、优化、CV、Stacking、TOPSIS
├── predict/          # gpse predict：特征对齐与表型预测
├── models/           # 模型注册与 Optuna 搜索空间（回归/分类 optimizer）
├── tasks/            # 任务层（分类标签编码、分类指标）
├── config/           # 包内置 YAML 配置与模型常量
├── tools/            # 内部辅助脚本（表型分析）
└── utils/            # 配置加载、SNP ID、feature manifest、并行、日志
```

## 依赖

- **Python ≥ 3.10**（Poetry 构建）；核心库：scikit-learn、xgboost、lightgbm、catboost、optuna、cyvcf2、ngboost、pandas/numpy、loguru/rich、PyYAML；
- **外部工具**：PLINK ≥ 1.9（必需）；java + Beagle jar（仅插补时需要）。检查方式：`gpse convert --check-deps`。
