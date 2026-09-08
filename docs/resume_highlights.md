# GPSE 简历项目描述

> 本文档用于简历投递场景，包含完整版与精简版两种篇幅，可直接复制使用。
> 所有亮点均可在仓库中找到对应实现，经得起面试追问。

---

## 完整版

**`GPSE`** : **可复现的基因组选择 Stacking 集成预测管线（个人主导设计与开发）**

**面向园艺作物基因组育种场景的端到端 ML 基础设施，让每一次模型交付可复现、可对齐、可跨项目复用**

https://github.com/xsx123123/GPSE （PyPI: `pip install gpse`）

**项目亮点：**

- **端到端子命令架构**：`gpse {convert, train, predict, batch, tools}` 五层子命令贯穿全流程 —— 从原始 `VCF/PLINK` 数据转换、QC 过滤、LD 修剪、基因型-表型匹配，到 `Optuna` 超参优化、多模型评估、`TOPSIS` 排序与 `Stacking` 集成预测，一条命令链完成育种分析交付；
- **可移植的 SNP 身份契约**：全链路统一使用 `chr<chrom>_<chromStart>_<chromEnd>` 规范 SNP ID（UCSC 零基半开坐标），训练时持久化有序 `feature_manifest.json`，预测端按清单重排特征、报告缺失/多余位点、缺省按缺失码 `3` 填充并输出 `alignment.json` 对齐报告，从契约层面解决训练-推理特征错配，实现跨用户、跨项目的模型安全复用；
- **确定性可复现设计**：CV 折划分使用独立种子 `random.Random(42)`，重跑同输入产生完全一致的折布局；每个 CV fold 与 Optuna trial 均新建独立估计器杜绝状态串扰；`6 物种 × 18 性状 × 14 模型 × 50 次重复 × 5 折 CV` 的历史结果提供 SHA256 输入校验 + 指标基线 TSV 双级复现验证；
- **配置驱动的模型注册表**：`14+` 机器学习算法（`RF / XGBoost / LightGBM / CatBoost / SVR / MLP / ElasticNet / GBLUP` 等）统一注册，原生支持回归（连续性状）与分类（离散性状）双任务模式；外部工具（`PLINK / Beagle / Java`）路径全部 YAML 声明，项目级 `gpse.yaml` 与本机 `gpse.local.yaml` 分层隔离；
- **多准则模型治理**：内置 `TOPSIS` + 熵权法对多指标（R²、RMSE、训练时长等）综合排序，自动遴选 Top-N 基模型构建 `Stacking` 集成（折级元特征 + 元模型），避免单指标选型的片面性；
- **批量自适应调度**：`batch` 模块提供多性状批量运行器与自适应调度器，支持并行 repeat 执行、代表性重复选取与结果合并，配合复现 SOP 将大规模历史实验迁移成本降至最低。

---

## 精简版（篇幅紧张时使用）

**`GPSE`** : **可复现的基因组选择 Stacking 集成预测管线（个人主导设计与开发）**
https://github.com/xsx123123/GPSE （PyPI: `pip install gpse`）

- 设计并实现 `convert / train / predict / batch / tools` 端到端子命令管线，覆盖 `VCF/PLINK` 数据转换、QC、`Optuna` 超参优化到 `Stacking` 集成预测全流程；
- 提出 `chr<chrom>_<chromStart>_<chromEnd>` 规范 SNP 身份契约 + `feature_manifest.json` 特征清单对齐机制，实现模型跨用户、跨项目安全复用；
- 以独立 CV 种子与逐 fold 独立估计器保证确定性复现，完成 `6 物种 × 18 性状 × 14 模型 × 50 重复 × 5 折` 历史结果的双级（SHA256 + 指标基线）复现验证；
- 集成 `TOPSIS` 熵权多准则排序与 Top-N `Stacking` 集成，支持 `14+` 算法、回归/分类双任务，批量模块支持多性状自适应并行调度。

---

## 面试追问速查（亮点 → 实现位置）

| 亮点 | 关键实现 |
|------|----------|
| SNP 身份契约 | `gpse/utils/snp_ids.py`；manifest 写入于 `convert/genotype_matrix.py` 与 `train` 的 `feature_manifest.json` |
| 预测对齐报告 | `gpse predict` 输出 `predictions.csv` + `predictions.alignment.json` |
| 确定性 CV | 折划分固定 `random.Random(42)`；每 fold/trial 新建估计器 |
| 双级复现验证 | `docs/6species_reproduction_validation.md`、`docs/6species_input_sha256.txt`、`docs/6species_result_baseline.tsv` |
| 配置分层 | `gpse.yaml`（项目级）/ `gpse.local.yaml`（本机，gitignored）；`gpse/config/topsis.yaml` |
| 批量调度 | `gpse/batch/adaptive_scheduler.py`、`runner.py`、`merge.py` |
| 线程治理 | `gpse/cli.py` 顶层预设 BLAS/MKL/OpenMP 线程环境变量；`threadpoolctl` 包裹 `fit()` |
