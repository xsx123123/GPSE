<p align="center">
  <img src="logo.png" alt="GPSE Logo" width="250">
</p>

<h1 align="center">GPSE（Genomic Prediction with Stacking Ensemble）</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-Alpha-orange" alt="Status">
</p>

<p align="center"><strong>用于园艺作物的基因组预测与 Stacking Ensemble 框架。</strong></p>

<p align="center">
  English: <a href="../README.md">README.md</a> | 简体中文
</p>

GPSE 是一个面向基因组选择与预测的机器学习流水线，覆盖从原始基因组数据（VCF/PLINK）预处理，到超参数优化、模型评估、TOPSIS 排名和 Stacking Ensemble 预测的完整流程。

## 🌟 核心特性

* **完整数据流水线**：支持 VCF 到 PLINK 二进制格式转换、SNP 提取、数值矩阵生成，以及基因型与表型的精确匹配。
* **广泛算法支持**：支持 14+ 种机器学习算法，包括 Random Forest、XGBoost、LightGBM、CatBoost、SVR、MLP、ElasticNet 等。
* **双任务模式**：原生支持 **回归**（连续性状）与 **分类**（离散性状）。
* **自动超参数调优**：集成 Optuna，支持高效的多线程参数优化。
* **稳健评估**：通过多次重复 K 折交叉验证提升稳定性与可复现性。
* **模型排序与筛选**：内置 **TOPSIS**（Technique for Order of Preference by Similarity to Ideal Solution）并结合熵权法进行多指标排名。
* **Stacking Ensemble**：自动集成 Top-N 表现最佳的模型以提升预测精度。

## 🛠️ 安装

### 前置要求

* Python >= 3.10
* [PLINK 1.9](https://www.cog-genomics.org/plink/)（用于基因组格式转换与 SNP 提取）

### 使用 Poetry 安装

```bash
git clone https://github.com/xsx123123/GPSE.git
cd gpse
poetry install
```

### 使用 pip 安装

```bash
pip install .
```

## 🚀 使用方法

GPSE 使用子命令结构：`gpse {convert,train,predict}`。

### 命令行示意

![GPSE 命令行帮助](command.png)

### 1. 数据转换与 QC（`gpse convert`）

`gpse convert` 负责所有基因型/表型预处理：格式转换、QC 过滤、LD pruning 和样本匹配，输出可直接用于训练的数值矩阵。

#### 流程概览

```text
Input                        Processing                     Output
─────                        ──────────                     ──────
samples.vcf            →  VCF → PLINK BED              →  {prefix}_genotype.csv
phenotype.txt/.csv     →  PED/MAP → numeric (0/1/2)    →  {prefix}_phenotype.csv
                            SNP filtering                   {prefix}_phenotype_scaler.json
                            Sample ID matching                 (仅在 --standardize-phenotype 时)
                            Column name cleaning
                            Phenotype Z-score (optional)
```

基因型编码：`00→0`（纯合参考），`01/10→1`（杂合），`11→2`（纯合替代），缺失→`3`。

#### 1.1 VCF + 表型 → 训练数据

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --direct \
    --out-prefix data/train
```

输出：

* `data/train_genotype.csv` - 数值矩阵（行=样本，列=SNP，值为 0/1/2）
* `data/train_phenotype.csv` - 清洗并匹配后的表型文件（ID + trait 值）

#### 1.2 表型标准化

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --direct \
    --standardize-phenotype \
    --out-prefix data/train
```

额外输出：`data/train_phenotype_scaler.json`（用于预测阶段逆标准化）

#### 1.3 提取指定 SNP

```bash
# 从 PLINK 二进制输入
gpse convert \
    --bfile plink_data \
    --extract snp_list.txt \
    --pheno phenotype.txt \
    --out-prefix data/train

# 从 SNP 列表目录批量提取
gpse convert \
    --bfile plink_data \
    --snp-dir snp_lists/ \
    --out-prefix data/train
```

#### 1.4 使用已有矩阵（跳过矩阵生成）

```bash
gpse convert \
    --matrix-file existing_genotype.csv \
    --pheno phenotype.txt \
    --out-prefix data/matched
```

#### 1.5 QC 过滤 + LD pruning

独立执行基因型层面的 QC（缺失率、MAF）、可选 Beagle imputation 和 LD pruning。

```bash
gpse convert \
    --run-qc \
    --input-prefix plink_data \
    --out-prefix data/qc_data \
    --snpmaxmiss 0.1 \
    --samplemaxmiss 0.1 \
    --maf 0.05 \
    --r2-cutoff 0.2
```

带 Beagle imputation：

```bash
gpse convert \
    --run-qc \
    --input-prefix plink_data \
    --out-prefix data/qc_data \
    --impute \
    --beagle-jar-path /path/to/beagle.jar
```

输出：`data/qc_data_pruned.bed/bim/fam`（可直接进入矩阵转换）

#### 1.6 PED/MAP 重编码为数值矩阵

```bash
gpse convert --recode-prefix plink_data
# 输出：plink_data.geno
```

#### 1.7 检查外部依赖

```bash
gpse convert --check-deps
```

#### 1.8 重命名表型列

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --trait-name Fruit_Weight \
    --direct \
    --out-prefix data/train
```

### 2. 模型训练（`gpse train`）

#### 2.1 使用预处理后的数据训练

```bash
gpse train \
    --geno_file data/train_genotype.csv \
    --pheno_file data/train_phenotype.csv \
    --target_trait Fruit_Weight \
    --task_type regression \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --use_stacking \
    --top_n_models 5 \
    --n_jobs 2 \
    --max_workers 4 \
    --results_dir output_results/
```

#### 2.2 一键式：预处理 + 训练

```bash
gpse train \
    --enable_preprocess \
    --preprocess_prefix data/train \
    --vcf_file samples.vcf \
    --raw_pheno_file phenotype.txt \
    --target_trait Fruit_Weight \
    --task_type regression \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --use_stacking \
    --results_dir output_results/
```

#### 2.3 仅预处理（不训练）

```bash
gpse train \
    --preprocess_only \
    --preprocess_prefix data/train \
    --vcf_file samples.vcf \
    --raw_pheno_file phenotype.txt \
    --target_trait Fruit_Weight
```

#### 2.4 分类任务

```bash
gpse train \
    --geno_file genotype.csv \
    --pheno_file phenotype.csv \
    --target_trait Disease_Resistance \
    --task_type classification \
    --n_classes 3 \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --results_dir classification_results/
```

### 3. 表型分析

快速分析表型数据，判断更适合回归还是分类。

```bash
python -m gpse.tools.analyze_phenotypes
```

### 4. 查看帮助

```bash
gpse --help
gpse --version
gpse convert --help
gpse train --help
```

## 📥 输入与输出格式

当前 GPSE 对 `gpse convert` 和 `gpse train` 定义了完整 I/O 约束。`gpse predict` 已有命令入口，但预测流程尚未实现。

### `gpse convert` 输入

`gpse convert` 用于准备可训练的基因型和表型文件。

| 输入类型 | 参数 | 格式要求 |
| --- | --- | --- |
| VCF | `--vcf samples.vcf` | 标准 VCF 文件，GPSE 会调用 PLINK 转成二进制 PLINK 文件。 |
| PLINK binary | `--bfile prefix` | 需要同时存在 `prefix.bed`、`prefix.bim`、`prefix.fam`。 |
| PLINK text | `--ped-file file.ped --map-file file.map` | 标准 PED/MAP 文件。 |
| 已有矩阵 | `--matrix-file genotype.csv` | 第一列为样本 ID 的 CSV 基因型矩阵。 |
| SNP 列表 | `--extract snp_list.txt` | 每行一个 SNP ID，传给 PLINK `--extract`。 |
| SNP 列表目录 | `--snp-dir snp_lists/` | 目录下包含多个 `.txt` SNP 列表文件。 |

表型文件通过 `--pheno` 传入。文件应包含表头，第一列为样本 ID，第二列为目标性状。GPSE 会先尝试按 tab 读取，失败后再按逗号读取。转换时只保留前两列。缺失值（`NaN` 和字符串 `NA`）会被删除。可通过 `--trait-name` 重命名目标性状列。

### `gpse convert` 输出

仅生成基因型矩阵时，输出：

```text
{out_prefix}.csv
```

格式示例：

```csv
ID,SNP1,SNP2,SNP3
sample1,0,1,2
sample2,1,3,0
```

基因型编码：

| 复合基因型 | 编码值 |
| --- | --- |
| `00` | `0` |
| `01` 或 `10` | `1` |
| `11` | `2` |
| 缺失或未知 | `3` |

启用表型/基因型匹配后，输出推荐训练文件：

```text
{out_prefix}_genotype.csv
{out_prefix}_phenotype.csv
```

匹配后的表型文件示例：

```csv
ID,TraitName
sample1,12.3
sample2,9.8
```

启用 `--standardize-phenotype` 时，还会输出：

```text
{out_prefix}_phenotype_scaler.json
```

QC 模式（`--run-qc`）会输出 PLINK 前缀文件：

```text
{out_prefix}_raw.bed/.bim/.fam
{out_prefix}_qc.bed/.bim/.fam
{out_prefix}_qc_filled.bed/.bim/.fam
{out_prefix}_qc_filled.prune.in
{out_prefix}_pruned.bed/.bim/.fam
{out_prefix}_qc.log
```

Recode 模式（`--recode-prefix prefix`）输出：

```text
prefix.geno
```

### `gpse train` 输入

`gpse train` 消费的是 CSV 文件，通常直接来自 `gpse convert` 的输出。

| 参数 | 要求 |
| --- | --- |
| `--geno_file` | 需要包含 `ID` 列的 CSV 基因型矩阵；如果没有 `ID`，会尝试 `--cv_id_column`。 |
| `--pheno_file` | 与基因型文件共享同一 ID 列的 CSV 表型表。 |
| `--target_trait` | 表型表中的目标性状列名。 |
| `--task_type` | `regression` 或 `classification`，默认 `regression`。 |
| `--n_classes` | 分类任务必填，且必须至少为 2。 |

推荐训练输入：

```bash
gpse train \
    --geno_file data/train_genotype.csv \
    --pheno_file data/train_phenotype.csv \
    --target_trait Fruit_Weight
```

训练时，GPSE 只保留 genotype 和 phenotype 中共同存在的样本 ID，并按统一顺序排序；所有非 ID 的 genotype 列都会作为特征，内部会规范化为 `feature_0`、`feature_1` 等名称。

分类标签可以是字符串，也可以是不连续数值。GPSE 会将其编码为连续整数类，并保存编码器：

```text
{results_dir}/label_encoder.pkl
```

如果未提供 `--cv_file`，GPSE 会自动生成：

```text
{results_dir}/cv_folds/{target_trait}_cv_{n_repeats}x{n_splits}.csv
```

CV 文件包含样本 ID 作为索引，以及每个 repeat 对应的折分配列：

```csv
ID,TraitName,cv0,cv1
sample1,12.3,0,3
sample2,9.8,1,4
```

折编号范围为 `0` 到 `n_splits - 1`。

### `gpse train` 输出

默认结果目录为 `optimization_results_v2/`。

主要输出：

```text
{results_dir}/model_comparison.csv
{results_dir}/cv_folds/{target_trait}_cv_{n_repeats}x{n_splits}.csv
{results_dir}/{model_name}/summary_results.json
{results_dir}/{model_name}/repeat_{i}/repeat_results.json
{results_dir}/{model_name}/repeat_{i}/all_predictions.json
```

启用 `--save_models` 时：

```text
{results_dir}/{model_name}/repeat_{i}/fold_{j}_model.pkl
```

代表性模型输出：

```text
{results_dir}/{model_name}/representative_model/model.pkl
{results_dir}/{model_name}/representative_model/info.json
```

回归任务标准化输出：

```text
{results_dir}/phenotype_scaler.json
```

Stacking 输出：

```text
{results_dir}/ensemble_stacking/stacking_ensemble_model.pkl
{results_dir}/ensemble_stacking/stacking_results.pkl
```

TOPSIS 排名输出：

```text
{results_dir}/model_comparison_topsis.csv
{results_dir}/model_comparison_topsis_simple.csv
```

### `gpse predict`

`gpse predict` 目前接受 `--model`、`--geno-file` 和 `--out`，但预测流程尚未实现，因此输入输出格式还未最终定型。

## 📁 源码结构

项目按三个工作流命令组织：`convert`、`train` 和 `predict`。命令相关代码放在对应子包中，共享支撑代码放在 `config`、`models`、`tasks`、`tools` 和 `utils` 中。

### `gpse/`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包元数据，当前导出 `__version__`。 |
| `cli.py` | 顶层命令路由器，仅负责 `gpse {convert,train,predict}` 的参数路由和共享 CLI 参数。 |

### `gpse/config/`

配置常量与打包的 YAML 默认值。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 配置数据类和常量的公共导出。 |
| `constants.py` | 数据类与不可变常量，包括文件名、目录名、精度设置和线程环境变量名。 |
| `_topsis_config.py` | 读取 TOPSIS 配置、验证 criteria/weights、记录运行环境，并保存代表性模型。 |
| `default.yaml` | 默认应用/日志配置。 |
| `software.yaml` | 软件元数据与外部工具定义，用于转换/QC 依赖检查。 |
| `topsis.yaml` | 用于回归/分类模型排序的 TOPSIS 指标、方向和权重。 |

### `gpse/convert/`

`gpse convert` 的实现：基因型/表型转换、QC、LD pruning 和外部工具执行。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出 `GenomicDataProcessor`。 |
| `cli.py` | 转换模式 CLI 解析器与分发器，包含 QC、重编码和依赖检查。 |
| `external.py` | 外部工具发现、路径解析、版本检查和命令执行辅助函数。 |
| `processor.py` | 主转换处理器：VCF/PLINK 转换、SNP 提取、数值矩阵生成、表型清洗、样本匹配和表型标准化。 |
| `qc.py` | PLINK/Beagle QC 工具：格式转换、基因型过滤、imputation、LD pruning 和 PED/MAP 数值重编码。 |

### `gpse/train/`

`gpse train` 的实现：模型训练、重复 CV、优化、模型排名和 Stacking Ensemble。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 对 `GenomicPredictorV2`、`StackingEnsemble` 和 `TOPSISEvaluator` 的懒加载导出。 |
| `cli.py` | `gpse train` 的 CLI 解析器与分发器，包括训练参数、预处理参数、校验和训练流程启动。 |
| `predictor.py` | 训练总控类，负责初始化任务优化器、日志、目录，并绑定训练子模块方法。 |
| `_data_io.py` | 训练数据加载、基因型/表型对齐、表型标准化和逆标准化。 |
| `_model_tools.py` | 模型创建、默认参数查找、参数过滤，以及回归/分类默认指标回退。 |
| `_fold_training.py` | 单折训练、预测、指标计算、折级日志和折级指标平均。 |
| `_ensemble.py` | fold 集成预测逻辑和集成指标计算。 |
| `_optimization.py` | 基于 Optuna 的交叉验证超参数优化。 |
| `_repeat_training.py` | repeat 级编排、并行 repeat 执行、统计汇总、代表性 repeat 选择和结果保存。 |
| `_cv_manager.py` | CV fold 文件的创建/加载，以及基于预定义 CV 分配生成 folds。 |
| `_pipeline.py` | 顶层 `run_all_models` 流程：运行选定模型、生成比较表、执行 TOPSIS，并可选训练 stacking。 |
| `stacking.py` | 可选的 stacking stage：加载基础模型、生成 meta-features、训练 meta-model、评估并保存产物。 |
| `topsis.py` | TOPSIS evaluator 和可选 CLI，用于根据配置好的指标和权重对模型排名。 |

### `gpse/predict/`

`gpse predict` 的实现占位。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Prediction 包标记。 |
| `__main__.py` | 支持 `python -m gpse.predict`。 |
| `cli.py` | 未来预测流程的 CLI 占位，目前会提示 prediction 尚未实现。 |

### `gpse/models/`

模型注册表和优化器/搜索空间定义。这里定义模型如何构建、Optuna 如何采样参数，但不直接运行完整训练流水线。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 回归和分类优化器的懒加载导出。 |
| `regression_model_optimizer.py` | 回归模型注册、Optuna 搜索空间、参数过滤、模型工厂和默认参数。 |
| `classification_model_optimizer.py` | 分类模型注册、Optuna 搜索空间、参数过滤、模型工厂和默认参数。 |
| `model_optimizers.py` | 兼容旧代码的回归优化器导入 shim。新代码建议直接使用 `regression_model_optimizer.py`。 |
| `classification_models.py` | 兼容旧代码的分类优化器导入 shim。新代码建议直接使用 `classification_model_optimizer.py`。 |

### `gpse/tasks/`

训练阶段共享的任务级运行辅助代码。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | `GenomicClassifier` 的懒加载导出。 |
| `classification.py` | 分类任务运行支持：标签编码/解码、概率转标签、分类指标、结果汇总，以及对 `ClassificationModelOptimizer` 的委派。 |

### `gpse/tools/`

独立于主工作流的工具脚本。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | tools 包标记。 |
| `analyze_phenotypes.py` | 用于检查表型分布、辅助判断任务类型的独立分析脚本。 |

### `gpse/utils/`

跨工作流共享的通用工具代码。这里应只放通用支撑代码，train/convert/predict 特有业务逻辑应放到对应子包。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 日志和通用基因组工具的懒加载导出。 |
| `configuration.py` | YAML 配置加载与合并辅助，支持打包默认值和可选的项目/用户覆盖。 |
| `dependency_checker.py` | 通用外部依赖检测与版本检查工具。 |
| `genomic_utils.py` | 通用训练辅助：指标计算、CV 文件辅助、结果表生成、随机种子生成、目录创建、fold 工具和 TOPSIS 封装。 |
| `log_utils.py` | Loguru/Rich 日志初始化、子进程日志设置和日志收集。 |
| `logo.py` | 基于 Rich 的 logo 与欢迎面板渲染。 |
| `print_utils.py` | 可复用的 Rich 表格/面板/列输出工具。 |
| `version.py` | 版本、依赖、系统环境和外部工具信息输出。 |

## 📦 主要依赖

* `scikit-learn`
* `xgboost`
* `lightgbm`
* `catboost`
* `optuna`
* `pandas` & `numpy`
* `rich` & `loguru`（用于美观的 CLI 输出和日志）

## 📝 最近更新

* **线程控制与启动性能**（`2026-06-03`）
  * 在导入 numpy/scipy 之前设置 6 个环境变量，修复 BLAS/MKL 线程池忽略 `--n_jobs` 的问题。
  * 在所有 `model.fit()` 调用周围增加 `threadpoolctl.threadpool_limits()` 作为运行时安全网。
  * 将 `train/`、`models/`、`tasks/` 和 `utils/` 下的 `__init__.py` 改为懒加载，以避免 `gpse --help` 载入整个 ML 栈。
  * 重命名 CLI 参数：`--threads` → `--n_jobs`，`--parallel_jobs` → `--max_workers`。
  * 为 `histgradientboost_reg` 和 `knn_reg` 补充 `--n_jobs`。
  * 修复 `gpse 42` 彩蛋的重复面板和 Rich 标记未渲染问题。

* **导入系统统一**（`2026-06-03`）
  * 移除了 `cli.py` 和训练预测器中的所有 `sys.path` hack。
  * 统一为绝对包路径导入（`from gpse.xxx import ...`）。
  * 为 `config/`、`convert/`、`train/`、`models/`、`tasks/`、`utils/` 和 `tools/` 配置了正确的 `__init__.py` 导出。

* **`GenomicPredictorV2` 模块化重构**（`2026-06-03`）
  * 将单体的 `GenomicPredictorV2` 拆分为 `gpse/train/` 下的多个聚焦模块。
  * 将 TOPSIS 运行时配置迁移到 `gpse/config/_topsis_config.py`，YAML 默认值迁移到 `gpse/config/topsis.yaml`。
  * 将所有中文注释、docstring 和日志翻译为英文。
  * 将 `ModelConfig`、`ClassificationModelConfig`、`NumpyEncoder` 移至 `config/constants.py`。

## 📄 许可证

本项目采用 MIT License，详见 `LICENSE` 文件。

## 👥 作者

* XIAOLIU <1468835852@qq.com>
* JZHANG <zhangjian199567@outlook.com>
