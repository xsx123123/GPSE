# GPSE 待办计划 v3：特征选择与预测可靠性

> 生成日期：2026-07-10
> 更新日期：2026-07-14
> 背景：当前训练流程默认直接使用全部 SNP 构建模型。在 SNP 数量远大于样本数的场景下，部分模型容易出现训练结果异常、泛化不稳、不同 repeat 波动大或 Stacking/TOPSIS 结果难解释的问题。

> 实施状态（2026-07-19）：P0 的训练 SNP manifest、预测列对齐和模型加载兼容层已完成；P1 的 `none` / `variance` / `univariate` / `mutual_info` / `tree_importance` / `lasso` / `elasticnet` / `pca` / `svd` 特征选择和可选 `none` / `mean` 基因型插补已接入 Optuna、fold CV、fold ensemble、representative/deployment artifact 与 stacking OOF。`--select_percentile` 可替代 `--select_k`，按训练 fold 的特征数保留指定比例，并写入 artifact metadata。`GenotypePreprocessor -> model` sklearn Pipeline 已用于 Optuna、CV fold、Stacking 严格 OOF、representative 和 deployment 的构建，并以 artifact v3 持久化；树模型使用 scaler passthrough，尺度敏感模型使用 `StandardScaler`。预测优先调用完整 Pipeline，同时兼容旧组件/tuple。训练结束后额外生成 CSV/JSON/Markdown/HTML 结果 bundle，集中展示 metrics、CV 选择审计和 artifact 路径。

---

## 一、核心判断

使用全部 SNP 不一定必然过拟合。对于 Ridge / GBLUP / ElasticNet 等强正则化模型，全 SNP 建模在基因组预测中可以是合理路线。

但在当前 GPSE 的多模型流程中，同时包含 RF、GBDT、XGBoost、SVM、MLP、Stacking 等模型。当 `SNP 数 >> 样本数` 时，直接把全部 SNP 喂给所有模型，确实很容易导致：

- 训练集指标很好，但测试集或外部数据表现不稳定；
- 不同 repeat / fold 之间波动很大；
- TOPSIS 排名被偶然高分模型带偏；
- Stacking 元学习器学习到噪声模式；
- 最终模型对新数据预测时泛化能力弱。

因此 v3 的重点是：**加入特征选择，但必须避免数据泄漏。**

---

## 二、当前 Feature Engineering 与 Preprocessing 基线

在新增特征选择和统一 Pipeline 之前，需要先明确 GPSE 当前已经具备哪些预处理能力，以及哪些能力仍然缺失。以下内容以现有代码的实际调用为准，不把文档规划计入已实现功能。

### 2.1 当前已实现能力总览

| 能力 | 当前状态 | 默认行为 | 主要位置 |
| --- | --- | --- | --- |
| SNP 加性编码 | 已实现 | convert 后生成 `0/1/2` 数值特征，缺失编码为 `3` | `gpse/convert/genotype_matrix.py` |
| 指定 SNP 提取 | 已实现 | 仅传入 `--extract` 或 SNP 目录时启用 | `gpse/convert/genotype_matrix.py` |
| SNP 缺失率过滤 | 已实现 | `--run-qc` 时启用，默认阈值 `--geno 0.1` | `gpse/convert/qc.py` |
| 样本缺失率过滤 | 已实现 | `--run-qc` 时启用，默认阈值 `--mind 0.1` | `gpse/convert/qc.py` |
| MAF 过滤 | 已实现 | `--run-qc` 时启用，默认 `--maf 0.05` | `gpse/convert/qc.py` |
| LD pruning | 已实现 | `--run-qc` 时启用，默认 `50 / 10 / r²=0.2` | `gpse/convert/qc.py` |
| 基因型插补 | 已实现、可选 | `--impute` 时使用 Beagle，未配置时回退 PLINK filling | `gpse/convert/qc.py` |
| 基因型/表型样本对齐 | 已实现 | convert 和 train 均会保留共有样本并统一顺序 | `gpse/convert/phenotype.py`、`gpse/train/_data_io.py` |
| 表型缺失值处理 | 已实现 | 删除缺失或不可转换的目标值样本 | `gpse/convert/phenotype.py`、`gpse/train/_data_io.py` |
| SNP 特征标准化 | 已实现 | 每个 CV fold 内使用 `StandardScaler` | `gpse/utils/genomic_utils.py`、`gpse/train/_optimization.py` |
| 回归表型标准化 | 已实现、可选 | `--standardize_phenotype` 时进行 Z-score | `gpse/train/_data_io.py` |
| 分类标签编码 | 已实现 | 使用 `LabelEncoder` 转为从 0 开始的整数 | `gpse/tasks/classification.py` |
| Stacking 元特征 | 已实现、可选 | `--use_stacking` 时以基础模型预测构造元特征 | `gpse/train/stacking.py` |
| Stacking 元模型 Pipeline | 已实现 | `StandardScaler -> Ridge/LogisticRegression` | `gpse/train/stacking.py` |

### 2.2 基因型编码与缺失值语义

当前 PED/MAP compound genotype 使用加性编码：

```text
00 -> 0    纯合参考
01 -> 1    杂合
10 -> 1    杂合
11 -> 2    纯合突变
缺失 -> 3
```

该编码把每个 SNP 转换成一个数值特征，是当前最核心的基因组特征构建步骤。

需要特别注意：如果没有在 QC 阶段启用 `--impute`，缺失基因型会以数值 `3` 进入模型。这不是统计意义上的缺失值插补，而是把缺失当作额外数值状态。v3 需要明确是否继续保留该语义，或者在 Pipeline 中增加可配置的缺失值处理策略。

### 2.3 当前 QC 与无监督 SNP 过滤

启用 `gpse convert --run-qc` 后，当前流程为：

```text
VCF / PLINK
  -> 统一为 PLINK BED
  -> 可选 Beagle / PLINK 基因型插补
  -> SNP 缺失率过滤
  -> 样本缺失率过滤
  -> MAF 过滤
  -> LD pruning
  -> 提取 pruning 后 SNP
  -> 0/1/2 加性编码
```

默认 QC 参数：

```text
--snpmaxmiss 0.1
--samplemaxmiss 0.1
--maf 0.05
--ld-window 50
--ld-step 10
--r2-cutoff 0.2
```

MAF 和 LD pruning 属于现有的无监督特征过滤能力，但只有显式启用 `--run-qc` 才执行，不应与未来训练阶段的 fold 内监督式特征选择混为一谈。

通过 `--extract` 或 `--snp-dir` 提供外部 SNP 清单，也可以得到预定义特征子集；该能力不使用当前训练数据的表型，因此不属于训练时自动特征选择。

### 2.4 当前训练数据预处理

训练加载阶段当前会执行：

1. 自动读取 CSV、Parquet 或 Feather。
2. 检查目标性状列和样本 ID 列。
3. 保留基因型与表型的共有样本。
4. 表型重复 ID 保留第一条记录。
5. 按样本 ID 排序，确保 `X` 和 `y` 对齐。
6. 回归任务将字符串目标强制转换为数值，不可转换值变成 `NaN`。
7. 删除目标值缺失的样本。
8. 分类任务使用 `LabelEncoder` 并保存 `label_encoder.pkl`。
9. 回归任务可选标准化表型并保存 `phenotype_scaler.json`。
10. 把真实 SNP 列名重命名为 `feature_0...feature_n`。

第 10 步会丢失训练矩阵中的原始 SNP 标识和顺序，是未来 predict 无法可靠对齐新样本 SNP 的主要风险之一。

### 2.5 当前普通模型特征标准化

普通模型在每个 CV fold 内执行：

```text
训练 fold -> StandardScaler.fit_transform
验证 fold -> StandardScaler.transform
测试集    -> StandardScaler.transform
```

Optuna 调参过程同样在各自 fold 内拟合 scaler，因此普通模型当前没有明显的 fold 间缩放泄漏。

但当前实现仍存在以下不足：

- 所有模型都进行标准化，包括通常不需要缩放的 RF、GBDT、XGBoost、LightGBM 和 CatBoost。
- scaler 和模型以 `(model, scaler)` tuple 保存。
- ensemble、Stacking 和未来 predict 必须手动拆包并调用 `transform`。
- 没有统一的普通模型 sklearn Pipeline。
- 训练矩阵出现缺失值时目前只记录 warning，没有训练阶段 imputer。

### 2.6 当前显式与隐式派生特征

当前明确实现的派生特征主要是 Stacking 元特征：

```text
原始 SNP
  -> Base Model A prediction
  -> Base Model B prediction
  -> Base Model C prediction
  -> Meta-feature matrix
  -> StandardScaler
  -> Ridge / LogisticRegression meta-model
```

Kernel Ridge 和 RBF SVM 等模型还会通过核函数进行隐式非线性映射，但它们属于模型内部行为，不会显式生成和持久化新的 SNP 特征列。

### 2.7 当前尚未实现能力

现有训练代码中尚未实现：

- `VarianceThreshold`；
- `SelectKBest` / `SelectPercentile`；
- `f_regression` / `f_classif`；
- Mutual Information 特征选择；（已实现：`--feature_selection mutual_info`）
- Lasso / ElasticNet 系数驱动的 SNP 筛选；（已实现：`--feature_selection lasso|elasticnet`）
- RF / XGBoost feature importance 驱动的 SNP 筛选；（已实现第一版：`--feature_selection tree_importance`，使用 fold-local ExtraTrees）
- PCA / Truncated SVD；（已实现：`--feature_selection pca|svd`；`--select_k` 表示输出组件数）
- SNP 交互项、显性效应或上位性特征；
- `SimpleImputer` / KNN Imputer / Iterative Imputer；
- 原始 SNP schema 持久化和预测对齐；
- 普通模型统一 sklearn Pipeline；
- 可直接执行的完整 `gpse predict` 流程。

因此，v3 不是从零增加预处理，而是在已有 QC、编码和标准化基础上，补齐**训练阶段特征选择、产物 schema、统一 Pipeline 和预测对齐**。

---

## 三、最重要的原则：特征选择不能泄漏

不能先在全数据上做监督式特征选择，再做 CV 或 train/test split。

错误流程：

```text
全体样本 X,y -> 监督式选 SNP -> CV / 测试集评估
```

这种做法会把验证集或测试集的表型信息提前泄漏进特征集合，导致评估结果虚高。

正确流程：

```text
每个 fold:
  只用训练 fold 拟合 selector
  用同一个 selector transform 训练 fold / 验证 fold / 测试集
  再 fit scaler + model
```

代表性模型可以在全量训练数据上重新拟合 selector + scaler + model，但它只能作为最终预测模型；泛化性能仍应以 CV / test 结果为准。

---

## 四、建议新增能力

### 4.1 训练侧新增特征选择模块

新增文件：

```text
gpse/train/_feature_selection.py
```

建议先支持三类策略：

| 策略 | 是否监督 | 适用场景 | 说明 |
| --- | --- | --- | --- |
| `none` | 否 | 默认兼容 | 不做特征选择 |
| `variance` | 否 | 快速过滤低信息 SNP | 去掉单态或近似单态 SNP |
| `univariate` | 是 | 第一版监督式筛选 | 回归用 `f_regression`，分类用 `f_classif` |

后续可扩展：

- `mutual_info`
- `lasso`
- `elasticnet`
- `tree_importance`
- `pca`
- `svd`

### 4.2 CLI 参数建议

在 `gpse train` 中新增：

```bash
--feature_selection none|variance|univariate
--select_k 5000
--select_percentile 10
--variance_threshold 0.0
```

可选扩展：

```bash
--feature_selection_metric f_regression|f_classif|mutual_info
```

建议默认：

```text
--feature_selection none
```

原因：保持旧版本行为兼容，不改变用户已有结果。

### 4.3 训练流程接入点

需要接入的地方：

- `gpse/train/_optimization.py`
  - Optuna CV 内部每个 fold 必须单独 fit selector。
- `gpse/train/_fold_training.py`
  - 单 fold 训练时 fit selector，并保存到模型产物。
- `gpse/train/_ensemble.py`
  - fold-ensemble 重新训练或加载模型时必须同步使用 selector。
- `gpse/config/_topsis_config.py`
  - representative model 保存时必须保存全量训练数据上拟合的 selector。
- `gpse/train/stacking.py`
  - base model 是 `(model, scaler, selector)` 或 dict 时，predict/create meta-features 必须先走 selector。

### 4.4 普通模型统一使用 sklearn Pipeline

当前普通模型虽然已经在每个 CV fold 内单独拟合 `StandardScaler`，没有明显的 fold 间缩放泄漏，但训练、保存和预测仍依赖手动维护 `(model, scaler)` 二元组。加入 selector 后，如果继续扩展为 `(model, scaler, selector)`，会使 `_optimization.py`、`_fold_training.py`、`_ensemble.py`、`stacking.py` 和未来的 `predict` 出现更多格式判断与重复转换逻辑。

v3 建议把普通模型统一封装为 sklearn `Pipeline`：

```python
Pipeline(
    [
        ("selector", selector),
        ("scaler", scaler),
        ("model", model),
    ]
)
```

执行规则：

- 每个 CV fold 都必须重新创建并 `fit` 一条独立 Pipeline。
- Pipeline 只能在训练 fold 上执行 `fit`，验证 fold 和测试集只能调用 `predict` / `predict_proba`。
- `selector=none` 时使用 `"passthrough"`。
- 不需要标准化的树模型可使用 `scaler="passthrough"`，避免无意义的数据复制和计算。
- Ridge、Lasso、ElasticNet、SVM、KNN、MLP 等尺度敏感模型默认使用 `StandardScaler`。
- 模型工厂应统一返回可训练 estimator，调用侧不再手动执行 `fit_transform` / `transform`。
- Optuna 参数仍由底层模型工厂生成；如改用 `Pipeline.set_params()`，模型参数统一使用 `model__` 前缀。

建议新增统一构建入口 `gpse/train/_model_pipeline.py`，负责判断模型是否需要标准化、创建 selector 和 scaler、组装 `selector -> scaler -> model`，并提供统一结构与参数校验。

迁移后调用方式应统一为：

```python
estimator = self.create_training_pipeline(model_name, params)
estimator.fit(X_fold_train, y_fold_train)
y_fold_pred = estimator.predict(X_fold_val)
```

调用侧不再拆包模型或手动执行 selector/scaler transform。

---

## 五、模型产物格式升级

当前模型产物是：

```python
joblib.dump((model, scaler), model_path)
```

建议升级为包含完整 Pipeline 和训练元数据的 dict，避免继续扩展 tuple：

```python
joblib.dump(
    {
        "pipeline": fitted_pipeline,
        "feature_snps": feature_snps,
        "selected_snps": selected_snps,
        "task_type": self.task_type,
        "feature_selection": feature_selection_config,
    },
    model_path,
)
```

兼容要求：

- 加载模型时继续支持旧格式 `(model, scaler)`。
- 新格式优先使用 `{'pipeline': fitted_pipeline, ...}` dict。
- 兼容层应把旧 `(model, scaler)` 适配为统一 estimator 接口，上层业务不再判断 tuple。
- 如果旧模型没有 `feature_snps`，predict 必须要求用户显式提供训练 SNP 清单。
- 新模型应能直接调用 `artifact['pipeline'].predict(X)`。

---

## 六、必须同时修复：训练 SNP 清单持久化

特征选择依赖训练时的原始 SNP 顺序。

当前 `gpse/train/_data_io.py` 会把真实 SNP ID 重命名为 `feature_0...`，这会导致 predict 无法可靠对齐新样本数据。

v3 必须新增：

```text
{results_dir}/feature_snps.json
```

内容示例：

```json
{
  "n_features": 123456,
  "snps": ["snp1", "snp2", "snp3"]
}
```

写入时机：

```text
读取 geno matrix 后，X.columns 被重命名前
```

---

## 七、predict 侧必须配套

特征选择不是只改训练就够。`gpse predict` 必须遵循同一条 pipeline：

```text
新样本 genotype matrix
  -> 按 feature_snps.json 对齐训练 SNP 顺序
  -> 缺失 SNP 填充
  -> selector.transform
  -> scaler.transform
  -> model.predict
  -> 回归反标准化 / 分类 label decode
```

否则即使训练时选出了 SNP，预测时仍然会因为列顺序、缺失 SNP 或 ID 不一致得到错误结果。

predict 输出建议同时写：

```text
predictions.csv
predictions_alignment_report.json
```

alignment report 至少包含：

- 训练 SNP 总数；
- 新数据匹配 SNP 数；
- 缺失 SNP 数；
- 覆盖率；
- 缺失 SNP 示例；
- 使用的填充策略；
- 最终进入模型的特征数。

---

## 八、实现优先级

### P0：预测可靠性的前置修复

1. 保存训练有序 SNP 清单 `feature_snps.json`。
2. 新增模型加载兼容层，支持旧 tuple 和新 dict。
3. 确定 predict 的输入矩阵列对齐规则。

### P1：最小可用特征选择

1. 新增 `_feature_selection.py`。
2. 新增 `_model_pipeline.py`，统一构建 `selector -> scaler -> model`。
3. 支持 `none / variance / univariate`。
4. 在 `_fold_training.py` 和 `_optimization.py` 中按 fold fit Pipeline，避免泄漏。
5. representative model 保存完整 Pipeline 和 selected SNP。

### P2：Stacking / ensemble 兼容

1. `_ensemble.py` 直接加载和调用 Pipeline，不再拆包 `(model, scaler)`。
2. `stacking.py` 支持新模型产物格式，并移除新格式的 scaler/selector 特殊分支。
3. TOPSIS / model comparison 中记录特征选择配置。

### P3：predict 完整打通

1. 实现 `gpse/predict`。
2. 支持训练 SNP 对齐。
3. 支持 selector + scaler + model pipeline。
4. 支持 `phenotype_scaler.json` 反标准化。
5. 支持 `label_encoder.pkl` 分类标签解码。

---

## 九、验收标准

### 9.1 防泄漏验收

- 监督式 selector 在每个 CV fold 内只对训练 fold `fit`。
- 验证 fold 和测试集只能 `transform`，不能参与 selector 拟合。
- Optuna 调参过程也遵守同样规则。
- selector、scaler 和 model 由同一条 Pipeline 在 fold 内完成拟合。
- 不允许在进入 CV 前对全量 `X` 调用 selector/scaler 的 `fit` 或 `fit_transform`。

### 9.2 产物验收

- `results_dir/feature_snps.json` 存在。
- 新模型产物包含 fitted Pipeline / feature_snps / selected_snps / task_type。
- Pipeline 可直接对已完成 SNP 对齐的特征矩阵调用 `predict` / `predict_proba`。
- 旧 `(model, scaler)` 模型仍能加载。

### 9.3 predict 验收

- 新样本 SNP 顺序被强制重排到训练顺序。
- 多余 SNP 被忽略。
- 缺失 SNP 被填充并记录到 alignment report。
- alignment report 记录匹配 SNP 数和 feature coverage；`--min-feature-coverage` 可将低覆盖率升级为预测前硬错误。
- 覆盖率低于阈值时给出明确报错或告警。

### 9.4 性能验收

- 小型模拟数据可跑通 `train -> representative model -> predict`。
- 特征选择开启后，测试指标不应因为泄漏而异常虚高。
- `pytest` 通过，并新增覆盖以下场景的测试：
  - Pipeline 在每个 fold 内独立拟合；
  - 尺度敏感模型包含 `StandardScaler`；
  - 树模型的 scaler 使用 `passthrough`；
  - selector 在 fold 内 fit；
  - predict SNP 重排；
  - 缺失 SNP 填充；
  - 旧模型格式兼容。

---

## 十、推荐第一版落地方案

第一版不要直接做复杂的 GWAS/Lasso/PCA 全家桶。建议先落地：

```text
feature_snps.json
新模型 Pipeline + metadata dict 格式
none / variance / univariate
CV fold 内 fit selector + scaler + model Pipeline
predict 对齐训练 SNP 顺序
```

这样能先解决最关键的两个问题：

1. 全 SNP 输入导致的高维噪声和潜在过拟合；
2. 新样本预测时 SNP 列顺序和训练不一致导致的错误预测。
