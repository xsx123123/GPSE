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

---

## 附录 A：2026-08-05 代码审计与自适应调度器

> 本节记录 2026-08-05 对全仓库进行的一次系统性排查，包含已落地的修复与仍待处理的技术债。

### A.1 P0 严重缺陷（已修复）

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| P0-1 | `svm_clf` 搜索空间错误使用 `LinearSVC` 参数（`loss/dual/penalty`），导致每次 trial 抛 `TypeError` | `gpse/config/models.yaml:292-309` | 改为 `SVC` 的有效参数 `kernel/C/gamma/degree/tol`，保留 `random_state` fixed |
| P0-2 | HPO 后 `study.best_params` 不包含 DSL `type: fixed` 项，`create_model` 未回灌 `default_params`，导致所有 DSL 模型 `random_state` 丢失，可复现性被静默破坏 | `gpse/train/_optimization.py:206` + `gpse/config/_model_registry.py:264-296` | `create_model` 先加载 YAML `default_params` 再用调用方 params 覆盖 |
| P0-3 | `max_trials = min(n_trials, 100)` 把用户显式传入的 `--trials 500` 静默截断，无任何告警 | `gpse/train/_optimization.py:193` | 直接使用 `self.n_trials`，信任 CLI 输入 |
| P0-4 | `--task_type` 在 CLI 被 `required=True` 阻塞，`phenotype_info.json` 自动推断逻辑完全不可达 | `gpse/train/workflow.py:74-80` | 改为 `default=None`，在 `gpse/train/cli.py` 内尝试从 `phenotype_info.json` 推断，既未提供也无法推断时才报错 |
| P0-5 | `threadpoolctl` 在 4 处被无条件 import，但未在 `pyproject.toml` 中声明，干净 `pip install gpse` 训练必崩 | `pyproject.toml:43-62` | 加入 `threadpoolctl = "^3.5.0"` |
| P0-6 | Optuna 早停逻辑反了：`best_value <= recent_best_value` 在“全局最优恰好在最近窗口内（仍在改进）”时也会触发停止 | `gpse/train/_optimization.py:161-178` | 改为 `recent_best_value <= best_value - 1e-9` 才停止（窗口内无任何超越历史最优），并加入浮点容差 |

### A.2 P1 真实缺陷（已修复）

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| P1-1 | `ngboost` 声明为依赖但全仓 0 引用，白付编译成本 | `pyproject.toml:55`、`requirements.yaml:31` | ✅ 已从两处依赖声明中移除 |
| P1-2 | `scripts/` 下 `genomic_classification.py` / `genomic_prediction_v2.py` 等 9 个 v2 旧脚本已被 `gpse/` 替代，无引用 | `scripts/` | ✅ 已确认无引用；Bash 审批被阻断，需用户手动 `rm -rf scripts/` 完成物理删除 |
| P1-3 | `requirements.yaml` 漂移：`lifelines/matplotlib/plotly/SQLAlchemy/alembic/formulaic/sympy/nvidia-nccl-cu12/Mako/greenlet/narwhals` 均与主流程无关 | `requirements.yaml` | ✅ 已收敛为 `pyproject.toml` 直接依赖的 conda lock 快照，加注释说明 |
| P1-4 | `gpse.yaml:15` 硬编码 `/home/jzhang/software/Beagle/...jar`，示例 shipped 即不可用 | `gpse.yaml` | ✅ 已替换为 `/path/to/beagle.jar` 占位符并提示通过 `gpse.local.yaml` 覆盖 |
| P1-5 | `gpse/__init__.py:5-6` `importlib_metadata` 回退分支在 Python 3.10+ 下既无必要也未声明依赖 | `gpse/__init__.py` | ✅ 已移除 try/except，仅保留 `from importlib.metadata import version` |
| P1-6 | `gpse/utils/genomic_utils.py:40,47` 裸 `except:` 吞掉 `KeyboardInterrupt/SystemExit` | `gpse/utils/genomic_utils.py` | ✅ 已改为 `except Exception:` |
| P1-7 | 文件名拼错 `gpse/utils/paralle.py` → 应为 `parallel.py`，4 处 import 跟着错 | `gpse/utils/paralle.py` + 4 import 点 + `tests/test_parallelism.py` + `gpse/utils/__init__.py` | ✅ 新增 `parallel.py`，所有 import 全部改正；`paralle.py` 保留为向后兼容 shim（re-export），建议下一版本物理删除 |
| P1-8 | `predict/core.py:_find_result_file` 无限向上遍历 `.parents` 找 `label_encoder.pkl`，祖先目录同名文件会被静默加载导致反变换错误 | `gpse/predict/core.py:24-29` | ✅ 加入 `max_up=3` 深度限制，文档注释说明上限原因 |

### A.3 P2 测试与质量（部分完成）

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| P2-1 | 8 个测试文件覆盖 ~40 模块，`convert`/`predict` CLI、Beagle QC、VCF 样本交集校验均无测试 | `tests/` | 🔶 已新增 `tests/test_audit_fixes.py`（覆盖 svm_clf 搜索空间合法性、trials 无硬上限、无全局 RNG seed、threadpoolctl 依赖、paralle shim、predict 搜索深度、调度器并发度数学、flag upsert）；端到端 `train → predict` 测试仍待补充 |
| P2-2 | `_optimization.py:76` 在并行 Optuna callback 里调 `np.random.seed()`，污染全局 RNG | `gpse/train/_optimization.py:76` | ✅ 已移除全局 seed；可复现性由 `TPESampler(seed=opt_seed)` 与 P0-2 的 `random_state` 合并保证 |
| P2-3 | `train/cli.py:345-355` 的分类手动校验分支因 `--task_type` required 而成为死代码 | `gpse/train/cli.py` | ✅ P0-4 修复后该分支已激活：推断出的 classification 同样走 `n_classes` 校验；回归时传 `--n_classes` 仅告警 |

### A.4 新增能力：内存感知自适应批量调度器

新增文件：

- `gpse/batch/adaptive_scheduler.py` — Python 调度器本体；
- `batch/launch_lowdim_adaptive.sh` — 薄壳启动脚本；
- `batch/lowdim_core_adaptive.example.yaml` — 11 个低维核心性状的示例配置。

设计要点（对应 `docs/update.26.8.5.md` 的全部 6 条要求）：

1. **资源检测**：`/proc/meminfo.MemAvailable` 优先，兼容 cgroup v2 `memory.max / memory.current / memory.events`；CPU 预算由 `--max-threads` 约束。
2. **并发度计算**：`min(⌊MAX_THREADS/threads_per_trait⌋, ⌊(MemAvailable - min_free_mem)/mem_per_trait⌋, max_concurrent_traits)`；内存不足时等待重试而非强行启动。
3. **队列式执行**：每个性状独立日志、PID 文件、起止时间戳；通过 `scheduler_status.json` 持久化，重启时自动跳过已成功（`reports/run_summary.json` 完整）的性状。
4. **自动降级**：优先减少并发度，降到 1 后再减半 `--threads/--max_workers` 直到 `--min-threads-per-trait`；每次降级写 `MEMORY_PRESSURE` 日志；内存恢复后每次最多 +1 并带 120 s 冷却。
5. **GPSE 并行模型**：每个性状固定 `--n_jobs 1 --repeat_workers 1`，仅通过 `--max_workers` 并行不同模型，避免乘法式进程膨胀；最终传入的 GPSE 参数被记录到 TSV。
6. **失败识别**：OOM/Killed/MemoryError/BrokenProcessPool 关键词判为 `FAILED_MEM`，最多自动重试 2 次，每次重试前先降级；其他失败标为 `FAILED_NON_MEMORY`，不重试。
7. **输出**：`scheduler_status.tsv` 含 `species,trait,status,attempt,exit_code,start_time,end_time,threads,max_workers,mem_available_gb,error_category,log_file,result_dir`；`scheduler.log` 记录降级 / 升级 / 启动事件。
8. **可用性**：`--dry-run` 打印当前可用内存、计算出的安全并发度、所有待执行命令，但不启动训练；实验协议（`random_seed=42`、50 repeats、5 folds、相同 CV、相同模型列表、top-3 stacking）在示例 YAML 中固化。

使用方式：

```bash
# 预览
MAX_THREADS=90 MIN_FREE_MEM_GB=64 MEM_PER_TRAIT_GB=32 THREADS_PER_TRAIT=8 \
    bash batch/launch_lowdim_adaptive.sh --dry-run

# 正式跑
MAX_THREADS=90 MIN_FREE_MEM_GB=64 MEM_PER_TRAIT_GB=32 THREADS_PER_TRAIT=8 \
    bash batch/launch_lowdim_adaptive.sh
```

默认初始保守值：`MAX_CONCURRENT_TRAITS=2`，稳定后由调度器根据实时可用内存自动上调。

### A.5 后续建议

1. 手动执行 `rm -rf scripts/` 完成 P1-2 物理删除（Bash 自动审批被阻断，调度器与代码均未引用该目录）。
2. 下一版本彻底删除 `gpse/utils/paralle.py` 兼容 shim，把 P1-7 从"重命名 + shim"收敛为"物理删除"。
3. 补充 P2-1 剩余的端到端测试，至少覆盖 `train → representative → predict` 的完整链路。
4. 自适应调度器在首次真实运行后，根据实测内存峰值调整 `MEM_PER_TRAIT_GB` 与 `MIN_FREE_MEM_GB`。

---

## 附录 B：2026-08-05 第二轮深度排查

> 对 convert/train/predict/batch/models/config 全域的第二次深度审计（三个并行审计代理分区扫描 + 关键项人工复核）。以下问题均为**第一轮未发现**的新问题；P0 三项已通过 grep 人工验证。

### B.1 P0 严重缺陷（新发现，待修复）

| # | 问题 | 位置 | 证据 |
|---|------|------|------|
| B-P0-1 | `detect_phenotype_type` / `save_phenotype_info` 在 convert 流程**零调用**，`{prefix}_{trait}_phenotype_info.json` 从不生成；`train` 侧（A-P0-4 的自动推断）依赖该文件。且 `phenotype.py:315` 用 `np.nan` 但模块未 import numpy，一旦接入即 NameError | `gpse/convert/phenotype.py:292,394,315` ↔ `gpse/train/cli.py:346` | grep 全仓：两个函数仅有定义无调用点 |
| B-P0-2 | scaler 文件名不对齐：convert 写 `{prefix}_{trait}_scaler.json`，`predict/core.py` 只找字面 `phenotype_scaler.json` → convert 侧标准化永不反变换；且 convert 支持 minmax（存 min/max），predict 硬取 `["std"]/["mean"]` → KeyError | `gpse/convert/processor.py:350,358` ↔ `gpse/predict/core.py:286-290` | 两套命名各自为政 |
| B-P0-3 | train 从不写独立 `phenotype_scaler.json`（scaler 仅存于 `split_manifest.json`），predict 只认该文件名 → `--standardize_phenotype` 训练出的模型预测结果停留在标准化尺度 | `gpse/train/_pipeline.py:52,61` ↔ `gpse/predict/core.py:286` | grep 全仓无 `phenotype_scaler.json` 写点 |

### B.2 P1 真实缺陷（新发现，待修复）

**convert / 数据面**

| # | 问题 | 位置 |
|---|------|------|
| B-P1-1 | canonical SNP ID 两侧不一致：convert 走 MAP `end=pos+1`，predict 走 VCF `end=pos+len(REF)` → indel 永不对齐；`normalize_chromosome` 保大小写，Chr1≠chr1 | `gpse/utils/snp_ids.py:14,35,60` ↔ `gpse/predict/core.py:97` |
| B-P1-2 | feature_manifest 只记坐标不记 REF/ALT，编码锚定 PLINK A1/A2；predict 仅按位置对齐 → REF/ALT 翻转导致 0↔2 静默反转且无法检测 | `gpse/utils/feature_manifest.py:28-37` |
| B-P1-3 | 样本 ID 归一化三处不一致：matrix 侧 `rstrip('_')`、overlap 检查用原始 VCF ID、pheno 侧无处理；PED 重复样本被 dict 静默覆盖，pheno 重复 ID 致 geno/pheno 行数错位 | `gpse/convert/genotype_matrix.py:268-269,284`、`processor.py:247-258,333-336` |
| B-P1-4 | `_compress_stdout` 同时作用于 stderr：含 `--vcf:` 的行整行丢弃、≥3 个百分号截断 → 可能吞掉真实 PLINK 错误 | `gpse/convert/external.py:184-185,200-204,379,393` |
| B-P1-5 | `-101`（缺失）编码仍回退 `3`，越出 [-1,1] 语义；`03`/`30` 半缺失基因型无显式处理 | `gpse/convert/genotype_matrix.py:275` |

**train / 训练面**

| # | 问题 | 位置 |
|---|------|------|
| B-P1-6 | `_apply_threads_budget` 先于 task_type 自动推断执行；未传 `--task_type` 时 `_resolve_model_count` 走 classification 分支，模型数算错（6 vs 15），`--threads` 派生并行度错误 | `gpse/train/cli.py:153,229,345` |
| B-P1-7 | `_save_representative_model` 无条件调用，`save_representative` 标志从未被读取（已验证）；且用全量 X,y 重拟合但 info.json 只记录 train-CV 指标，无 `trained_on: all_samples` 警示 | `gpse/train/_repeat_training.py:502`、`predictor.py:202`、`config/_topsis_config.py:176` |
| B-P1-8 | 并行 repeat 按 `as_completed` 完成顺序 append，`selected_repeat_idx` 是列表位置而非 repeat_idx；`--optuna_per_repeat --repeat_workers>1` 时 `final_predictions.csv` 会取错 repeat 目录 | `gpse/train/_repeat_training.py:435,469`、`_pipeline.py:152` |
| B-P1-9 | 外部 `--cv_file` 经 `read_csv(index_col=0)` 后索引 dtype 被重推断（数字 ID→int64），与 parquet 来源 str 型 X.index 不匹配 → 所有 fold 静默为空，无 dtype 诊断 | `gpse/train/_cv_manager.py:67,196` |
| B-P1-10 | 用户 CV 文件不校验与当前 n_splits/n_repeats 一致性；fold 数较少时高 fold_idx 的 val_mask 为空，静默产生退化 fold | `gpse/train/_cv_manager.py:188-212` |
| B-P1-11 | 缺失的 fold pkl（失败 fold）被静默跳过、无警告，用少于 n_splits 个成员计算并上报 ensemble 指标 | `gpse/train/_ensemble.py:136-143` |

**predict / batch / models**

| # | 问题 | 位置 |
|---|------|------|
| B-P1-12 | `transform_artifact_input` 中 `scaler.transform` 无 None 检查；树模型 `scaler_=None`（passthrough），legacy 分支必崩（训练侧 `_ensemble.py:130` 有检查） | `gpse/predict/core.py:218` ↔ `train/_model_pipeline.py:62` |
| B-P1-13 | 概率平均未按 `classes_` 对齐到全类空间（`tasks/classification.py:351-358` 有此逻辑而 predict 无），缺类成员时形状错/类别错位；legacy 分支无 `hasattr(predict_proba)` 检查 | `gpse/predict/core.py:229-243,261` |
| B-P1-14 | lightgbm_clf：n_classes=None 时 `create_model` 强置 objective=binary，与 `get_default_params`（multiclass）及 `_lightgbm_clf_params` 矛盾；`models.yaml:347` num_class 占位符解析为 None 传入 | `gpse/config/_model_registry.py:295-302` ↔ `:329-333` |
| B-P1-15 | `safe_kernel_ridge.py` 全局 monkey-patch `linalg.solve`：并发/嵌套 fit 时 finally 恢复错对象，可永久泄漏 patched 版本 | `gpse/models/safe_kernel_ridge.py:32-36` |
| B-P1-16 | adaptive_scheduler：`max_threads < threads_per_trait` 时 by_cpu=0，任务永不启动且无日志（静默死循环）；SIGTERM/SIGINT 后存在 PENDING 任务仍返回 0 | `gpse/batch/adaptive_scheduler.py:155,579,587-593` |

### B.3 P2 测试与质量（新发现，待修复）

| # | 问题 | 位置 |
|---|------|------|
| B-P2-1 | TOPSIS 熵权法：全零列权重 NaN（0/0）、负值准则取 log 得 NaN、单行时 1/log(1) 除零；手动权重不校验负数/和为 0；非有限分数过滤后长数组赋回短 df 列错位 | `gpse/train/topsis.py:37-44,123-127,135-145` |
| B-P2-2 | 基因型重复 ID 不去重（只去表型），数量恰好相等时同一基因型可跨 fold（泄漏），否则报误导性"样本数不匹配" | `gpse/train/_data_io.py:114-131` |
| B-P2-3 | `prepare_cv_data(seed=42)` 硬编码，无视 `--random_seed` | `gpse/train/_cv_manager.py:77` |
| B-P2-4 | stacking 多分类 meta 特征取 `proba.max()` 丢失类别信息；`_model_factory=None` 后 dump 无 try/finally，异常时 factory 永久丢失 | `gpse/train/stacking.py:75,184-187` |
| B-P2-5 | result bundle 直接写出，无临时文件 + rename 原子性，重跑静默覆盖 | `gpse/train/_results.py:226-256` |
| B-P2-6 | `standardize_phenotype` 时 fold 级 MSE 在标准化尺度、ensemble 指标反标准化，同一报告混尺度且标 "original" | `gpse/train/_repeat_training.py:243-246` ↔ `_ensemble.py:214`、`_pipeline.py:95` |
| B-P2-7 | 兜底 holdout 不传 X/split_strategy，structure_aware 下退化为随机切分 | `gpse/train/_repeat_training.py:404-408` |
| B-P2-8 | `--enable_preprocess` 未给 `--raw_pheno_file` 时 `os.path.exists(None)` 抛未捕获 TypeError | `gpse/train/cli.py:325,340` |
| B-P2-9 | `unpack_model_artifact` 仅覆盖 v3 dict 与 2-tuple；缺 "scaler" 键的 dict 或其他长度 tuple 直接 TypeError，无版本提示 | `gpse/train/_feature_selection.py:236-242` |
| B-P2-10 | adaptive_scheduler：失败分类扫描累积日志（"ab" 追加），前次 MEM 残留关键词误判后续失败；子串 "oom" 误匹配 room/zoom；对自定义 gpse_command 无条件追加 gpse flag；降级后 threads_per_trait 永不回升 | `gpse/batch/adaptive_scheduler.py:280,301,604-607,504-524` |
| B-P2-11 | adaptive_scheduler YAML：defaults 里设 results_dir 会让所有性状共用同一目录；dry-run 也 mkdir 日志目录；缺 name 直接 KeyError | `gpse/batch/adaptive_scheduler.py:744-747` |
| B-P2-12 | `classification_model_optimizer.py` `_svc_clf_params` 仍是 LinearSVC 参数遗留，builtin 引用即崩 | `gpse/models/classification_model_optimizer.py:211-220` |
| B-P2-13 | `batch/runner.py` 进程内串行调 train_main 无内存感知，与 adaptive_scheduler YAML schema 互不兼容（两套批处理并存）；`merge.py` 源 CSV 已有 "Trait" 列时 insert 抛错中止整体合并 | `gpse/batch/runner.py:228`、`merge.py:55` |
| B-P2-14 | `_ensemble.py` info.json 声称 `trained_on: all_samples`，成员实为 K-1/K 折训练 | `gpse/train/_ensemble.py:56-58,86` |
| B-P2-15 | TRAIT_INVALID_CHARS 缺 `( ) ' * ? ; $`；n_unique==1 判为单类 classification；缺失标记不含 "-9"/"Na" | `gpse/convert/validators.py:25-28`、`phenotype.py:345` |
| B-P2-16 | run_command 无 timeout；Beagle 失败时临时文件残留；match_genotype_phenotype 恒用 read_csv 读 geno（parquet 输入即坏） | `gpse/convert/external.py:350,368`、`qc.py:288`、`phenotype.py:155` |
| B-P2-17 | analyze_phenotypes.py 硬编码 melon 路径，类别阈值 15 与 detect_phenotype_type 的 20 不一致 | `gpse/tools/analyze_phenotypes.py:84-88` |
| B-P2-18 | 魔数：`_ensemble.py:47` `random_seed+10_000`、`_repeat_training.py:78` `n_splits=32`；genotype_matrix.py:232-239 死代码 | 各处 |

### B.4 建议修复顺序

1. **B-P0-1/2/3（数据正确性闭环）**：在 convert 流程接入 `detect_phenotype_type` + `save_phenotype_info`（含 numpy import 修复）；统一 scaler 产物命名为 `phenotype_scaler.json`（train 与 convert 双写，predict 兼容 minmax 与 zscore）。
2. **B-P1-6/7/9（训练正确性）**：把 threads 派生挪到 task_type 推断之后；让 `save_representative` 标志真正生效；CV 索引统一 `astype(str)` 对齐。
3. **B-P1-12/13（predict 健壮性）**：scaler None 检查 + classes_ 全类空间对齐。
4. **B-P1-16/B-P2-10/11（调度器）**：by_cpu=0 显式报错、中断返回非 0、失败分类只扫本次日志段、dry-run 不 mkdir。
5. **B-P1-1/2（SNP 身份体系）**：统一 canonical end 规则并在 manifest 中记录 REF/ALT，predict 侧校验方向。

