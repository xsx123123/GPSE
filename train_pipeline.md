# GPSE Train Pipeline 逻辑评审文档

> 审查日期：2026-07-18
> 审查范围：`gpse train` 的数据转换、hold-out、交叉验证、Optuna、模型训练、测试评估、TOPSIS 和 Stacking 流程
> 文档目的：向项目成员说明当前代码的真实执行逻辑，并讨论是否需要重构为更严格的数据科学训练流程。

---

## 1. 结论摘要

理想的训练流程应当是：

```text
VCF/PLINK 原始数据
    ↓
基因型转换与必要的数据清洗
    ↓
固定 hold-out 切分：Train / Test
    ↓
仅在 Train 内生成 CV folds
    ↓
Optuna 仅使用 Train-CV 优化超参数
    ↓
使用最佳参数在完整 Train 上重新训练 evaluation model
    ↓
Test 只进行一次最终评估
    ↓
确认评估结果后，可使用 Train + Test 重训 deployment model
```

当前 GPSE 已经实现了上述流程中的主要步骤，但还存在以下重要差异：

1. VCF 转换只有在显式使用 `--enable_preprocess` 时才会执行。
2. CV 分组文件目前在完整样本集上生成，之后才根据 hold-out 训练集筛选。
3. Optuna 的确只在 hold-out 训练集的 CV folds 上优化，没有直接使用测试集。
4. 最佳参数会用于后续 fold 模型训练和测试集评估。
5. TOPSIS/Stacking 模型选择目前会使用测试集指标，造成测试集参与模型选择。
6. representative model 会在全部 `X, y` 上重新训练，其中包含 hold-out 测试样本。
7. 分类任务的 hold-out 切分目前没有使用 `stratify=y`。
8. 表型标准化发生在 hold-out 之前，均值和标准差可能包含测试集信息。

因此，当前流程可以正常运行，但如果要把测试结果作为严格的独立泛化性能，应进一步隔离 Train、Validation/CV 和 Test 的职责。

---

## 2. 当前代码实际执行流程

### 2.1 输入模式

`gpse train` 当前支持两种输入模式。

#### 模式 A：直接训练已有矩阵

```bash
gpse train \
    --geno_file genotype.csv \
    --pheno_file phenotype.csv \
    --target_trait Trait \
    --task_type regression
```

该模式不会处理 VCF，直接读取已经转换完成的基因型矩阵。

#### 模式 B：预处理后立即训练

```bash
gpse train \
    --enable_preprocess \
    --preprocess_prefix data/train \
    --vcf_file samples.vcf.gz \
    --raw_pheno_file phenotype.csv \
    --target_trait Trait \
    --task_type regression
```

该模式在 `gpse/train/cli.py` 中创建 `GenomicDataProcessor`，执行：

```text
VCF → PLINK → PED/MAP → canonical SNP matrix → phenotype matching → train
```

因此，当前实现并不是“只要传入 VCF，train 就自动转换”，而是必须显式启用 `--enable_preprocess`。

---

### 2.2 数据加载

训练入口调用：

```text
GenomicPredictorV2.run_all_models()
    → load_data()
```

`load_data()` 负责：

1. 读取基因型矩阵和表型文件；
2. 按样本 ID 取交集并排序；
3. 删除缺失表型样本；
4. 保留真实 SNP 列名；
5. 分类任务执行标签编码；
6. 可选地进行表型标准化；
7. 保存训练特征顺序到 `feature_manifest.json`。

当前有序 SNP 清单输出为：

```text
<results_dir>/feature_manifest.json
```

该文件用于后续 `gpse predict` 对齐其他用户的 VCF。

---

### 2.3 CV 分组生成

当前 pipeline 在加载数据后立即执行：

```text
prepare_cv_folds(pheno_data, target_trait)
```

默认生成：

```text
<results_dir>/cv_folds/<trait>_cv_<repeats>x<folds>.csv
```

CV 分配现在使用独立的固定随机源：

```python
random.Random(42)
```

因此在以下条件相同时，即使删除 CV 缓存文件，重新生成的 fold 也保持一致：

- 样本及样本顺序相同；
- `n_repeats` 相同；
- `n_splits` 相同；
- seed 保持为 `42`。

需要注意：CV 文件目前是在完整样本集上生成，之后在每个 repeat 中通过训练集样本 ID 进行筛选。测试样本虽然存在于 CV 文件中，但不会被映射进实际的训练 fold。

---

### 2.4 Hold-out 切分

每个模型开始 repeated training 时，先生成测试集索引：

```python
train_test_split(
    range(len(X)),
    test_size=self.test_size,
    random_state=self.random_seed,
)
```

默认配置中：

```text
use_same_test_set = True
```

因此：

```text
所有模型 + 所有 repeat → 使用相同的 hold-out 测试集
```

如果通过程序化 API 设置：

```python
use_same_test_set=False
```

则不同 repeat 会根据 `random_seed + repeat_idx` 生成不同测试集。

当前 CLI 参数使用 `store_true` 且默认值已经是 `True`，因此 CLI 暂时没有直接关闭该选项的方式。

---

### 2.5 Hold-out 后的 CV

每个 repeat 的实际顺序为：

```text
完整 X, y
    ↓
prepare_train_test_data()
    ↓
X_train, y_train, X_test, y_test
    ↓
generate_cv_folds_from_file(X_train, ...)
```

CV fold 的 ID 会映射到 `X_train`，测试样本无法进入训练 fold。

因此从实际训练数据边界看：

```text
Optuna 和 fold training 只使用 X_train / y_train
```

但从代码结构看，CV assignment 文件是在 hold-out 之前生成的，流程表达不够清晰。

---

## 3. Optuna 超参数优化

每个模型、每个 repeat 都会单独执行一次 Optuna，除非使用：

```bash
--use_default_params
```

Optuna objective 的流程为：

```text
最佳参数候选 trial
    ↓
遍历 X_train 的 CV folds
    ↓
只在 fold_train 上拟合 StandardScaler
    ↓
转换 fold_train 和 fold_validation
    ↓
使用全新的 estimator 训练 fold_train
    ↓
在 fold_validation 上计算指标
    ↓
返回多个 fold 的平均指标
```

回归任务优化目标：

```text
平均 Pearson correlation
```

分类任务优化目标：

```text
平均 Accuracy
```

Optuna 使用确定性 sampler seed：

```text
random_seed + repeat_idx × 100
```

### 已修复：fold 间模型复用

之前一个 trial 内的多个 fold 会复用同一个 estimator 对象。对于支持 `warm_start` 或存在内部状态残留的模型，可能造成跨 fold 污染。

现在已经修改为：

```text
每一个 Optuna fold → 创建一个全新的 estimator
每一个最终评估 fold → 创建一个全新的 estimator
```

---

## 4. 最佳参数确认后的训练

Optuna 返回：

```text
best_params
best_value
n_trials
early_stopped
```

之后当前代码会使用 `best_params`，在相同的 `X_train` CV folds 上重新训练模型：

```text
best_params
    ↓
fold 1: fold_train → fold_validation + hold-out test
fold 2: fold_train → fold_validation + hold-out test
...
fold N: fold_train → fold_validation + hold-out test
```

每个 fold 的模型和 scaler 可以保存为：

```text
<model>/repeat_<i>/fold_<j>_model.pkl
```

随后 fold ensemble 对 hold-out test 进行预测：

```text
每个 fold model 对 X_test 预测
    ↓
回归：预测值平均
分类：预测概率平均 / 多数投票
    ↓
计算 hold-out test 指标
```

这一部分基本符合：

```text
Optuna 确定参数 → 使用最佳参数训练模型 → hold-out 评估
```

但这里保存的是多个 fold model，并不是一个在完整 hold-out Train 上重新拟合的单一 evaluation model。

---

## 5. Representative Model

每个模型完成所有 repeat 后，会选择 ensemble 表现最接近平均值的 repeat，并获取该 repeat 的参数。

随后 representative model 会执行：

```text
StandardScaler.fit_transform(完整 X)
model.fit(完整 X, 完整 y)
```

也就是说，该模型使用了：

```text
Train + hold-out Test
```

### 该设计适合的用途

- 最终部署；
- 给其他用户进行表型预测；
- 在确认算法和超参数后充分利用全部训练样本。

### 不适合的用途

- 再次计算独立 hold-out 性能；
- 把 representative model 本身称为“未见过测试集的模型”；
- 直接把旁边 `info.json` 的 CV/ensemble 指标解释为该全量模型的独立泛化指标。

建议未来明确区分：

```text
evaluation_model
    只在 hold-out Train 上训练
    用于生成公平的 Test 性能

deployment_model
    在 Train + Test 全量数据上重训
    用于交付和外部预测
```

---

## 6. 审查时发现的问题（历史记录）

> 本节记录 2026-07-18 审查时的初始状态。除 6.7 的严格原始 VCF train-only QC 外，以下问题已在第 12 节列出的后续改造中处理；不要将本节描述视为当前实现状态。

### 6.1 高风险：测试集参与模型选择

目前 TOPSIS 和 Stacking 选择 base models 时，会使用：

```text
Test Pearson
Test Accuracy
Test F1
Test AUC
```

之后又使用相同的 hold-out test 评估 Stacking。

实际逻辑为：

```text
测试集指标
    ↓
选择 top models
    ↓
训练 Stacking
    ↓
再次在相同测试集上报告性能
```

这会使最终 Stacking 指标偏乐观，因为测试集已经参与了模型选择。

#### 建议

模型选择应改用：

```text
Optuna best CV score
CV validation mean/std
OOF prediction performance
```

测试集只能在模型、参数、base model 集合全部确定后评估一次。

---

### 6.2 高风险：代表模型指标与模型训练数据不一致

`representative_model/model.pkl` 使用全量数据重训，但 `info.json` 保存的是之前 fold ensemble 的测试表现。

它们不是同一个估计器，也不是同一个训练数据范围。

#### 建议

在元数据中明确记录：

```json
{
  "artifact_role": "deployment_model",
  "trained_on": "all_samples",
  "performance_source": "holdout_fold_ensemble",
  "performance_belongs_to_saved_model": false
}
```

或者分别保存 evaluation model 和 deployment model。

---

### 6.3 中高风险：分类 hold-out 没有分层抽样

当前分类 hold-out 没有使用：

```python
stratify=y
```

在类别不平衡或样本较少时，可能出现：

- 某个类别没有进入训练集；
- 某个类别没有进入测试集；
- 测试指标不稳定；
- AUC 无法计算；
- 模型训练失败。

#### 建议

分类任务使用：

```python
train_test_split(
    indices,
    test_size=test_size,
    random_state=random_seed,
    stratify=y,
)
```

并在切分前验证每个类别的样本量是否足以支持 hold-out 和 CV。

---

### 6.4 中风险：表型标准化发生在 hold-out 之前

当前 `load_data()` 会先对完整 `y` 计算：

```text
mean
std
```

然后才进行 hold-out。

因此测试集表型参与了标准化参数计算。

#### 建议

严格流程应调整为：

```text
先 hold-out
    ↓
只使用 y_train 计算 mean/std
    ↓
转换 y_train 和 y_test
    ↓
训练和评估
```

如果最后生成 deployment model，可以再使用全量 `y` 计算 deployment scaler。

---

### 6.5 中风险：CV 文件在完整样本集上生成

当前 CV 文件包含 hold-out test 样本的 fold 标记，虽然实际训练时这些样本会被排除。

#### 建议

更清晰的实现是：

```text
固定 hold-out
    ↓
保存 train_ids.txt / test_ids.txt
    ↓
仅为 train IDs 生成 cv_folds.csv
```

这样可以从文件层面证明测试样本从未出现在 CV 中。

---

### 6.6 中风险：CLI 无法关闭固定测试集

`--use_same_test_set` 当前是 `store_true`，但默认值也是 `True`。

因此用户无法通过 CLI 表达：

```text
use_same_test_set=False
```

#### 建议

改为互斥参数：

```text
--use-same-test-set
--vary-test-set-by-repeat
```

或者使用 Python 3.9+ 的：

```python
action=argparse.BooleanOptionalAction
```

支持：

```text
--use-same-test-set
--no-use-same-test-set
```

---

### 6.7 严格评估下的基因型 QC 边界

VCF 转换、缺失率过滤、MAF 过滤和 LD pruning 当前通常在 hold-out 前对全部样本执行。

这些操作不直接使用表型标签，因此不属于明显的监督学习泄漏。但如果要求完全独立测试集，测试样本的基因型分布仍会影响：

- SNP 是否被保留；
- MAF；
- 缺失率；
- LD pruning 结果；
- 插补参数。

#### 可选严格方案

```text
先按样本切分原始 VCF
    ↓
只在 Train VCF 上确定 QC/SNP 集合
    ↓
将相同 SNP 集合和处理参数应用到 Test VCF
```

该方案更严格，但实现和运行成本明显更高。项目需要根据“生产预测”还是“严格论文评估”决定是否采用。

---

## 7. 推荐重构流程

推荐将训练划分为三个明确阶段。

### 阶段 A：数据准备

```text
VCF/PLINK
    ↓
canonical SNP matrix
    ↓
样本 ID 和表型对齐
    ↓
保存 feature_manifest.json
```

### 阶段 B：模型评估

```text
固定 hold-out split
    ├── Train
    │     ↓
    │   仅 Train 生成 CV
    │     ↓
    │   Optuna
    │     ↓
    │   使用 CV/OOF 指标选择 base models
    │     ↓
    │   在完整 Train 上训练 evaluation model/stacking
    │
    └── Test
          ↓
        最终只评估一次
```

该阶段产生：

```text
split_manifest.json
cv_folds.csv
optuna_best_params.json
evaluation_model.pkl
holdout_metrics.json
```

### 阶段 C：部署模型

当 hold-out 性能确认后：

```text
best model + best params
    ↓
Train + Test 全量重训
    ↓
deployment_model.pkl
```

该模型用于：

```text
gpse predict
```

部署模型元数据必须说明：

```text
该模型使用了全部已知样本，不能再用于独立测试性能评估。
```

---

## 8. 推荐的最终目录结构

```text
results/
├── feature_manifest.json
├── split_manifest.json
├── train_ids.txt
├── test_ids.txt
├── cv_folds/
│   └── trait_cv_train_only.csv
├── models/
│   ├── rf_reg/
│   │   ├── best_params.json
│   │   ├── cv_metrics.json
│   │   ├── evaluation_model.pkl
│   │   └── deployment_model.pkl
│   └── xgboost_reg/
├── stacking/
│   ├── selected_models.json
│   ├── evaluation_model.pkl
│   └── deployment_model.pkl
└── reports/
    ├── holdout_metrics.csv
    ├── model_comparison_cv.csv
    └── model_comparison_holdout.csv
```

---

## 9. 需要团队确认的设计问题

建议团队重点讨论以下问题：

1. `gpse train` 是否应该在传入 `--vcf_file` 时自动启用预处理，而不再要求 `--enable_preprocess`？
2. hold-out 是否应该始终只切分一次并在所有 model/repeat 中固定？
3. `n_repeats` 应表示重复 CV、重复模型随机初始化，还是重复 hold-out？
4. TOPSIS 应只使用 CV/validation 指标，还是允许使用 hold-out 指标？
5. 是否需要同时保存 evaluation model 和 deployment model？
6. 分类任务是否默认使用 stratified hold-out 和 StratifiedKFold？
7. 表型标准化参数是否必须只在 hold-out Train 上拟合？
8. 论文级严格评估是否要求 QC/LD pruning 也只使用 Train VCF？
9. 如果模型缺失大量 SNP，`predict` 应继续填充、发出强警告，还是直接拒绝预测？

---

## 10. 当前验证状态

截至 2026-07-18，已完成以下验证：

```text
Python compileall                通过
完整 pytest                      17 passed
回归训练 smoke test             通过
三分类训练 smoke test           通过
Optuna 真实训练路径              通过
Stacking smoke test             通过
固定 CV seed=42 可重复性         通过
VCF canonical SNP 对齐          通过
缺失/多余 SNP 汇总              通过
```

本次审查期间已修复：

1. CV 折生成未固定 seed；
2. 非共享测试集索引丢失；
3. 多分类 fold ensemble 使用类别编号平均；
4. Optuna early stopping 未真正停止；
5. 分类预测明细为空；
6. Optuna 和最终 CV folds 之间复用 estimator。

---

## 11. 最终建议

当前 `gpse train` 已经可以正常完成：

```text
数据加载/转换
→ hold-out
→ CV
→ Optuna
→ 最佳参数 fold training
→ fold ensemble
→ hold-out 评估
→ 全量 deployment model
```

但如果目标是获得严格、可发表、可复核的独立测试性能，建议优先完成以下三项：

1. **模型选择完全移出 hold-out test，TOPSIS/Stacking 只使用 CV/OOF 指标。**
2. **明确区分 evaluation model 和 deployment model。**
3. **分类任务使用 stratified hold-out/CV，并将表型标准化移到 hold-out 之后。**

完成这三项后，训练流程的数据边界会更加清晰，测试集也能真正承担“最终、一次性、独立评估”的职责。

---

## 12. 2026-07-18 已完成的流程优化

本轮根据 `docs/26.7.18/GPSE训练流程优化方案.md` 完成了 P0、P1、P3 的核心训练边界改造，并补充了可复核的 P2 基线与调参审计。

### 12.1 已落地内容

1. **Stacking base model 选择移出测试集**：`model_comparison_cv.csv` 仅包含训练集 CV validation 的均值与标准差；TOPSIS 仅接收这些列。TOPSIS 异常时的回退排序同样仅使用 CV 均值（高优先级）和 CV 标准差（低优先级）。
2. **严格 OOF stacking**：每个 base model 使用已冻结的训练侧参数，在 `X_train` 内重新执行 K 折训练；每个 meta-train 样本都由未见过该样本的 fold model 预测。随后每个 base model 仅在完整 `X_train` 上重训一次以构造 `Z_test`；meta-learner 从不使用测试样本进行拟合或选择。
3. **不再加载 representative model 构造 stacking 特征**：representative model 会接触全量样本，不能用于 hold-out 评估；新版 stacking 从 `create_model(model_name, frozen_params)` 创建全新的 base model。
4. **固定 split 与 manifest**：默认共享 hold-out 时，pipeline 在所有模型训练前只生成一次 split，并写入 `split_manifest.json`、`train_ids.txt`、`test_ids.txt`。模型/repeat/stacking 共用该索引。
5. **分类分层**：hold-out 使用 `train_test_split(..., stratify=y)`，训练 CV 和 stacking OOF CV 使用 `StratifiedKFold`。样本量不足以完成分层时抛出包含类别计数的可操作错误。
6. **CLI 语义完整**：`--use_same_test_set` 使用 `BooleanOptionalAction`，可明确传入 `--no-use_same_test_set`。
7. **表型标准化仅拟合训练标签**：`load_data()` 保留原始表型；启用 `--standardize_phenotype` 后，每个 repeat 仅由 `y_train` 拟合 scaler 并变换 `y_train/y_test`。即使使用 `--enable_preprocess`，预处理阶段也会保留原始表型并将标准化延后到 hold-out 之后；`--preprocess_only --standardize_phenotype` 会被拒绝，避免写出由全量样本拟合的表型 scaler。共享 hold-out 的 scaler 写入 `split_manifest.json`，repeat 级 scaler 写入 `repeat_results.json`。
8. **CV 文件真实 train-only**：默认在 hold-out 后的 `X_train` 上生成 KFold/StratifiedKFold，并在每个 repeat 输出 `cv_train_only.csv`；用户显式传入 `--cv_file` 时仍可复用外部 folds。
9. **同构 deployment ensemble**：每个模型使用冻结参数在全量样本上重训 K 个成员，产出 `deployment_ensemble/member_*.pkl` 与五项 metadata；`gpse predict` 自动加载并对回归取均值、对分类取概率均值。
10. **稳健 Optuna 与 repeat 语义**：默认第一个 repeat 调参、后续 repeat 复用冻结参数；目标为 `mean(CV) - λ·std(CV)`（`--cv_stability_penalty` 默认 0.5），硬上限 100 trials，并保存 `optuna_best_params.json` 与 top-5 trials。
11. **GBLUP 基线**：新增 `gblup_reg`，以标准化 marker 的 linear-kernel `KernelRidge` 实现 GBLUP 等价的固定无调参基线，自动进入统一 hold-out/CV/比较表。

### 12.2 当前最终 evaluation pipeline

```text
输入基因型矩阵 + 表型
    ↓
对齐样本、删除缺失表型、写 feature_manifest.json
    ↓
固定 hold-out（分类：stratified）
    ↓
写 split_manifest.json / train_ids.txt / test_ids.txt
    ↓
每个模型、每个 repeat：仅在 X_train 上生成/筛选 CV folds
    ↓
Optuna 仅使用 train-CV 选择参数（mean − λ·std；默认 repeat 1 调参后冻结）
    ↓
用冻结参数训练 fold models，并以 fold ensemble 评估固定 X_test
    ↓
汇总 train-CV validation mean/std → model_comparison_cv.csv
    ↓
TOPSIS 仅使用 CV mean/std 选择 stacking base models
    ↓
每个被选 base model 在 X_train 内 K 折重训 → OOF meta features Z_train
    ↓
base model 在完整 X_train 重训 → Z_test
    ↓
Ridge / LogisticRegression meta-learner 在 Z_train 拟合
    ↓
对 Z_test 评估一次并写 stacking 结果；该指标不再参与任何选择
    ↓
全量数据上用同一冻结参数重训 K-member deployment ensemble
    ↓
gpse predict 对成员预测取均值/平均概率
```

### 12.3 生成的关键产物

```text
<results_dir>/
├── feature_manifest.json
├── split_manifest.json
├── train_ids.txt
├── test_ids.txt
├── <model>/repeat_<n>/cv_train_only.csv
├── <model>/repeat_<n>/optuna_best_params.json
├── <model>/deployment_ensemble/
│   ├── member_1.pkl ... member_K.pkl
│   └── info.json
├── model_comparison.csv                 # hold-out 汇总，仅用于报告
├── model_comparison_cv.csv              # stacking 选择的唯一输入
├── model_comparison_cv_topsis.csv
├── model_comparison_cv_topsis_simple.csv
├── stacking_selected_models.json
├── ensemble_stacking/
│   ├── stacking_ensemble_model.pkl
│   └── stacking_results.pkl
└── reports/
    ├── model_comparison_holdout.csv
    ├── holdout_metrics.csv
    ├── holdout_metrics.json
    ├── model_leaderboard.csv
    ├── run_summary.json
    ├── training_report.md
    └── training_report.html
```

### 12.4 后续完善与报告口径

12. **structure-aware split 与树模型约束**：`--split_strategy structure_aware` 使用标准化基因型 PCA（最多 10 个主成分）和 KMeans 聚类，再以簇为不可拆分 group 生成 hold-out；`split_manifest.json` 记录 PCA 维数、簇数、簇规模和实际 test 比例。使用相同输入分别执行 `--split_strategy random` 与 `--split_strategy structure_aware`，即可并列报告两种泛化口径。主要树模型搜索空间统一收紧为 `max_depth ≤ 6`，并提高叶节点/child-weight 的下界以降低小样本过拟合。
13. **最终 hold-out 报告与原尺度误差**：所有选择完成后输出 `reports/model_comparison_holdout.csv`、`reports/holdout_metrics.csv` 和 `reports/holdout_metrics.json`。同一结果目录会保留 random 与 structure-aware 两种策略的行，并显式标记 `gblup_reg` 基线。启用表型标准化时，fold ensemble 的 Spearman/MSE/RMSE/MAE 先反变换到原始表型尺度再汇总进报告。
14. **不平衡分类指标与样本门槛**：Optuna 分类目标使用 balanced accuracy；每个 binary/multiclass hold-out、fold ensemble 和最终报告都包含 PR-AUC。分类数据在切分前要求每个类别至少有 `n_splits + 1` 个样本，避免无法同时支撑 hold-out 与分层 CV 的隐性失败。
15. **structure-aware CV**：当 `--split_strategy structure_aware` 生效且未指定外部 `--cv_file` 时，训练集内重新执行 PCA/KMeans；分类使用 `StratifiedGroupKFold`，回归使用 `GroupKFold`。每个遗传簇只会出现在训练侧或验证侧，避免近缘材料跨 CV 折。
16. **代表性与部署参数选择**：多个 repeat 的代表性模型和 `deployment_ensemble` 统一按训练侧 CV 选择（validation 均值最高、标准差较低和 repeat 序号为确定性 tie-breaker）。hold-out/ensemble 测试指标只保留在最终报告中，绝不参与参数、repeat 或部署模型选择；`training_selection` 与 representative metadata 均记录 `source: train_only_cv`。
17. **fold-local SNP 特征选择**：`--feature_selection none|variance|univariate|mutual_info|tree_importance|lasso|elasticnet|pca|svd` 支持默认全特征、统计/模型驱动筛选及降维；`--select_k` 或优先级更高的 `--select_percentile` 控制保留 SNP 数或输出组件数，`--variance_threshold` 控制低方差过滤。每个 Optuna/CV/stacking OOF/deployment 分区仅在其训练侧拟合 selector，再对验证、hold-out 与预测数据执行 transform。新模型 artifact 保存 selector 和选中 SNP 列表或组件名，`gpse predict` 自动复用。
18. **fold-local 基因型插补**：`--genotype_imputation none|mean` 默认保留原始编码；选择 `mean` 时，将 `--missing_genotype_code`（默认 `3`）和 NaN 视为缺失，仅以当前训练分区的每个 SNP 均值拟合 `SimpleImputer`。artifact 持久化 imputer 与编码配置；预测端对缺失 SNP 使用 artifact 的编码对齐，并在 alignment report 记录插补配置。
19. **统一结果模块**：训练完成后生成只读的 `reports/model_leaderboard.csv`、`reports/run_summary.json`、`reports/training_report.md` 和 `reports/training_report.html`。报告汇总最终 hold-out 指标、train-only CV 选择记录、模型 artifact 路径、split manifest、特征选择/插补配置及失败模型；该模块仅消费已冻结结果，绝不参与模型选择。
20. **预测 SNP 覆盖率保护**：`gpse predict` 的 alignment report 记录训练 SNP 数、匹配数、缺失数、覆盖率和低覆盖率告警。默认兼容旧行为并继续填充缺失 SNP；设置 `--min-feature-coverage <0-1>` 后，覆盖率不足时会在预测前明确拒绝，避免静默生成不可信结果。
21. **统一模型 Pipeline artifact**：新保存模型使用 sklearn-compatible `GenotypePreprocessor -> model` Pipeline；`GenotypePreprocessor` 在同一训练分区内完成可选缺失基因型插补、SNP 筛选和标准化。Optuna、CV fold、Stacking 严格 OOF、representative 与 deployment 构建均直接使用该 Pipeline；artifact 同时保留组件字段以继续兼容旧的 `(model, scaler)` 加载路径。
22. **模型特定缩放策略**：树模型（RF、GBDT、XGBoost、LightGBM、CatBoost、AdaBoost、HistGradientBoosting）在统一 Pipeline 中使用 scaler passthrough，避免无意义的数据复制；线性、核方法、SVM、KNN 和 MLP 继续使用 `StandardScaler`。artifact 的 `feature_scaling` 字段记录实际策略。

当前 hold-out 指标已经不再用于任何模型、参数、repeat 或 stacking 的选择，也不参与 meta-learner 训练。
