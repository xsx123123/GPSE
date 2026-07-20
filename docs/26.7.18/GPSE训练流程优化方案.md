# GPSE 训练流程优化方案

> 版本：v1.0
> 日期：2026-07-18
> 依据：《GPSE Train Pipeline 逻辑评审文档》（2026-07-18）+ 第二轮独立评审意见
> 目标：将 `gpse train` 从「能跑且大体正确」提升为「可发表、可复核的严格独立评估流程」
> 使用方式：本文档可直接作为重构任务书，按 P0 → P3 顺序实施；每个任务卡可单独交给 CC / Claude Code 执行

> 实施状态（2026-07-18）：P0-1、P0-2、P1-1（含 balanced accuracy、PR-AUC 与类别门槛）、P1-2、P1-3、P2-1（PCA/KMeans group hold-out + group-aware CV）、P2-2、P2-3、P2-4、P3-1、P3-2、P3-3 已完成。所有选择（含 representative/deployment repeat）均在训练侧冻结，最终 hold-out CSV/JSON 仅作报告；structure-aware 与 random 结果通过分别运行两次 CLI 并列报告。完成细节和实际 artifact 见 `train_pipeline.md` 第 12 节。

---

## 0. 优化目标

当前 pipeline 的核心防线（Optuna 仅用 train CV、per-fold 新建 estimator、StandardScaler 仅在 fold_train 拟合、固定测试集）是正确的。本次优化解决四类问题：

1. **阻断性泄漏**：测试集参与了模型选择（TOPSIS/Stacking），这是唯一一处会让 headline 指标失效的真泄漏；
2. **口径一致性**：报告的指标必须描述实际交付的模型，evaluation 与 deployment 的职责、形式、元数据要分清；
3. **审稿人必查项**：分层抽样、表型标准化位置、群体结构对照、GBLUP baseline；
4. **工程整洁度**：CV 文件职责、CLI 表达力、manifest 完备性。

完成 P0 + P1 后，测试集可承担「最终、一次性、独立评估」的职责；完成 P2 后，结果达到 GP/GS 领域方法学论文的审稿标准。

---

## 1. 问题总览与优先级

| 编号 | 问题 | 风险 | 优先级 | 改动范围 |
|------|------|------|--------|----------|
| P0-1 | TOPSIS/Stacking 使用 test 指标选 base models | 高（指标失效） | P0 | `model_selection`、`stacking` |
| P0-2 | Stacking 缺少明确的 OOF 协议 | 高（隐性泄漏） | P0 | `stacking` |
| P1-1 | 分类 hold-out 未分层抽样 | 中高 | P1 | `data_split` |
| P1-2 | 表型标准化在 hold-out 之前 | 中 | P1 | `load_data` |
| P1-3 | evaluation 指标与 deployment 模型形式不一致 | 中高 | P1 | `representative model` |
| P2-1 | 随机划分未考虑遗传相关性/群体结构 | 高（论文说服力） | P2 | `data_split`、报告层 |
| P2-2 | 缺少 GBLUP/rrBLUP baseline | 中（论文说服力） | P2 | `models` |
| P2-3 | Optuna 目标函数在小样本 CV 上不稳 | 中 | P2 | `optuna objective` |
| P2-4 | repeat 语义混杂（参数搜索随机性 + 训练随机性） | 中 | P2 | 配置与文档 |
| P3-1 | CV 文件在完整样本集上生成再筛选 | 低 | P3 | `prepare_cv_folds` |
| P3-2 | CLI 无法关闭固定测试集 | 低 | P3 | `cli.py` |
| P3-3 | split/feature manifest 不完备 | 低 | P3 | 输出层 |

---

## 2. P0：阻断性泄漏修复

### 2.1 【P0-1】模型选择完全移出测试集

**现状**

```text
Test Pearson / Accuracy / F1 / AUC
    ↓
TOPSIS 排序选 top base models
    ↓
训练 Stacking
    ↓
在同一个 test 上报告 Stacking 性能   ← 测试集被用了两次
```

测试集先参与「选谁进 stacking」，再参与「评价 stacking」，最终指标必然偏乐观，且偏乐观的程度无法量化——这是审稿人一票否决级别的问题。

**改法**

TOPSIS 的输入指标只允许来自训练侧，按优先级取：

1. Optuna `best_value`（即 mean CV 指标，已在 train-only folds 上计算，零额外成本）；
2. 最终 fold 模型的 CV validation mean/std（用于稳定性加权）；
3. OOF 预测指标（见 2.2，训练 stacking 时天然产出）。

```python
# 改造前（禁止）
topsis_input[model] = {
    "pearson": metrics["test_pearson"],
    "r2": metrics["test_r2"],
}

# 改造后
topsis_input[model] = {
    "pearson": optuna_result["best_value"],          # mean CV Pearson
    "stability": 1.0 / (cv_std + 1e-8),              # 可选稳定性准则
}
```

**配套约束**

- `holdout_metrics.json` 的生成时机推迟到「所有选择全部冻结」之后；
- 在代码中约定：任何以 `test_` 开头的指标字典，只允许出现在 `evaluate_final()` 内部，禁止传入 `model_selection/` 与 `stacking/` 模块（可在 code review 与 lint 层面检查）；
- Stacking 训练完成后，对 test 只评估一次，且该结果不再回流到任何选择逻辑。

**验收**：全局搜索 `test_` 指标引用点，确认 `model_selection`、`stacking` 的输入中不存在任何 test 派生量。

---

### 2.2 【P0-2】Stacking 补 OOF 协议

**问题**

文档只规定了「不能用 test 选 base model」，但没有规定 meta-learner 的训练特征怎么来。Stacking 最常见的翻车点有两个：

1. 用 base model 对 train 的 **in-sample 预测**训练 meta-learner → 严重过拟合，stacking 反而不如最好的单模型；
2. 用 base model 对 test 的预测参与 meta-learner 拟合 → 泄漏。

**正确协议**

```text
对每个 base model m（参数已由 Optuna 冻结）：
    在 X_train 的 K 折 CV 上：
        fold k：用 fold_train_k 训练 model_m_k
                对 fold_val_k 预测 → 得到该折的 OOF 预测
    拼接 K 折 OOF → meta_feature_m（长度 = len(X_train)）

meta 特征矩阵 Z = [meta_feature_1, ..., meta_feature_M]
meta-learner g（建议 Ridge / LogisticRegression，低容量）：
    g.fit(Z, y_train)

最终评估（唯一一次触碰 test）：
    每个 base model 在完整 X_train 上重训 → 对 X_test 预测 → Z_test
    stacking 预测 = g.predict(Z_test)
    计算 test 指标，写入 holdout_metrics.json
```

**关键不变量（写进代码注释与测试）**

- meta-learner 看到的每一行训练特征，都是由「没见过该行样本」的 base model 产生的；
- `Z_test` 的每一列，由「在完整 X_train 上重训」的 base model 产生；
- meta-learner 用低容量模型（Ridge/LogisticRegression），正则强度可在 train 内再套一层小 CV 选择，**不允许**用 test 选。

**验收**：构造一个「完美记忆」的 dummy base model（train 上预测完美、泛化极差），stacking 的 OOF 指标应显著差于其 in-sample 指标——若两者接近，说明 OOF 协议被击穿。

---

## 3. P1：低成本、高审稿收益

### 3.1 【P1-1】分类任务 stratified hold-out + StratifiedKFold

```python
train_test_split(
    indices,
    test_size=test_size,
    random_state=random_seed,
    stratify=y if task_type == "classification" else None,
)
```

配套：

- CV 生成处同步使用 `StratifiedKFold`（分类）/ `KFold`（回归）；
- 切分前校验：每个类别样本数 ≥ `n_splits + 1` 且足以支撑 hold-out（建议每类 ≥ 10，否则拒绝切分并给出明确报错）；
- 类别不平衡时，Optuna 优化目标从 Accuracy 改为 **balanced accuracy** 或 **AUC**，报告层加 **PR-AUC**。

### 3.2 【P1-2】表型标准化移到 hold-out 之后

**现状**：`load_data()` 对完整 `y` 计算 mean/std → 再 hold-out，测试表型参与了标准化参数。实际泄漏量很小（仅 2 个自由度），但改动成本极低，且是审稿人顺手会挑的点。

**改法**

```text
load_data() 只保留原始 y
    ↓
hold-out split
    ↓
y_mean, y_std = fit(y_train)     # 只用训练集
    ↓
y_train_std, y_test_std = transform(...)
    ↓
训练与评估
```

配套：

- `y_scaler` 参数（mean/std）写入 `split_manifest.json`，保证可复现；
- deployment 阶段允许用全量 `y` 重新计算 deployment scaler，但必须在 metadata 中标注 `scaler_fitted_on: "all_samples"`；
- 多环境/多性状场景：按环境/性状分组内标准化。

### 3.3 【P1-3】evaluation 与 deployment artifact 形式对齐

**问题**：hold-out 指标来自 K 个 fold model 的 ensemble，而 representative model 是单个全量模型——报告的指标描述的不是交付的模型。

**推荐方案：deployment 也做成 fold-ensemble**

```text
评估确认后，在完整数据（Train + Test）上：
    重新分 K 折（新 seed）
    每折训练一个 model_k（best_params 冻结）
    保存 K 个 pkl + 各自的 scaler
gpse predict 时：
    K 个模型分别预测 → 回归取均值 / 分类取概率均值
```

好处：

- 交付形式与评估形式一致，指标归属清晰；
- K-fold ensemble 在全量数据上通常比单模型略好、方差更小；
- predict 端复用现有 feature_manifest 对齐逻辑，只是多加载几个 pkl。

**metadata 规范**（每个 artifact 的 `info.json` 必须包含）

```json
{
  "artifact_role": "deployment_ensemble",
  "n_members": 5,
  "trained_on": "all_samples",
  "performance_source": "holdout_fold_ensemble",
  "performance_belongs_to_saved_model": true,
  "note": "holdout 指标来自同构的 train 侧 fold-ensemble；本 artifact 用全量数据重训，不可再用于独立评估"
}
```

---

## 4. P2：论文说服力

### 4.1 【P2-1】群体结构感知划分（structure-aware split）

**动机**：GS 领域最大的隐性乐观来源。随机 hold-out 下，测试集中的近缘材料（同一组合后代、姊妹系、克隆）会让 Pearson 系统性偏高——流程上测试集没参与训练，但「泛化到新种质」的 claim 仍然虚高。做 GP/GS 的审稿人必查。

**方案**：两种划分并行，结果同时报告。

```text
划分 A（现有）：random hold-out
划分 B（新增）：structure-aware hold-out
    1. 由基因型构建 GRM（PLINK --make-grm-gz）或 PCA
    2. 对 GRM/PC 做层次聚类或 KMeans，得到簇标签
    3. 按簇整体划入 train 或 test（同一簇不跨边界）
    4. 控制 test 比例与 random split 一致
```

**报告要求**

- 主表同时给出 A、B 两种划分的 test 指标；
- B 的指标是唯一允许声称「泛化到新种质」的数字；A 的指标注明「泛化到同群体随机个体」；
- CV folds 同理可加 structure-aware 版本作为 supplementary（fold 内不放同簇材料）。

**验收**：`split_manifest.json` 记录 `split_strategy: random | structure_aware`、聚类参数、每簇样本数。

### 4.2 【P2-2】GBLUP / rrBLUP baseline

**动机**：基于 GRM 的 GBLUP 几乎没有超参搜索压力，在 n << p 的 GS 场景经常赢过树模型。没有它，审稿人无法判断 ML 模型是否带来真实增益——直接关系到 GPSE 的卖点成立。

**实施**

- 新增 `gblup` 模型类：由 genotype 构建 GRM，解 mixed model（可调用 `sommer`/`rrBLUP` 或直接用 `sklearn` 的 kernel ridge 以 GRM 为核矩阵，数学上等价）；
- 走完全相同的 hold-out / CV / 评估协议（GBLUP 无需 Optuna，`--use_default_params` 路径即可）；
- 报告层：`model_comparison_holdout.csv` 必须包含 GBLUP 行。

### 4.3 【P2-3】Optuna 目标稳健化

小样本下 mean CV Pearson 方差大，纯 mean 选出的 best trial 可能是运气好的 trial；且几百轮搜索本身就在对 CV 过拟合。

**改法**

- 目标函数改为 `mean - λ * std`（λ 默认 0.5，可配置），偏好稳健参数；
- 收紧树模型搜索空间：`max_depth ≤ 6`、`min_child_weight/min_samples_leaf` 设下界；
- trials 设硬上限（如 ≤ 100），并记录 `n_trials` 供报告；
- 保存 top-5 trials 的参数，检查参数稳定性（若 top trials 参数差异巨大，在报告中注明该模型超参面平坦/敏感）。

### 4.4 【P2-4】明确 repeat 语义

当前每个 repeat 单独跑 Optuna，repeat 间方差混杂「调参随机性」与「训练随机性」。二选一并写进 methods：

- **方案甲（推荐用于论文主表）**：repeat 1 调参，参数冻结后用于全部 repeats → repeats 衡量「模型训练稳定性」；
- **方案乙**：每 repeat 重新调参 → repeats 衡量「整条 AutoML 流水线的稳定性」。

无论选哪种，`config` 中增加显式开关 `optuna_per_repeat: true | false`，报告文案与之一致。

---

## 5. P3：工程整洁

### 5.1 【P3-1】CV train-only 生成

现状「全样本生成再按 train ID 筛选」无泄漏，但有两个副作用：fold 大小不再均衡（每折被随机抽走 test 样本）；极端情况下某折样本过少。

```text
固定 hold-out
    ↓
保存 train_ids.txt / test_ids.txt
    ↓
仅对 train IDs 生成 cv_folds.csv（seed=42 不变）
```

收益：文件层面即可证明测试样本从未出现在 CV 中；fold 均衡；代码表达与数据边界一致。

### 5.2 【P3-2】CLI 互斥参数

```python
parser.add_argument(
    "--use_same_test_set",
    action=argparse.BooleanOptionalAction,   # Python 3.9+
    default=True,
)
# 支持 --use-same-test-set / --no-use-same-test-set
```

### 5.3 【P3-3】manifest 完备性

每次训练产出：

```text
feature_manifest.json   # 有序 SNP 清单（已有）
split_manifest.json     # 新增：split 策略、seed、train/test IDs、y_scaler 参数、聚类参数（若 structure-aware）
```

---

## 6. 重构后完整流程

```text
阶段 A：数据准备
    VCF/PLINK → canonical SNP matrix → 样本/表型对齐
    → feature_manifest.json
    （QC/MAF/缺失率/LD pruning 维持全样本执行：无监督、领域惯例，
      论文级严格模式可选「Train VCF 定 QC 集合 → 应用到 Test」）

阶段 B：模型评估（test 只被触碰一次）
    固定 hold-out（分类 stratify；可选 structure-aware）
    → split_manifest.json / train_ids.txt / test_ids.txt
        ↓
    y_scaler 仅用 y_train 拟合
        ↓
    train-only CV folds
        ↓
    Optuna（目标 mean − λ·std，trials 上限）
        ↓
    best_params 冻结 → K 折 fold training
        ↓
    ┌───────────────────────────────┐
    │ TOPSIS：仅用 CV/OOF 指标选 base │
    │ Stacking：OOF 特征训练 meta     │
    └───────────────────────────────┘
        ↓
    冻结一切选择 → fold-ensemble 对 test 评估一次
    → holdout_metrics.json

阶段 C：部署
    best model + frozen params
    → 全量数据 K 折 fold-ensemble 重训
    → deployment_ensemble/（metadata 标注 trained_on: all_samples）
    → gpse predict 加载 K 个成员取均值
```

---

## 7. 目标目录结构

```text
results/
├── feature_manifest.json
├── split_manifest.json
├── train_ids.txt
├── test_ids.txt
├── cv_folds/
│   └── <trait>_cv_train_only.csv
├── models/
│   ├── gblup/                       # P2-2 新增 baseline
│   │   ├── cv_metrics.json
│   │   └── fold_models/
│   ├── rf_reg/
│   │   ├── optuna_best_params.json
│   │   ├── optuna_top5_trials.json
│   │   ├── cv_metrics.json
│   │   ├── evaluation_folds/      # train 侧 fold ensemble（产生 holdout 指标）
│   │   └── deployment_ensemble/   # 全量数据 fold ensemble（交付物）
│   └── xgboost_reg/
├── stacking/
│   ├── selected_models.json       # 选择依据：CV/OOF 指标（禁止 test）
│   ├── oof_predictions.csv
│   ├── meta_learner.pkl
│   └── deployment_ensemble/
└── reports/
    ├── holdout_metrics.csv                # 含 random / structure_aware 两行
    ├── model_comparison_cv.csv
    └── model_comparison_holdout.csv       # 含 GBLUP 行
```

---

## 8. 验收标准

实施完成后逐项核验：

```text
[x] 全局检索：model_selection、stacking 与 representative/deployment 选择中无任何 test 派生指标
[x] Stacking OOF 测试：完美记忆 dummy model 的 OOF 指标 << in-sample 指标
[x] 分类任务：hold-out 与每折的类别比例与全样本一致（stratify 生效）
[x] y_scaler 参数仅由 y_train 计算，且已写入 split_manifest.json
[x] 默认生成的 cv_train_only.csv 中不含任何 hold-out test 样本
[x] deployment_ensemble 的 info.json 五项 metadata 齐全
[x] 同一 best_params 下，evaluation 与 deployment 均为 K 成员 ensemble 形式
[x] 报告支持在同一结果目录累计 random 与 structure_aware 两种划分的 holdout 指标
[x] model_comparison_holdout.csv 含 GBLUP baseline
[x] optuna_best_params.json 记录目标函数形式（mean − λ·std）与 n_trials
[x] 回归/分类 smoke test、固定 seed 可重复性测试已覆盖并持续由 pytest 回归验证
```

---

## 9. 实施任务拆分（可直接下发）

| 任务 | 内容 | 依赖 | 预估 |
|------|------|------|------|
| T1 (P0) | TOPSIS 输入切换到 CV/OOF 指标；test 指标禁入选择模块 | 无 | 0.5 天 |
| T2 (P0) | Stacking 重写为 OOF 协议 + 低容量 meta-learner + dummy model 测试 | T1 | 1 天 |
| T3 (P1) | stratified hold-out / StratifiedKFold / 类别数校验 / 不平衡指标 | 无 | 0.5 天 |
| T4 (P1) | y 标准化移到 hold-out 后；scaler 入 manifest | 无 | 0.5 天 |
| T5 (P1) | deployment fold-ensemble + metadata 规范 + predict 端多成员加载 | T4 | 1 天 |
| T6 (P2) | structure-aware split（GRM/PCA 聚类 → 按簇划分）+ 双口径报告 | 无 | 1–2 天 |
| T7 (P2) | GBLUP baseline 接入统一评估协议 | 无 | 1 天 |
| T8 (P2) | Optuna 目标 mean − λ·std、搜索空间收紧、top-5 trials 落盘 | 无 | 0.5 天 |
| T9 (P2) | `optuna_per_repeat` 开关 + methods 口径说明 | T8 | 0.5 天 |
| T10 (P3) | CV train-only 生成；CLI BooleanOptionalAction；split_manifest | 无 | 0.5 天 |

建议顺序：T3/T4/T10 先行（独立、零风险）→ T1/T2（核心）→ T5 → T6–T9。

---

## 10. Methods 写作预埋

重构完成后，论文 methods 可直接按此口径撰写：

1. 固定 hold-out（分类分层；另设 structure-aware 划分作为稳健性对照），测试集在所有模型与超参冻结后仅评估一次；
2. 超参数经 Optuna 在训练集 K 折 CV 上优化，目标为 mean − 0.5·std 的 fold 指标；
3. 模型选择（含 stacking base model 筛选）仅使用训练集 CV/OOF 指标；
4. Stacking 的 meta 特征来自训练集 out-of-fold 预测，meta-learner 为岭回归/逻辑回归；
5. 表型标准化参数仅由训练集拟合；
6. 最终交付模型为全量数据上的 K 折 ensemble，与评估形式同构；
7. 以 GBLUP 为无调参 baseline 对照全部 ML 模型。

---

## 11. 范围外说明（有意不做）

- **QC/LD pruning 仅基于 Train VCF**：属于可选的论文级极致严格方案，实现与运行成本高，且 MAF/缺失率过滤不接触表型、属领域惯例。维持现状，但在 methods 中如实说明 QC 在全样本上执行；若审稿人明确提出，再启用 6.7 节的严格模式。
- **重复 hold-out（multiple random splits）**：当前 `use_same_test_set=True` 的 paired 设计利于模型间比较，保留；structure-aware 划分已提供第二个独立视角，足以支撑论文结论。
