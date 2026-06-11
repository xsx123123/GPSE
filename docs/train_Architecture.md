# GPSE 训练流程架构审查报告

> 审查范围：`gpse/train/` 核心训练管道  
> 审查重点：标准表型预测模型构建流程（带超参数调优）的符合度与数据泄露风险

---

## 总评

GPSE 的**核心调优哲学**和标准流程**基本一致**，但在**数据划分策略**、**超参数搜索方式**和**最终模型的交付形态**上有几个关键差异。

---

## 第一阶段：准备与定义 ✅ 基本一致

| 标准流程 | GPSE 实际做法 |
|---------|-------------|
| 读取突变矩阵 + 表型数据 | `load_data()` 载入基因型/表型文件，取交集样本，对齐索引 |
| 定义参数搜索空间（如 SVM 的 C 和 kernel） | `*_model_optimizer.py`（回归/分类各自一份）通过 Optuna 的 `suggest_*` 定义搜索空间 |

**差异点**：GPSE 用的是 **Optuna（贝叶斯优化 / TPE）**，而不是 GridSearchCV（暴力网格遍历）。这不是错误，而是更高效的实现。

---

## 第二阶段：核心调优循环（k-fold CV）⚠️ 部分符合，但有结构差异

### 2.1 数据划分策略：GPSE 比标准流程多了一层 "外层 Test Split"

**标准流程描述**：
> 将**当前可用数据**（即 100% 数据）划分为 k 份，然后做 k-fold CV。

**GPSE 实际流程**（见 `run_model_multiple_repeats`）：

```python
# 1. 先从完整 X 中 hold out 20% 作为固定的 test_indices
_, test_indices = train_test_split(range(len(X)), test_size=0.2, ...)

# 2. 剩下的 80% 作为 X_train / y_train
# 3. 仅在 X_train（80%）上做 k-fold CV 用于超参数调优和 fold 训练
```

也就是说，GPSE 采用的是 **Train(80%) / Test(20%) → 在 Train 上做 k-fold CV** 的嵌套结构。这实际上比"把全部数据扔进 k-fold"更符合机器学习最佳实践（保留了完全独立的 Test Set 用于无偏估计），但**确实和字面描述的标准流程不一致**。

### 2.2 内层 CV 循环：✅ 符合标准

在 `optimize_model_parameters()`（`_optimization.py`）中，对于每一个 Optuna trial（即"候选参数组合"）：

```python
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_fold_train = X.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]
    ...
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)  # 只在 train fold fit
    X_fold_val_scaled = scaler.transform(X_fold_val)           # transform val
    model.fit(X_fold_train_scaled, y_fold_train)
    score = evaluate_on(val)   # Pearson r 或 accuracy
mean_score = np.mean(scores)   # 均分即该参数组合的最终成绩
```

这与标准流程的 **"k-1 训练 / 1 验证 / 循环 k 次 / 求平均"** 完全一致。

### 2.3 特征筛选（Data Leakage 检查）：✅ 当前无此步骤，故无泄露

> ⚠️ **高能防错点**：如果 GPSE 里有降维或特征筛选（比如选出 Top 相关的突变位点），这一步必须只在 k-1 的"训练集"上计算权重/P值，然后把选出来的位点规则套用到那个留出的"验证集"上。绝对不能在第一阶段用全量数据提前做筛选，否则会造成严重的"数据泄露（Data Leakage）"。

**结论：当前 GPSE 核心训练管道中没有任何特征筛选或降维操作。**

从 `load_data()` 到 `optimize_model_parameters()` 再到 `_train_single_fold()`，所有特征都是原样送入模型的。`StandardScaler` 虽然在每个 fold 的 train 上 fit、在 val 上 transform，但这只是缩放，不是筛选。

因此**目前不存在"特征筛选导致的数据泄露"**。但如果未来打算加入 SNP 筛选（比如按 P-value 或相关性选 Top-K 位点），务必遵守上述防错规则：只在 `X_fold_train` 上算权重，再把选出来的列索引套用到 `X_fold_val` 上。

---

## 第三阶段：模型决断（Model Selection）✅ 符合

Optuna 的 `study.best_params` 就是"对比参数池里所有组合的平均得分，选出最高分"。GPSE 在优化结束后会记录：

```python
best_params = study.best_params
```

并在日志中输出最优 CV 成绩。这与标准第三阶段完全一致。

---

## 第四阶段：终极合体（Final Training）⚠️ 交付形态有差异

**标准流程描述**：
> 废弃旧模型 → 仅保留最优参数 → 在 **100% 完整数据** 上训练最终模型 → 对外服务。

GPSE 在这里做了**两件事**，而不是一件事：

| 标准期望 | GPSE 实际做法 |
|---------|-------------|
| 丢弃 CV 中所有临时模型 | ⚠️ **没有完全丢弃**。如果 `save_models=True`，每个 fold 的模型都会被保存为 `.pkl` |
| 用 100% 数据训练一个最终模型 | ✅ **确实做了**。在 `_save_representative_model()` 中，用 `best_params` 在完整 `X`（注意：这里的 X 是包含之前 hold out 的 20% test set 的 **100% 全量数据**）上训练并保存为 `representative_model/model.pkl` |
| 这个最终模型就是对外服务的实体 | ✅ 是的，representative model 就是用于部署的模型 |

**但有一个细微但重要的问题**：

因为 GPSE 之前 hold out 了一个 Test Set（20%）用于评估，而这个 Test Set 的评估并不是由"representative model"完成的——它是由**第二阶段保留下来的 k 个 fold 模型对 Test Set 做 ensemble 平均预测**来完成的（见 `_compute_ensemble_predictions()`）。

换句话说：
- **Test 指标** → 来自 k 个 fold 模型的平均预测（模型只在 80% Train 的子集上训练过，所以 Test 评估是无偏的）。
- **Representative 模型** → 在 100% 数据上重新训练，只用于保存/部署，**不参与任何已报告指标的评估**。

从工程角度看这是合理的（deployment 模型应该用全量数据），但严格来说，如果以后用这个 `representative_model` 回头去预测那 20% Test Set，就会造成数据泄露（因为 representative model 见过它们）。

---

## 额外发现：Stacking Ensemble 中的 Scaler 复用

如果使用了 `--use_stacking`，在 `StackingEnsemble.create_meta_features()` 中：

```python
actual_model, scaler = model   # scaler 来自 representative model（在 100% 数据上 fit）
fold_model = clone(actual_model)
X_fold_train_scaled = scaler.transform(X_fold_train)
fold_model.fit(X_fold_train_scaled, y_fold_train)
meta_train[val_idx, i] = fold_model.predict(X_fold_val_scaled)
```

这里用来生成 meta-feature 的 `scaler` 是 representative model 附带的、在 **100% 数据（含 Test）** 上 fit 的 scaler。虽然 `fold_model` 本身是重新 clone 并在 fold train 上训练的，但 scaler 的参数（mean/std）包含了全部样本的信息。这在严格意义上属于**轻微的 data leakage**（虽然对 StandardScaler 来说影响通常很小）。

---

## 总结对照表

| 环节 | 是否符合标准流程 | 说明 |
|------|----------------|------|
| 数据载入 + 参数空间定义 | ✅ 符合 | Optuna 替代了 GridSearch，更高效 |
| **外层数据划分** | ⚠️ **不符合** | GPSE 先切 80/20 Test，再在 80% 上做 k-fold；标准流程是全量数据直接 k-fold |
| 内层 k-fold CV 循环 | ✅ 符合 | k-1 训练 / 1 验证 / 均分 |
| **特征筛选防泄露** | ✅ **当前无风险** | 代码中没有任何特征筛选；但如果未来添加，必须放在 fold 内部 |
| Scaler 防泄露 | ✅ 符合 | `fit` 只在 fold train，`transform` 在 val |
| 模型选择（Best Params） | ✅ 符合 | Optuna `study.best_params` |
| 最终模型训练于 100% 数据 | ✅ 符合 | `_save_representative_model` 在完整 `X, y` 上训练 |
| **废弃旧模型** | ⚠️ **不完全符合** | Fold 模型被保留用于 ensemble 评估和 stacking；representative 是额外训练的 |
| **最终模型与 Test 评估的关系** | ⚠️ **需注意** | Test 指标来自 fold ensemble，不是 representative model；representative model 见过 Test 数据，不应再用于评估同一个 Test Set |

---

## 一句话结论

> GPSE 的训练流程在"防数据泄露"和"CV 内层循环"上是**正确且安全的**，但它采用了 **Train/Test Split → 在 Train 上做 k-fold CV → 用 fold 模型 ensemble 评估 Test → 另用 100% 数据训练 representative 模型** 的嵌套结构，这和字面描述的"全量数据直接做 k-fold CV，然后拿最优参数在 100% 数据上训一个最终模型"在**数据划分层级**和**模型交付逻辑**上存在差异。如果未来加入特征筛选，请务必确保筛选逻辑发生在每个 fold 的 `train_idx` 内部。
