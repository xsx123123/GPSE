# GPSE `train` / `predict` 模块审查报告

> 审查日期：2026-07-10
> 审查范围：`gpse/train/` 全部模块 + `gpse/predict/` 桩代码 + 相关 `gpse/convert/`、`gpse/utils/` 支撑代码
> 目的：(1) 核对 `train` 模块现阶段逻辑与代码是否准确；(2) 设计 `predict` 模块，尤其解决"他人 VCF 的 variant ID 与训练时不一致"的对齐问题。

---

## 第一部分：`train` 模块逻辑与代码核对

### 1.1 结论速览

**train 模块在默认 CLI 用法下逻辑是正确、可用的**，README 中描述的整体流程（加载 → CV → Optuna 调参 → 逐折训练 → fold-ensemble → 重复统计 → TOPSIS → Stacking）与代码一致。但存在 **1 个致命的设计缺陷（对 predict 影响极大）**、**若干潜在 bug（当前默认路径不触发，但改配置即暴露）** 和 **几个正确性隐患**。下面分级列出。

---

### 1.2 训练主流程（与 README 核对：准确）

实际调用链，与 README 的架构图一致：

```
cli.py main()
  └─ GenomicPredictorV2(...)                         # predictor.py
       └─ run_all_models()                           # _pipeline.py
            ├─ load_data()                           # _data_io.py
            ├─ prepare_cv_folds()                    # _cv_manager.py
            ├─ run_model_multiple_repeats() ×N model # _repeat_training.py
            │    └─ train_and_evaluate_model_for_repeat() ×n_repeats
            │         ├─ optimize_model_parameters() # _optimization.py (Optuna)
            │         ├─ _train_single_fold() ×folds # _fold_training.py
            │         ├─ _compute_ensemble_predictions() # _ensemble.py
            │         └─ _save_representative_model() # config/_topsis_config.py
            ├─ create_comparison_table → model_comparison.csv
            ├─ call_topsis_evaluator → *_topsis.csv   # topsis.py
            └─ StackingEnsemble.fit (可选)            # stacking.py
```

产物结构（README 描述准确）：

| 产物 | 内容（实测） |
| --- | --- |
| `{model}/repeat_i/fold_j_model.pkl` | `joblib.dump((model, scaler))` —— **元组 `(sklearn估计器, StandardScaler)`** |
| `{model}/representative_model/model.pkl` | 同上，`(model, scaler)` 元组；在**全量 X,y** 上重新拟合 |
| `{model}/representative_model/info.json` | `{repeat_idx, ensemble_pearson/accuracy, difference_from_avg}` |
| `ensemble_stacking/stacking_ensemble_model.pkl` | `joblib.dump(self)` —— **整个 `StackingEnsemble` 对象**（内嵌 base_models、meta_model、selected_models、task_type） |
| `label_encoder.pkl` | 分类任务的 `LabelEncoder` |
| `phenotype_scaler.json` | 回归标准化时的 `{mean, std, applied}` |

---

### 1.3 ⚠️ 致命设计缺陷（对 predict 决定性影响）

**特征（SNP）身份在训练过程中被完全丢弃，且不写入任何模型产物。**

- `gpse/train/_data_io.py:142`：
  ```python
  X.columns = [f"feature_{i}" for i in range(X.shape[1])]
  ```
  加载数据后，真实 SNP ID（来自 convert 生成的矩阵表头，即 `.map` 第 2 列）被**按位置**重命名为 `feature_0, feature_1, ...`。
- 每个 fold / representative 模型都是在 `scaler.fit_transform(X)` 的 **numpy 数组** 上 `fit` 的（`_fold_training.py`、`_topsis_config.py:142-145`），因此 sklearn 估计器**连 `feature_names_in_` 都没有**。
- 保存的 `.pkl` 只有 `(estimator, scaler)`，**没有任何地方记录列名、SNP ID、列顺序**。

**后果**：模型对输入的唯一约束是"列的数量和顺序必须与训练时逐位一致"。一旦用他人 VCF 预测，只要 SNP 集合/顺序不同，模型会：
1. 列数不同 → 直接报错；
2. 列数恰好相同但含义错位 → **静默给出完全错误的预测**（最危险）。

→ 这正是你问题 2 的根源，predict 模块必须自己补上"SNP 对齐层"。**强烈建议同时修一个小改动**：在 convert/train 阶段把训练用的 SNP 列表（有序）持久化成 sidecar 文件（详见第二部分 §2.4）。

---

### 1.4 潜在 Bug（当前默认 CLI 路径不触发，但需知晓）

| # | 位置 | 问题 | 触发条件 | 严重度 |
| --- | --- | --- | --- | --- |
| B1 | `_repeat_training.py:98,174-177` | 当 `use_same_test_set=False` 时，`prepare_train_test_data` 内部生成 test_indices 但**不回传**，调用方局部 `test_indices` 仍为 `None`，`len(test_indices)` / `list(test_indices)` 抛 `TypeError` | 需 `use_same_test_set=False` | **CLI 无法触发**：该 flag 是 `action="store_true"` 且 `default=True`，命令行永远为 True。仅程序化调用会踩雷 |
| B2 | `_optimization.py:149-166` | Optuna 早停回调 `return True` **不会**停止 `study.optimize`（Optuna 需 `study.stop()`）。`--patience` 实际是空操作 | 始终 | 中：调参不会提前停，只是浪费算力，不影响正确性 |
| B3 | `_ensemble.py:66,73-76` | 多分类 fold-ensemble 用 `round(mean(整数类别标签))` 投票 + 二列 one-hot 兜底，**多分类（>2 类）时错误** | `task_type=classification` 且 `n_classes>2` | 高（限多分类）：会产生错误的 ensemble_pred（如类0与类2平均成类1）。二分类正常 |
| B4 | `stacking.py:153,192` | `for model_name in self.selected_models:` 循环内 `self.selected_models.remove(...)`，迭代中改列表会跳过元素 | 某个 base 模型加载失败时 | 中：仅在模型缺失时错误跳过 |
| B5 | `stacking.py:315,331` | `LogisticRegression(multi_class='ovr')` 在新版 sklearn 已弃用/移除 | 分类 stacking + 新 sklearn | 中：可能告警或报错 |

---

### 1.5 正确性隐患（非崩溃，但影响结果解读）

1. **representative model 的指标与模型不匹配 + 数据泄漏**
   `_topsis_config.py:141-148`：representative `model.pkl` 在**全量 X,y（含测试集）** 上重训；但旁边 `info.json` 里的 `ensemble_pearson` 是 CV-ensemble 的分数，来自**另一个**模型。→ 保存的估计器的"真实泛化能力"并非 info.json 所示；且该模型见过全部样本，**适合用于对外预测，不适合用于报告性能**。（对 predict 是好事：representative model 用了全部数据，泛化更稳。）

2. **CV 折分配不可复现**
   `genomic_utils.py` 的 `prepare_cv_data` 用 Python `random.shuffle`，**未用 `self.random_seed` 播种**。同 seed 重跑会得到不同折 —— 唯一救命的是 CV 文件默认被缓存复用。若删掉 CV 文件重跑，折会变。

3. **CV 的 ID→位置映射静默丢样本**
   `_cv_manager.py:109-115`：`if idx in X_index_map` 会静默剔除对不上的 ID，只有 debug 日志。若 CV 文件与 X 的 ID 不完全一致，折可能变小甚至为空，无告警。

4. **fold 间复用同一个 `model` 对象**
   `_repeat_training.py:119` 创建一次，逐折 `model.fit` 原地重训。对标准 sklearn 估计器安全（`.fit` 会重置），但对支持 `warm_start` 的模型是潜在污染源。

5. **分类任务 `all_predictions.json` 为空**
   `_fold_training.py`：分类分支不往 `all_predictions` 追加，分类任务该文件恒为空 `{"train":[],"val":[],"test":[]}`，与回归行为不一致。

6. **TOPSIS 中间列错位**
   `topsis.py:132-145`：`df` 被过滤/排序后，`data_proc`/`data_norm` 未同步重排，`_positive`/`_norm` 诊断列与行错位。`TOPSIS_Score`/`Rank` 本身正确（排序前已赋值）。

---

### 1.6 `train` 模块小结

- **README 描述与实现一致，默认 CLI 用法可正确训练回归/二分类模型。**
- 立即需修：无（默认路径无崩溃）。
- 建议修（按优先级）：B3 多分类 ensemble > 隐患 §1.5.2 CV 播种 > B4/B5 stacking 分类健壮性 > B2 早停 > B1 死代码。
- **对 predict 最关键的一条：§1.3 特征身份丢失** —— 决定了 predict 模块的核心设计。

---

## 第二部分：`predict` 模块搭建方案

### 2.1 现状

`gpse/predict/cli.py` 仅是桩代码：解析 `--model / --geno-file / --out`，然后 `parser.error("predict is not implemented yet")`。需要从零实现预测逻辑。

### 2.2 predict 需要做的事（数据流）

```
他人 VCF ──convert(复用现有管线)──> 他人基因型矩阵(列=他人SNP ID)
                                          │
                                          ▼
                          ┌────────── SNP 对齐层（新增，核心）──────────┐
                          │ 用"训练时的有序 SNP 列表"重排/补齐/校验     │
                          │  - 交集/缺失 SNP 处理                        │
                          │  - 列顺序严格对齐训练顺序                    │
                          └─────────────────────────────────────────────┘
                                          │  (对齐后的 X, 列顺序==训练)
                                          ▼
              加载模型.pkl ──> scaler.transform(X) ──> estimator.predict()
                                          │
                          (回归且标准化过) 用 phenotype_scaler.json 反标准化
                          (分类) 用 label_encoder.pkl 把整数类别解码回原标签
                                          │
                                          ▼
                                   预测输出 CSV
```

### 2.3 variant ID 不一致问题 —— 解决方案

这是本次的重点。分三层解决：

#### (A) 训练侧：持久化"训练 SNP 清单"（推荐新增，一劳永逸）

当前 `.pkl` 不含 SNP 名（见 §1.3）。最稳妥的做法是在训练时把有序 SNP 列表存下来。有两种落地方式：

- **最小改动**：在 `_data_io.py` 重命名列 **之前**，把 `list(X.columns)`（即真实 SNP ID，顺序即训练顺序）写到 `{results_dir}/feature_snps.json`。predict 时读它作为"金标准列顺序"。
- **顺带修 §1.3**：让 representative model 保存时把 `feature_names` 一并存进产物（如 `joblib.dump({"model":m,"scaler":s,"feature_snps":cols,"task_type":..}, ...)`）。这样模型自解释、最健壮。

> 若不想改训练侧代码，也可退而用 **convert 阶段的基因型矩阵表头** 作为训练 SNP 清单来源——因为训练用的 geno_file 表头就是有序 SNP ID，predict 时要求用户同时提供"训练用的 geno_file（或其表头）"即可。但把清单固化进 results_dir 更省心。

#### (B) 预测侧：SNP 对齐算法（predict 模块核心逻辑）

拿到训练有序 SNP 列表 `train_snps`（长度 = 模型输入维度）和他人矩阵 `df_new`（列 = 他人 SNP ID）后：

```python
def align_snps(df_new, train_snps, missing_fill=3):
    # df_new: index=样本ID, columns=他人SNP ID, 值∈{0,1,2,3}
    new_cols = set(df_new.columns)
    present = [s for s in train_snps if s in new_cols]
    missing = [s for s in train_snps if s not in new_cols]

    # 1) 按训练顺序重排交集列
    aligned = df_new.reindex(columns=train_snps)   # 缺失列 → NaN

    # 2) 缺失 SNP 处理：填充为"缺失基因型"编码 3（与训练 missing 编码一致）
    #    或填充等位基因频率/众数——3 最保守，让模型把它当缺失看
    aligned = aligned.fillna(missing_fill)

    # 3) 报告对齐情况（供用户判断可信度）
    coverage = len(present) / len(train_snps)
    return aligned[train_snps], {
        "n_train_snps": len(train_snps),
        "n_matched": len(present),
        "n_missing": len(missing),
        "coverage": coverage,
        "missing_examples": missing[:20],
    }
```

关键点：
- **`reindex(columns=train_snps)` 是对齐的核心**：无论他人 SNP 顺序如何、是否多出无关 SNP，都会被强制重排成训练顺序，多余列自动丢弃，缺失列变 NaN。
- **缺失 SNP 填充策略**：默认填 `3`（训练时 missing 的编码），最保守；也可提供 `--missing-fill mode/mean` 选项。**必须暴露覆盖率**，coverage 太低（如 <0.8）应告警甚至拒绝。
- **ID 命名规则差异**：如果两边 SNP 命名规则不同（如 `chr1:12345` vs `1_12345_A_T` vs rsID），纯字符串交集会全空。此时需要**按位置（染色体+坐标）对齐**而非按 ID 字符串：
  - 让 predict 支持从 `.bim`/`.map` 读取 `chr:pos`（+ 可选 ref/alt）作为对齐键；
  - 提供 `--match-by {id, position}`，position 模式下用 `(chrom, pos)`（必要时校正等位基因方向 allele flip）对齐。
  - 这需要 predict 也接受 VCF/PLINK 输入并复用 `GenomicDataProcessor` 生成带位置信息的矩阵。

#### (C) 等位基因方向（allele flip）——进阶但重要

即便位置对齐，不同数据集的 REF/ALT 可能相反，导致 `0↔2` 翻转。若能拿到训练与预测双方的 ref/alt，应检测并对翻转的 SNP 做 `2-x` 校正。首版可先不做，但在文档里明确标注"假设双方 ref/alt 一致"，并在 §2.5 里列为 roadmap。

### 2.4 predict CLI 建议接口

```bash
gpse predict \
    --model output_results/ensemble_stacking/stacking_ensemble_model.pkl \
    # 或 --model output_results/xgboost_reg/representative_model/model.pkl
    --geno-file others.vcf            # 支持 VCF / PLINK / 已转好的矩阵
    --train-snps output_results/feature_snps.json  # 训练有序SNP清单(或自动从模型内读)
    --match-by id                     # id | position
    --missing-fill 3                  # 缺失SNP填充: 3 | mode | mean
    --min-coverage 0.8                # 覆盖率阈值,低于则告警/中止
    --task-type regression            # 可从模型/info自动推断
    --phenotype-scaler output_results/phenotype_scaler.json  # 回归反标准化(可选)
    --label-encoder output_results/label_encoder.pkl         # 分类解码(可选)
    --out predictions.csv
```

输出 `predictions.csv`：

```csv
ID,predicted_value        # 回归(已反标准化)
sample_x,12.7
```
```csv
ID,predicted_label,proba_class0,proba_class1   # 分类(已解码原标签)
sample_x,Resistant,0.2,0.8
```
另写一个 `predictions_alignment_report.json`（覆盖率、缺失 SNP 数、填充策略），保证可追溯。

### 2.5 加载模型的两种情形（实测契约）

1. **单模型** `representative_model/model.pkl` 或 `fold_j_model.pkl`：
   ```python
   model, scaler = joblib.load(path)          # 元组 (estimator, StandardScaler)
   X_scaled = scaler.transform(X_aligned)     # X_aligned 列序==训练序
   pred = model.predict(X_scaled)             # 分类可用 predict_proba
   ```
2. **Stacking** `stacking_ensemble_model.pkl`：
   ```python
   ensemble = joblib.load(path)               # 一个 StackingEnsemble 对象
   pred = ensemble.predict(X_aligned)         # base_models 已内嵌,内部自带 scaler.transform
   ```
   注意：`StackingEnsemble.predict` 内部对每个 base 模型走 `(model,scaler)` 元组分支 → `scaler.transform(X)` → 生成 meta 特征 → meta_model 预测。**同样只按位置对齐，无 SNP 名校验**，所以对齐层必须在喂进去之前完成。

### 2.6 实施步骤（建议顺序）

1. **(训练侧小改)** 在 `_data_io.py` 重命名列前落盘 `feature_snps.json`（有序真实 SNP ID）。低风险、收益大。
2. **(predict 骨架)** 实现 `gpse/predict/` 下：`_io.py`（加载模型/清单/矩阵）、`_align.py`（SNP 对齐，§2.3B）、`_predict.py`（单模型 + stacking 两条路径）、改写 `cli.py`。
3. **(输入多态)** predict 接受矩阵/VCF/PLINK：VCF/PLINK 复用 `GenomicDataProcessor` 转矩阵（不需 pheno）。
4. **(反变换)** 回归接 `phenotype_scaler.json` 反标准化；分类接 `label_encoder.pkl` 解码。
5. **(健壮性)** 覆盖率报告 + `--min-coverage` 闸门 + 对齐报告 JSON。
6. **(进阶 roadmap)** position 对齐、allele-flip 校正、缺失填充策略扩展。

### 2.7 顺带建议修复的 train 侧问题（与 predict 相关）

- **§1.3**：把 SNP 有序清单持久化（predict 前置依赖）。
- **B3 多分类 ensemble**：若目标支持多分类预测，必须先修 `_ensemble.py` 的投票逻辑（改用 `argmax(ensemble_proba)`）。
- **B5 stacking 分类**：新版 sklearn 下 `multi_class='ovr'` 需处理，否则分类 stacking 模型可能训不出/预测报错。

---

## 附：关键文件/行号索引

| 主题 | 位置 |
| --- | --- |
| 特征名被丢弃 | `gpse/train/_data_io.py:142` |
| representative 全量重训+保存 | `gpse/config/_topsis_config.py:141-148` |
| fold 模型保存 `(model,scaler)` | `gpse/train/_fold_training.py:142-143` |
| stacking 整对象保存 | `gpse/train/stacking.py:456` |
| stacking predict 契约 | `gpse/train/stacking.py:371-419` |
| test_indices 死代码崩溃 | `gpse/train/_repeat_training.py:98,174` |
| Optuna 早停无效 | `gpse/train/_optimization.py:149-166` |
| 多分类 ensemble 错误 | `gpse/train/_ensemble.py:66,73-76` |
| CV 折未播种 | `gpse/utils/genomic_utils.py`（`prepare_cv_data`，`random.shuffle`） |
| 基因型矩阵表头=真实SNP ID | `gpse/convert/genotype_matrix.py:228,254`（读 `.map` 第2列） |
| predict 桩 | `gpse/predict/cli.py:53` |

---

## 补充：人工复核记录（2026-07-10）

> 复核人：Kimi Code CLI
> 复核方式：直接读取当前 `HEAD` 对应源码，逐条核对报告结论。

### 总体结论

**审计报告中列出的问题在当前代码中基本全部真实存在**。默认 CLI 路径下训练仍可跑通，但报告中的 1 个致命设计缺陷、5 个潜在 bug、6 个正确性隐患均已确认。

### 逐项核对

#### 1. 致命设计缺陷 §1.3 —— 确认存在

- `gpse/train/_data_io.py:142` 确实把 `X.columns` 重命名为 `feature_0, feature_1, ...`，真实 SNP ID 被丢弃。
- 训练全流程使用 `scaler.fit_transform(X)` 后的 numpy 数组拟合，sklearn 估计器不会记录 `feature_names_in_`：
  - `_fold_training.py:46-53`
  - `_topsis_config.py:141-145`
  - `_optimization.py:90-91`
- 产物仅保存 `(model, scaler)` 元组，未记录列名或 SNP 顺序：
  - `_fold_training.py:143`
  - `_topsis_config.py:148`

**影响**：predict 模块必须自行实现 SNP 对齐层；否则他人 VCF 列数/列序不一致时会报错或给出静默错误结果。

#### 2. 潜在 Bug —— 全部确认存在

| 编号 | 核对结果 | 关键位置 |
|------|----------|----------|
| **B1** | **存在**。`prepare_train_test_data` 不返回 `test_indices`；当 `use_same_test_set=False` 传入 `None` 时，`_repeat_training.py:98` 的 `len(test_indices)` 会抛 `TypeError`。 | `_repeat_training.py:95-98` |
| **B2** | **存在**。回调仅 `return True`，未调用 `study.stop()`，`--patience` 实际为空操作。 | `_optimization.py:149-166` |
| **B3** | **存在**。多分类时对整数标签取平均后四舍五入；且 lines 73-76 硬编码 2 列 one-hot，>2 类时 shape 错误。 | `_ensemble.py:66,73-76` |
| **B4** | **存在**。`for model_name in self.selected_models:` 循环内执行 `self.selected_models.remove(model_name)`，会跳过元素。 | `stacking.py:153,192` |
| **B5** | **存在**。`LogisticRegression(multi_class='ovr')` 已在新版 sklearn 中弃用/移除。 | `stacking.py:315,331` |

#### 3. 正确性隐患 —— 全部确认存在

| 编号 | 核对结果 | 关键位置 |
|------|----------|----------|
| §1.5.1 | **存在**。representative model 在完整 `X, y` 上重训，但 `info.json` 记录的是 CV-ensemble 指标，二者并非同一模型。 | `_topsis_config.py:141-148` |
| §1.5.2 | **存在**。`prepare_cv_data` 中 `random.shuffle(phe_index)` 前未播种，同 seed 重跑折会不同。 | `gpse/utils/genomic_utils.py:68` |
| §1.5.3 | **存在**。`generate_cv_folds_from_file` 用 `if idx in X_index_map` 静默过滤，无告警。 | `gpse/utils/genomic_utils.py:271-272` |
| §1.5.4 | **存在**。`_repeat_training.py:119` 只创建一次 `model`，逐 fold 原地 `fit`；对 `warm_start` 模型有污染风险。 | `_repeat_training.py:119` |
| §1.5.5 | **存在**。`_fold_training.py` 分类分支未调用 `save_fold_predictions_and_plots`，`all_predictions.json` 恒为空。 | `_fold_training.py:61-90` |
| §1.5.6 | **存在**。`topsis.py` 在 `df` 过滤排序后，直接把 `data_proc`/`data_norm` 按列位置赋值，中间诊断列与行错位。 | `gpse/train/topsis.py:132-145` |

#### 4. predict 模块现状 —— 确认存在

- `gpse/predict/cli.py:53` 仍是桩代码，直接 `parser.error("predict is not implemented yet")`。

### 默认 CLI 是否仍能训练？

**可以**。原因是：
- `use_same_test_set` 默认 `True`（`workflow.py:155`），B1 不触发；
- 二分类场景下 B3 的 `round(mean(labels))` 和 2 列 one-hot 刚好正确；
- B2/B4/B5 不影响默认回归/二分类流程跑通。

但改配置、做多分类、使用新版 sklearn 或程序化调用时，上述 bug 会暴露。

### 修复优先级建议

1. **最高**：§1.3 持久化训练 SNP 清单（predict 的前置依赖，必须做）。
2. **高**：B3 多分类 ensemble 投票逻辑、B5 新版 sklearn `multi_class` 兼容。
3. **中**：§1.5.2 CV 播种、§1.5.3 丢样本告警、B4 迭代中删列表。
4. **低**：B1 死代码清理、B2 早停真正生效、§1.5.6 TOPSIS 列对齐。


---

## 补充：Codex 复核记录（2026-07-10）

> 复核方式：直接读取当前工作区源码，按报告列出的缺陷逐项检索和查看上下文；随后运行现有测试套件。
> 工作区状态：`docs/train_predict_audit_report.md` 当前为未跟踪文件，复核过程中未修改训练/预测源码。

### 结论

当前 `gpse` 中，报告列出的主要问题仍然存在。默认训练路径仍可运行，但 `predict` 相关的核心前置问题（训练 SNP 身份和顺序未持久化）尚未解决，`gpse predict` 也仍未实现。

### 本次确认仍存在的问题

| 问题 | 当前状态 | 证据 |
| --- | --- | --- |
| 训练 SNP 身份丢失 | **存在**。训练数据在进入模型前仍被重命名为 `feature_0...`，真实 SNP ID 未保存。 | `gpse/train/_data_io.py:142` |
| 模型产物不带 SNP 清单 | **存在**。fold / representative 模型仍只保存 `(model, scaler)`。 | `gpse/train/_fold_training.py:143`；`gpse/config/_topsis_config.py:148` |
| `predict` 模块未实现 | **存在**。CLI 解析后直接报错。 | `gpse/predict/cli.py:53` |
| B1 `use_same_test_set=False` 崩溃风险 | **存在但默认 CLI 不触发**。`prepare_train_test_data()` 内部生成的 `test_indices` 不回传，调用方仍可能对 `None` 调 `len()` / `list()`。 | `gpse/train/_repeat_training.py:98,174-177`；`gpse/utils/genomic_utils.py:403-436` |
| B2 Optuna 早停无效 | **存在**。callback 只 `return True`，没有调用 `study.stop()`。 | `gpse/train/_optimization.py:149-166` |
| B3 多分类 ensemble 错误 | **存在**。仍对整数类别预测求均值再四舍五入，并且 fallback probability matrix 硬编码 2 列。 | `gpse/train/_ensemble.py:66,73-76` |
| B4 Stacking 迭代中修改列表 | **存在**。遍历 `self.selected_models` 时执行 `remove()`。 | `gpse/train/stacking.py:153-192` |
| representative model 指标与模型不完全对应 | **存在**。模型在全量 `X,y` 上重新拟合，但 `info.json` 记录的是 CV-ensemble 指标。 | `gpse/config/_topsis_config.py:141-148` |
| CV 折分不可复现 | **存在**。`prepare_cv_data()` 使用 `random.shuffle()`，函数签名和调用处未传入 `random_seed`。 | `gpse/utils/genomic_utils.py:52-75` |
| CV 文件 ID 不匹配时静默丢样本 | **存在**。ID 映射用列表推导中的 `if idx in X_index_map` 静默过滤。 | `gpse/train/_cv_manager.py:109-115`；`gpse/utils/genomic_utils.py:269-272` |
| fold 间复用同一个模型对象 | **存在**。每个 repeat 只创建一次 `model`，再传入每个 fold 训练。 | `gpse/train/_repeat_training.py:119-144` |
| 分类 `all_predictions.json` 为空 | **存在**。分类分支只计算指标，不调用 `save_fold_predictions_and_plots()` 写预测明细。 | `gpse/train/_fold_training.py:61-90` |
| TOPSIS 中间诊断列错位 | **存在**。`df` 过滤/排序后，`data_proc` / `data_norm` 未同步重排即赋值。 | `gpse/train/topsis.py:131-145` |

### 对原报告 B5 的补充校正

`LogisticRegression(multi_class="ovr")` 的兼容性风险存在，但在当前项目锁定和本地环境的 `scikit-learn==1.7.2` 下不是立即报错：

- `pyproject.toml` / `requirements.yaml` 均指定 `scikit-learn 1.7.2`。
- 本地验证 `LogisticRegression(multi_class="ovr")` 可以构造并 fit。
- fit 时会产生弃用警告：`multi_class` 从 sklearn 1.5 起 deprecated，预计 sklearn 1.8 移除。

因此 B5 当前应表述为：**当前环境为弃用警告，升级 sklearn 1.8 后可能变为硬错误；建议提前改为默认策略或显式 `OneVsRestClassifier(LogisticRegression(...))`。**

相关位置：

- `gpse/train/stacking.py:315`
- `gpse/train/stacking.py:331`

### 现有测试结果

本次运行：

```bash
python3 -m pytest -q
```

结果：

```text
13 passed in 0.33s
```

注意：现有测试未覆盖上述 train / predict 审计问题，因此测试通过只能说明当前已有测试集通过，不能说明这些问题已解决。

### 当前修复建议（按实际阻塞程度）

1. 先在训练侧持久化有序 SNP 清单，例如在 `_data_io.py` 重命名前保存 `feature_snps.json`，这是实现可靠 `predict` 的前置条件。
2. 实现 `gpse/predict` 的模型加载、矩阵读取、SNP 对齐、覆盖率报告、回归反标准化和分类 label 解码。
3. 修复多分类相关逻辑：`_ensemble.py` 投票/概率聚合，以及 stacking 的 `multi_class="ovr"` 兼容写法。
4. 修复可复现性与静默失败问题：CV 播种、CV ID 不匹配告警、Optuna early stop、列表遍历中删除元素。
5. 最后处理诊断与报告一致性问题：分类预测明细、TOPSIS 中间列对齐、representative model 指标说明。
