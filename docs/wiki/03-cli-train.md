# 03. 子命令 `gpse train`

训练基因组预测模型：Optuna 超参优化、重复交叉验证、TOPSIS 排名、Stacking 集成。实现位于 `gpse/train/`（CLI：`gpse/train/cli.py`，参数解析：`gpse/train/workflow.py`）。

## 最小示例

```bash
gpse train --task_type regression \
  --geno_file geno.csv --pheno_file pheno.csv --target_trait yield
```

分类任务额外要求 `--n_classes`：

```bash
gpse train --task_type classification --n_classes 3 \
  --geno_file geno.csv --pheno_file pheno.csv --target_trait grade
```

## 参数分组

### 基本

| 参数 | 默认 | 说明 |
|------|------|------|
| `--task_type` | 必需 | `regression` / `classification` |
| `--geno_file` `--pheno_file` `--target_trait` | 必需（常规模式） | 基因型矩阵、表型文件、目标性状 |
| `--n_classes` | — | 分类任务类别数 |
| `--results_dir` | `optimization_results_v2` | 结果输出目录 |
| `--random_seed` | 42 | 随机种子 |

### 模型与优化

| 参数 | 默认 | 说明 |
|------|------|------|
| `--models` | 全部 | 见下方模型清单 |
| `--trials` | 100 | Optuna 试验次数 |
| `--patience` | 20 | 早停耐心 |
| `--use_default_params` | False | 跳过优化，使用默认参数 |
| `--optuna_per_repeat` / `--no-optuna_per_repeat` | False | 每个 repeat 独立调参 |
| `--cv_stability_penalty` | 0.5 | CV 稳定性惩罚系数 |
| `--save_models` / `--save_representative` | — | 保存模型 / 代表性模型 |

回归模型（15）：`elasticnet_reg, gbdt_reg, svr_reg, mlp_reg, knn_reg, rf_reg, xgboost_reg, adaboost_reg, lightgbm_reg, catboost_reg, kernelridge_reg, gblup_reg, histgradientboost_reg, sgd_reg, lasso_reg`
分类模型（6）：`rf_clf, xgboost_clf, lightgbm_clf, catboost_clf, svm_clf, mlp_clf`

### 交叉验证

| 参数 | 默认 | 说明 |
|------|------|------|
| `--test_size` | 0.2 | 固定 hold-out 比例 |
| `--train_folds` | 5 | 训练集 CV 折数（n_splits） |
| `--n_repeats` | 100 | 重复次数 |
| `--cv_file` / `--force_new_cv` | — | 复用/强制重建 CV 划分 |
| `--cv_id_column` | `ID` | CV 文件中的 ID 列 |
| `--use_same_test_set` / `--no-use_same_test_set` | True | 各模型共享同一测试集 |
| `--split_strategy` | `random` | `random` / `structure_aware`（配 `--structure_clusters`） |

### 特征与插补

| 参数 | 默认 | 说明 |
|------|------|------|
| `--feature_selection` | `none` | `variance / univariate / mutual_info / tree_importance / lasso / elasticnet / pca / svd` |
| `--select_k` | 5000 | 特征数上限 |
| `--select_percentile` / `--variance_threshold` | — | 百分位 / 方差阈值 |
| `--genotype_imputation` | `none` | `none` / `mean` |
| `--missing_genotype_code` | 3 | 缺失基因型编码 |
| `--standardize_phenotype` | False | 表型标准化 |

### Stacking 集成

| 参数 | 默认 | 说明 |
|------|------|------|
| `--use_stacking` | False | 启用 Stacking |
| `--top_n_models` | 5 | 基模型数量（按 TOPSIS 排名选取） |
| `--cv_folds` | 5 | OOF 元特征折数 |

### 并行

| 参数 | 默认 | 说明 |
|------|------|------|
| `--threads` | — | 一键自动推导以下三项 |
| `--n_jobs` | 1 | 模型内线程 |
| `--max_workers` | 1 | 模型级并行 |
| `--repeat_workers` | 1 | repeat 级并行 |

### 预处理模式

`--enable_preprocess` 在训练前内联执行 convert 流程；`--preprocess_only` 仅执行预处理。配套参数与 `gpse convert` 对应：`--preprocess_prefix --vcf_file --bfile --ped_file --map_file --extract_file --snp_dir --direct_convert --matrix_file --raw_pheno_file --skip_matrix_conversion --skip_phenotype_match --skip_data_clean --load_matrix_info --plink_path --plink_out`。

## 训练产物

`results_dir` 下包含（详见 `train_pipeline.md` §12）：

- `feature_manifest.json` / `split_manifest.json`：特征与数据划分清单；
- `deployment_ensemble/member_*.pkl`：可部署集成模型；
- `model_comparison_cv*.csv`：模型对比排行榜；`reports/`：评估报告。

## 对应 API

见 [06. API 参考 → train](06-api-reference.md#gpsetrain)。
