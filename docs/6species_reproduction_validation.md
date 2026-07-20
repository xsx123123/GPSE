# GPSE 六物种历史结果复现与验证 SOP

## 1. 文档目标

本文档用于使用当前仓库中的新版 `gpse`，复现并验证以下历史结果目录：

```text
/data/xiaoliu/gpse_6pecies_model
```

历史实验包含：

- 6 个物种
- 18 个连续性状
- 14 个回归模型
- 5 折交叉验证
- 50 次重复
- 固定 30% hold-out 测试集
- Top-3 Stacking 集成

本文档的目标是实现两级复现：

1. **科学结果对齐**：输入、数据划分、模型集合和主要指标与历史结果可比较。
2. **新版模型重建**：生成带 `feature_manifest.json`、split manifest 和可安全预测模型的新产物。

> 严格的 pickle 或浮点逐字节复现暂时无法保证。历史日志只确认使用过 Python 3.8，没有完整的 Conda 环境导出或 `pip freeze`。

## 2. 开工前的关键结论

1. **直接训练，不需要 `gpse convert`**：基因型已经是数值矩阵，表型已经拆分并完成 Z-score 标准化。
2. **不要再次标准化表型**：命令中不要添加 `--standardize_phenotype`。
3. **必须显式指定历史 14 个模型**：新版额外增加了 `gblup_reg`，不限制模型会改变 TOPSIS 和 Stacking 候选集。
4. **历史性状级 CV 文件应作为复现输入**：使用各结果目录中的 `*_cv_50x5.csv`，不要使用数据集自带的 `6specise/cv/*_CVFs.csv` 替代。
5. **不要覆盖历史目录**：新版结果统一写入新的 `reproduction_6species/` 目录。
6. **maize 的 CatBoost 是后补训练**：三个 maize 主结果目录和对应的 `_retrain` 目录需要合并理解。

## 3. 目录与辅助文件

建议先定义三个目录：

```bash
export GPSE_HIST_ROOT=/data/xiaoliu/gpse_6pecies_model
export GPSE_DATA_ROOT="$GPSE_HIST_ROOT/6specise"
export GPSE_REPRO_ROOT="$PWD/reproduction_6species"

mkdir -p "$GPSE_REPRO_ROOT"
```

仓库内已经准备好两个验证文件：

| 文件 | 用途 |
| --- | --- |
| `docs/6species_input_sha256.txt` | 校验基因型、表型、原始 CV 和历史性状级 CV |
| `docs/6species_result_baseline.tsv` | 18 个性状的历史核心指标基线 |

## 4. 输入文件

每个性状的完整训练输入只有三个文件：

1. 一个物种级基因型矩阵
2. 一个性状级表型文件
3. 一个历史性状级 50×5 CV 文件

### 4.1 数据集清单

| 物种 | 基因型文件 | 样本数 | SNP 数 | 性状 |
| --- | --- | ---: | ---: | --- |
| maize | `6specise/geno/maize_geno.csv` | 391 | 244,781 | FT、HT、YLD |
| rice | `6specise/geno/rice_geno.csv` | 327 | 57,542 | FT、HT、YLD |
| sorghum | `6specise/geno/sorghum_geno.csv` | 451 | 56,299 | HT、MO、YLD |
| soy | `6specise/geno/soy_geno.csv` | 5,014 | 4,234 | HT、R8、YLD |
| spruce | `6specise/geno/spruce_geno.csv` | 1,722 | 6,930 | DBH、DE、HT |
| switchgrass | `6specise/geno/switchgrass_geno.csv` | 514 | 217,150 | AN、HT、ST |

### 4.2 文件路径规则

基因型：

```text
$GPSE_DATA_ROOT/geno/<species>_geno.csv
```

表型：

```text
$GPSE_DATA_ROOT/pheno_split/<species>_<trait>.csv
```

历史性状级 CV：

```text
$GPSE_HIST_ROOT/multi_traits_results_<species>_<trait>/cv_folds/<trait>_cv_50x5.csv
```

例如 `maize_FT`：

```text
基因型  $GPSE_DATA_ROOT/geno/maize_geno.csv
表型    $GPSE_DATA_ROOT/pheno_split/maize_FT.csv
CV      $GPSE_HIST_ROOT/multi_traits_results_maize_FT/cv_folds/FT_cv_50x5.csv
```

### 4.3 不需要作为训练输入的文件

- `6specise/pheno/*.csv`：原始多性状表型，仅在重新拆分表型时使用。
- `6specise/cv/*_CVFs.csv`：数据集自带的 100 组 CV，不是历史 GPSE 结果实际保存的性状级 CV。
- `6specise/6specise_G3.zip`：数据压缩备份。
- 历史 PNG、PDF、热图和汇总图：均为二次结果。
- 历史 `model.pkl`：属于验证对象，不是重新训练输入。

## 5. 输入数据处理要求

### 5.1 基因型矩阵

- 第一列必须保持为 `ID`。
- SNP 列名和列顺序必须保持不变。
- 不需要重新编码，不需要 VCF/PLINK 转换。
- 不要删除、排序或重命名 SNP 列。
- 直接通过 `--geno_file` 输入新版训练流程。

这些历史矩阵使用数值型基因型编码，可直接用于模型训练。新版 `gpse convert` 输出的 `0/1/2/3` 编码规则不应被强行套用到这批已处理矩阵上。

### 5.2 表型文件

- 每个文件只包含 `ID` 和一个性状列。
- 表型已完成 Z-score 标准化，均值约为 0、标准差约为 1。
- 无缺失表型，无重复 ID。
- 不要传 `--standardize_phenotype`，避免二次标准化。

### 5.3 ID 对齐

基因型和表型的 ID 集合必须完全一致。运行单个性状前可快速检查：

```bash
python - <<'PY'
import os
import pandas as pd

data_root = os.environ["GPSE_DATA_ROOT"]
geno = pd.read_csv(f"{data_root}/geno/maize_geno.csv", usecols=["ID"])
pheno = pd.read_csv(f"{data_root}/pheno_split/maize_FT.csv", usecols=["ID"])

geno_ids = set(geno["ID"].astype(str))
pheno_ids = set(pheno["ID"].astype(str))

assert not geno["ID"].duplicated().any(), "基因型存在重复 ID"
assert not pheno["ID"].duplicated().any(), "表型存在重复 ID"
assert geno_ids == pheno_ids, (
    f"ID 不一致: geno_only={len(geno_ids - pheno_ids)}, "
    f"pheno_only={len(pheno_ids - geno_ids)}"
)
print(f"ID check OK: {len(geno_ids)} samples")
PY
```

## 6. CV 文件与测试集的关系

### 6.1 CV 文件内容

历史 `FT_cv_50x5.csv` 类文件包含：

```text
ID,FT,cv0,cv1,...,cv49
```

- `cv0` 对应第 1 次重复。
- `cv49` 对应第 50 次重复。
- 每个值为 `0–4`，代表该样本在当前重复中所属的验证折。

因此它固定的是**训练集内部每次重复的 5 折划分**。

### 6.2 CV 文件不单独决定测试集

最终 30% hold-out 测试集由以下参数共同决定：

```text
--test_size 0.3
--random_seed 42
--use_same_test_set
```

新版先生成固定 hold-out 测试集，再根据 ID 将外部 CV 分配应用到剩余训练样本。完整复现需要同时保持：

- 相同输入文件
- 相同样本 ID 和行顺序
- `--test_size 0.3`
- `--random_seed 42`
- `--use_same_test_set`
- 相同 `--cv_file`

### 6.3 CV 参数必须匹配文件

历史 CV 文件只有 `cv0–cv49`，且 fold 值为 `0–4`，因此必须配合：

```text
--n_repeats 50
--n_splits 5
```

如果只做冒烟测试，可以使用同一个 CV 文件并设置 `--n_repeats 2`，此时只读取 `cv0` 和 `cv1`。

## 7. 历史实验参数

| 参数 | 历史值 | 新版复现设置 |
| --- | --- | --- |
| task type | regression | `--task_type regression` |
| model count | 14 | 显式传 `--models` |
| parameters | default | `--use_default_params` |
| hold-out test size | 0.3 | `--test_size 0.3` |
| CV folds | 5 | `--n_splits 5` |
| repeats | 50 | `--n_repeats 50` |
| random seed | 42 | `--random_seed 42` |
| same test set | enabled | `--use_same_test_set` |
| stacking | enabled | `--use_stacking` |
| stacking models | Top-3 | `--top_n_models 3` |
| stacking folds | 5 | `--cv_folds 5` |
| representative model | enabled | `--save_representative` |

历史 14 个模型：

```text
elasticnet_reg
gbdt_reg
svr_reg
mlp_reg
knn_reg
rf_reg
xgboost_reg
adaboost_reg
lightgbm_reg
catboost_reg
kernelridge_reg
histgradientboost_reg
sgd_reg
lasso_reg
```

新版还包含 `gblup_reg`。本轮历史对齐不要加入该模型。

## 8. 运行前检查

### 8.1 校验输入 SHA256

```bash
sha256sum -c docs/6species_input_sha256.txt
```

清单覆盖 6 个基因型、18 个拆分表型、6 个原始 CV 和 18 个历史性状级 CV。所有行都应显示 `OK`。

### 8.2 记录新版运行环境

```bash
mkdir -p "$GPSE_REPRO_ROOT/run_metadata"

git rev-parse HEAD > "$GPSE_REPRO_ROOT/run_metadata/git_commit.txt"
python --version > "$GPSE_REPRO_ROOT/run_metadata/python_version.txt" 2>&1
python -m pip freeze > "$GPSE_REPRO_ROOT/run_metadata/pip_freeze.txt"
gpse --version > "$GPSE_REPRO_ROOT/run_metadata/gpse_version.txt" 2>&1
```

环境记录是解释新版与历史指标差异的必要材料。

## 9. 推荐运行流程

### 9.1 定义历史模型列表

以下数组适用于 Bash 和 Zsh：

```bash
GPSE_HIST_MODELS=(
  elasticnet_reg gbdt_reg svr_reg mlp_reg knn_reg rf_reg xgboost_reg
  adaboost_reg lightgbm_reg catboost_reg kernelridge_reg
  histgradientboost_reg sgd_reg lasso_reg
)
```

### 9.2 第一步：maize_FT 冒烟测试

先验证输入、CV 解析、模型执行和产物结构：

```bash
gpse train \
  --geno_file "$GPSE_DATA_ROOT/geno/maize_geno.csv" \
  --pheno_file "$GPSE_DATA_ROOT/pheno_split/maize_FT.csv" \
  --target_trait FT \
  --task_type regression \
  --results_dir "$GPSE_REPRO_ROOT/maize_FT_smoke" \
  --models svr_reg kernelridge_reg \
  --use_default_params \
  --test_size 0.3 \
  --n_splits 5 \
  --n_repeats 2 \
  --random_seed 42 \
  --cv_file "$GPSE_HIST_ROOT/multi_traits_results_maize_FT/cv_folds/FT_cv_50x5.csv" \
  --use_same_test_set \
  --threads 2
```

冒烟测试只用于验证流程，不与 50 次重复的历史最终指标直接比较。

### 9.3 第二步：maize_FT 完整复现

```bash
gpse train \
  --geno_file "$GPSE_DATA_ROOT/geno/maize_geno.csv" \
  --pheno_file "$GPSE_DATA_ROOT/pheno_split/maize_FT.csv" \
  --target_trait FT \
  --task_type regression \
  --results_dir "$GPSE_REPRO_ROOT/maize_FT" \
  --models "${GPSE_HIST_MODELS[@]}" \
  --use_default_params \
  --test_size 0.3 \
  --n_splits 5 \
  --n_repeats 50 \
  --random_seed 42 \
  --cv_file "$GPSE_HIST_ROOT/multi_traits_results_maize_FT/cv_folds/FT_cv_50x5.csv" \
  --use_same_test_set \
  --use_stacking \
  --top_n_models 3 \
  --cv_folds 5 \
  --save_representative \
  --threads 4
```

### 9.4 第三步：扩展到全部性状

| 物种 | 性状 | 新版结果目录 | 历史 CV 文件 |
| --- | --- | --- | --- |
| maize | FT | `maize_FT` | `multi_traits_results_maize_FT/cv_folds/FT_cv_50x5.csv` |
| maize | HT | `maize_HT` | `multi_traits_results_maize_HT/cv_folds/HT_cv_50x5.csv` |
| maize | YLD | `maize_YLD` | `multi_traits_results_maize_YLD/cv_folds/YLD_cv_50x5.csv` |
| rice | FT | `rice_FT` | `multi_traits_results_rice_FT/cv_folds/FT_cv_50x5.csv` |
| rice | HT | `rice_HT` | `multi_traits_results_rice_HT/cv_folds/HT_cv_50x5.csv` |
| rice | YLD | `rice_YLD` | `multi_traits_results_rice_YLD/cv_folds/YLD_cv_50x5.csv` |
| sorghum | HT | `sorghum_HT` | `multi_traits_results_sorghum_HT/cv_folds/HT_cv_50x5.csv` |
| sorghum | MO | `sorghum_MO` | `multi_traits_results_sorghum_MO/cv_folds/MO_cv_50x5.csv` |
| sorghum | YLD | `sorghum_YLD` | `multi_traits_results_sorghum_YLD/cv_folds/YLD_cv_50x5.csv` |
| soy | HT | `soy_HT` | `multi_traits_results_soy_HT/cv_folds/HT_cv_50x5.csv` |
| soy | R8 | `soy_R8` | `multi_traits_results_soy_R8/cv_folds/R8_cv_50x5.csv` |
| soy | YLD | `soy_YLD` | `multi_traits_results_soy_YLD/cv_folds/YLD_cv_50x5.csv` |
| spruce | DBH | `spruce_DBH` | `multi_traits_results_spruce_DBH/cv_folds/DBH_cv_50x5.csv` |
| spruce | DE | `spruce_DE` | `multi_traits_results_spruce_DE/cv_folds/DE_cv_50x5.csv` |
| spruce | HT | `spruce_HT` | `multi_traits_results_spruce_HT/cv_folds/HT_cv_50x5.csv` |
| switchgrass | AN | `switchgrass_AN` | `multi_traits_results_switchgrass_AN/cv_folds/AN_cv_50x5.csv` |
| switchgrass | HT | `switchgrass_HT` | `multi_traits_results_switchgrass_HT/cv_folds/HT_cv_50x5.csv` |
| switchgrass | ST | `switchgrass_ST` | `multi_traits_results_switchgrass_ST/cv_folds/ST_cv_50x5.csv` |

每个性状只需要替换 `species`、`trait`、`--results_dir` 和 `--cv_file`。

## 10. 新版线程设置

### 10.1 推荐使用 `--threads`

新版通过 `--threads N` 接收总并行预算，并自动推导：

- `n_jobs`：单个训练任务内部线程数
- `max_workers`：并行模型数
- `repeat_workers`：同一模型并行 repeat 数

当前策略保持 `n_jobs=1`，优先并行不同模型，再把剩余预算分配给 repeats。

对于 14 个模型、50 次重复，典型分配为：

| 参数 | 自动分配结果 |
| --- | --- |
| `--threads 2` | `n_jobs=1, max_workers=2, repeat_workers=1` |
| `--threads 4` | `n_jobs=1, max_workers=4, repeat_workers=1` |
| `--threads 8` | `n_jobs=1, max_workers=8, repeat_workers=1` |
| `--threads 16` | `n_jobs=1, max_workers=14, repeat_workers=1`，有效并行度 14 |
| `--threads 28` | `n_jobs=1, max_workers=14, repeat_workers=2` |
| `--threads 50` | `n_jobs=1, max_workers=14, repeat_workers=3`，有效并行度 42 |

实际分配会写入启动日志。`--threads` 不能超过当前进程通过 CPU affinity 检测到的可用核心数。

### 10.2 资源建议

- 首轮冒烟测试：`--threads 2`
- maize、switchgrass 首次完整运行：`--threads 4`
- rice、sorghum：确认内存后可提高至 `--threads 8`
- soy、spruce：SNP 较少，但样本较多，仍需观察 MLP、KernelRidge 和 KNN 内存
- 不建议未测内存就直接使用 `--threads 50`

### 10.3 严格模拟历史并行结构

历史脚本近似采用：

```text
n_jobs=1
max_workers=1
repeat_workers=50
```

即模型依次运行，同一个模型的 50 次重复并行。只有在需要研究并行拓扑影响时才显式设置：

```bash
--n_jobs 1 --max_workers 1 --repeat_workers 50
```

该设置曾导致 maize CatBoost 进程池异常，不建议作为第一轮方案。

## 11. 新版输出验收

### 11.1 P0：输入与划分

必须检查：

- 输入 SHA256 全部通过。
- 日志中的样本数和 SNP 数与第 4 节一致。
- `feature_manifest.json` 存在，SNP 数与输入矩阵一致。
- `split_manifest.json` 存在。
- `train_ids.txt` 和 `test_ids.txt` 存在。
- 固定测试集比例约为 30%。
- 外部 CV 文件成功加载，没有缺少 `cv0–cv49` 的报错。

### 11.2 P1：模型完整性

完整运行应满足：

- 14 个基础模型均有结果目录。
- 每个模型完成 50 次重复。
- 每次重复完成 5 折训练。
- 每个模型存在 `summary_results.json`。
- 启用 representative model 后存在对应模型产物。
- Stacking 目录和结果文件存在。

核心文件包括：

```text
model_comparison.csv
model_comparison_cv.csv
model_comparison_cv_topsis.csv
model_comparison_cv_topsis_simple.csv
reports/model_comparison_holdout.csv
stacking_selected_models.json
<model>/summary_results.json
<model>/repeat_<n>/repeat_results.json
<model>/repeat_<n>/all_predictions.json
<model>/representative_model/model.pkl
<model>/representative_model/info.json
<model>/deployment_ensemble/
ensemble_stacking/stacking_ensemble_model.pkl
ensemble_stacking/stacking_results.pkl
```

### 11.3 P2：数值结果

历史核心指标见：

```text
docs/6species_result_baseline.tsv
```

建议按以下顺序比较：

1. 模型集合是否一致。
2. 每个模型的 `Test Pearson` 和标准差。
3. 每个模型的验证集/CV Pearson 和标准差。
4. Top-3 模型集合及顺序。
5. Stacking hold-out Pearson。
6. 最后抽查 repeat 和 fold 级预测。

### 11.4 P3：预测可用性

历史模型缺少新版 `feature_manifest.json`，不能安全确认 SNP 顺序。新版重训后应验证：

```bash
gpse predict \
  --model "$GPSE_REPRO_ROOT/maize_FT/<model>/representative_model/model.pkl" \
  --geno-file "$GPSE_DATA_ROOT/geno/maize_geno.csv" \
  --out "$GPSE_REPRO_ROOT/maize_FT/prediction_check.csv"
```

同时检查生成的 alignment report 中：

- `feature_coverage` 为 1.0
- `missing_model_snp_count` 为 0
- 样本数与输入一致

## 12. 验收阈值

### 12.1 严格历史环境复跑

适用于相同历史脚本、Python/依赖、CV 和并行设置：

| 检查项 | 要求 |
| --- | --- |
| 输入 SHA256 | 完全一致 |
| CV 文件 | 完全一致 |
| 模型、repeat、fold 数 | 完全一致 |
| 指标差异 | 优先要求完全一致；非确定性场景绝对差不超过 `1e-6` |

### 12.2 新版科学结果对齐

| 检查项 | 初始标准 |
| --- | --- |
| 输入、ID、SNP 顺序、CV | 必须一致 |
| 单模型 Test Pearson | 与历史值绝对差不超过 `0.02`，超过则告警 |
| Top-3 模型集合 | 应尽量一致 |
| TOPSIS 排序 | 结合均值和标准差解释变化 |
| Stacking | 单独比较，不要求与旧实现逐字节一致 |

超过阈值时按以下顺序排查：

1. 是否错误地再次标准化表型。
2. 是否误用了 `6specise/cv/*_CVFs.csv`。
3. `test_size`、随机种子和固定测试集是否一致。
4. 是否意外加入 `gblup_reg`。
5. CV 文件是否按 ID 正确匹配。
6. Python、scikit-learn、XGBoost、LightGBM、CatBoost 版本差异。
7. 并行度和底层 BLAS/OpenMP 非确定性。

不要通过反复更换随机种子来追历史指标。

## 13. 历史目录特殊情况

### 13.1 maize CatBoost 后补训练

maize 的 FT、HT、YLD 首次运行 CatBoost 时，50 个并发进程发生进程池异常。后补结果位于：

```text
multi_traits_results_maize_FT_retrain
multi_traits_results_maize_HT_retrain
multi_traits_results_maize_YLD_retrain
```

三个 maize 主目录只有 13 个 `summary_results.json` 和 13 个 representative model，但主目录的 `model_comparison.csv` 已包含后补 CatBoost 指标。验证历史完整性时，需要将主目录与对应 `_retrain` 目录合并理解。

### 13.2 历史脚本不是 melon 目录原样副本

以下文件在历史结果目录和 melon 目录之间存在差异：

```text
ensemble_stacking.py
genomic_data_pipeline.py
genomic_prediction_v2.py
genomic_utils.py
```

严格历史代码复跑时，应归档 `/data/xiaoliu/gpse_6pecies_model` 根目录中的脚本，而不是从 `/home/xiaoliu/data/gpse_3species/melon` 复制替代。

## 14. 推荐执行顺序

1. 设置 `GPSE_HIST_ROOT`、`GPSE_DATA_ROOT` 和 `GPSE_REPRO_ROOT`。
2. 执行 SHA256 校验。
3. 保存当前 Git commit、Python 版本和依赖环境。
4. 运行 `maize_FT` 的 2 模型 × 2 repeats 冒烟测试。
5. 检查输入维度、CV、split manifest 和 feature manifest。
6. 完整运行 `maize_FT` 的 14 模型 × 50 repeats。
7. 对比历史基线、TOPSIS 和 Stacking。
8. 使用新版模型执行一次矩阵预测和 feature alignment 验证。
9. 依次扩展至其余 17 个性状。
10. 全部性状通过后，再重建跨物种热图、箱线图和总排名报告。

## 15. 完成标准

单个性状只有同时满足以下条件才算完成：

- 输入 SHA256 和 ID 对齐通过。
- 固定测试集与历史设置一致。
- 50×5 CV 成功应用。
- 14 个模型训练完整。
- TOPSIS 和 Stacking 产物完整。
- 核心指标完成历史对比并记录差异。
- 新版模型包含 feature manifest。
- `gpse predict` 对原矩阵的 SNP 覆盖率为 100%。
- 运行命令、Git commit、Python 和依赖版本已归档。

