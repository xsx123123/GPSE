# GPSE 验证脚本说明

本目录用于验证 GPSE 的多物种、多性状批量训练，以及玉米性状的专项对照实验。当前脚本均为回归任务，并使用 GPSE 支持的全部 15 个回归模型。

## 目录内容

目录中的文件分为两类：

1. `*_batch_config.yaml` 与 `*_batch.sh`：六个物种的多性状批量验证。
2. `6_special_validation_*.sh`：玉米 FT、HT 性状的专项对照验证。

## 环境要求

运行前请确认：

- 已安装当前项目及其训练依赖，终端中可以直接执行 `gpse`。
- 配置中的基因型、表型和 CV 划分文件真实存在且可读。
- 输出目录的父目录存在并具有写入权限。
- LightGBM、XGBoost、CatBoost 等可选模型依赖已经安装。
- 机器资源足够。当前配置使用 `threads: 60`，可根据实际 CPU 数量调整。

建议在项目根目录运行脚本。必要时先安装当前项目：

```bash
pip install -e .
gpse --help
```

## 批量多性状验证

每个物种由一个 YAML 配置文件和一个启动脚本组成。启动脚本先执行 `--dry_run` 打印将要运行的命令，确认无误后再正式训练。

| 物种 | 配置文件 | 启动脚本 | 验证性状 | 启用 stacking 的性状 |
| --- | --- | --- | --- | --- |
| Maize | `maize_batch_config.yaml` | `6_species_validation_maize_batch.sh` | HT、FT、YLD | FT |
| Rice | `rice_batch_config.yaml` | `6_species_validation_rice_batch.sh` | YLD、FT、HT | FT |
| Sorghum | `sorghum_batch_config.yaml` | `6_species_validation_sorghum_batch.sh` | YLD、MO、HT | MO |
| Soy | `soy_batch_config.yaml` | `6_species_validation_soy_batch.sh` | HT、R8、YLD | R8 |
| Spruce | `spruce_batch_config.yaml` | `6_species_validation_spruce_batch.sh` | DBH、HT、DE | HT |
| Switchgrass | `switchgrass_batch_config.yaml` | `6_species_validation_switchgrass_batch.sh` | ST、HT、AN | HT |

运行单个物种，例如玉米：

```bash
bash tests/validation/6_species_validation_maize_batch.sh
```

只预览批量命令、不执行训练：

```bash
gpse batch \
  --config tests/validation/maize_batch_config.yaml \
  --dry_run
```

绕过启动脚本，直接执行批量训练：

```bash
gpse batch --config tests/validation/maize_batch_config.yaml
```

依次运行六个物种：

```bash
for script in tests/validation/6_species_validation_*_batch.sh; do
  bash "$script" || exit 1
done
```

### Batch YAML 结构

`defaults` 定义所有性状共享的训练参数，`traits` 定义性状名称及其个性化覆盖参数：

```yaml
defaults:
  geno_file: /path/to/geno.csv
  pheno_file: /path/to/multi_trait_pheno.csv
  task_type: regression
  use_default_params: true
  test_size: 0.3
  n_splits: 5
  n_repeats: 2
  random_seed: 42
  use_same_test_set: true
  threads: 60
  models:
    - elasticnet_reg
    - gbdt_reg
  results_root: /path/to/results/species_batch

traits:
  - name: TRAIT_A
  - name: TRAIT_B
    use_stacking: true
```

每个性状的输出目录由 GPSE 自动生成：

```text
<results_root>/<trait_name>/
```

例如玉米配置中的 FT 结果会写入：

```text
/data/jzhang/project/Temp/gpse_26.6.4_test/gpse_new_validation/6species/maize_batch/FT/
```

## 当前回归模型

六个 batch 配置以及玉米专项脚本均使用以下 15 个回归模型：

| 模型名称 | 模型类型 |
| --- | --- |
| `elasticnet_reg` | Elastic Net |
| `gbdt_reg` | Gradient Boosting Decision Tree |
| `svr_reg` | Support Vector Regression |
| `mlp_reg` | Multi-layer Perceptron Regressor |
| `knn_reg` | K-nearest Neighbors Regressor |
| `rf_reg` | Random Forest Regressor |
| `xgboost_reg` | XGBoost Regressor |
| `adaboost_reg` | AdaBoost Regressor |
| `lightgbm_reg` | LightGBM Regressor |
| `catboost_reg` | CatBoost Regressor |
| `kernelridge_reg` | Kernel Ridge Regression |
| `gblup_reg` | Genomic BLUP |
| `histgradientboost_reg` | Histogram Gradient Boosting Regressor |
| `sgd_reg` | Stochastic Gradient Descent Regressor |
| `lasso_reg` | Lasso Regression |

所有性状默认继承 YAML `defaults.models` 中的完整列表。如果只想让某个性状运行部分模型，可以在对应性状下覆盖 `models`：

```yaml
traits:
  - name: TRAIT_A
    models: [rf_reg, gblup_reg]
```

## 分类模型

`gpse batch` 同样支持分类任务。分类配置应设置 `task_type: classification`，并使用分类模型，不能将回归模型和分类模型混合在同一个任务中。

可用分类模型：

```text
rf_clf
xgboost_clf
lightgbm_clf
catboost_clf
svm_clf
mlp_clf
```

分类配置示例：

```yaml
defaults:
  geno_file: /path/to/geno.csv
  pheno_file: /path/to/pheno.csv
  task_type: classification
  n_classes: 3
  models:
    - rf_clf
    - xgboost_clf
    - lightgbm_clf
    - catboost_clf
    - svm_clf
    - mlp_clf
  results_root: /path/to/classification_results

traits:
  - name: CLASS_TRAIT
```

## 玉米专项验证

专项脚本直接调用 `gpse train`。每个脚本通常连续运行两组实验：

1. 基线训练，不启用 stacking。
2. 启用 `--use_stacking` 的 stacking + TOPSIS 训练。

| 脚本 | 性状 | 外部 CV 文件 | 单变量特征选择 | 主要用途 |
| --- | --- | --- | --- | --- |
| `6_special_validation_maize_FT.sh` | FT | 是 | 否 | FT 基线与 stacking 对照 |
| `6_special_validation_maize_FT_noCV.sh` | FT | 否 | 否 | FT 内部 CV 基线与 stacking 对照 |
| `6_special_validation_maize_FT_feature_selection_univariate.sh` | FT | 是 | 是，保留 5000 个特征 | 验证外部 CV 下的单变量特征选择 |
| `6_special_validation_maize_FT_feature_selection_univariate_noCV.sh` | FT | 否 | 是，保留 5000 个特征 | 验证内部 CV 下的单变量特征选择 |
| `6_special_validation_maize_HT.sh` | HT | 是 | 否 | HT 基线与 stacking 对照 |

运行示例：

```bash
bash tests/validation/6_special_validation_maize_FT.sh
```

带单变量特征选择的 FT 验证：

```bash
bash tests/validation/6_special_validation_maize_FT_feature_selection_univariate.sh
```

带有 `noCV` 的脚本名称表示不传入外部 `--cv_file`，由训练流程自行生成交叉验证划分，并不表示完全关闭交叉验证。

## 公共训练参数

当前验证配置主要使用以下参数：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `task_type` | `regression` | 执行回归任务 |
| `use_default_params` | `true` | 使用模型默认参数，不运行完整超参数搜索 |
| `test_size` | `0.3` | 30% 样本作为测试集 |
| `n_splits` | `5` | 五折交叉验证 |
| `n_repeats` | `2` | 重复训练两次 |
| `random_seed` | `42` | 固定随机种子 |
| `use_same_test_set` | `true` | 不同模型使用相同测试集 |
| `threads` | `60` | 总线程预算，由 GPSE 分配给模型和重复任务 |
| `use_stacking` | 按性状或实验启用 | 执行 stacking，并结合 TOPSIS 选择基础模型 |
| `feature_selection` | `univariate` | 专项脚本中使用单变量特征选择 |
| `select_k` | `5000` | 保留排名前 5000 个特征 |

## 修改数据和输出路径

这些验证文件包含当前服务器的绝对路径。在其他环境运行前，至少需要修改：

- `geno_file` 或 `--geno_file`：基因型 CSV 文件。
- `pheno_file` 或 `--pheno_file`：表型 CSV 文件。
- `results_root` 或 `--results_dir`：结果输出目录。
- `--cv_file`：专项实验使用的外部 CV 划分文件。

批量配置使用多列表型文件，`traits[].name` 必须与表型文件中的目标列名完全一致。专项脚本目前使用按性状拆分的表型文件。

## 注意事项

- 六个物种、三个性状、15 个模型会产生大量训练任务，运行时间和磁盘占用都可能较大。
- 同一个输出目录中已有结果时，执行前应确认是否允许覆盖或追加。
- `use_stacking` 会增加额外训练和模型选择开销。
- `use_default_params: true` 适合流程验证；如果需要正式调参评估，应根据实验设计调整该选项和 trials 数量。
- 外部 CV 文件中的样本标识必须与当前基因型、表型数据一致。
- `6_special_validation_maize_HT.sh` 当前引用的 CV 文件名是 `FT_cv_50x5.csv`。运行 HT 验证前应确认该文件是否确实为预期划分，或将其改为正确的 HT CV 文件。

## 快速检查

检查所有 batch shell 脚本的 Bash 语法：

```bash
for script in tests/validation/*_batch.sh; do
  bash -n "$script" || exit 1
done
```

预览所有物种的 batch 命令：

```bash
for config in tests/validation/*_batch_config.yaml; do
  gpse batch --config "$config" --dry_run || exit 1
done
```

建议先完成路径检查和 `--dry_run`，再启动完整验证。
