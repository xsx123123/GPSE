# GPSE pipeline 模块测试（cucumber 原始 VCF + 表型）

测试目的：验证 `gpse pipeline`（convert + train 串联）在真实数据上能否端到端跑通。

## 测试发现的问题与修复（2026-09-10）

**问题**：convert 阶段写 parquet/feather 矩阵时，基因型编码用 numpy `U2` 字符串数组
（`gpse/convert/genotype_matrix.py`），导致 parquet 中全部 SNP 列为 string 类型。
train 阶段读入后特征列全是 object dtype，XGBoost 直接报错
`DataFrame.dtypes for data must be int, float, bool or category`，
且报错信息会把 11 万列名全部打印出来。CSV 输出不受影响（pandas 读取时自动推断数值），
所以旧的 CSV 流程没暴露该问题；parquet 是默认输出格式（`--out-format parquet`），
因此默认 pipeline 路径必现。

**修复**：`gpse/convert/genotype_matrix.py` 两处 parquet/feather 写出前，
将基因型 DataFrame 转为 `np.int8`（剂量编码 -1/0/1/2/3 均可容纳）。
修复后 `cucumber_Days_to_flower_genotype.parquet` 为 1 列 string（ID）+ 113854 列 int8，
train 阶段正常运行。

## 数据

- 基因型：`/data/jzhang/project/Temp/gpse_26.6.4_test/raw_data/cucumber_raw_SNP.vcf.gz`
  （113,854 个 SNP × 1,234 个样本）
- 表型：`/data/jzhang/project/Temp/gpse_26.6.4_test/raw_data/cucumber_phenotype_format.csv`
  （836 个样本，SampleID 与 VCF 样本名全部匹配）

注意：使用 `cucumber_phenotype_format.csv` 而不是 `cucumber_phenotype.csv`，
后者的 SampleID 含空格（如 `PI 176954`），与 VCF 样本名（`PI_176954`）不一致。

## 运行方式

```bash
bash tests/pipeline/run_cucumber_pipeline_test.sh
```

脚本内核心命令（为缩短冒烟测试时间，模型/轮次都做了缩减）：

```bash
gpse pipeline \
  --vcf raw_data/cucumber_raw_SNP.vcf.gz \
  --pheno raw_data/cucumber_phenotype_format.csv \
  --out-prefix GPSE_test_result/pipeline/cucumber \
  --out-format parquet \
  --target_trait Days_to_flower \
  --task_type regression \
  --models rf_reg xgboost_reg svr_reg \
  --trials 5 --train_folds 3 --n_repeats 2 --patience 5 \
  --random_seed 42 \
  --use_stacking --top_n_models 3 --cv_folds 3 \
  --threads 8 --n_jobs 2 --max_workers 2 --repeat_workers 2 \
  --results_dir GPSE_test_result/pipeline/train_results_Days_to_flower
```

## 输出

- 测试输出目录：`/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE_test_result/pipeline/`
  - `cucumber_<trait>_genotype.parquet` / `cucumber_<trait>_phenotype.parquet`：convert 阶段按性状输出的匹配矩阵
  - `train_results_Days_to_flower/`：train 阶段结果（优化报告、预测、stacking 等）
  - `gpse_pipeline_*.log`、`run_cucumber_pipeline_test.log`：运行日志
