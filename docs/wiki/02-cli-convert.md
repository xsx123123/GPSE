# 02. 子命令 `gpse convert`

把 VCF + 表型数据转换为可训练的数值矩阵，并提供 QC、重编码与依赖检查。实现位于 `gpse/convert/`（CLI：`gpse/convert/cli.py`，工作流：`gpse/convert/workflow.py`）。

## 四种运行模式

| 模式 | 触发方式 | 功能 |
|------|----------|------|
| pipeline（默认） | 直接执行 | VCF→PLINK→PED/MAP→数值矩阵→表型匹配 |
| qc | `--run-qc` | 质控过滤 + 可选 Beagle 插补 + LD pruning |
| recode | `--recode-prefix PREFIX` | PED/MAP 复合基因型 → 加性数值编码 |
| deps | `--check-deps` | 检查 PLINK / Java / Beagle 外部依赖 |

## pipeline 模式

```bash
gpse convert --vcf data.vcf.gz --pheno pheno.csv --out-prefix out/run \
  [--extract snps.txt] [--snp-dir DIR] [--direct] \
  [--skip-matrix] [--skip-match] [--skip-clean] \
  [--out-format csv|parquet|feather] [--allow-extra-chr] \
  [--trait-name yield] [--standardize-phenotype]
```

- 必需：`--vcf`、`--pheno`、`--out-prefix`；
- 中间步骤可跳级：`--bfile / --ped-file / --map-file / --matrix-file` 从任意中间产物继续；
- `--skip-*` 用于跳过已完成步骤；`--out-format` 默认 parquet。

## qc 模式

```bash
gpse convert --run-qc --input-prefix PLINK_PREFIX \
  [--snpmaxmiss 0.1] [--samplemaxmiss 0.1] [--maf 0.05] \
  [--r2-cutoff 0.2] [--ld-window 50] [--ld-step 10] [--impute]
```

- `--snpmaxmiss` / `--samplemaxmiss`：SNP / 样本最大缺失率；
- `--maf`：最小等位基因频率；`--r2-cutoff` + `--ld-window` + `--ld-step`：LD pruning 参数；
- `--impute`：启用 Beagle 插补（需配置 `beagle_jar_path` 与 java）。

## recode 模式

```bash
gpse convert --recode-prefix PLINK_PREFIX
```

将 PED/MAP 中的复合基因型（如 `A A` / `A G`）转为 0/1/2 加性数值编码。

## deps 模式

```bash
gpse convert --check-deps [--plink-path plink] [--java-path java] \
  [--beagle-jar-path beagle.jar] [--require-beagle]
```

## 通用参数

- `-t/--threads`（默认 10）：线程数；
- `--config PATH`：追加配置文件；`--no-project-config`：禁用项目配置自动加载；
- `-l/--log-level`：日志级别。

## 对应 API

见 [06. API 参考 → convert](06-api-reference.md#gpseconvert)。
