# 04. 子命令 `gpse predict`

用训练产物对新样本做表型预测。实现位于 `gpse/predict/`（CLI：`gpse/predict/cli.py`，核心：`gpse/predict/core.py`）。

## 用法

```bash
gpse predict --model MODEL --vcf-file new.vcf.gz --out predictions.csv \
  [--report report.json] [--missing-value 3.0] [--min-feature-coverage 0.8]
```

| 参数 | 必需 | 说明 |
|------|------|------|
| `--model` | 是 | `model.pkl` 文件 / `deployment_ensemble` 目录 / results 目录（自动解析） |
| `--vcf-file` / `--geno-file` | 是（二选一） | 新样本的 VCF 或基因型矩阵文件 |
| `--out` | 是 | 输出预测 CSV |
| `--report` | 否 | 特征对齐报告 JSON，默认 `<out>.alignment.json` |
| `--missing-value` | 否（默认 3.0） | 缺失基因型的填充值 |
| `--min-feature-coverage` | 否（0–1） | 特征覆盖率下限，低于则拒绝预测 |

## 特征对齐机制

- 训练时以 canonical SNP ID（`chr<chrom>_<chromStart>_<chromEnd>`，见 `gpse/utils/snp_ids.py`）记录特征，并写入 `feature_manifest.json`；
- 预测时按 manifest 对齐新样本的 SNP 列：缺失特征以 `--missing-value` 填充，多余特征丢弃；
- 对齐统计（覆盖率、缺失 SNP 列表）写入对齐报告 JSON；覆盖率低于 `--min-feature-coverage` 时报错退出，防止在低质量对齐上产生误导性预测。

## 输出

- `predictions.csv`：样本 ID + 预测表型；
- `<out>.alignment.json`：对齐报告（feature coverage、缺失/多余特征）。

## 对应 API

见 [06. API 参考 → predict](06-api-reference.md#gpsepredict)。
