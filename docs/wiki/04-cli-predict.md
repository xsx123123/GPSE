# 04. Subcommand `gpse predict`

Predicts phenotypes of new samples from training artifacts. Implemented in `gpse/predict/` (CLI: `gpse/predict/cli.py`, core: `gpse/predict/core.py`).

## Usage

```bash
gpse predict --model MODEL --vcf-file new.vcf.gz --out predictions.csv \
  [--report report.json] [--missing-value 3.0] [--min-feature-coverage 0.8]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--model` | yes | A `model.pkl` file / `deployment_ensemble` directory / results directory (auto-resolved) |
| `--vcf-file` / `--geno-file` | yes (one of the two) | VCF or genotype matrix file of the new samples |
| `--out` | yes | Output prediction CSV |
| `--report` | no | Feature alignment report JSON; defaults to `<out>.alignment.json` |
| `--missing-value` | no (default 3.0) | Fill value for missing genotypes |
| `--min-feature-coverage` | no (0–1) | Minimum feature coverage; lower coverage rejects the input |

## Feature Alignment

- At training time, features are recorded as canonical SNP IDs — `chr<chrom>_<chromStart>_<chromEnd>` (zero-based, half-open coordinates; see `gpse/utils/snp_ids.py`) — and written to `feature_manifest.json`;
- At prediction time, the new samples' SNP columns are aligned against the manifest: missing features are filled with `--missing-value`, extra features are dropped;
- Alignment statistics (matched/missing/extra SNP counts, feature coverage, missing SNP list) are written to the alignment report JSON; if coverage falls below `--min-feature-coverage`, the command exits with an error instead of producing misleading predictions on a low-quality alignment.

## Output

- `predictions.csv`: sample IDs + predicted phenotypes;
- `<out>.alignment.json`: alignment report (feature coverage, matched/missing/extra features).

## Typical Workflow

A common pattern is to hold out a test set with `gpse tools split` before training, then score it afterwards:

```bash
gpse tools split --geno geno.csv --pheno pheno.csv --out-prefix data/split --test-ratio 0.2
gpse train --geno_file data/split_train_geno.csv --pheno_file data/split_train_pheno.csv ...
gpse predict --model optimization_results_v2 --geno-file data/split_test_geno.csv --out test_predictions.csv
```

## Corresponding API

See [06. API Reference → predict](06-api-reference.md#gpsepredict).
