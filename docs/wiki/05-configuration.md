# 05. Configuration

## Project-level `gpse.yaml`

Placed in the working directory (or `gpse.local.yaml`, or given explicitly via `--config PATH`); declares external tools and convert settings:

```yaml
external_tools:
  - name: plink
    cmd: plink            # absolute paths are allowed
  - name: java
    cmd: java
    required: false
convert:
  beagle_jar_path: /path/to/beagle.jar   # only used with --impute
```

Loading logic (`gpse/utils/configuration.py`): deep-merged in the order **package defaults → project `gpse.yaml` → `gpse.local.yaml` → CLI `--config`**; named lists are overridden by `name`. `--no-project-config` disables automatic project config loading.

## Package Configs

| File | Contents |
|------|----------|
| `gpse/config/software.yaml` | Software metadata + external tool declarations (plink: `--version` probe, min_version 1.9, required; java: optional) |
| `gpse/config/default.yaml` | Software metadata + logging config (log_level, more_info, Label) |
| `gpse/config/topsis.yaml` | TOPSIS multi-criteria model ranking configuration (see below) |

## TOPSIS Configuration (`topsis.yaml`)

The TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) evaluator ranks trained models using multiple performance criteria. The configuration lives in `gpse/config/topsis.yaml` and defines which metrics participate in ranking and their relative importance.

### Schema

```yaml
tasks:
  regression:       # or "classification"
    criteria:
      - name: "Test Pearson"      # must match a column in model_comparison.csv
        type: max                  # "max" = higher is better, "min" = lower is better
        weight: 0.8                # relative importance (normalised internally)
```

### Weight Semantics

- Weights are normalised at runtime — only the **ratio** between non-zero weights matters.
- Criteria with `weight: 0` are **reference-only**: they appear in the config for documentation and can be activated by setting a positive weight in a custom config file. They do **not** participate in the TOPSIS ranking computation.
- At least one criterion must have `weight > 0`.

### Default Criteria

**Regression** (11 criteria, 2 active):

| Criterion | Type | Weight | Status |
|-----------|------|--------|--------|
| Test Pearson | max | 0.8 | **active** |
| Test Pearson (std) | min | 0.2 | **active** |
| Test Spearman | max | 0 | reference |
| Test Spearman (std) | min | 0 | reference |
| Validation Pearson | max | 0 | reference |
| Validation Pearson (std) | min | 0 | reference |
| Test MSE | min | 0 | reference |
| Test MSE (std) | min | 0 | reference |
| Ensemble Pearson | max | 0 | reference |
| Ensemble Pearson (std) | min | 0 | reference |
| Training Time (s) | min | 0 | reference |

**Classification** (11 criteria, 2 active):

| Criterion | Type | Weight | Status |
|-----------|------|--------|--------|
| Test Accuracy | max | 0.8 | **active** |
| Test Accuracy (std) | min | 0.2 | **active** |
| Test F1 | max | 0 | reference |
| Test F1 (std) | min | 0 | reference |
| Test AUC | max | 0 | reference |
| Test AUC (std) | min | 0 | reference |
| Validation Accuracy | max | 0 | reference |
| Validation Accuracy (std) | min | 0 | reference |
| Ensemble Accuracy | max | 0 | reference |
| Ensemble Accuracy (std) | min | 0 | reference |
| Training Time (s) | min | 0 | reference |

### Custom Config File

Users can supply their own TOPSIS config via `--topsis_config`:

```bash
gpse train \
    --geno_file data/genotype.parquet \
    --pheno_file data/phenotype.csv \
    --target_trait Fruit_Weight \
    --task_type regression \
    --topsis_config my_topsis.yaml
```

Example custom config that also considers Spearman correlation and training time:

```yaml
tasks:
  regression:
    criteria:
      - name: "Test Pearson"
        type: max
        weight: 0.5
      - name: "Test Pearson (std)"
        type: min
        weight: 0.1
      - name: "Test Spearman"
        type: max
        weight: 0.3
      - name: "Training Time (s)"
        type: min
        weight: 0.1
```

When `--topsis_config` is omitted, the built-in `gpse/config/topsis.yaml` is used. The active configuration (source file, active criteria, and reference criteria) is printed at the start of every training run.

## Batch Config (`gpse batch`)

`gpse batch` reads its own YAML config, separate from `gpse.yaml`. It has two sections: `defaults` (any `gpse train` option plus the batch-only `results_root`) and `traits` (each entry requires `name` = target trait column and may override any option, or be skipped with `enabled: false`). See [03. `gpse train` → Batch Multi-Trait Training](03-cli-train.md#batch-multi-trait-training-gpse-batch) for the full schema. A ready-to-edit template lives at `batch/batch_config.example.yaml`; real-world validation configs are under `tests/validation/*_batch_config.yaml`.

## Manifest Files in Training Artifacts

- `feature_manifest.json`: canonical SNP ID list of the training features, used by `gpse predict` for alignment (`gpse/utils/feature_manifest.py`);
- `split_manifest.json`: hold-out / CV split record, guaranteeing reproducibility.
