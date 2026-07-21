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
| `gpse/config/topsis.yaml` | TOPSIS evaluation criteria: classification `Test Accuracy (max, 0.8) / Test Accuracy (std) (min, 0.2)`; regression `Test Pearson (max, 0.8) / Test Pearson (std) (min, 0.2)` |

## Batch Config (`gpse batch`)

`gpse batch` reads its own YAML config, separate from `gpse.yaml`. It has two sections: `defaults` (any `gpse train` option plus the batch-only `results_root`) and `traits` (each entry requires `name` = target trait column and may override any option, or be skipped with `enabled: false`). See [03. `gpse train` → Batch Multi-Trait Training](03-cli-train.md#batch-multi-trait-training-gpse-batch) for the full schema. A ready-to-edit template lives at `batch/batch_config.example.yaml`; real-world validation configs are under `tests/validation/*_batch_config.yaml`.

## Manifest Files in Training Artifacts

- `feature_manifest.json`: canonical SNP ID list of the training features, used by `gpse predict` for alignment (`gpse/utils/feature_manifest.py`);
- `split_manifest.json`: hold-out / CV split record, guaranteeing reproducibility.
