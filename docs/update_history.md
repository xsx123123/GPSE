# GPSE Update History

Project: **GPSE** — Genomic Prediction with Stacking Ensemble for horticultural crops
Repository: https://github.com/xsx123123/GPSE
Current version: **0.0.4a** (as of 2026-09-07)

This document summarizes each released version: a short **Description**, the corresponding
**Upgrade Code** (version bump in `pyproject.toml` plus key interface code), and the detailed
**Update Content** (features, fixes, performance, docs).

---

## v0.0.1 — Initial Release (2026-06-03)

### Description
First packaged release of GPSE. Established the installable Python package framework
(Poetry/`pyproject.toml`), the unified `gpse` CLI entry point, external tool dependency
checking (PLINK/Beagle), package restructuring with unified loguru-based logging, and lazy
imports to keep CLI startup fast.

### Upgrade Code
```toml
# pyproject.toml
[tool.poetry]
name = "gpse"
version = "0.0.1"

[tool.poetry.scripts]
gpse = "gpse.cli:main"
```

### Update Content
- **feat**: initialize the `gpse` package framework (`a5e06fc`)
- **feat**: external tool dependency checker with correct version comparison for non-semver tools like PLINK (`d15dbb0`, `cf540b2`)
- **refactor**: extract CLI entry point and modernize argument parsing (`3c30951`)
- **refactor**: reorganize package structure; unify logging into a single file; split `GenomicPredictorV2` into focused sub-modules (`df154fc`, `737abae`)
- **fix**: enforce thread control via early environment variables and `threadpool_limits` (`abb3e76`)
- **fix**: lazy imports in `__init__.py` to avoid loading heavy dependencies at startup (`121c4bf`)

---

## v0.0.2 — Prediction Alignment & Training Hardening (2026-07-20)

### Description
Major feature release. Introduced the **canonical SNP ID contract** (`chr<chrom>_<start>_<end>`,
UCSC zero-based half-open) shared across `convert`, `train`, and `predict`, implemented the
`gpse predict` subcommand for VCF/matrix inputs with feature-manifest alignment, saved final
hold-out predictions from the CV-TOPSIS top model, and delivered a large training/logging
hardening pass (Rich console output, model-level parallelism, `--threads`, explicit
`--task_type`).

### Upgrade Code
```toml
# pyproject.toml
-version = "0.0.1"
+version = "0.0.2"
```

Canonical SNP ID convention (`gpse/utils/snp_ids.py`):
```text
chr<chrom>_<chromStart>_<chromEnd>        # zero-based, half-open
VCF chr1:100, REF=A   ->  chr1_99_100
VCF chrX:7,  REF=AT   ->  chrX_6_8
```

New prediction workflow:
```bash
gpse predict \
    --model results/ \
    --vcf-file new_samples.vcf.gz \
    --out predictions.csv
# outputs: predictions.csv + predictions.alignment.json
```

### Update Content
- **feat(train)**: save final hold-out predictions from the CV-TOPSIS rank-1 model to `reports/final_predictions.csv`; CV-TOPSIS ranking now runs regardless of `--use-stacking` (`ceafb6c`)
- **feat(convert/train/predict)**: canonical SNP IDs, adjacent `<matrix-stem>.features.json` manifest, and ordered `feature_manifest.json` saved at training; prediction reorders input features, reports missing/extra SNPs, and fills absent SNPs with missing code `3` (configurable via `--missing-value`) (`baf898c`)
- **feat(convert)**: Azodi 2019 conversion mode (−1/0/1 encoding, min-max phenotype scaling) (`1ed2b1b`)
- **feat(train)**: model-level parallelism, `--threads` shorthand, graceful Ctrl+C shutdown, leftover thread-budget recycling into `n_jobs` (`bedaa3f`, `727a11e`, `bcacd92`)
- **refactor!**: require explicit `--task_type`; drop phenotype auto-detection (`8c23499`)
- **feat(logging)**: switch console output to Rich/loguru, compact fold/repeat log lines, config rendered as a rich panel (`4c33d25`, `48d48be`, `a804c31`)
- **fix(train)**: reproducible CV fold assignment with dedicated seed 42; fresh estimator per fold; Optuna early-stopping condition; multi-class fold ensemble via averaged probabilities/majority vote; drop NaN-target samples; scipy-related robustness fixes (`baf898c`, `eee2130`)
- **fix(models)**: tighten `kernelridge_reg` search space; improve SVR convergence; scope CatBoost outputs to results directory (`4fa75b0`, `20553c1`, `c0c791c`)
- **docs**: wiki framework for CLI subcommands and API reference (`6835627`)
- **test**: `tests/test_snp_id_alignment.py`; full suite at 16 passed

---

## v0.0.3a1 — Batch Pipeline & Validation Suite (2026-07-21)

### Description
Alpha release focused on throughput and reproducibility: the new `gpse batch` subcommand for
YAML-driven multi-trait training, the `gpse tools split` utility for train/test splitting, a
workaround for the scipy-OpenBLAS segfault in KernelRidge, and a full 6-species validation
suite (15 regression models across cucumber/melon/watermelon and the lettuce MAGIC population).

### Upgrade Code
```toml
# pyproject.toml
-version = "0.0.2"
+version = "0.0.3a1"
```

New subcommands:
```bash
gpse batch --config batch_config.yaml        # multi-trait YAML-driven training
gpse batch --config batch_config.yaml --dry-run   # single-line command + derived config preview
gpse tools split ...                          # train/test data splitting
```

### Update Content
- **feat(batch)**: `gpse batch` subcommand for multi-trait YAML-driven training; unified batch logging with train style; pretty-printed dry-run commands; stop remaining tasks on Ctrl+C (`c4b0ce0`, `e19517d`, `b72ee41`, `409b152`)
- **feat(batch)**: merge per-trait summary tables across traits after a batch run (`597f1d7`)
- **feat(tools)**: `gpse tools split` subcommand for train/test data splitting (`74c9b36`)
- **fix(train)**: work around scipy-OpenBLAS segfault in KernelRidge solve (`08a226e`)
- **chore(validation)**: batch configs + runner scripts for 3 cucurbit species and the lettuce MAGIC population (3D traits + GWAS significant-SNP sets); all 15 regression models in 6-species configs (`445bab6`, `a35b8cf`)
- **docs**: PyPI install instructions and version badge; Documentation section with wiki links; all wiki pages translated to English (`1bfd689`, `445bab6`)

---

## v0.0.4a — YAML-Driven Model Registry & Performance (2026-08-20)

### Description
Latest alpha release. Adds a **YAML-driven model registry** so hyperparameter search spaces are
declarative and configurable via `--model_config`, expands the TOPSIS criteria configuration with
`--topsis_config`, optimizes hot-path bottlenecks in the training/conversion pipeline, and lets
`gpse convert` accept PLINK binary files (`--bfile`) as an alternative genotype source. Also ships
refinements to prediction alignment, SNP-ID utilities, and feature manifests.

### Upgrade Code
```toml
# pyproject.toml
-version = "0.0.3a1"
+version = "0.0.4a"
```

New CLI options:
```bash
gpse train  --model_config my_models.yaml     # YAML-driven model registry / search spaces
gpse train  --topsis_config my_criteria.yaml  # expanded TOPSIS criteria configuration
gpse convert --bfile my_plink_prefix ...      # PLINK binary files as genotype source
```

### Update Content
- **feat(models)**: YAML-driven model registry with `--model_config` option (`32d107a`)
- **feat(topsis)**: expanded criteria config; new `--topsis_config` CLI option (`386aa5a`)
- **perf**: optimize hot-path bottlenecks in the training and conversion pipeline (`d3c9e5d`)
- **fix(registry)**: make inline search-space param functions picklable for parallel workers (`aa76e5e`)
- **fix(convert)**: allow `--bfile` as an alternative genotype source; clarify CLI help (`e7eff28`)
- **release**: refinements across `convert` (genotype matrix/processor), `predict` (core/cli),
  `train` (CLI/data IO/workflow), SNP-ID utilities, and feature manifests; extended
  `tests/test_snp_id_alignment.py` (`b99e741`)

---

## Version Timeline

| Version | Date       | Theme                                        | Release commit |
|---------|------------|----------------------------------------------|----------------|
| 0.0.1   | 2026-06-03 | Initial package framework and `gpse` CLI     | `17c34bd`      |
| 0.0.2   | 2026-07-20 | Canonical SNP IDs, `gpse predict`, hardening | `ceafb6c`      |
| 0.0.3a1 | 2026-07-21 | `gpse batch`/`tools split`, validation suite | `445bab6`      |
| 0.0.4a  | 2026-08-20 | YAML model registry, TOPSIS config, perf     | `b99e741`      |
