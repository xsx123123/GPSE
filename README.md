<p align="center">
  <img src="docs/logo.png" alt="GPSE Logo" width="250">
</p>

<h1 align="center">GPSE (Genomic Prediction with Stacking Ensemble)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-Alpha-orange" alt="Status">
</p>

<p align="center"><strong>Genomic Prediction with Stacking Ensemble for horticultural crops.</strong></p>

GPSE is a comprehensive, machine-learning-based pipeline for genomic selection and prediction. It provides end-to-end functionalities from raw genomic data (VCF/PLINK) preprocessing to hyperparameter optimization, model evaluation, TOPSIS ranking, and Stacking Ensemble prediction.

## 🌟 Key Features

* **Complete Data Pipeline**: Seamlessly convert VCF files to PLINK binary formats, extract specific SNPs, convert to numerical matrices, and accurately match genotypes with phenotypes.
* **Broad Algorithm Support**: Supports 14+ robust machine learning algorithms including Random Forest, XGBoost, LightGBM, CatBoost, SVR, MLP, ElasticNet, and more.
* **Dual Task Modes**: Native support for both **Regression** (continuous traits) and **Classification** (categorical/discrete traits).
* **Automated Hyperparameter Tuning**: Integrated with Optuna for efficient, automated, multi-threaded parameter optimization.
* **Robust Evaluation**: Performs multiple repeats of K-Fold cross-validation to ensure model stability and reproducibility.
* **Model Ranking & Selection**: Built-in **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) evaluation utilizing Entropy Weight Method for multi-criteria model ranking.
* **Stacking Ensemble**: Automatically ensembles the Top-N performing models to maximize prediction accuracy.

## 🛠️ Installation

### Prerequisites

* Python >= 3.10
* [PLINK 1.9](https://www.cog-genomics.org/plink/) (Required for genomic data format conversion and SNP extraction)

### Install via Poetry

This project uses Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/<project>/gpse.git
cd gpse

# Install dependencies using Poetry
poetry install
```

### Install via pip

```bash
pip install .
```

## 🚀 Usage

GPSE uses a subcommand architecture: `gpse {convert,train,predict}`.

### Command-Line Preview

![GPSE command-line help](docs/command.png)

### 1. Data Conversion & QC (`gpse convert`)

`gpse convert` handles all genotype/phenotype preprocessing: format conversion, QC filtering, LD pruning, and sample matching. The output is training-ready numerical matrices.

#### Pipeline Overview

```
Input                        Processing                     Output
─────                        ──────────                     ──────
samples.vcf            →  VCF → PLINK BED              →  {prefix}_genotype.csv
phenotype.txt/.csv     →  PED/MAP → numeric (0/1/2)    →  {prefix}_phenotype.csv
                            SNP filtering                   {prefix}_phenotype_scaler.json
                            Sample ID matching                 (only with --standardize-phenotype)
                            Column name cleaning
                            Phenotype Z-score (optional)
```

Genotype encoding: `00→0` (homozygous ref), `01/10→1` (heterozygous), `11→2` (homozygous alt), missing→`3`.

#### 1.1 VCF + Phenotype → Training Data

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --direct \
    --out-prefix data/train
```

Output files:
- `data/train_genotype.csv` — numeric matrix (rows=samples, columns=SNPs, values=0/1/2)
- `data/train_phenotype.csv` — cleaned, sample-matched phenotype (ID + trait value)

#### 1.2 With Phenotype Standardization

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --direct \
    --standardize-phenotype \
    --out-prefix data/train
```

Additional output: `data/train_phenotype_scaler.json` (mean/std for inverse transform during prediction).

#### 1.3 Extract Specific SNPs

```bash
# From PLINK binary input
gpse convert \
    --bfile plink_data \
    --extract snp_list.txt \
    --pheno phenotype.txt \
    --out-prefix data/train

# Batch extraction from a directory of SNP list files
gpse convert \
    --bfile plink_data \
    --snp-dir snp_lists/ \
    --out-prefix data/train
```

#### 1.4 Use Existing Matrix (Skip Matrix Generation)

```bash
gpse convert \
    --matrix-file existing_genotype.csv \
    --pheno phenotype.txt \
    --out-prefix data/matched
```

#### 1.5 QC Filtering + LD Pruning

Runs genotype-level QC (missing rate, MAF), optional Beagle imputation, and LD pruning as a standalone step.

```bash
gpse convert \
    --run-qc \
    --input-prefix plink_data \
    --out-prefix data/qc_data \
    --snpmaxmiss 0.1 \
    --samplemaxmiss 0.1 \
    --maf 0.05 \
    --r2-cutoff 0.2
```

With Beagle imputation:

```bash
gpse convert \
    --run-qc \
    --input-prefix plink_data \
    --out-prefix data/qc_data \
    --impute \
    --beagle-jar-path /path/to/beagle.jar
```

Output: `data/qc_data_pruned.bed/bim/fam` (LD-pruned PLINK binary, ready for matrix conversion).

#### 1.6 Recode PED/MAP to Numeric

```bash
gpse convert --recode-prefix plink_data
# Output: plink_data.geno
```

#### 1.7 Check External Dependencies

```bash
gpse convert --check-deps
```

#### 1.8 Rename Phenotype Trait

```bash
gpse convert \
    --vcf samples.vcf \
    --pheno phenotype.txt \
    --trait-name Fruit_Weight \
    --direct \
    --out-prefix data/train
```

### 2. Model Training (`gpse train`)

#### 2.1 Train with Pre-processed Data

```bash
gpse train \
    --geno_file data/train_genotype.csv \
    --pheno_file data/train_phenotype.csv \
    --target_trait Fruit_Weight \
    --task_type regression \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --use_stacking \
    --top_n_models 5 \
    --n_jobs 2 \
    --max_workers 4 \
    --results_dir output_results/
```

#### 2.2 One-Stop: Preprocessing + Training

```bash
gpse train \
    --enable_preprocess \
    --preprocess_prefix data/train \
    --vcf_file samples.vcf \
    --raw_pheno_file phenotype.txt \
    --target_trait Fruit_Weight \
    --task_type regression \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --use_stacking \
    --results_dir output_results/
```

#### 2.3 Preprocessing Only (No Training)

```bash
gpse train \
    --preprocess_only \
    --preprocess_prefix data/train \
    --vcf_file samples.vcf \
    --raw_pheno_file phenotype.txt \
    --target_trait Fruit_Weight
```

#### 2.4 Classification Task

```bash
gpse train \
    --geno_file genotype.csv \
    --pheno_file phenotype.csv \
    --target_trait Disease_Resistance \
    --task_type classification \
    --n_classes 3 \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --results_dir classification_results/
```

### 3. Analyze Phenotypes

Quickly analyze phenotype data to determine the appropriate task type (Regression vs Classification).

```bash
python -m gpse.tools.analyze_phenotypes
```

### 4. Show Help

```bash
gpse --help
gpse --version
gpse convert --help
gpse train --help
```

## 📁 Source Layout

The package is organized around the three workflow commands: `convert`, `train`,
and `predict`. Runtime-specific code lives in those command packages; shared
support code lives in `config`, `models`, `tasks`, `tools`, and `utils`.

### `gpse/`

| File | Scope |
| --- | --- |
| `__init__.py` | Package metadata, currently exposes `__version__`. |
| `cli.py` | Thin top-level command router for `gpse {convert,train,predict}`. It defines shared CLI flags, routes subcommands, and delegates workflow logic to the relevant package. |

### `gpse/config/`

Configuration constants and packaged YAML defaults.

| File | Scope |
| --- | --- |
| `__init__.py` | Public exports for config dataclasses and constants. |
| `constants.py` | Dataclasses and immutable model/training constants, including filenames, directory names, precision settings, and thread environment variable names. |
| `_topsis_config.py` | Loads TOPSIS task configuration, validates criteria/weights, logs runtime settings, and saves representative models. It is consumed by the training predictor. |
| `default.yaml` | Default application/logging configuration. |
| `software.yaml` | Package metadata and external tool definitions used by conversion/QC dependency checks. |
| `topsis.yaml` | Task-specific TOPSIS criteria, directions, and weights for regression/classification model ranking. |

### `gpse/convert/`

Implementation for `gpse convert`: genotype/phenotype conversion, QC, LD pruning,
and external tool execution.

| File | Scope |
| --- | --- |
| `__init__.py` | Public convert package export for `GenomicDataProcessor`. |
| `cli.py` | CLI parser and dispatcher for conversion modes such as full conversion, QC, recoding, and dependency checks. |
| `external.py` | External-tool discovery, configured path resolution, version checks, and command execution helpers. |
| `processor.py` | Main conversion processor: VCF/PLINK conversion, SNP extraction, numeric matrix generation, phenotype cleanup, sample matching, and phenotype standardization. |
| `qc.py` | PLINK/Beagle QC utilities: format conversion, genotype filtering, imputation, LD pruning, and PED/MAP numeric recoding. |

### `gpse/train/`

Implementation for `gpse train`: model training, repeated CV, optimization,
model ranking, and stacking ensemble training.

| File | Scope |
| --- | --- |
| `__init__.py` | Lazy exports for `GenomicPredictorV2`, `StackingEnsemble`, and `TOPSISEvaluator`. |
| `cli.py` | CLI parser and dispatcher for `gpse train`, including training arguments, preprocessing options, validation, and training workflow launch. |
| `predictor.py` | Main training orchestrator class. It initializes task-specific optimizers, logging, directories, and binds the training submodule functions as methods. |
| `_data_io.py` | Training data loading, genotype/phenotype alignment, phenotype standardization, and inverse standardization. |
| `_model_tools.py` | Unified model creation, default-parameter lookup, parameter filtering, and default metric fallbacks for regression/classification. |
| `_fold_training.py` | Single-fold model training, prediction, metric calculation, fold logging, and fold-level metric averaging. |
| `_ensemble.py` | Fold-ensemble prediction logic and ensemble metric calculation. |
| `_optimization.py` | Optuna-based hyperparameter optimization over CV folds. |
| `_repeat_training.py` | Repeat-level orchestration, parallel repeat execution, summary statistics, representative repeat selection, and repeat result saving. |
| `_cv_manager.py` | CV fold file creation/loading and fold generation from predefined CV assignments. |
| `_pipeline.py` | Top-level `run_all_models` workflow. Runs selected models, creates comparison tables, performs TOPSIS selection, and optionally trains stacking ensembles. |
| `stacking.py` | Optional stacking ensemble stage. Loads trained base models, creates meta-features, trains the meta-model, evaluates, and saves ensemble artifacts. |
| `topsis.py` | TOPSIS evaluator and optional CLI. Ranks trained models from comparison CSV outputs using configured criteria and weights. |

### `gpse/predict/`

Implementation placeholder for `gpse predict`.

| File | Scope |
| --- | --- |
| `__init__.py` | Prediction package marker. |
| `__main__.py` | Enables `python -m gpse.predict`. |
| `cli.py` | CLI stub for future prediction workflows. Parses model/genotype/output arguments and currently reports that prediction is not implemented yet. |

### `gpse/models/`

Model registries and optimizer/search-space definitions. These modules define
how models are constructed and how Optuna proposes parameters; they do not run
the full training pipeline by themselves.

| File | Scope |
| --- | --- |
| `__init__.py` | Lazy exports for regression and classification optimizers. |
| `regression_model_optimizer.py` | Regression model registry, Optuna search spaces, parameter filtering, model factories, and default parameters. |
| `classification_model_optimizer.py` | Classification model registry, Optuna search spaces, parameter filtering, model factories, and default parameters. |
| `model_optimizers.py` | Backward-compatible regression optimizer import shim. New code should use `regression_model_optimizer.py`. |
| `classification_models.py` | Backward-compatible classification optimizer import shim. New code should use `classification_model_optimizer.py`. |

### `gpse/tasks/`

Task-specific runtime helpers that are shared by training components.

| File | Scope |
| --- | --- |
| `__init__.py` | Lazy export for `GenomicClassifier`. |
| `classification.py` | Classification-specific runtime support: label encoding/decoding, probability-to-label conversion, classification metrics, result summaries, and delegation to `ClassificationModelOptimizer`. |

### `gpse/tools/`

Standalone helper scripts that are useful outside the main workflow commands.

| File | Scope |
| --- | --- |
| `__init__.py` | Tools package marker. |
| `analyze_phenotypes.py` | Standalone phenotype analysis helper for inspecting trait distributions and deciding whether traits are better treated as regression or classification targets. |

### `gpse/utils/`

Shared utilities used across workflow packages. This package should contain
generic support code only; train/convert/predict-specific business logic should
live in the corresponding workflow package.

| File | Scope |
| --- | --- |
| `__init__.py` | Lazy exports for logging and shared genomic utility functions. |
| `configuration.py` | YAML configuration loading and merge helpers for packaged defaults plus optional project/user overrides. |
| `dependency_checker.py` | Generic external dependency detection and version-check helpers used by conversion tooling. |
| `genomic_utils.py` | Shared training helpers: metric calculations, CV file helpers, result-table generation, seed generation, directory creation, fold utilities, and TOPSIS wrapper dispatch. |
| `log_utils.py` | Loguru/Rich logger initialization, subprocess logging setup, and subprocess log collection. |
| `logo.py` | Rich-based logo and welcome panel rendering. |
| `print_utils.py` | Reusable Rich table/panel/column printing helpers. |
| `version.py` | Version, dependency, system, and external-tool reporting helpers. |

## 📦 Core Dependencies

* `scikit-learn`
* `xgboost`
* `lightgbm`
* `catboost`
* `optuna`
* `pandas` & `numpy`
* `rich` & `loguru` (for beautiful CLI output and logging)

## 📝 Recent Updates

* **Thread Control & Startup Performance** (`2026-06-03`)
  * Fixed BLAS/MKL thread pools ignoring `--n_jobs` by setting all 6 environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`) **before** numpy/scipy import.
  * Added `threadpoolctl.threadpool_limits()` as a runtime safety net around all `model.fit()` calls.
  * Switched `__init__.py` files in `train/`, `models/`, `tasks/`, and `utils/` to **lazy imports** (`__getattr__`) so `gpse --help` no longer loads the entire ML stack.
  * Renamed CLI args for clarity: `--threads` → `--n_jobs`, `--parallel_jobs` → `--max_workers`.
  * Added `--n_jobs` to `histgradientboost_reg` and `knn_reg` (previously missing thread control).
  * Fixed easter egg (`gpse 42`) duplicate face and unrendered rich markup.

* **Import System Unification** (`2026-06-03`)
  * Removed all `sys.path` hacks from `cli.py` and the training predictor.
  * Unified all imports to absolute package paths (`from gpse.xxx import ...`).
  * Populated `__init__.py` files with proper exports for `config/`, `convert/`, `train/`, `models/`, `tasks/`, `utils/`, and `tools/`.

* **Modular Refactor of `GenomicPredictorV2`** (`2026-06-03`)
  * Split the monolithic `GenomicPredictorV2` class into focused training modules under `gpse/train/`:
    * `predictor.py` – main `GenomicPredictorV2` training orchestrator
    * `_data_io.py` – genotype / phenotype loading & standardization
    * `_model_tools.py` – model creation, default parameters & metric fallbacks
    * `_fold_training.py` – single CV fold training, logging & averaging
    * `_ensemble.py` – fold-ensemble prediction & metrics
    * `_optimization.py` – Optuna hyper-parameter optimization
    * `_repeat_training.py` – repeat-level training orchestration & parallel execution
    * `_cv_manager.py` – CV fold preparation & file generation
    * `_pipeline.py` – top-level `run_all_models` pipeline (including TOPSIS + Stacking)
    * `stacking.py` – optional stacking ensemble training
    * `topsis.py` – TOPSIS model ranking
  * Moved TOPSIS runtime configuration to `gpse/config/_topsis_config.py` and YAML defaults to `gpse/config/topsis.yaml`.
  * All Chinese comments, docstrings, and log messages translated to **English**.
  * Moved `ModelConfig`, `ClassificationModelConfig`, `NumpyEncoder` into `config/constants.py`.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👥 Authors

* XIAOLIU <1468835852@qq.com>
* JZHANG <zhangjian199567@outlook.com>
