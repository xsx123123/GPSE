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
  * Switched `__init__.py` files in `core/`, `models/`, `utils/` to **lazy imports** (`__getattr__`) so `gpse --help` no longer loads the entire ML stack.
  * Renamed CLI args for clarity: `--threads` → `--n_jobs`, `--parallel_jobs` → `--max_workers`.
  * Added `--n_jobs` to `histgradientboost_reg` and `knn_reg` (previously missing thread control).
  * Fixed easter egg (`gpse 42`) duplicate face and unrendered rich markup.

* **Import System Unification** (`2026-06-03`)
  * Removed all `sys.path` hacks from `cli.py` and `prediction_v2.py`.
  * Unified all imports to absolute package paths (`from gpse.xxx import ...`).
  * Populated `__init__.py` files with proper exports for `config/`, `core/`, `models/`, `utils/`, `tools/`.

* **Modular Refactor of `GenomicPredictorV2`** (`2026-06-03`)
  * Split the monolithic `GenomicPredictorV2` class in `gpse/core/prediction_v2.py` into **9 focused sub-modules** under `gpse/core/`:
    * `_data_io.py` – genotype / phenotype loading & standardization
    * `_model_tools.py` – model creation, default parameters & metric fallbacks
    * `_fold_training.py` – single CV fold training, logging & averaging
    * `_ensemble.py` – fold-ensemble prediction & metrics
    * `_optimization.py` – Optuna hyper-parameter optimization
    * `_repeat_training.py` – repeat-level training orchestration & parallel execution
    * `_cv_manager.py` – CV fold preparation & file generation
    * `_pipeline.py` – top-level `run_all_models` pipeline (including TOPSIS + Stacking)
    * `_topsis_config.py` – TOPSIS configuration, representative model saving & environment logging
  * All Chinese comments, docstrings, and log messages translated to **English**.
  * Moved `ModelConfig`, `ClassificationModelConfig`, `NumpyEncoder` into `config/constants.py`.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👥 Authors

* XIAOLIU <1468835852@qq.com>
* JZHANG <zhangjian199567@outlook.com>
