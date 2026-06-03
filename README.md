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

GPSE provides a unified command-line interface via the `gpse` executable (installed automatically with `pip install .` or `poetry install`).

### 1. Data Preprocessing Only

Convert raw VCF/PLINK data to numerical matrices and match genotypes with phenotypes.

```bash
gpse --preprocess_only \
    --preprocess_prefix processed_data \
    --vcf_file path/to/genotypes.vcf \
    --raw_pheno_file path/to/phenotypes.txt \
    --target_trait Trait_Name \
    --plink_path plink
```

### 2. Preprocessing + Model Training (One-Stop)

Run preprocessing and model training in a single command.

```bash
gpse --enable_preprocess \
    --preprocess_prefix processed_data \
    --vcf_file path/to/genotypes.vcf \
    --raw_pheno_file path/to/phenotypes.txt \
    --target_trait Trait_Name \
    --task_type regression \
    --n_splits 5 \
    --n_repeats 10 \
    --trials 50 \
    --use_stacking \
    --top_n_models 5 \
    --results_dir output_results/
```

### 3. Model Training Only (With Pre-processed Data)

If you already have genotype and phenotype CSV files:

```bash
gpse \
    --geno_file processed_data_genotype.csv \
    --pheno_file processed_data_phenotype.csv \
    --target_trait Trait_Name \
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

### 4. Classification Task

```bash
gpse \
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

### 5. Analyze Phenotypes

Quickly analyze phenotype data to determine the appropriate task type (Regression vs Classification).

```bash
python -m gpse.tools.analyze_phenotypes
```

### 6. Show Help

```bash
gpse --help
gpse --version
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
