# 01. Overview

## What GPSE Does

GPSE (Genomic Prediction with Stacking Ensemble) is a genomic selection machine learning pipeline aimed at horticultural crop breeding. Starting from raw VCF/PLINK genotype data, it covers:

1. **Data conversion**: VCF → PLINK → PED/MAP → additive numeric matrix → phenotype matching;
2. **Quality control**: missing-rate/MAF filtering, optional Beagle imputation, LD pruning;
3. **Model training**: Optuna hyperparameter optimization + repeated cross-validation + TOPSIS multi-metric ranking;
4. **Ensemble learning**: strict out-of-fold (OOF) Stacking that prevents data leakage;
5. **Prediction deployment**: canonical SNP ID feature alignment to predict phenotypes of new samples from training artifacts.

## Key Features

- **15 regression + 6 classification algorithms**: ElasticNet, GBDT, SVR, MLP, KNN, RF, XGBoost, AdaBoost, LightGBM, CatBoost, KernelRidge, GBLUP baseline, and more;
- **Dual task modes**: `--task_type regression | classification`;
- **Reproducible / leak-free**: fixed hold-out + train-only CV + strict OOF stacking (see `train_pipeline.md`);
- **Cross-user model reuse**: canonical SNP IDs (`chr<chrom>_<chromStart>_<chromEnd>`) and `feature_manifest.json`;
- **Parallel training**: `--threads` automatically derives `n_jobs / max_workers / repeat_workers`;
- **TOPSIS multi-criteria evaluation**: entropy or manual weights, balancing accuracy and stability;
- **Batch multi-trait training**: `gpse batch` runs the full training workflow for many traits from one YAML config;
- **Train/test splitting**: `gpse tools split` carves out a held-out test set for later `gpse predict` scoring.

## Architecture

```
gpse/
├── cli.py            # CLI entry point (argparse + rich-argparse), routes subcommands
├── convert/          # gpse convert: data conversion / QC / recode / deps
├── train/            # gpse train: training, optimization, CV, Stacking, TOPSIS
├── predict/          # gpse predict: feature alignment and phenotype prediction
├── batch/            # gpse batch: YAML-driven multi-trait training + result merging
├── models/           # Model registries and Optuna search spaces (regression/classification)
├── tasks/            # Task layer (classification label encoding, classification metrics)
├── config/           # Package YAML configs and model constants
├── tools/            # gpse tools: train/test splitting + phenotype analysis helper
└── utils/            # Config loading, SNP IDs, feature manifest, parallelism, logging
```

## Dependencies

- **Python ≥ 3.10** (Poetry build); core libraries: scikit-learn, xgboost, lightgbm, catboost, optuna, cyvcf2, ngboost, pandas/numpy, loguru/rich, PyYAML;
- **External tools**: PLINK ≥ 1.9 (required); java + Beagle jar (only needed for imputation). Check with `gpse convert --check-deps`.
