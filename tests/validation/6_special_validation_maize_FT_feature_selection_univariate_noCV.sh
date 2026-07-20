echo "===== [$(date '+%F %T')] Run 1/2: baseline (no stacking) -> maize_FT_feature_selection_univariate ====="

gpse train \
  --geno_file /data/xiaoliu/gpse_6pecies_model/6specise/geno/maize_geno.csv \
  --pheno_file /data/xiaoliu/gpse_6pecies_model/6specise/pheno_split/maize_FT.csv \
  --target_trait FT \
  --task_type regression \
  --results_dir /data/jzhang/project/Temp/gpse_26.6.4_test/gpse_new_validation/6species/maize_FT_feature_selection_univariate_noCV \
  --models elasticnet_reg gbdt_reg svr_reg mlp_reg knn_reg rf_reg xgboost_reg adaboost_reg lightgbm_reg catboost_reg \
           kernelridge_reg gblup_reg histgradientboost_reg sgd_reg lasso_reg \
  --use_default_params \
  --test_size 0.3 \
  --n_splits 5 \
  --n_repeats 2 \
  --random_seed 42 \
  --use_same_test_set \
  --threads 60 \
  --feature_selection univariate --select_k 5000

echo "===== [$(date '+%F %T')] Run 1/2 done: baseline ====="
echo "===== [$(date '+%F %T')] Run 2/2: stacking+TOPSIS -> maize_FT_feature_selection_univariate_topsis ====="

gpse train \
  --geno_file /data/xiaoliu/gpse_6pecies_model/6specise/geno/maize_geno.csv \
  --pheno_file /data/xiaoliu/gpse_6pecies_model/6specise/pheno_split/maize_FT.csv \
  --target_trait FT \
  --task_type regression \
  --results_dir /data/jzhang/project/Temp/gpse_26.6.4_test/gpse_new_validation/6species/maize_FT_feature_selection_univariate_topsis_noCV \
  --models elasticnet_reg gbdt_reg svr_reg mlp_reg knn_reg rf_reg xgboost_reg adaboost_reg lightgbm_reg catboost_reg \
           kernelridge_reg gblup_reg histgradientboost_reg sgd_reg lasso_reg \
  --use_default_params \
  --test_size 0.3 \
  --n_splits 5 \
  --n_repeats 2 \
  --random_seed 42 \
  --use_stacking \
  --use_same_test_set \
  --threads 60 \
  --feature_selection univariate --select_k 5000

echo "===== [$(date '+%F %T')] Run 2/2 done: stacking+TOPSIS ====="
echo "===== [$(date '+%F %T')] All runs finished ====="