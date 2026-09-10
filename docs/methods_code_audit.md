# GPSE Methods 代码核对报告（methods_code_audit）

> 审查日期：2026-09-08
> 审查对象：仓库 `main` 分支（HEAD `a6a03bd`，GPSE 0.0.4a1）
> 审查目的：依据**实际调用路径**核对论文框架描述，为 Methods 修订提供证据。
> 审查方法：通读 `gpse/train`、`gpse/convert`、`gpse/predict`、`gpse/models`、`gpse/config`、`gpse/tasks` 及配置/验证脚本；追踪 CLI → `run_all_models` → 各子模块的真实调用链；区分"代码默认值 / 可选功能 / 论文实验实际配置"。
> 说明：仓库中**没有论文源文件**（无 .tex/.docx/Methods 稿），因此未直接修改原稿；英文修订稿见 `docs/methods_revision.md`。

---

## 0. 执行摘要（最重要结论）

1. **"熵权 TOPSIS" 与实现不符**：训练管线实际使用**固定手工权重 0.8/0.2** 的 TOPSIS（准则 = 训练侧 CV 均值 + CV 标准差）。熵权法已实现（`gpse/train/topsis.py:34-44`）但训练路径**从不调用**（`gpse/utils/genomic_utils.py:134` 硬编码 `use_entropy_weights=False`；`gpse/train/_pipeline.py:362-371` 传入 `manual_weights="0.8,0.2"`）。熵权仅能通过独立 CLI `python -m gpse.train.topsis --use_entropy_weights` 手动启用。
2. **"五阶段"顺序描述遗漏了核心模型拟合环节**：实际的顺序是 ① 数据加载/预处理 → ②（可选、折内）特征筛选/降维 → ③ **候选模型在训练分区内做超参优化 + 重复 K 折训练 + 折集成评估**（框架未列出）→ ④ TOPSIS 排名 → ⑤ Stacking → ⑥ 预测/部署重拟合。"dimensionality reduction and marker filtering" 不是独立的第 2 顺序阶段：默认关闭（`feature_selection=none`），是折内预处理流水线的一部分；convert 侧 QC/LD 剪枝是另一个可选入口且作用于**全队列**。
3. **CV/OOF 设计总体符合"训练折内拟合"的声明**（部分符合，需限定范围）：训练侧的缺失值插补、特征筛选、降维、标准化均在每条训练折上拟合后应用于对应验证折/测试折；表型标准化在 hold-out 训练标签上拟合。但 convert 侧 QC/LD 剪枝/Beagle 插补（若启用）在**划分前的全量数据**上进行，不受该声明覆盖。
4. **`--topsis_config` 实际不影响排名**：`gpse/config/topsis.yaml` 与 `get_topsis_configuration()` 在真实排名路径中未被调用（criteria/权重硬编码在 `_pipeline.py:365-368`），该配置目前仅用于日志展示。
5. **predict 不做表型反标准化**：当前训练管线不写出 `phenotype_scaler.json`（scaler 保存在 `split_manifest.json` 内），而 `gpse/predict/core.py:286` 只查找 `phenotype_scaler.json`。使用 `--standardize_phenotype` 训练的模型，预测输出停留在标准化尺度。
6. 论文实验（6 物种复现）实际配置：回归任务、14（历史）/15（新版）个模型、**默认参数（未用 Optuna）**、30% hold-out、5 折、50 次重复、seed 42、固定测试集、Top-3 stacking——与代码默认值（20% test、100 repeats、Top-5、Optuna 100 trials）**不同**，论文中必须写实验值而非代码默认值。历史实验的表型 z-score 标准化在上游全量数据上完成（当前版本已改为仅训练分区拟合）。

---

## 1. 系统结构与入口

| 项 | 实际实现 | 证据 |
| --- | --- | --- |
| CLI 入口 | `gpse {convert,train,predict,batch,mcp}`；`scripts/` 为旧版兼容脚本，非主路径 | `pyproject.toml:71-73`；`gpse/cli.py` |
| 训练编排 | `GenomicPredictorV2`（方法绑定子模块）→ `run_all_models()` | `gpse/train/predictor.py:105-313`；`gpse/train/_pipeline.py:206` |
| "Interaction Layer" | 代码中不存在该术语；最接近的对应物是 CLI + MCP 服务器（`gpse mcp`，将 convert→train→predict 封装为 MCP 工具） | `gpse/mcp/server.py`；全仓库 grep 无 "Interaction Layer" |
| 任务类型 | 回归/分类由用户通过 `--task_type` 指定；`--n_classes` 必填（分类）。`gpse convert` 的表型类型自动检测（`detect_phenotype_type`）写出 `phenotype_info.json`，但**当前 train CLI 不读取它** | `gpse/train/cli.py:355-366`（注释明确 "user-specified, no auto-detect"）；`gpse/convert/phenotype.py:292-391` |

---

## 2. 逐项核对表

结论取值：**已证实 / 部分符合 / 不符合 / 无法确认**。

### 2.1 数据读取与预处理

| # | 框架/论文陈述 | 实际实现 | 文件:行号 | 配置依据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1.1 | 基因型与表型数据加载 | 基因型矩阵 + 表型文件（CSV/Parquet/Feather，首列或 `ID` 列为样本 ID）；按 ID 取交集、表型重复 ID 保留首条、双向排序对齐；回归目标强制数值化（非数值→NaN）；缺失表型样本剔除；重复 SNP 列名报错 | `gpse/train/_data_io.py:130-292`（交集 169-185，重复 187-194，对齐 196-206，数值化 214-216，缺失剔除 244-262） | `gpse/train/_data_io.py` | 已证实 |
| 1.2 | 原始数据转换（VCF/PLINK→矩阵） | 可选路径 `gpse convert`（或 `gpse train --enable_preprocess`）：VCF→PLINK BED（`plink --vcf --make-bed --double-id`）→ PED/MAP（`--recode compound-genotypes 01 --output-missing-genotype 3`）→ 数值矩阵。加性编码 `00→0, 01/10→1, 11→2, 缺失→3`（可选 `-101` 中心化编码）。SNP ID 默认用规范化坐标 `chr<C>_<start>_<end>`（UCSC 风格 0-based 半开区间），可 `--preserve_vcf_snp_ids` 保留 VCF 原始 ID | `gpse/convert/genotype_matrix.py:32-50, 88-107, 140-230, 405-520`；`gpse/utils/snp_ids.py:17-30` | `gpse/convert/processor.py:439-814` | 已证实 |
| 1.3 | 预编码数值 VCF 直读 | 若 VCF 的 GT 已是 0/1/2 剂量，跳过 PLINK 往返直接读矩阵（QC/LD/插补在此模式下不可用） | `gpse/convert/processor.py:569-609`；`gpse/convert/genotype_matrix.py:370-405` | commit a339f01 | 已证实 |
| 1.4 | 基因型质控 | `--run-qc` 时 PLINK：`--geno 0.1 --mind 0.1 --maf 0.05`（均可覆盖）；默认**不运行**。注意：`gpse train --enable_preprocess` 不传 `run_qc`，即训练内嵌预处理**不含 QC** | `gpse/convert/qc.py:367-385`；`gpse/convert/processor.py:624-650`；`gpse/train/cli.py:280-299`（preprocess_kwargs 无 run_qc） | `gpse/utils/cli_display.py:342-352` | 已证实（但为可选功能） |
| 1.5 | LD 剪枝 | `--run-qc` 时 `--indep-pairwise 50 10 0.2`（window/step/r² 可覆盖）；作用于**全队列**（划分前） | `gpse/convert/qc.py:390-417` | 同上 | 已证实（可选；全队列） |
| 1.6 | 缺失基因型插补 | 两种互斥机制：(a) convert 侧 Beagle（`--impute`，可选，全队列）；(b) train 侧折内均值插补（`--genotype_imputation mean`，把编码 3 视为缺失后按折训练均值填充），默认 `none`（缺失保留编码 3） | `gpse/convert/qc.py:199-284`；`gpse/train/_feature_selection.py:89-121` | `gpse/train/workflow.py:271-281` | 已证实 |
| 1.7 | 表型缺失/匹配 | convert 侧按性状 dropna 后与基因型取交集、统一顺序输出；6 物种实验直接用预先拆分好的单性状表型文件 | `gpse/convert/phenotype.py:110-115, 131-192` | 复现 SOP `docs/6species_reproduction_validation.md` §4-5 | 已证实 |
| 1.8 | 类别编码 | 分类任务：`LabelEncoder` 在**全部样本**标签上拟合（0 起连续整数），保存 `label_encoder.pkl`；预测时逆变换。这是确定性的词表映射，不是分布统计拟合 | `gpse/tasks/classification.py:76-115`；`gpse/predict/core.py:282-284` | — | 已证实（注意拟合范围） |
| 1.9 | 表型标准化 | `--standardize_phenotype`（仅回归）：z-score，**在 hold-out 训练标签上拟合**，再变换测试标签；集成指标用保存的 mean/std 反变换回原始尺度。默认关闭。convert 侧另有全量 z-score/min-max（`--pheno-scale`），与训练侧标准化是两个入口 | `gpse/train/_repeat_training.py:243-250`；`gpse/train/_ensemble.py:212-218`；`gpse/convert/phenotype.py:195-281` | `gpse/train/workflow.py:316-319` | 已证实 |

### 2.2 降维与标记筛选（框架"第二阶段"）

| # | 框架/论文陈述 | 实际实现 | 文件:行号 | 配置依据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 2.1 | "dimensionality reduction and marker filtering" 作为顺序第 2 阶段 | **不是独立顺序阶段**。训练侧特征筛选/降维是折内 sklearn Pipeline 的一个步骤（`imputer→selector→scaler→model`），默认 `none`（保留全部 SNP） | `gpse/train/_model_pipeline.py:38-99`；`gpse/train/workflow.py:229-252` | `feature_selection=none` 默认 | 部分符合 |
| 2.2 | 筛选方法 | 可选：`variance`（VarianceThreshold）、`univariate`（f_classif/f_regression 的 SelectKBest）、`mutual_info`、`tree_importance`（ExtraTrees 200 棵 + SelectFromModel）、`lasso`/`elasticnet`（LassoCV/ElasticNetCV/LogisticRegressionCV + SelectFromModel，内部 CV=min(5, 每类样本数)）、`pca`、`svd`。`select_k` 默认上限 5000，`select_percentile` 可按比例覆盖 | `gpse/train/_feature_selection.py:124-219` | `gpse/train/workflow.py:246-264` | 已证实 |
| 2.3 | 是否使用表型信息 | `univariate/mutual_info/tree_importance/lasso/elasticnet` 使用训练折标签；`variance/pca/svd` 不使用 | 同上 | — | 已证实 |
| 2.4 | 拟合范围 | **逐折拟合**：Optuna 目标、折训练、stacking OOF、部署集成中的每一条流水线都在各自的训练子集上拟合 selector（及 imputer/scaler），再变换验证/测试数据 | `gpse/train/_optimization.py:98-104`；`gpse/train/_fold_training.py:52-70`；`gpse/train/stacking.py:100-111`；`gpse/train/_ensemble.py:56-67` | — | 已证实（限训练侧） |
| 2.5 | convert 侧标记过滤 | QC/MAF/LD 剪枝在**全队列**（含未来测试样本）上计算，属群体级操作，不受"仅训练折"声明覆盖；6 物种实验未使用该路径（输入为外部预处理矩阵） | `gpse/convert/qc.py:300-419` | 复现 SOP §5.1 | 部分符合（需在论文中限定） |

### 2.3 候选模型训练与评价

| # | 框架/论文陈述 | 实际实现 | 文件:行号 | 配置依据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 3.1 | 回归/分类模型集合 | 回归 15 个：ElasticNet、Lasso、SGD、LinearSVR、AdaBoost、KNN、RF、LightGBM、HistGBR、GBDT、XGBoost、CatBoost、KernelRidge（RBF）、MLP、gblup_reg（线性核 KernelRidge，`alpha=1`，恒用默认参数，作基因组基线）；分类 6 个：RF、SVC、XGBoost、LightGBM、CatBoost、MLP | `gpse/config/models.yaml`（全文件）；`gpse/models/safe_kernel_ridge.py` | — | 已证实 |
| 3.2 | 评价指标 | 回归：Pearson（主）、Spearman、R²、MSE、RMSE、MAE；分类：Accuracy、Precision/Recall/F1（weighted）、AUC（二类 macro / 多类 ovr-weighted）、PR-AUC、log_loss。Optuna 目标用 Pearson r（回归）或 balanced accuracy（分类） | `gpse/utils/genomic_utils.py:28-57`；`gpse/tasks/classification.py:117-222`；`gpse/train/_optimization.py:108-137` | — | 已证实 |
| 3.3 | 超参优化 | Optuna TPE + MedianPruner，默认 ≤100 trials，早停 patience 20；目标 = mean(CV) − λ·std(CV)，λ 默认 0.5；CV 为训练分区上的 K 折。默认 `optuna_per_repeat=False`：仅在第 1 个 repeat 上搜索，参数冻结后用于其余 repeat | `gpse/train/_optimization.py:26-249`（目标 149-159，早停 162-184，采样器 187-196） | `gpse/train/workflow.py:95-143` | 已证实 |
| 3.4 | 搜索空间 | 每模型 YAML 声明（int/float/categorical/fixed）或 builtin param_func（条件空间） | `gpse/config/models.yaml:37-385` | — | 已证实 |
| 3.5 | 模型比较规则 | 逐 repeat 汇总各折指标均值±std → `summary_results.json`；跨 repeat 求均值；比较表 `model_comparison.csv` 按 Ensemble 指标排序（仅展示） | `gpse/utils/genomic_utils.py:153-251`；`gpse/train/_repeat_training.py:467-616` | — | 已证实 |

### 2.4 TOPSIS

| # | 框架/论文陈述 | 实际实现 | 文件:行号 | 配置依据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 4.1 | "entropy-weighted TOPSIS" | 训练路径调用 `call_topsis_evaluator(..., manual_weights="0.8,0.2", min_transform="neglog")`，且 `use_entropy_weights=False` **硬编码**。熵权法函数存在但主路径不调用 | `gpse/train/_pipeline.py:362-371`；`gpse/utils/genomic_utils.py:126-135`；`gpse/train/topsis.py:34-44, 117-128` | — | **不符合**（需改写或启用熵权） |
| 4.2 | 决策矩阵 | 模型 × 2 准则：[CV 均值（max），CV 标准差（min）]，CV 均值/标准差来自各 repeat 训练侧验证折指标在 repeat 间的聚合（`training_selection`）；**不含测试集指标** | `gpse/train/_pipeline.py:344-360`；`gpse/train/_repeat_training.py:595-602` | — | 已证实（训练侧） |
| 4.3 | min 准则变换 | `neglog`：`-log(x + 1e-12)`（另有 reciprocal/minmax_inv 选项） | `gpse/train/topsis.py:50-62` | `_pipeline.py:368` | 已证实 |
| 4.4 | 归一化 | 列向量 L2 归一化：`x / sqrt(Σx²)`（先做 min 变换再归一化） | `gpse/train/topsis.py:63-65` | — | 已证实 |
| 4.5 | 权重 | 固定 0.8/0.2（归一化后） | `gpse/train/topsis.py:121-128` | `_pipeline.py:367` | 已证实（手工权重） |
| 4.6 | 理想解/贴近度 | 加权矩阵列 max/min 为理想/负理想解；欧氏距离 D⁺/D⁻；贴近度 `D⁻/(D⁺+D⁻)`；降序排名 | `gpse/train/topsis.py:68-76` | — | 已证实 |
| 4.7 | 排名用途 | Top-N（默认 5；实验 3）进入 stacking 基模型；排名同时决定 `final_predictions.csv`（rank-1 模型选中 repeat 的折集成测试预测） | `gpse/train/_pipeline.py:381-382, 113-203` | `gpse/train/workflow.py:310-315` | 已证实 |
| 4.8 | `--topsis_config` | `topsis.yaml`/`get_topsis_configuration()` 在排名路径**未被调用**，criteria 硬编码于 `_pipeline.py`；该参数目前仅影响日志 | `gpse/config/_topsis_config.py:110-129`（无调用方）；grep 证实 `_pipeline.py` 不引用 | — | 不符合（配置开关失效，应注明或修复） |

### 2.5 Stacking 与预测

| # | 框架/论文陈述 | 实际实现 | 文件:行号 | 配置依据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 5.1 | OOF 预测作为元模型训练输入 | 对每个基模型：在 **hold-out 训练分区**上重新做 K 折（默认 5 折，KFold/StratifiedKFold，seed 42），逐折重训完整流水线（含预处理/特征筛选），验证折预测构成元特征矩阵；随后基模型在**全训练分区**上重训，用于测试预测与部署 | `gpse/train/stacking.py:91-128` | `cv_folds=5` 默认 | 已证实 |
| 5.2 | OOF 预测值形态 | 回归：`model.predict()`（连续值）；分类：二类取 `P(正类)`，多类取 `max(predict_proba)` | `gpse/train/stacking.py:72-76` | — | 已证实 |
| 5.3 | 元模型 | 回归 Ridge(alpha=1)；分类 LogisticRegression(C=1, max_iter=1000)；元特征先 StandardScaler；**无调参** | `gpse/train/stacking.py:130-139` | — | 已证实 |
| 5.4 | 元模型评估 | 在 hold-out 测试集上单次评估（基模型全训练分区重训后生成测试元特征）；不把元模型训练集得分当泛化性能报告（训练侧指标单独存放） | `gpse/train/stacking.py:158-189`；`gpse/train/_pipeline.py:443-464` | — | 已证实 |
| 5.5 | 最终重拟合 | 每模型另存：`representative_model/`（选中 repeat 的参数在全量 X,y 上重拟合）与 `deployment_ensemble/`（全量数据上的 K 折集成，seed+10000）；两者明确标注"部署用，不得在原 hold-out 上评估" | `gpse/config/_topsis_config.py:132-212`（175-176 全量拟合）；`gpse/train/_ensemble.py:32-105`（info.json 83-98） | `--save_models/--save_representative` | 已证实 |
| 5.6 | 新样本预测路径 | `gpse predict`：解析产物（部署集成目录/stacking pkl/representative/model.pkl）→ 读 `feature_manifest.json` 特征顺序 → VCF（规范化 SNP ID）或矩阵输入 → `reindex` 对齐列，缺失 SNP 填 3（或插补码）→ 复用保存的 imputer/selector/scaler（pipeline.transform）→ 集成：回归取成员均值、分类取 `argmax(mean(proba))` → LabelEncoder 逆变换 → 输出 CSV + 对齐报告 | `gpse/predict/core.py:32-322`（对齐 112-139，集成 221-269，逆编码 282-284） | — | 已证实 |
| 5.7 | 预测期表型反标准化 | 查找 `phenotype_scaler.json`——**当前训练管线从不写该文件**（scaler 在 `split_manifest.json` 内，predict 不读取）。`--standardize_phenotype` 训练的模型预测值停留在标准化尺度 | `gpse/predict/core.py:286-290`；grep 证实无写入方（仅旧版 `scripts/genomic_prediction_v2.py:510`） | — | 不符合（功能缺口） |

---

## 3. CV、OOF 与数据泄漏专项回答

**Q1 数据何时划分？是否有独立测试集/外层 CV？**
有独立 hold-out 测试集，无外层嵌套 CV。`run_all_models` 在训练开始前一次性创建固定 hold-out（`use_same_test_set=True` 默认；`_pipeline.py:246-271`），所有模型、所有 repeat 共用；划分清单（`train_ids.txt`/`test_ids.txt`/`split_manifest.json`）落盘。测试集仅用于最终评估与报告（`_write_holdout_reports`，`_pipeline.py:474-479`），不参与超参选择、repeat 选择、TOPSIS 排名与 stacking 基模型选择（`_repeat_training.py:142-170`、`_pipeline.py:343-397`）。注意 `use_same_test_set=False` 时每个 repeat 各自重新划分（`_repeat_training.py:225-235`），此时 stacking 另用 base seed 生成一个划分（`_pipeline.py:407-416`），与各 repeat 的划分不同。

**Q2 折数、重复、种子、划分方式？**
默认 5 折 × 100 repeat（实验：5 折 × 50 repeat），seed 42。回归 KFold(shuffle)，分类 StratifiedKFold(shuffle)（`_cv_manager.py:118-165`）；逐 repeat 种子 = base+repeat_idx。可选 `structure_aware`：训练分区上 PCA(≤10 PC) + KMeans 聚类 → 回归 GroupKFold / 分类 StratifiedGroupKFold，保持遗传结构簇完整（`_cv_manager.py:24-39`）；hold-out 也可选结构感知（PCA/KMeans + GroupShuffleSplit，`_repeat_training.py:61-110`）。外部 `--cv_file`（仅回归生效；分类忽略 cv 文件改用 StratifiedKFold，`_cv_manager.py:167-186`）：读取 `cv{r}` 列的折分配，映射到训练分区样本。

**Q3 模型比较/超参优化/最终评估是否同折？是否嵌套？**
非嵌套。超参优化与折训练使用**同一组训练分区 CV 折**（同一 repeat 内，`_optimization.py:72` 与 `_repeat_training.py:256` 都调用 `generate_cv_folds_from_file`）；因此用于 TOPSIS/repeat 选择的 `cv_mean` 是在同折上调参得到的验证分数，属常见的乐观估计（选择偏差），但测试集评估独立于该选择。最终性能在 hold-out 测试集上给出。

**Q4 缺失处理/标准化/降维/筛选是否逐折拟合？**
是（训练侧）：`GenotypePreprocessor.fit`（imputer→selector→scaler）在每条训练折上拟合，`transform` 应用到验证折与测试集（`_model_pipeline.py:53-71`；`_fold_training.py:52-70`；`_optimization.py:98-107`；stacking OOF 折 `stacking.py:100-111`）。表型标准化在 hold-out 训练标签上拟合（`_repeat_training.py:243-246`）。树类模型跳过特征标准化（`_model_pipeline.py:22-35`）。
**不属于折内拟合的操作**（需在论文中区分）：(a) convert 侧 QC/MAF/LD 剪枝/Beagle 插补（全队列）；(b) 基因型 0/1/2/3 编码（逐样本确定性规则）；(c) 分类 LabelEncoder（全样本标签词表，确定性映射）；(d) 外部 cv_file 的折分配本身（预先定义）。

**Q5 OOF 如何生成与拼接？**
仅 stacking 使用 OOF：对每个基模型独立做 5 折 KFold（整个训练分区），`meta_train[val_idx, col] = prediction_feature(fold_pipeline, X_val)` 逐块填入 (n_train × n_base) 矩阵（`stacking.py:95-111`）。每个训练样本的元特征来自未见过该样本的基模型实例。**限制**：基模型超参数来自全训练分区 CV 上的选择，基模型集合（Top-N）来自全训练分区 CV 的 TOPSIS 排名——因此 OOF 仅对"基模型权重"无偏，对"超参+模型选择"并非完全独立；不应表述为"无偏预测"。

**Q6 回归/分类的 OOF 输出？**
回归：连续预测值 `predict()`。分类：`predict_proba`——二类取正类概率列，多类取每样本最大类概率（`stacking.py:72-76`）。

**Q7 TOPSIS 排名在哪些数据上计算？最终评估是否独立？**
排名只用训练侧 CV 指标（`model_comparison_cv.csv`，`_pipeline.py:344-360`）。最终 hold-out 报告在 stacking 与所有选择冻结之后写出（`_write_holdout_reports` 文档字符串明确 "after all model/stacking choices are frozen"）。测试指标在每个 fold 模型上同时被计算（`avg_test_*`）但仅作报告，不反馈到任何选择。

**Q8 元模型在哪训练/调参/评估？**
元模型在训练分区的 OOF 矩阵上训练（Ridge/Logistic，固定超参，无调参）；在 hold-out 测试集上单次评估（`stacking.py:158-189`）。训练侧得分（`train_metrics`）单独保存，未冒充泛化性能。

**Q9 最终基模型是否全量重拟合？推理是否复用训练期变换？**
是。`representative_model` 与 `deployment_ensemble` 在全部样本（含原测试样本）上重拟合，并显式标注不得在原 hold-out 上评估（`_ensemble.py:83-98`）。推理复用保存的 imputer/selector/scaler、特征顺序（`feature_manifest.json`）与 LabelEncoder（`predict/core.py:210-284`）。**例外**：表型反标准化参数未被 predict 读取（见 2.5/5.7）。

---

## 4. 论文实验实际配置（vs 代码默认值）

依据 `docs/6species_reproduction_validation.md`（复现 SOP）与 `tests/validation/`、`batch/` 脚本：

| 参数 | 代码默认 | 6 物种实验（历史/复现） | 依据 |
| --- | --- | --- | --- |
| task_type | regression | regression（18 个性状，6 物种） | SOP §1/§7 |
| 模型 | 全部（回归 15） | 显式 14 个（历史；新版另含 gblup_reg 共 15） | SOP §7 |
| 超参 | Optuna 100 trials | `--use_default_params`（**默认参数，未调参**） | SOP §7 |
| hold-out | 0.2 | 0.3 | SOP §7 |
| 折数 | 5 | 5 | SOP §7 |
| repeat | 100 | 50（冒烟测试 2） | SOP §7 |
| seed | 42 | 42 | SOP §7 |
| 固定测试集 | True | True | SOP §7 |
| stacking Top-N | 5 | 3 | SOP §7 |
| stacking CV 折 | 5 | 5 | SOP §7 |
| 特征筛选 | none | none（另做了 univariate/select_k=5000 的专项验证） | `6_special_validation_maize_FT_feature_selection_univariate.sh` |
| 表型标准化 | 关闭（训练侧） | 上游已完成 z-score；不传 `--standardize_phenotype` | SOP §5.2 |
| CV 文件 | 自动生成 | 历史 50×5 性状级 CV 文件（`--cv_file`） | SOP §6 |

**⚠ 待作者确认**：论文报告的结果对应哪套配置（历史 14 模型默认参数 vs 新版 15 模型；是否包含 Optuna 调参的运行；univariate 特征筛选实验是否进入正文）。

---

## 5. 影响论文准确性/评估可信度的问题（按重要性排序）

1. **"entropy-weighted TOPSIS" 与实现不符**（2.4/4.1）。要么改论文措辞为"固定权重（0.8/0.2）的 TOPSIS"，要么修改代码启用熵权并重跑。当前状态下论文不能声称熵权。
2. **五阶段描述遗漏候选模型训练与超参优化环节**，且把特征筛选/降维表述为独立顺序阶段。建议按 §6 修订句改写。
3. **`--topsis_config`/`topsis.yaml` 未接入排名路径**：论文若描述"TOPSIS 准则与权重可由配置文件定义"则不成立（当前硬编码）。
4. **predict 缺表型反标准化**：`--standardize_phenotype` 训练的模型预测值尺度错误（未写 `phenotype_scaler.json`）。
5. **stacking OOF 的独立性范围**：OOF 仅对基模型权重成立；超参与基模型集合选择使用了全训练分区 CV。论文不应称 OOF 为"无偏"或暗示消除全部选择偏差。
6. **convert 侧 QC/LD/插补为全队列操作**：若论文实验用到该路径，需在 Methods 中如实描述其作用范围（群体级、划分前）。
7. **实验配置 ≠ 代码默认值**：论文必须写 30%/50/Top-3/默认参数等实验值；若写了 Optuna 调参需确认对应运行真实存在。
8. （次要）`avg_test_*` 指标是"各折模型在测试集上的平均"，与折集成指标（`avg_ensemble_*`）并存，论文引用时应明确用哪一列。
9. （次要）分类的 LabelEncoder 在全样本上拟合（确定性词表，通常可接受，但严格说测试标签参与了编码词表）。
10. （次要，测试维护）`tests/test_audit_fixes.py` 2 项失败：pyproject 未声明 threadpoolctl；`_find_result_file` 已无搜索深度上限。133 项通过。

---

## 6. 框架概述原句与建议替换句

| 原句 | 问题 | 建议替换 |
| --- | --- | --- |
| "The pipeline comprises five stages: genotypic and phenotypic data loading and preprocessing; dimensionality reduction and marker filtering; entropy-weighted TOPSIS evaluation of candidate models; stacking ensemble construction; and phenotype prediction." | 遗漏模型拟合/调参核心环节；特征筛选非顺序阶段；熵权不属实 | "The core pipeline proceeds in six steps: (i) genotypic and phenotypic data loading, sample matching and preprocessing; (ii) optional, fold-local marker filtering and dimensionality reduction; (iii) benchmarking of candidate models under repeated K-fold cross-validation on the training partition, with optional Optuna hyperparameter optimization; (iv) TOPSIS ranking of candidate models from training-side cross-validation statistics; (v) construction of a stacking ensemble from the top-ranked models using out-of-fold predictions; and (vi) phenotype prediction from refitted deployment models." |
| "entropy-weighted TOPSIS evaluation of candidate models" | 权重为固定 0.8/0.2 | "TOPSIS evaluation of candidate models with fixed criteria weights (0.8 for cross-validation performance, 0.2 for cross-validation stability)" — 或启用熵权后保留原句 |
| "Cross-validation supports model benchmarking and hyperparameter optimization." | 属实但需注明实验配置 | 保留，但补充："In the experiments reported here, candidate models were run with default hyperparameters"（若属实，待作者确认） |
| "Out-of-fold predictions provide the training inputs for the stacking meta-model." | 属实，需限定 | "Out-of-fold predictions of the base models, generated by re-training each base model within a K-fold split of the training partition, provide the input features for the stacking meta-model." |
| "Data-dependent preprocessing, dimensionality reduction and marker filtering are fitted or determined exclusively within the corresponding training folds." | 需限定范围（不含 convert 侧全队列 QC；不含超参/模型选择） | "Within the training pipeline, data-dependent preprocessing (genotype imputation, feature scaling), marker filtering and dimensionality reduction are fitted on each training fold and applied unchanged to the corresponding validation and test data; genotype quality control and LD pruning, when performed during data conversion, operate on the full cohort before the hold-out split." |

---

## 7. 待作者确认清单

1. 论文结果对应的运行：(a) 历史 14 模型默认参数运行，还是新版 15 模型（含 gblup_reg）？(b) 是否有任何结果来自 Optuna 调参运行？(c) univariate/select_k=5000 特征筛选实验是否进入正文？
2. 论文是否需要声称"熵权 TOPSIS"？若是，需先修改代码（把 `use_entropy_weights=True` 接入 `_pipeline.py` 调用或接通 `--topsis_config`）并重跑，否则按固定权重描述。
3. 分类任务是否有论文实验（当前复现 SOP 只有回归）？若有，需补充其配置（n_classes、分层、AUC 口径）。
4. `--standardize_phenotype` 是否用于任何论文实验？（影响是否需要修复 predict 反标准化缺口。）
5. 结构感知划分（`structure_aware`）是否用于任何论文实验？
6. 引用文献：Optuna (Akiba et al., 2019)、TOPSIS (Hwang & Yoon, 1981)、熵权法、Stacking (Wolpert, 1992)、scikit-learn、XGBoost/LightGBM/CatBoost 的版本与引用需按实际环境（`pip freeze`，见 SOP §8.2）核实后再写入。
