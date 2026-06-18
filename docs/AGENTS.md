# AGENTS.md — GPSE 项目协作速查手册

> 本文档面向后续接入的 AI 助手（Claude / Cursor / Kimi 等），用于快速理解项目架构、技术约束和协作规范，避免重复扫描全仓库。

---

## 一、项目概述

**GPSE**（Genomic Prediction with Stacking Ensemble）是一个面向园艺作物的基因组选择（Genomic Selection）机器学习流水线。

- **一句话目标**：将原始基因组数据（VCF/PLINK）自动转换为数值矩阵，通过 14+ 种 ML 算法进行超参数优化与交叉验证，最终利用 TOPSIS 排名和 Stacking 集成学习输出预测模型。
- **目标用户**：植物育种、数量遗传学、生物信息学研究人员。
- **当前状态**：Alpha 阶段；`convert` 和 `train` 已可用，`predict` 仅 CLI stub（未实现）。

---

## 二、技术栈

| 类别 | 技术/依赖 | 版本/说明 |
|------|----------|----------|
| 语言 | Python | >= 3.10（支持 3.10/3.11/3.12） |
| 包管理 | Poetry | `pyproject.toml` 为主；`requirements.yaml` 为锁定快照 |
| 数据科学 | pandas, numpy, scipy | pandas ^2.3, numpy ^2.2 |
| ML 核心 | scikit-learn, XGBoost, LightGBM, CatBoost, NGBoost | 均为最新稳定版 |
| 超参优化 | Optuna | ^4.9，使用 TPE + MedianPruner |
| 基因组 IO | cyvcf2, pyarrow | VCF 解析；Parquet/Feather 读写 |
| CLI / UI | rich, rich-argparse, loguru | 彩色终端、富文本帮助、结构化日志 |
| 外部依赖 | PLINK 1.9, Java（可选） | PLINK 必须；Java 仅在 Beagle 基因型插补时需要 |
| 测试 | pytest, black, flake8, mypy | 测试目录 `tests/` 当前仅 1 个文件 |

**虚拟环境**：项目使用 Poetry 管理，无固定的 `venv/` 路径。开发时建议 `poetry install` + `poetry shell`。

---

## 三、架构总览

### 3.1 顶层目录

```
gpse/
├── cli.py                  # 根路由：gpse {convert,train,predict}
├── config/                 # 常量、YAML 配置、TOPSIS 配置
├── convert/                # gpse convert — 数据转换 / QC / 表型处理（独立业务域）
├── models/                 # 模型注册表、Optuna 搜索空间（回归 14 + 分类 6）
├── predict/                # gpse predict — 仅 stub，未实现
├── tasks/                  # 分类任务运行时支持（标签编码等）
├── train/                  # gpse train — 训练 / CV / TOPSIS / Stacking（独立业务域）
├── tools/                  # 独立工具脚本（如 analyze_phenotypes）
└── utils/                  # 跨模块通用工具：日志、依赖检查、打印、版本
scripts/                    # 旧脚本（向后兼容），将逐步合并到 convert/
batch/                      # 批处理脚本
tests/                      # 测试（当前仅 1 个文件，覆盖严重不足）
docs/                       # 文档、logo、截图
gpse.yaml                   # 项目级运行时配置（外部工具路径等）
```

### 3.2 子目录独立业务逻辑说明

以下子目录拥有**独立的 CLI 入口和业务闭环**，后续可在其内部补充子级 `AGENTS.md`：

- **`gpse/convert/`**：VCF/PLINK/PED → 数值矩阵、QC 过滤、LD pruning、Beagle 插补、表型匹配。
  - 核心类：`GenomicDataProcessor`（`processor.py`）— 薄编排器， delegating 到 `genotype_matrix.py`, `phenotype.py`, `qc.py`, `validators.py`。
- **`gpse/train/`**：训练流水线全链路。
  - 核心类：`GenomicPredictorV2`（`predictor.py`）— 通过**方法绑定**（method binding）将 `_data_io.py`, `_model_tools.py`, `_fold_training.py` 等子模块函数挂载为自身方法。
- **`gpse/models/`**：模型定义与搜索空间。目前回归和分类各自有一个硬编码优化器类，计划重构为 YAML 驱动的 `ModelRegistry`。
- **`gpse/predict/`**：当前仅 `cli.py` stub。预测功能为**已知大坑**，涉及 feature schema 对齐、缺失 SNP 处理、等位基因方向校验等，均未实现。

### 3.3 数据流 / 调用链

```
CLI (cli.py)
  │
  ├── gpse convert
  │      ├── external.py / PLINK / Beagle  —  格式转换与 QC
  │      ├── processor.py / GenomicDataProcessor
  │      │      ├── genotype_matrix.py  —  VCF→BED→PED→numeric matrix
  │      │      ├── phenotype.py        —  表型读取、样本匹配、标准化、自动类型检测
  │      │      └── validators.py       —  列名校验、trait 名合法性检查
  │      └── 输出：{prefix}_{trait}_genotype.parquet
  │                {prefix}_{trait}_phenotype.csv
  │                {prefix}_{trait}_phenotype_info.json   (任务类型 + n_classes)
  │
  └── gpse train
         ├── [可选] 复用 convert 做预处理 (enable_preprocess)
         ├── predictor.py / GenomicPredictorV2
         │      ├── _data_io.py          —  加载 geno/pheno，标准化
         │      ├── _cv_manager.py       —  K-Fold CV 文件生成
         │      ├── _optimization.py     —  Optuna 超参优化
         │      ├── _fold_training.py    —  单 fold 训练、评估
         │      ├── _repeat_training.py  —  多 repeat 并行（ProcessPoolExecutor）
         │      ├── _pipeline.py         —  run_all_models() 顶层循环
         │      ├── topsis.py            —  TOPSIS 多准则排名
         │      └── stacking.py          —  Stacking Ensemble（Top-N 模型集成）
         └── 输出：optimization_results_v2/
                   ├── model_comparison.csv
                   ├── {model}/repeat_{i}/fold_{j}_model.pkl
                   └── ensemble_stacking/stacking_ensemble_model.pkl
```

---

## 四、核心入口与配置

### 4.1 CLI 命令

| 命令 | 状态 | 说明 |
|------|------|------|
| `gpse convert` | 已实现 | 数据转换 / QC / 表型匹配 |
| `gpse train` | 已实现 | 模型训练 / 优化 / 集成 |
| `gpse predict` | **未实现** | 仅解析参数，执行时直接报错 |
| `gpse --version` | 已实现 | |
| `python -m gpse.tools.analyze_phenotypes` | 已实现 | 分析表型分布，辅助判断 regression vs classification |

### 4.2 关键配置文件

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | Poetry 依赖、Black 配置、pytest 配置、CLI 入口脚本 (`gpse = gpse.cli:main`) |
| `gpse.yaml` | 运行时配置：外部工具路径（如 PLINK、Beagle jar）。可被 `gpse.local.yaml` 覆盖（后者在 `.gitignore` 中） |
| `gpse/config/software.yaml` | 外部工具元信息（版本要求、检测命令） |
| `gpse/config/topsis.yaml` | TOPSIS 排名默认准则与权重 |
| `gpse/config/default.yaml` | 默认日志与显示配置 |

### 4.3 线程控制（极易踩坑）

项目对 BLAS/MKL/OpenMP 线程池有严格控制，防止多进程场景下线程爆炸：

- **CLI 顶层** (`gpse/cli.py` L15-28)：在导入 numpy/scipy 之前，通过预解析 `--n_jobs` 设置 6 个环境变量：`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`。
- **运行时**：`model.fit()` 调用被 `threadpoolctl.threadpool_limits()` 包裹。
- **并行策略**：`n_jobs` = 单模型线程数（通常 1）；`max_workers` = 模型级并行数；`repeat_workers` = 单模型内部 repeat 级并行数。

**AI 协作提示**：若新增涉及 numpy/scipy 的模块，确保在模块导入前完成线程环境变量设置；不要直接使用 `os.cpu_count()` 作为默认线程数。

---

## 五、开发规范

### 5.1 命名约定

- 模块名：`snake_case.py`
- 私有子模块：以 `_` 前缀，如 `_fold_training.py`, `_data_io.py`
- 类名：`PascalCase`，如 `GenomicPredictorV2`, `GenomicDataProcessor`
- 常量/配置：`UPPER_SNAKE_CASE` 或放在 `config/constants.py` 的 dataclass 中
- CLI 参数：`--snake-case`，已统一：`--n_jobs`, `--max_workers`, `--out_prefix`

### 5.2 代码风格

- **Black**：行宽 100（`pyproject.toml` 已配置）。
- **导入**：**必须使用绝对导入** `from gpse.xxx import ...`。已彻底移除所有 `sys.path` hack。
- **Lazy Import**：`train/`, `models/`, `tasks/`, `utils/` 的 `__init__.py` 使用 `__getattr__` 延迟导入，确保 `gpse --help` 不加载整个 ML 栈。
- **类型注解**：项目中有部分 typing，但未严格执行 mypy（存在 `type: ignore`）。新增代码建议加注解。

### 5.3 错误处理模式

- **业务错误**（如 trait 名非法、VCF/表型样本不匹配）：抛 `ValueError` 或 `FileNotFoundError`，由 CLI 捕获后打印**干净日志**（无 Python traceback），exit code = 1。
- **运行时意外异常**：打印完整 traceback，便于调试。
- **外部命令失败**：`external.py` 中的 `run_command` 会捕获 `subprocess.CalledProcessError`，压缩 PLINK 的进度条垃圾日志，并输出友好提示。

### 5.4 提交规范

项目当前无强制 commit hook，但近期 commit 风格：

```
feat(convert): auto-detect phenotype type
fix(convert): filter glued PLINK/VCF progress lines
refactor(convert): simplify file path display
```

---

## 六、构建与测试

### 6.1 安装依赖

```bash
# 推荐：Poetry
poetry install

# 或 pip
pip install -e .
```

### 6.2 外部依赖检查

```bash
gpse convert --check-deps   # 检查 PLINK、Java 等
```

### 6.3 运行测试

```bash
pytest tests/
```

> **警告**：当前 `tests/` 下仅 `test_called_process_error_logging.py` 一个文件，测试覆盖极低。任何重构前必须手动跑端到端验证。

### 6.4 打包发布

```bash
poetry build   # 生成 wheel + sdist
```

---

## 七、AI 协作提示（本项目特有的上下文）

### 7.1 领域术语与数据格式

| 术语 | 含义 |
|------|------|
| SNP | 单核苷酸多态性，特征列 |
| VCF | Variant Call Format，原始基因型输入 |
| PLINK BED/BIM/FAM | 二进制基因型格式，中间态 |
| PED/MAP | PLINK 文本格式，用于 recode 到数值 |
| QC | Quality Control：按缺失率（--geno/--mind）、MAF（--maf）过滤 |
| LD Pruning | 去除连锁不平衡（高度相关）的 SNP |
| Beagle Imputation | 基因型插补，填充缺失位点 |
| Compound Genotype | PLINK `--recode compound-genotypes 01` 输出的双字符基因型 |
| Genotype 编码 | `00→0`（纯合 REF）, `01/10→1`（杂合）, `11→2`（纯合 ALT）, missing→`3` |
| TOPSIS | 多准则决策排名算法，基于熵权法 |
| Stacking | 用 Top-N 基模型的预测值作为元特征，训练二级元学习器（Ridge/LogisticRegression） |
| phenotype_info.json | `gpse convert` 自动输出的元数据，含 `task_type`（regression/classification）、`n_classes` |

### 7.2 容易踩的坑

1. **缺失值编码不一致（已知 BUG）**
   - `convert/processor.py` 的 `GENO_DICT` fallback 是 `'3'`。
   - `convert/qc.py` 的 `recode_to_numeric()` fallback 是 `'NaN'`。
   - 若用户混用两条路径，同一缺失位点编码不同。**修复方向：统一为 `3`**。

2. **VCF / 表型样本 ID 不匹配**
   - `processor.py` 在转换前用 `cyvcf2` 做防御性检查。若交集小于较小集合的大小，直接 abort 并打印两边示例 ID。
   - 常见原因：VCF 中叫 `Ames_12781`，表型中叫 `Chipper`。

3. **Trait 名合法性**
   - 空格、`%`、`:`、`/`、`|`、括号、引号、逗号均非法。必须在转换**开始前**校验，否则 ValueError 干净退出。

4. **PLINK stdout 进度条污染日志**
   - PLINK 用 `\r` 输出 `0%1%2%...99%done`，`external.py` 已做压缩处理。新增外部工具调用时应复用 `run_command`。

5. **Stacking 模型加载路径脆弱**
   - `stacking.py` 中 `load_and_select_models()` 搜索 6 级路径 + 遍历 50 个 repeat，极易找不到模型。这是已知技术债务。

6. **`predict` 未实现，但涉及大量设计约束**
   - 需要保存 `feature_schema.json`（SNP ID、顺序、REF/ALT、编码规则、scaler）。
   - 预测时必须做 feature 对齐，不能重新做 LD pruning（会改变 feature 空间）。
   - 缺失 feature 策略建议默认 `strict`（缺就报错）。

7. **`GenomicPredictorV2` 的方法绑定模式**
   - 类体中大量使用 `load_data = load_data` 这类绑定，将子模块函数变为方法。修改子模块签名时，需同步检查 `predictor.py` 的绑定列表。

8. **自动表型类型检测（2026-06-10 新增）**
   - `convert/phenotype.py` 中的 `detect_phenotype_type()`：
     - 二元性状（≤2 唯一值）→ classification
     - 字符串标签 → classification
     - 整数且 ≤20 类、每类 ≥5 样本 → classification
     - 其余 → regression
   - `gpse train` 会自动读取 `{prefix}_{trait}_phenotype_info.json` 推断 `task_type` 和 `n_classes`。

### 7.3 重构路线图（技术债务）

来自 `todo_v2.md` 的关键计划：

- **Phase 1-3**：将硬编码模型配置（14 回归 + 6 分类）外置到 YAML，统一为 `ModelRegistry`。当前每个模型加新需改 4 处代码。
- **Phase 4**：修复 Stacking 模型加载；TOPSIS 权重外置。
- **Phase 5**：补充单元测试和端到端测试。
- **P6.x**：实现 `predict` 子命令 + `feature_schema.json` + ModelPackage 抽象。

---

## 八、待办 / 已知问题

| 优先级 | 问题 | 状态 | 位置 |
|--------|------|------|------|
| P0 | `predict` 子命令完全未实现 | 未开始 | `gpse/predict/` |
| P0 | 模型缺少 `feature_schema.json`，无法安全预测新数据 | 未开始 | `train/` |
| P1 | 缺失值编码不统一（`3` vs `NaN`） | 待修复 | `convert/qc.py`, `convert/processor.py` |
| P1 | 测试覆盖极低（仅 1 个测试文件） | 待补充 | `tests/` |
| P2 | Stacking 模型加载路径脆弱（6 级搜索） | 待修复 | `train/stacking.py` |
| P2 | 模型参数硬编码，扩展成本高 | 规划中 | `models/` |
| P2 | `create_model()` if/elif 链不可扩展 | 规划中 | `train/_model_tools.py` |
| P3 | 部分缺失基因型 `03`/`30` 未显式处理 | 低优先级 | `convert/processor.py` |

---

## 九、如何使用本文件

> **这是给 AI 助手的阅读指南。**

1. **首次接触项目**：优先阅读 **"三、架构总览"** 和 **"七、AI 协作提示"**，快速建立项目心智模型。
2. **需要修改 convert/QC/数据转换相关代码**：重点看 **"3.2 子目录独立业务逻辑说明"** 中 `gpse/convert/` 的说明，以及 **"7.2 容易踩的坑"** 第 1、2、3、4 条。
3. **需要修改 train/模型训练相关代码**：重点看 `gpse/train/` 的数据流、方法绑定模式、线程控制；修改 `_model_tools.py` 或 `models/` 时注意硬编码的搜索空间。
4. **需要新增模型算法**：先查阅 `todo_v2.md` 中的 YAML 外置化方案；在当前未重构完成前，仍需在 `regression_model_optimizer.py` / `classification_model_optimizer.py` 中同步改 4 处（注册、搜索空间、create_model 分支、默认参数）。
5. **需要实现 predict 功能**：必须先阅读 **"7.2 第 6 条"** 和 `todo_v2.md` 的 P6.x 章节。不能简单写个 `model.predict(X)`，必须补齐 feature schema 保存与对齐机制。
6. **遇到线程/性能问题**：立刻检查 **"4.3 线程控制"** 和 `cli.py` 的环境变量设置逻辑。

---

*本文档基于仓库状态 2026-06-10 生成。若项目结构发生显著变化，请同步更新本文件。*
