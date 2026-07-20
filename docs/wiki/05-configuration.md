# 05. 配置文件

## 项目级 `gpse.yaml`

放置于工作目录（或 `gpse.local.yaml`，或用 `--config PATH` 显式指定），用于声明外部工具与 convert 配置：

```yaml
external_tools:
  - name: plink
    cmd: plink            # 可写绝对路径
  - name: java
    cmd: java
    required: false
convert:
  beagle_jar_path: /path/to/beagle.jar   # 仅 --impute 时使用
```

加载逻辑（`gpse/utils/configuration.py`）：按 **包内默认 → 项目 `gpse.yaml` → `gpse.local.yaml` → CLI `--config`** 顺序深合并，named-list 按 `name` 覆盖；`--no-project-config` 关闭项目配置自动加载。

## 包内置配置

| 文件 | 内容 |
|------|------|
| `gpse/config/software.yaml` | 软件元信息 + 外部工具声明（plink：`--version` 探测、min_version 1.9、必需；java：可选） |
| `gpse/config/default.yaml` | 软件元信息 + 日志配置（log_level、more_info、Label） |
| `gpse/config/topsis.yaml` | TOPSIS 评价指标：分类 `Test Accuracy (max, 0.8) / Test Accuracy (std) (min, 0.2)`；回归 `Test Pearson (max, 0.8) / Test Pearson (std) (min, 0.2)` |

## 训练产物中的清单文件

- `feature_manifest.json`：训练特征的 canonical SNP ID 列表，供 predict 对齐（`gpse/utils/feature_manifest.py`）；
- `split_manifest.json`：hold-out / CV 划分记录，保证可复现。
