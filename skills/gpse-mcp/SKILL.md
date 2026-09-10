---
name: gpse-mcp
version: 1.0.0
description: "GPSE MCP Server（gpse-mcp / gpse mcp）：把 GPSE 基因组预测全流程（convert → train → predict → batch → tools）封装为标准 MCP 工具，供 AI Agent 通过 stdio 驱动。当用户需要通过 MCP 调用 GPSE 做基因型数据转换/QC、模型训练（含 Optuna 超参优化、TOPSIS、Stacking 集成）、性状预测、多性状批量训练，或管理长时训练后台任务（查状态/拉日志/停任务）时使用。实现位于 gpse/mcp/server.py；修改 MCP 工具或排查 MCP 调用问题时也使用本 skill。"
metadata:
  requires:
    bins: ["gpse-mcp"]
  cliHelp: "gpse mcp --help"
---

# gpse-mcp

GPSE 内置的 MCP (Model Context Protocol) 服务器，实现见 `gpse/mcp/server.py`，
console script 入口 `gpse-mcp = gpse.mcp.server:main`（pyproject.toml）。

## 启动与接入

stdio 传输，两种等价启动方式：

```bash
gpse-mcp        # console script
gpse mcp        # CLI 子命令
```

MCP client 配置示例：

```json
{
  "mcpServers": {
    "gpse": {
      "command": "gpse-mcp"
    }
  }
}
```

**CRITICAL — stdout 必须保持纯 JSON-RPC**：server 进程禁止向 stdout 打印任何内容
（无 logo、无 stdout 日志）。给 server 或 `gpse mcp` 子命令加功能时，任何输出只能走
stderr 或返回给 client 的 tool result。

## 执行模型（排查问题时先读这段）

- 每个 tool 调用都是一次子进程：`[sys.executable, "-m", "gpse.cli", ...]`（list 形式，
  不走 shell，参数无注入风险）。同步命令默认超时 300s，stdout/stderr 各截取尾部
  最多 8000 字符返回。
- 所有路径参数可以是绝对路径，或相对于 **MCP server 进程的工作目录**——不一定是
  用户当前 shell 的 cwd。传参时优先用绝对路径避免歧义。
- 长时命令（`gpse_train` / `gpse_batch`）默认转入**后台任务**：立即返回 8 位 hex
  `job_id`，日志与元数据在 `$GPSE_MCP_JOBS_DIR`（默认 `~/.gpse/mcp_jobs/<job_id>/`，
  含 `meta.json` + `output.log`）。任务在独立 session 中运行，停止时杀整个进程组
  （SIGTERM → 5s 宽限 → SIGKILL）。

## 工具清单

| 工具 | 作用 | 关键参数 |
|------|------|----------|
| `gpse_version` | 返回已安装 GPSE 版本（读包元数据，非 `gpse --version`） | 无 |
| `gpse_help` | 获取 CLI 帮助文本，**先用它发现 flag** | `command`: convert/train/predict/batch/tools/"" |
| `gpse_convert` | 基因型/表型 → 训练用数值矩阵（QC/LD pruning） | `vcf`/`bfile`/`ped_file`+`map_file`/`matrix_file`（四选一源）、`pheno`、`out_prefix` |
| `gpse_train` | 训练 + Optuna 优化 + TOPSIS + Stacking；**默认后台任务** | `task_type`（必填）、`geno_file`、`pheno_file`、`target_trait`、`n_classes`（分类必填）、`wait` |
| `gpse_predict` | 用训练好的模型预测新样本（SNP-ID 对齐） | `model`、`geno_file`、`out`（均必填） |
| `gpse_batch` | 多性状 YAML 批量训练；**默认后台任务** | `config`（必填）、`dry_run`、`wait` |
| `gpse_tools` | 工具子命令（如 `split` 划分训练/测试集） | `command`、`args` |
| `gpse_job_list` | 列出全部后台任务（最新在前，状态实时刷新） | 无 |
| `gpse_job_status` | 查单个任务状态：running/finished/failed/exited/stopped | `job_id` |
| `gpse_job_log` | 读任务日志尾部 | `job_id`、`tail`（默认 4000 字符） |
| `gpse_job_stop` | 停任务（整进程组 SIGTERM→SIGKILL） | `job_id` |

MCP resources：`gpse://config/default`（默认配置）、`gpse://jobs/{job_id}/log`（任务日志）。

## 典型工作流

1. **先探活**：`gpse_version` 确认 server 可达。
2. **发现参数**：不确知 flag 时先 `gpse_help("<command>")`，再通过各工具的
   `extra_args`（字符串列表，原样透传）传入，如 `["--run-qc", "--maf", "0.05"]`。
   不要凭记忆猜 flag。
3. **convert → train → predict**：
   - `gpse_convert(vcf=..., pheno=..., out_prefix=...)` → 产出
     `{prefix}_{trait}_genotype.parquet` / `_phenotype.csv` / `_phenotype_info.json`。
   - `gpse_train(task_type="regression", geno_file=..., pheno_file=...,
     target_trait=...)` → 拿到 `job_id` 后**轮询**：`gpse_job_status` +
     `gpse_job_log` 看进度；失败时读日志定位；不要乱停。只有小型/快速验证才用
     `wait=True` 同步跑（受 timeout 限制，默认 300s）。
   - `gpse_predict(model=<results 目录或 .pkl>, geno_file=..., out=...)`。
4. **多性状**：写 YAML config 走 `gpse_batch`，先 `dry_run=True, wait=True`
   打印将要执行的命令确认无误，再正式后台跑。
5. 任务结束后向用户汇报时带上 `job_id` 和关键输出路径，方便追溯。

## 修改 MCP server 时的约束

- 新增 tool：参数一律走 `_append_opt` / `_build_*_args` 构造参数列表（保持 list
  形式、无 shell），必填参数在 tool 内先做显式校验并返回 `{"ok": False, "error": ...}`
  而不是抛异常。
- 新增长时操作：复用 `_start_job` / `_load_job` / `_refresh_job` 的后台任务机制，
  不要自己起裸 `Popen`。
- `job_id` 格式固定为 8 位 hex（`_JOB_ID_RE`），校验逻辑不要放宽。
- 测试在 `tests/test_mcp_server.py`：参数构造器（`_build_*_args`）是纯函数，直接
  单测；job 管理用 `GPSE_MCP_JOBS_DIR` 指向 tmp 目录隔离。

## 相关文档

- `README.md` §6 "MCP Server (`gpse mcp`)"：面向用户的接入说明。
- `docs/AGENTS.md`：GPSE 项目整体架构、convert/train 数据流与常见坑（缺失值编码、
  trait 名合法性、线程控制等）——驱动 convert/train 前建议先读。
