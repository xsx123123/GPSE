请优化当前 GPSE 批量训练启动脚本，使其具备“内存感知的自适应并行降级”能力。

  背景：
  - 当前一次启动 11 个核心性状任务，每个任务内部再启用多个模型 worker。
  - 虽然总 CPU 线程预算设为约 90 核，但所有任务会同时加载大型基因型矩阵并创建进程池，容易
  造成内存瞬时峰值、进程被系统 OOM killer 终止。
  - 目标不是固定使用 90 核，而是在不超过 CPU 预算的前提下，优先保证任务稳定完成；内存不足
  时自动降低并发度，而不是让整个批次失败。

  请实现一个可恢复、可审计的批量调度器，要求如下：

  1. 资源检测与并行度计算
  - 启动前读取可用内存（优先 `/proc/meminfo` 的 `MemAvailable`，并兼容 cgroup v2 的
  `memory.max`、`memory.current`、`memory.events`）。
  - 获取 CPU 核数和用户设定的最大线程预算，例如 `MAX_THREADS=90`。
  - 支持参数：
    - `--max-threads`：CPU 总线程上限。
    - `--min-free-mem-gb`：必须保留的最小可用内存，如 32 GB。
    - `--mem-per-trait-gb`：每个并行性状任务的保守内存预算，如 20–40 GB。
    - `--threads-per-trait`：单个性状任务内部线程数。
    - `--max-concurrent-traits`：可选的硬上限。
  - 实际可并行性状数必须取以下限制的最小值：
    1) `floor(MAX_THREADS / threads_per_trait)`
    2) `floor((MemAvailable - min_free_mem) / mem_per_trait)`
    3) `max_concurrent_traits`（若指定）
  - 如果剩余内存不足以安全启动一个任务，则等待并定期重试，而不是强行启动。

  2. 队列式执行
  - 不要一次性启动所有性状。
  - 将性状命令放入待执行队列；只在资源条件满足时启动下一个任务。
  - 每个性状任务独立写入：
    - 结果目录
    - 标准输出/错误日志
    - PID 文件
    - 启动时间、结束时间、退出码
    - 实际线程数、启动时可用内存
  - 保持已完成任务不重跑；能够识别“完成、运行中、失败、未启动”状态。
  - 程序中断后再次执行时，应自动跳过已成功完成的性状，只重试失败或未完成的性状。

  3. 自动降级策略
  - 默认从用户指定的最大并行度开始。
  - 当出现以下任一情形时，暂停提交新任务并自动降低并行度：
    - `MemAvailable < min_free_mem_gb`
    - cgroup `memory.events` 中的 `oom` 或 `oom_kill` 计数增加
    - 子任务异常退出，且日志中包含 OOM、Killed、MemoryError、BrokenProcessPool 等内存相关
    关键词
  - 降级顺序：
    1) 优先减少“同时运行的性状任务数”；
    2) 如已只有一个性状任务，仍因内存不足失败，则降低该任务的内部 `--threads` /
    `--max_workers`；
    3) 降至 `--min-threads-per-trait` 后仍失败，则记录为失败，不无限重试。
  - 每次降级都要写入总调度日志，例如：
    `MEMORY_PRESSURE: concurrency 4 -> 3; threads_per_trait remains 8; MemAvailable=...`
  - 内存恢复到安全阈值后，可以谨慎恢复并行度，但每次最多增加 1 个任务，并设置冷却时间，避
  免抖动。

  4. GPSE 并行模型的正确配置
  - 注意 GPSE 的 `--threads` 会进一步派生 `n_jobs × max_workers × repeat_workers`。
  - 避免“多个性状任务 × 每任务多个模型 worker”造成乘法式进程膨胀。
  - 为稳定性，建议默认：
    - 每个 GPSE 性状任务：`--n_jobs 1`
    - `--repeat_workers 1`
    - 仅通过有限的 `--max_workers` 并行不同模型
  - 调度器必须记录每个任务最终实际传入的 GPSE 并行参数。
  - 不要修改实验协议：仍使用固定随机种子、50 repeats、5 folds、相同 CV 文件、相同模型列表
  和 top-3 stacking。

  5. 失败识别与重试
  - 正常完成的判据不能只看进程退出码：还需检查 `reports/run_summary.json` 存在，并验证全
  部预期模型完成、50 个 repeat 完整。
  - 内存相关失败可最多自动重试 2 次；每次重试前必须执行降级。
  - 非内存错误（输入文件缺失、参数非法、模型 bug 等）不要盲目重试，应标记为
  `FAILED_NON_MEMORY` 并输出日志摘要。
  - 汇总生成 `scheduler_status.tsv`，至少包括：

    `species,trait,status,attempt,exit_code,start_time,end_time,threads,max_workers,mem_a
    vailable_gb,error_category,log_file,result_dir`

  6. 安全性与可用性
  - 使用 Bash 或 Python 实现均可；优先选择易维护、可读性高的方案。
  - 不要删除既有结果目录或覆盖已完成结果。
  - 所有路径和资源参数通过环境变量或命令行参数配置。
  - 提供 `--dry-run`，可打印调度计划和根据当前内存计算出的并发度，但不启动训练。
  - 提供简洁 README/usage 示例，例如：
    `MAX_THREADS=90 MIN_FREE_MEM_GB=64 MEM_PER_TRAIT_GB=32 THREADS_PER_TRAIT=8 bash
    launch_lowdim_adaptive.sh`

  请先检查现有 `launch_lowdim_core_traits_90.sh` 和 `run_manuscript_missing_analyses.sh`
  的参数及结果目录结构，再在不破坏现有脚本的情况下新增自适应调度脚本。完成后用 `--dry-
  run` 验证并输出计算出的安全并行度。

  建议初始参数保守设置为：

  MAX_THREADS=90
  MIN_FREE_MEM_GB=64
  MEM_PER_TRAIT_GB=32
  THREADS_PER_TRAIT=8
  MIN_THREADS_PER_TRAIT=2
  MAX_CONCURRENT_TRAITS=2

  这样初始最多只同时跑 2 个性状，稳定后由调度器根据实时可用内存决定是否提高并发
