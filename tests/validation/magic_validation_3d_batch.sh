#!/bin/bash
# 验证 gpse batch：生菜 MAGIC 群体 3D 表型（9 个性状，LD 过滤基因型）
# 配置: tests/validation/magic_3d_batch_config.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/magic_3d_batch_config.yaml"

echo "===== [$(date '+%F %T')] Step 1/2: dry_run 预览批量命令 ====="
gpse batch --config "${CONFIG}" --dry_run

echo "===== [$(date '+%F %T')] Step 2/2: 正式执行批量训练 ====="
gpse batch --config "${CONFIG}"

echo "===== [$(date '+%F %T')] All batch runs finished ====="
