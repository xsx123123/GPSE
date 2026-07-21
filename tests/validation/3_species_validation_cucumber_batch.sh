#!/bin/bash
# 验证 gpse batch：cucumber 12 个连续性状批量训练
# 配置: tests/validation/cucumber_batch_config.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/cucumber_batch_config.yaml"

echo "===== [$(date '+%F %T')] Step 1/2: dry_run 预览批量命令 ====="
gpse batch --config "${CONFIG}" --dry_run

echo "===== [$(date '+%F %T')] Step 2/2: 正式执行批量训练 ====="
gpse batch --config "${CONFIG}"

echo "===== [$(date '+%F %T')] All batch runs finished ====="
