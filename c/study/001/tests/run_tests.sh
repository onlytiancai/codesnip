#!/bin/sh
# 一键运行全部测试：先增量编译，保证跑的是最新源码
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

make -s stats tests/test_stats

echo "=== 1/2 核心函数单元测试 ==="
./tests/test_stats
echo
echo "=== 2/2 CLI 端到端测试 ==="
./tests/test_cli.sh
echo
echo "全部测试通过"
