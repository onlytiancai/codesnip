#!/usr/bin/env bash
# 一键生成所有章节配图（双语：先 zh 再 en）
# 使用前先 pyenv activate qlib
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/Users/huhao/.pyenv/versions/3.11.9/bin/python3}"

cd "$SCRIPT_DIR/.."  # 切到 009 项目根

for LANG in zh en; do
  for f in scripts/gen_ch*.py; do
    echo "==> [$LANG] $f"
    "$PYTHON" "$f" "$LANG"
  done
done

echo ""
echo "✅ 全部配图生成完成（中英双语）"
echo "图片目录：assets/images/"
ls -1 assets/images/ | wc -l
