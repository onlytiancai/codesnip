#!/usr/bin/env bash
# 启动 Kokoro TTS Web Demo
set -euo pipefail

cd "$(dirname "$0")"

# 优先使用 pyenv 的 qlib 环境（如已安装）
if command -v pyenv >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(pyenv init -)" 2>/dev/null || true
  if pyenv versions --bare 2>/dev/null | grep -qx "qlib"; then
    echo "[run.sh] pyenv activating qlib ..."
    export PYENV_VERSION=qlib
  fi
fi

echo "[run.sh] python = $(which python)"
echo "[run.sh] python version = $(python -V)"

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"

echo "[run.sh] starting uvicorn on http://${HOST}:${PORT}"
exec python -m uvicorn app:app --host "${HOST}" --port "${PORT}" --log-level info
