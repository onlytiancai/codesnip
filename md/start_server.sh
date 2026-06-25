#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Serving ${DIR} on http://${HOST}:${PORT}"
echo "示例：http://${HOST}:${PORT}/show-md.html?md=how-llms-actually-work-zh-CN.md"
cd "$DIR"
exec python3 -m http.server "$PORT" --bind "$HOST"
