#!/usr/bin/env bash
# 手工测试 tiptap-rewrite-server 的 curl 集合。
# 前置：server 已运行在 http://127.0.0.1:8787。
# 用法：bash curl-tests.sh           # 全部跑一遍（核心 3 个）
#       bash curl-tests.sh healthz   # 只跑 /healthz
#       bash curl-tests.sh translate # 只跑 /api/translate
#       bash curl-tests.sh rewrite   # 只跑 /api/rewrite-word
#       bash curl-tests.sh errors    # 只跑错误路径

set -u
BASE="${BASE:-http://127.0.0.1:8787}"

hr() { printf '\n──────── %s ────────\n' "$1"; }
req() {
  local method="$1" path="$2" body="${3:-}"
  hr "$method $path${body:+  body=$body}"
  if [ -n "$body" ]; then
    curl -sS -w "\nHTTP %{http_code}\n" -X "$method" "$BASE$path" \
      -H 'Content-Type: application/json' -d "$body"
  else
    curl -sS -w "\nHTTP %{http_code}\n" -X "$method" "$BASE$path"
  fi
}

case "${1:-all}" in
  healthz)
    req GET /healthz
    ;;
  translate)
    hr "POST /api/translate  （单句）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"我昨天去公园。"}'

    hr "POST /api/translate  （多句）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"我昨天去公园，看到一只猫。它非常可爱。"}'

    hr "POST /api/translate  （显式 source=zh target=en）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"学习永无止境。","source":"zh","target":"en"}'
    ;;
  rewrite)
    hr "POST /api/rewrite-word  （不带 sourceSentence）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-word" \
      -H 'Content-Type: application/json' \
      -d '{"word":"cute","context":"It was extremely cute."}'

    hr "POST /api/rewrite-word  （带 sourceSentence）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-word" \
      -H 'Content-Type: application/json' \
      -d '{"word":"cute","context":"It was extremely cute.","sourceSentence":"它非常可爱。"}'

    hr "POST /api/rewrite-word  （带 word + context + sourceSentence）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-word" \
      -H 'Content-Type: application/json' \
      -d '{"word":"good","context":"This is a good idea.","sourceSentence":"这是个好主意。"}'
    ;;
  errors)
    hr "POST /api/translate  （缺 text → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' -d '{}'

    hr "POST /api/translate  （source=en 不支持 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"hello","source":"en","target":"en"}'

    hr "POST /api/rewrite-word  （缺 word → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-word" \
      -H 'Content-Type: application/json' -d '{}'

    hr "POST /api/rewrite-word  （sourceSentence 非字符串 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-word" \
      -H 'Content-Type: application/json' \
      -d '{"word":"good","sourceSentence":42}'
    ;;
  all|*)
    req GET /healthz
    req POST /api/translate '{"text":"我昨天去公园。"}'
    req POST /api/rewrite-word '{"word":"good","context":"This is a good idea.","sourceSentence":"这是个好主意。"}'
    ;;
esac