#!/usr/bin/env bash
# 手工测试 tiptap-rewrite-server 的 curl 集合。
# 前置：server 已运行在 http://127.0.0.1:8787，且本地 Ollama 已拉好模型。
# 用法：bash curl-tests.sh              # 全部跑一遍（每个能力一个代表 case）
#       bash curl-tests.sh healthz      # 只跑 /healthz
#       bash curl-tests.sh translate    # 只跑 /api/translate（含语气）
#       bash curl-tests.sh translate-reverse  # 只跑 /api/translate（英译中）
#       bash curl-tests.sh selection    # 只跑 /api/rewrite-selection（word / phrase）
#       bash curl-tests.sh sentence     # 只跑 /api/rewrite-sentence（4 个 operation + tone）
#       bash curl-tests.sh grammar      # 只跑 /api/grammar-check
#       bash curl-tests.sh instruction   # 只跑 /api/rewrite-instruction
#       bash curl-tests.sh dictionary   # 只跑 /api/dictionary/*
#       bash curl-tests.sh errors       # 只跑错误路径

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
    hr "POST /api/translate  （单句，默认语气）"
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

    hr "POST /api/translate  （tone=formal）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"教育对孩子很重要。","tone":"formal"}'

    hr "POST /api/translate  （tone=concise）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"教育对孩子很重要。","tone":"concise"}'

    hr "POST /api/translate  （tone=persuasive）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"我们都应该保护环境。","tone":"persuasive"}'
    ;;

  translate-reverse)
    hr "POST /api/translate  （英译中：source=en target=zh）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"Education is very important for children.","source":"en","target":"zh"}'
    ;;

  dictionary)
    hr "GET /api/dictionary/status  （词库路径 + 词数）"
    curl -sS -w "\nHTTP %{http_code}\n" "$BASE/api/dictionary/status"

    hr "GET /api/dictionary/lookup?word=important  （命中）"
    curl -sS -w "\nHTTP %{http_code}\n" \
      "$BASE/api/dictionary/lookup?word=important"

    hr "GET /api/dictionary/lookup?word=zzzz  （查不到 → 200 + found:false）"
    curl -sS -w "\nHTTP %{http_code}\n" \
      "$BASE/api/dictionary/lookup?word=zzzz"

    hr "GET /api/dictionary/lookup  （缺 word → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" "$BASE/api/dictionary/lookup"
    ;;

  selection)
    hr "POST /api/rewrite-selection  （kind=word）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' \
      -d '{"text":"good","kind":"word","context":"This is a good idea."}'

    hr "POST /api/rewrite-selection  （kind=word + sourceSentence）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' \
      -d '{"text":"good","kind":"word","context":"This is a good idea.","sourceSentence":"这是个好主意。"}'

    hr "POST /api/rewrite-selection  （kind=phrase：允许返回多词候选）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' \
      -d '{"text":"very important","kind":"phrase","context":"Education is very important for children.","sourceSentence":"教育对孩子很重要。"}'

    hr "POST /api/rewrite-selection  （kind=phrase：跨词的松散搭配）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' \
      -d '{"text":"in my opinion","kind":"phrase","context":"In my opinion, the plan is risky."}'
    ;;

  sentence)
    hr "POST /api/rewrite-sentence  （operation=rewrite，不带 tone）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Education is very important for children.","operation":"rewrite","sourceSentence":"教育对孩子很重要。"}'

    hr "POST /api/rewrite-sentence  （operation=rewrite + tone=formal）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Education is very important for children.","operation":"rewrite","tone":"formal"}'

    hr "POST /api/rewrite-sentence  （operation=rewrite + tone=persuasive）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Education matters.","operation":"rewrite","tone":"persuasive"}'

    hr "POST /api/rewrite-sentence  （operation=improve）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"He go to school last yesterday.","operation":"improve"}'

    hr "POST /api/rewrite-sentence  （operation=shorten）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Due to the fact that it was raining heavily, we had to cancel the outdoor activity.","operation":"shorten"}'

    hr "POST /api/rewrite-sentence  （operation=expand）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Regular exercise improves health.","operation":"expand"}'
    ;;

  grammar)
    # 响应是 {"text": "..."}，text 就是模型写的原文，界面原样显示，不解析
    hr "POST /api/grammar-check  （两处错误：动词形式 + 副词重复）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/grammar-check" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"He go to school last yesterday."}'

    hr "POST /api/grammar-check  （标点问题）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/grammar-check" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"I really like this book , it is very good"}'

    hr "POST /api/grammar-check  （正确句子，预期 text 是 NONE 之类，不是报错）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/grammar-check" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"She went to school yesterday."}'
    ;;

  instruction)
    hr "POST /api/rewrite-instruction  （Make it more academic.）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' \
      -d '{"text":"The results were kind of surprising.","instruction":"Make it more academic."}'

    hr "POST /api/rewrite-instruction  （Make it sound more natural. + context）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' \
      -d '{"text":"This is a very good idea.","instruction":"Make it sound more natural.","context":"I think this is a very good idea."}'

    hr "POST /api/rewrite-instruction  （中文指令：再短一点）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' \
      -d '{"text":"Due to the fact that it was raining, we cancelled the activity.","instruction":"再短一点"}'

    hr "POST /api/rewrite-instruction  （短语级：替换 very important）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' \
      -d '{"text":"very important","instruction":"Replace with a stronger, more academic expression.","context":"Education is very important for children."}'
    ;;

  errors)
    hr "POST /api/translate  （缺 text → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' -d '{}'

    hr "POST /api/translate  （source 和 target 相同 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"hello","source":"en","target":"en"}'

    hr "POST /api/translate  （source 非法 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"hello","source":"fr","target":"en"}'

    hr "POST /api/translate  （tone 不在枚举内 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/translate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"你好。","tone":"sarcastic"}'

    hr "POST /api/rewrite-selection  （缺 kind → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' -d '{"text":"good"}'

    hr "POST /api/rewrite-selection  （kind 非法 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' -d '{"text":"good","kind":"sentence"}'

    hr "POST /api/rewrite-selection  （context 非字符串 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-selection" \
      -H 'Content-Type: application/json' \
      -d '{"text":"good","kind":"word","context":42}'

    hr "POST /api/rewrite-sentence  （operation 不在枚举内 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Hello there.","operation":"polish"}'

    hr "POST /api/rewrite-sentence  （tone 不在枚举内 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-sentence" \
      -H 'Content-Type: application/json' \
      -d '{"sentence":"Hello there.","operation":"rewrite","tone":"sarcastic"}'

    hr "POST /api/grammar-check  （缺 sentence → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/grammar-check" \
      -H 'Content-Type: application/json' -d '{}'

    hr "POST /api/rewrite-instruction  （缺 instruction → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' -d '{"text":"Some text."}'

    hr "POST /api/rewrite-instruction  （instruction 为空白 → 400）"
    curl -sS -w "\nHTTP %{http_code}\n" -X POST "$BASE/api/rewrite-instruction" \
      -H 'Content-Type: application/json' -d '{"text":"Some text.","instruction":"   "}'
    ;;

  all|*)
    req GET /healthz
    req POST /api/translate '{"text":"教育对孩子很重要。","tone":"formal"}'
    req POST /api/rewrite-selection '{"text":"very important","kind":"phrase","context":"Education is very important for children."}'
    req POST /api/rewrite-sentence '{"sentence":"Education is very important for children.","operation":"rewrite","tone":"formal"}'
    req POST /api/grammar-check '{"sentence":"He go to school last yesterday."}'
    req POST /api/rewrite-instruction '{"text":"The results were kind of surprising.","instruction":"Make it sound more natural."}'
    ;;
esac
