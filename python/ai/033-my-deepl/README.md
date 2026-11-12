## 基本命令

设置不同模型，并打开日志

    ANTHROPIC_MODEL=MiniMax-M3.1-Flash-Preview python interactive_translate.py --debug

调试启动 server

    DEBUG=1 ANTHROPIC_API_KEY=sk-ant-... python -m uvicorn server:app --port 8000

手工测试 API

    # 翻译
    curl -X POST http://127.0.0.1:8000/api/translate \
        -H 'Content-Type: application/json' \
        -d '{"text":"埃隆·马斯克不在领英上。"}'

    # 重写
    curl -X POST http://127.0.0.1:8000/api/rephrase \
        -H 'Content-Type: application/json' \
        -d '{
            "source":"埃隆·马斯克不在领英上。",
            "current":"Elon Musk is not on LinkedIn.",
            "prefix":"Elon Musk",
            "target_word":"is"
        }'

## 翻译提示词

### 翻译提示词
You are a translator. Translate the given Chinese sentence into a single, natural English sentence. 
Preserve the source's meaning faithfully. 

Translate: 您的使用反馈,将影响我们下一步的优化方向

### 重写提示词

You are helping the user refine a translation by replacing the suffix starting at a specific position.

Given the Chinese source, the current English sentence, the fixed prefix, and the original target word, propose 4-8 alternative suffixes that rewrite the segment starting at the target position.

Rules:
- The PREFIX before the target position is FIXED. Do NOT include it in the candidates 
— only return the suffix that replaces the target word onward.
- Each candidate's FIRST word MUST be a DIFFERENT word from the original target word. The user explicitly wants the current word to be replaced (no ties, no same-word rephrasing).
- Mix two kinds of candidates so the user has a real choice:
  a) 'Word-only' candidates: keep every word AFTER the target word IDENTICAL to the current English sentence, and swap only the target word itself. Joined with the prefix, only that one word changes.
  b) 'Full-rewrite' candidates: freely rephrase the target word AND the words after it, producing noticeably different phrasing.

Aim for a roughly even split (e.g. half word-only, half full-rewrite).   

Do NOT make every candidate a full-rewrite.
- Use the Chinese source to keep the meaning consistent with the original.

content: 
Source (Chinese): 您的使用反馈,将影响我们下一步的优化方向
Current English sentence: Your feedback from using it will shape the direction of our next round of improvements.
Prefix to KEEP unchanged: 'Your feedback from using it will'
Original target word (each candidate's FIRST word must NOT equal this): 'shape'

Generate 4-8 alternative suffixes starting at the target position. Each must begin with a word DIFFERENT from the original target word. Produce a MIX of two styles:
(a) 'word-only' candidates that keep every word after the target IDENTICAL to the current English sentence and only swap thetarget word itself,
(b) 'full-rewrite' candidates that rephrase the target word AND the words after it.\nAim for a roughly even split (e.g. half word-only, half full-rewrite). 

Do NOT make every candidate a full-rewrite.

## 其它

腾讯翻译模型

    https://ollama.com/s2021008840/hy-mt2

    ollama run s2021008840/hy-mt2:1.8b

    curl http://localhost:11434/api/chat \
      -d '{
        "model": "s2021008840/hy-mt2:1.8b",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'

    node ollama-test.mjs
    python ollama-test.py

### ollama 重写

`ollama_rephrase.py` 是走本机 ollama 模型(`s2021008840/hy-mt2:1.8b`)的离线版
重写脚本——不需要 `ANTHROPIC_API_KEY`。设计原因:小模型跨长 prompt 维护约束
(「这个前缀必须保持原样」、「第一个词不能等于 X」)能力不足,所以本脚本让模型
只生成 target 之后的部分,prefix 由代码侧 prepend;输出仍为完整英文句子。

依赖:本机 ollama 进程在跑,目标模型已 pull;`ollama` Python 客户端已装。
注意 `~/.pyenv/versions/qlib/bin/python` 没装 `ollama`,要用:

    ~/.pyenv/versions/3.11.9/bin/python3.11 ollama_rephrase.py --help

调用样例(参数跟 translator / interactive_translate 保持一致):

    ~/.pyenv/versions/3.11.9/bin/python3.11 ollama_rephrase.py \
        --source '您的使用反馈,将影响我们下一步的优化方向' \
        --current 'Your feedback from using it will shape the direction of our next round of improvements.' \
        --prefix 'Your feedback from using it will' \
        --target 'shape'

    1. Your feedback from using it will influence the direction of our next round of improvements.
    2. Your feedback from using it will guide the direction of our next round of improvements.
    3. Your feedback from using it will determine the direction of our next round of improvements.
    ...

可用选项: `-n 6` (候选数,默认 6)、`--json` (输出 JSON 数组便于二次处理)、
`--model` (换其它 ollama 模型)、`--ollama-url` (远程 ollama)、`--debug`
(把 prompt + 模型原始回复打到 stderr)。

退出码:成功 0;缺参 argparse 错 2;ollama 不可达或解析出零候选 1。

### 交互式 REPL(ollama 后端)

`interactive_ollama_translate.py` 是 `interactive_translate.py`(走 Claude)
的本地对应版,流程一致,只是后端换成 ollama —— 不需要 `ANTHROPIC_API_KEY`。

    ~/.pyenv/versions/3.11.9/bin/python3.11 interactive_ollama_translate.py

REPL 跟 Claude 版完全对应:输入中文 → 翻译 → 按词号改写 → 选候选 → 更新句子。
命令:`/new` 重开、`/exit` 退出、`h` 帮助。`--debug` 把每次调 ollama 的
prompt + 原始回复打到 stderr 并关掉 spinner;`--model` 换模型;
`--ollama-url` 远程 ollama。退出后端不可达 / 模型没 pull 时启动直接 stderr 报错。

依赖与 ollama_rephrase.py 同(本机 ollama 进程 + 已 pull 的模型 + ollama python
客户端,默认 `~/.pyenv/versions/3.11.9/bin/python3.11` 装的是 0.6.1)。

