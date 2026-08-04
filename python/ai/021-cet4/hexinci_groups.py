#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""
CET-4 真题近义词组生成器
=======================

- 默认生成 5 组示例，输出到 cet4_hexinci_groups.json
- LLM 调用模式参考 minimax-test.py：模型 MiniMax-M3，过滤 thinking block
- 未来扩展点：把 DEMO_GROUPS 替换成「从 CET4luan_1.json.syno 提取的全部组」
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

ROOT      = Path(__file__).parent
OUT_JSON  = ROOT / "cet4_hexinci_groups.json"

# ---------------------------------------------------------------------------
# 5 个示例组（词全部来自 cet4_sijizhenti_hexinci.txt，不造词）
# ---------------------------------------------------------------------------

DEMO_GROUPS: list[dict] = [
    {
        "core_zh": "获得 / 取得",
        "core_en": "to get / obtain",
        # gain 不在词表里，全部用词表里有的；本组 4 词
        "words":   ["obtain", "acquire", "achieve", "secure"],
    },
    {
        "core_zh": "至关重要",
        "core_en": "extremely important",
        # indispensable 不在词表里，替换为 significant
        "words":   ["crucial", "vital", "essential", "critical", "significant"],
    },
    {
        "core_zh": "提升 / 增加",
        "core_en": "to raise / improve",
        # increase / improve 不在词表里，替换为 raise / advance
        "words":   ["enhance", "boost", "promote", "raise", "advance"],
    },
    {
        "core_zh": "阻止 / 禁止",
        "core_en": "to stop / forbid",
        # 原计划是"消除/去除"组，但 erase / delete / dismiss 均不在词表里。
        # 改为「阻止 / 禁止」语义（近义方向接近），5 词全部在词表
        "words":   ["ban", "prohibit", "block", "prevent", "hinder"],
    },
    {
        "core_zh": "冲突 / 分歧",
        "core_en": "disagreement / clash",
        # controversy / opposition 不在词表里，替换为 tension / division
        "words":   ["conflict", "dispute", "tension", "contest", "division"],
    },
]

# ---------------------------------------------------------------------------
# Prompt：明确要求 explanation_zh 覆盖共同点 / 不同点 / 用法 / 使用场景
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """你是 CET-4 词汇教学专家。下面是一组近义词：

核心义：{core_zh} ({core_en})
词：{words_csv}

请产出 JSON（不要任何额外文本、不要 ```json 围栏），严格匹配这个 schema：
{{
  "group_id": <int>,
  "core_meaning_zh": "...",
  "core_meaning_en": "...",
  "words": [...],
  "explanation_zh": "一段连贯的中文讲解文案，覆盖：①共同点（这组词共享的核心语义）②不同点（逐个词点出独有侧重）③用法（搭配、语域、典型搭配词）④使用场景（口语 / 书面 / 公文 / 学术 / 日常）。整体 150-260 字，CET-4 学习者读完能区分。",
  "differences": [
    {{"word":"...","nuance_zh":"1-2 句中文，讲该词独有的语感/搭配/语域","register":"正式|通用|口语"}}
  ],
  "examples": [
    {{"word":"...","en":"一个真实或自然的英文例句","zh":"中文翻译"}}
  ]
}}

硬性要求：
1. 每个 word 必须出现在 words 数组中
2. differences 长度 == words 长度
3. examples 长度 == words 长度；英文例句自然、可读，真实语境（非生造）
4. explanation_zh 必填，且必须涵盖共同点/不同点/用法/场景四个面向
5. 整体保持 CET-4 学习者能看懂的难度，不要用高深术语
6. **关键**：任何中文字符串字段内部禁止使用 ASCII 双引号 " ！所有引号都用中文「」或者干脆不写，必须保证输出是合法 JSON
"""


def call_llm(client: "anthropic.Anthropic", group: dict, group_id: int) -> dict:
    """单组近义词调一次 LLM，返回符合 schema 的 dict。"""
    prompt = PROMPT_TEMPLATE.format(
        core_zh   = group["core_zh"],
        core_en   = group["core_en"],
        words_csv = ", ".join(group["words"]),
    )

    msg = client.messages.create(
        model       = "MiniMax-M3",
        max_tokens  = 2000,
        temperature = 0.4,
        messages    = [{"role": "user", "content": prompt}],
    )

    # 过滤掉 thinking block，只取 text（参考 minimax-test.py）
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()

    # LLM 有时在 JSON 外面套 ```json``` 围栏，要剥掉
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 中文里偶尔会嵌入裸 ASCII 双引号破坏 JSON，做一次轻量容错后重试
        try:
            obj = json.loads(_strip_inner_ascii_quotes(raw))
        except json.JSONDecodeError as e:
            print(f"\n❌ group {group_id} JSON parse failed at line {e.lineno} col {e.colno}")
            print(f"raw (前 1500 字):\n{raw[:1500]}")
            raise

    obj["group_id"] = group_id
    return obj


def _strip_inner_ascii_quotes(s: str) -> str:
    """把夹在中文字符/字母数字之间的裸 ASCII 双引号换成「」，以挽救 JSON。"""
    out: list[str] = []
    n   = len(s)
    for i, ch in enumerate(s):
        if ch != '"':
            out.append(ch)
            continue
        prev_zh = i > 0 and "一" <= s[i - 1] <= "鿿"
        next_zh = i + 1 < n and "一" <= s[i + 1] <= "鿿"
        prev_al = i > 0 and (s[i - 1].isalnum() or s[i - 1] in "%+")
        next_al = i + 1 < n and (s[i + 1].isalnum() or s[i + 1] in "%+")
        # 紧贴汉字 或 紧贴字母数字 的双引号 → 视为非法嵌入，换成「」
        if (prev_zh and (next_zh or next_al)) or (prev_al and (next_zh or next_al)):
            out.append("「" if i % 2 == 0 else "」")
        else:
            out.append(ch)
    return "".join(out)


def main() -> None:
    client     = anthropic.Anthropic()
    groups_out = [call_llm(client, g, i + 1) for i, g in enumerate(DEMO_GROUPS)]

    payload = {
        "schema_version": "1.0",
        "total_groups":   len(groups_out),
        "groups":         groups_out,
    }

    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding = "utf-8",
    )

    print(f"✅ wrote {len(groups_out)} groups → {OUT_JSON}")


if __name__ == "__main__":
    main()
