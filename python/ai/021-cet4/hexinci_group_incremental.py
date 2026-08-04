#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""CET-4 真题词组增量分组器（auto 模式）。

未分组词 = 词表 - 已分组词；LLM 每次从中挑 3-5 个相关的词 + 写一段中文讲解，追加到 cet4_hexinci_groups_v2.json。

缓存策略：
- system prompt = 全部词表（按字母序固定；词表不变 → byte-identical → 命中率高）
- user   prompt = 当前已分组词列表（每次变）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

ROOT      = Path(__file__).parent
TXT       = ROOT / "cet4_sijizhenti_hexinci.txt"
JSON_PATH = ROOT / "cet4_hexinci_groups_v2.json"


def load_vocab() -> set[str]:
    return {ln.strip() for ln in TXT.read_text(encoding="utf-8").splitlines() if ln.strip()}


def load_db(path: Path = JSON_PATH) -> dict:
    if not path.exists():
        return _empty_db()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_db()


def _empty_db() -> dict:
    return {
        "created_at":         datetime.now().isoformat(timespec="seconds"),
        "updated_at":         None,
        "total_groups":       0,
        "total_words_grouped": 0,
        "vocab_size":         None,
        "remaining":          None,
        "groups":             [],
    }


def already_grouped(db: dict) -> set[str]:
    out: set[str] = set()
    for g in db.get("groups", []):
        out.update(g.get("words", []))
    return out


def save_db(db: dict, path: Path = JSON_PATH) -> None:
    grouped  = already_grouped(db)
    vocab_sz = len(load_vocab())
    db["total_groups"]        = len(db["groups"])
    db["total_words_grouped"] = len(grouped)
    db["vocab_size"]          = vocab_sz
    db["remaining"]           = vocab_sz - len(grouped)
    db["updated_at"]          = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_system_prompt(vocab_sorted: list[str]) -> str:
    return f"""你是 CET-4 词汇教学专家。

【全部词表（共 {len(vocab_sorted)} 个，按字母序）】
{", ".join(vocab_sorted)}

每次调用你会收到一条用户消息，包含当前「已分过组的词」列表。请从未分组的词（即【全部词表】 - 用户消息中的【已分过组的词】）里挑 3-5 个相关的词，并为这一组写一段连贯的中文讲解。

严格要求：
1. 选出的每一个词都必须出现在【全部词表】中，且不在用户消息的【已分过组的词】列表里
2. 不要造词、不要选列表外的词、不要加复数不要变形（如列表里是 "profitable" 就不准选 "profit"）
3. 选的 3-5 个词之间要有清晰关联（近义 / 同类事物 / 经常共现）
4. 写一段 200-500 字的中文讲解，覆盖共同点 / 各自侧重 / 搭配 / 使用场景
5. 用中文「」或纯文字，**不要在中文里嵌入 ASCII 双引号 "**，避免破坏 JSON 输出
6. 直接输出 JSON（不要 ```json``` 围栏，不要其它任何文字）

输出 schema（严格匹配）：
{{
  "theme_zh": "组主题中文，如「情感/喜乐类」",
  "title_en": "组主题英文，可选；不确定就传空串",
  "words": ["word1", "word2", ...],
  "explanation_zh": "200-500 字的中文讲解"
}}
"""


USER_PROMPT_TEMPLATE = """【当前已分过组的词（共 {grouped_count} 个，请勿选）】
{grouped_csv}

请按系统提示中的 schema，从【未分组的词】（=【全部词表】 -【已分过组的词】）中挑 3-5 个相关的词，并写一段中文讲解，直接输出 JSON。"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CET-4 增量分组（auto）")
    p.add_argument("--debug", action="store_true", help="打印 prompt / 响应 / 校验详情到 stderr")
    return p.parse_args()


def _run_auto() -> int:
    debug = parse_args().debug

    db          = load_db()
    vocab_set   = load_vocab()
    vocab_list  = sorted(vocab_set)
    grouped     = already_grouped(db)
    remaining   = vocab_set - grouped

    if len(remaining) < 3:
        raise SystemExit(f"❌ 剩余未分组词不足 3 个（剩 {len(remaining)}），auto 模式结束")

    system_prompt = _build_system_prompt(vocab_list)
    user_prompt   = USER_PROMPT_TEMPLATE.format(
        grouped_count = len(grouped),
        grouped_csv   = ", ".join(sorted(grouped)),
    )

    if debug:
        print(f"[debug] vocab={len(vocab_list)}  grouped={len(grouped)}  remaining={len(remaining)}", file=sys.stderr)
        print(f"[debug] system_chars={len(system_prompt)}  user_chars={len(user_prompt)}", file=sys.stderr)

    client    = anthropic.Anthropic()
    max_retry = 3
    last_err: str | None = None

    for attempt in range(1, max_retry + 1):
        if debug:
            print(f"\n[debug] ── attempt {attempt}/{max_retry} ──", file=sys.stderr)

        msg = client.messages.create(
            model       = "MiniMax-M3",
            system      = system_prompt,
            max_tokens  = 2000,
            temperature = 0.6,
            messages    = [{"role": "user", "content": user_prompt}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        if debug:
            print(f"[debug] LLM raw_chars={len(raw)}  stop_reason={msg.stop_reason}", file=sys.stderr)
            head, tail = raw[:300], raw[-300:]
            print(f"[debug] raw[:300]={head!r}", file=sys.stderr)
            if len(raw) > 600:
                print(f"[debug] raw[-300:]={tail!r}", file=sys.stderr)

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse failed at line {e.lineno} col {e.colno}: {raw[:200]}"
            if debug:
                print(f"[debug] JSONDecodeError: {e}", file=sys.stderr)
                print(f"[debug] raw (full):\n{raw}", file=sys.stderr)
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        theme_zh    = obj.get("theme_zh", "").strip()
        title_en    = (obj.get("title_en") or "").strip()
        words       = [w.strip() for w in obj.get("words", []) if w.strip()]
        explanation = (obj.get("explanation_zh") or "").strip()

        if debug:
            print(f"[debug] parsed: theme_zh={theme_zh!r}", file=sys.stderr)
            print(f"[debug] parsed: title_en={title_en!r}", file=sys.stderr)
            print(f"[debug] parsed: words={words}", file=sys.stderr)
            print(f"[debug] parsed: explanation_chars={len(explanation)}", file=sys.stderr)

        missing_fields = []
        if not theme_zh:    missing_fields.append("theme_zh")
        if not words:       missing_fields.append("words")
        if not explanation: missing_fields.append("explanation_zh")
        if missing_fields:
            last_err = f"LLM 返回字段不全: 缺失 {missing_fields}"
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        not_in_vocab = [w for w in words if w not in vocab_set]
        already_used = [w for w in words if w in grouped]
        duplicates   = sorted({w for w in words if words.count(w) > 1})
        if not_in_vocab or already_used or duplicates:
            last_err = f"LLM 选词不合法: not_in_vocab={not_in_vocab} already_grouped={already_used} duplicates={duplicates}"
            if debug:
                print(f"[debug] full_words={words}", file=sys.stderr)
                print(f"[debug] not_in_vocab={not_in_vocab}", file=sys.stderr)
                print(f"[debug] already_grouped={already_used}", file=sys.stderr)
                print(f"[debug] duplicates={duplicates}", file=sys.stderr)
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        if debug:
            print(f"[debug] ✓ valid group candidate", file=sys.stderr)

        new_id = (max((g["group_id"] for g in db["groups"]), default=0)) + 1
        db["groups"].append({
            "group_id":       new_id,
            "title_zh":       theme_zh,
            "title_en":       title_en,
            "words":          words,
            "explanation_zh": explanation,
        })
        save_db(db)
        print_summary(db, 1, len(words))
        return 0

    raise SystemExit(f"❌ auto 模式重试 {max_retry} 次都失败: {last_err}")


def print_summary(db: dict, batch_added: int, batch_words: int) -> None:
    grouped  = already_grouped(db)
    vocab_sz = len(load_vocab())
    print("\n────────── 概要 ──────────")
    print(f"本次提取: {batch_words} 个单词 / {batch_added} 组")
    print(f"累计:     共 {len(db['groups'])} 组 / {len(grouped)} 个单词已分组")
    print(f"词表:     {vocab_sz} 个，还剩 {vocab_sz - len(grouped)} 个未分组")


if __name__ == "__main__":
    raise SystemExit(_run_auto())