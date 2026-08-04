#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""
CET-4 真题词组增量分组器（schema v2.0）
========================================

只读 / 写的两个文件：
    1. 词表   : cet4_sijizhenti_hexinci.txt
    2. 累计库 : cet4_hexinci_groups_v2.json   ← auto / 单组 / 批处理 都写到这里

每组 3-5 词，不限于近义/同义，可以是同一类词（情感/人体/学科等），
也可以是经常共现的搭配词。

JSON schema v2.0:
    {
      "schema_version": "2.0",
      "total_groups": <int>,
      "total_words_grouped": <int>,
      "vocab_size": <int>,
      "remaining": <int>,
      "groups": [
        {
          "group_id": <int>,
          "title_zh": "...",
          "title_en": "...",
          "words": [...],
          "explanation_zh": "一段 ~200-500 字的中文讲解，覆盖共同点 / 各自侧重 / 搭配 / 使用场景"
        }
      ]
    }

用法：
    # auto 模式（无参）：让 LLM 从未分组的词里挑一组 + 写讲解
    hexinci_group_incremental.py

    # 单组（手动指定，--explanation 跳过 LLM）
    hexinci_group_incremental.py --theme "五官" --words cheek,organ,tumor \\
        --title "face anatomy" --explanation "..."

    # 批处理（JSON 列表）
    hexinci_group_incremental.py --groups-file new_groups.json

    # 只看当前进度
    hexinci_group_incremental.py --status

每次运行结束输出：
    本次提取: N 个单词 / M 组
    累计: 共 X 组 / Y 个单词已分组
    词表: Z 个，还剩 W 个未分组
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

ROOT            = Path(__file__).parent
TXT             = ROOT / "cet4_sijizhenti_hexinci.txt"
JSON_PATH       = ROOT / "cet4_hexinci_groups_v2.json"  # 唯一目标文件：auto / 单组 / 批处理 都写到这

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_vocab() -> set[str]:
    """读词表文件，返回单词集合（去重、保留顺序外的集合）。"""
    return {ln.strip() for ln in TXT.read_text(encoding="utf-8").splitlines() if ln.strip()}

def load_db(path: Path = JSON_PATH) -> dict:
    """读当前 JSON 数据库；不存在则返回空骨架。"""
    if not path.exists():
        return _empty_db()
    try:
        db = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_db()
    if db.get("schema_version") != "2.0":
        # 旧版本（含 differences / examples 的 v1）不兼容，直接丢弃
        return _empty_db()
    return db

def _empty_db() -> dict:
    return {
        "schema_version":     "2.0",
        "created_at":         datetime.now().isoformat(timespec="seconds"),
        "updated_at":         None,
        "total_groups":       0,
        "total_words_grouped":0,
        "vocab_size":         None,  # 首次写入时填
        "remaining":          None,
        "groups":             [],
    }

def already_grouped(db: dict) -> set[str]:
    """已经分过组的词（跨所有 group 并集）。"""
    out: set[str] = set()
    for g in db.get("groups", []):
        out.update(g.get("words", []))
    return out

def save_db(db: dict, path: Path = JSON_PATH) -> None:
    """重算概要字段，写回 JSON。"""
    grouped   = already_grouped(db)
    vocab_sz  = len(load_vocab())
    db["total_groups"]        = len(db["groups"])
    db["total_words_grouped"] = len(grouped)
    db["vocab_size"]          = vocab_sz
    db["remaining"]           = vocab_sz - len(grouped)
    db["updated_at"]          = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# LLM：调用 MiniMax-M3，生成单组 explanation_zh
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """你是 CET-4 词汇教学专家。下面要为一组词写一段连贯的中文讲解：

组类别（中文）：{theme_zh}
组类别（英文，可选）：{title_en}
词：{words_csv}

请产出一段连贯讲解，覆盖：
①共同点（这组词为什么归为一类 — 是近义？同类事物？经常共现？）
②每个词的独特侧重 / 典型搭配 / 语域（口语 / 书面 / 学术）
③常见中文释义、辨析建议
④使用场景示例（短例举）

要求：
- 长度 200-500 字
- 用中文「」或纯文字，不要在中文内容中嵌入 ASCII 双引号，避免破坏 JSON 输出
- 不需要 JSON 格式，直接输出一段（前后不要有 ```json 围栏）
- CET-4 学习者能看懂的难度
"""

def call_llm(client: "anthropic.Anthropic", theme_zh: str, title_en: str, words: list[str]) -> str:
    prompt = PROMPT_TEMPLATE.format(
        theme_zh = theme_zh,
        title_en = title_en or "（未提供）",
        words_csv = ", ".join(words),
    )
    msg = client.messages.create(
        model       = "MiniMax-M3",
        max_tokens  = 1500,
        temperature = 0.5,
        messages    = [{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()
    # 防 LLM 输出 ```json 围栏
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return raw

# ---------------------------------------------------------------------------
# Auto 模式：LLM 自动从未分组的词里挑一组 + 写讲解
# ---------------------------------------------------------------------------

# 控制 LLM 看到的"已分组摘要"数量。太多会爆 context；太少 LLM 容易选重复类别
AUTO_CONTEXT_RECENT_GROUPS = 80

AUTO_PROMPT_TEMPLATE = """你是 CET-4 词汇教学专家。

任务：从下面「未分组的候选词」里挑出 3-5 个相关的词，然后为这一组写一段连贯的中文讲解。

【已分过的组（最近 {recent_count} 组，摘要）— 避免选重复类别或重复词】
{grouped_summary}

【未分组的候选词（共 {remaining_count} 个）】
{remaining_csv}

严格要求：
1. **必须从「未分组的候选词」里挑 3-5 个，不要选已分过的组里出现过的词**
2. 选出的每一个词都必须在上面「未分组的候选词」列表中字符串完全匹配，**不要造词、不要选列表外的词、不要加复数不要变形**（如列表里是 "profitable" 就不准选 "profit"）
3. 选的 3-5 个词之间要有清晰关联，可以是：
   - 同近义词
   - 同一类事物（情感、人体、学科等）
   - 经常一起出现 / 搭配使用的词
4. 写一段 200-500 字的中文讲解，覆盖共同点 / 各自侧重 / 搭配 / 使用场景
5. 用中文「」或纯文字，**不要在中文里嵌入 ASCII 双引号 "，避免破坏 JSON 输出**
6. 直接输出 JSON（不要 ```json``` 围栏，不要其它任何文字）

输出 schema（严格匹配）：
{{
  "theme_zh": "组主题中文，如「情感/喜乐类」",
  "title_en": "组主题英文，可选；不确定就传空串",
  "words": ["word1", "word2", ...],
  "explanation_zh": "200-500 字的中文讲解"
}}
"""


def _grouped_summary(db: dict, max_recent: int = AUTO_CONTEXT_RECENT_GROUPS) -> str:
    """给 LLM 看的已分组组摘要：只含 title_zh + words 字段；只传最近 max_recent 组。"""
    recent = db["groups"][-max_recent:]
    summary = [
        {"title_zh": g.get("title_zh", ""), "words": g.get("words", [])}
        for g in recent
    ]
    return json.dumps(summary, ensure_ascii=False, indent=2)


def auto_pick_group(
    client: "anthropic.Anthropic",
    db: dict,
) -> dict:
    """让 LLM 自动从未分组的词里挑一组 + 写讲解；返回 {theme_zh, title_en, words, explanation_zh}。"""
    vocab     = load_vocab()
    grouped   = already_grouped(db)
    remaining = sorted(vocab - grouped)

    if len(remaining) < 3:
        raise SystemExit(f"❌ 剩余未分组词不足 3 个（剩 {len(remaining)}），无法自动凑组")

    prompt = AUTO_PROMPT_TEMPLATE.format(
        recent_count    = min(len(db["groups"]), AUTO_CONTEXT_RECENT_GROUPS),
        grouped_summary = _grouped_summary(db),
        remaining_count = len(remaining),
        remaining_csv   = ", ".join(remaining),
    )

    msg = client.messages.create(
        model       = "MiniMax-M3",
        max_tokens  = 2000,
        temperature = 0.6,
        messages    = [{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n❌ auto JSON parse failed at line {e.lineno} col {e.colno}", file=sys.stderr)
        print(f"raw (前 1500 字):\n{raw[:1500]}", file=sys.stderr)
        raise

    return obj

# ---------------------------------------------------------------------------
# 核心：提交一组
# ---------------------------------------------------------------------------

def _validate_words(
    words: list[str], vocab: set[str], grouped: set[str]
) -> list[str]:
    """校验词：必须都在词表里、且尚未被分组。多余空格剥掉。"""
    cleaned = [w.strip() for w in words if w.strip()]
    # 去重但保留顺序
    seen: set[str] = set()
    uniq: list[str] = []
    for w in cleaned:
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)

    missing  = [w for w in uniq if w not in vocab]
    conflict = [w for w in uniq if w in grouped]
    if missing:
        raise SystemExit(f"❌ 以下词不在词表里: {missing}")
    if conflict:
        raise SystemExit(f"❌ 以下词已分过组: {conflict}")
    if not 3 <= len(uniq) <= 5:
        raise SystemExit(f"❌ 每组必须 3-5 个词，现在是 {len(uniq)} 个: {uniq}")
    return uniq

def commit_one(
    db: dict,
    theme_zh: str,
    words: list[str],
    title_en: str = "",
    explanation: str | None = None,
    client: "anthropic.Anthropic | None" = None,
) -> int:
    """追加一组到 db，返回新组 id。"""
    vocab   = load_vocab()
    grouped = already_grouped(db)
    words   = _validate_words(words, vocab, grouped)

    if not explanation:
        if client is None:
            raise SystemExit("❌ 需要 explanation 但未提供 LLM client，且未传 explanation")
        explanation = call_llm(client, theme_zh, title_en, words)

    new_id = (max((g["group_id"] for g in db["groups"]), default=0)) + 1
    db["groups"].append({
        "group_id":       new_id,
        "title_zh":       theme_zh,
        "title_en":       title_en,
        "words":          words,
        "explanation_zh": explanation,
    })
    return new_id

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CET-4 增量分组")

    p.add_argument("--words", default="", help="逗号分隔的单词列表，如 cheek,chin,jaw")

    p.add_argument("--theme", default="", help="组主题中文，用于 title_zh 和讲解 prompt")
    p.add_argument("--title", default="", help="组主题英文，可选")

    p.add_argument(
        "--explanation",
        default=None,
        help="直接提供讲解文字，跳过 LLM 调用",
    )

    p.add_argument(
        "--groups-file",
        default=None,
        help="JSON 文件，包含 [{theme, title, words, explanation?}, ...] 列表，批量提交",
    )

    p.add_argument(
        "--status",
        action="store_true",
        help="只打印当前概要，不修改 JSON",
    )

    p.add_argument(
        "--auto",
        action="store_true",
        help="自动模式：让 LLM 从「未分组的词」里挑一组并写讲解。无参时也默认走 auto。",
    )

    return p.parse_args()


def _run_auto() -> int:
    """Auto 模式：让 LLM 从未分组的词里选一组 + 写讲解，输出到 JSON_PATH。

    - 「未分组的候选词」= 词表 - 已分组词
    - 给 LLM 的「已分组摘要」= JSON_PATH 里所有已存在的组
    """
    db = load_db()  # JSON_PATH

    # 算"未分组的候选词"
    vocab     = load_vocab()
    grouped   = already_grouped(db)
    remaining = sorted(vocab - grouped)
    if len(remaining) < 3:
        raise SystemExit(f"❌ 剩余未分组词不足 3 个（剩 {len(remaining)}），auto 模式结束")

    client    = anthropic.Anthropic()
    max_retry = 3
    last_err: str | None = None

    for attempt in range(1, max_retry + 1):
        prompt = _build_auto_prompt(db, remaining)
        msg = client.messages.create(
            model       = "MiniMax-M3",
            max_tokens  = 2000,
            temperature = 0.6,
            messages    = [{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse failed at line {e.lineno} col {e.colno}: {raw[:200]}"
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        theme_zh    = obj.get("theme_zh", "").strip()
        title_en    = (obj.get("title_en") or "").strip()
        words       = [w.strip() for w in obj.get("words", []) if w.strip()]
        explanation = (obj.get("explanation_zh") or "").strip()

        if not theme_zh or not words or not explanation:
            last_err = f"LLM 返回字段不全: theme_zh={theme_zh!r} words={words!r}"
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        # 额外做一次硬校验：选出的词必须全部在 remaining 里
        invalid = [w for w in words if w not in remaining]
        if invalid:
            last_err = f"LLM 选出了不在剩余候选中的词: {invalid}"
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        # 提交到累计库
        try:
            commit_one(db, theme_zh, words, title_en, explanation, None)
        except SystemExit as e:
            last_err = str(e)
            print(f"  ⚠️  attempt {attempt}/{max_retry}: {last_err}", file=sys.stderr)
            continue

        save_db(db)
        print_summary(db, 1, len(words))
        return 0

    raise SystemExit(f"❌ auto 模式重试 {max_retry} 次都失败: {last_err}")


def _build_auto_prompt(snapshot_db: dict, remaining: list[str]) -> str:
    """构造 auto 模式的 prompt：合并后的已分组摘要 + 过滤后的剩余词表。"""
    return AUTO_PROMPT_TEMPLATE.format(
        recent_count    = min(len(snapshot_db["groups"]), AUTO_CONTEXT_RECENT_GROUPS),
        grouped_summary = _grouped_summary(snapshot_db, AUTO_CONTEXT_RECENT_GROUPS),
        remaining_count = len(remaining),
        remaining_csv   = ", ".join(remaining),
    )


def _run_manual(db: dict, args: argparse.Namespace) -> int:
    """单组 / 批处理模式：写到累计库 JSON_PATH（即 v2.json）。"""
    # 准备 LLM client（如果需要）
    client: "anthropic.Anthropic | None" = None
    if args.groups_file:
        items = load_groups_file(args.groups_file)
        needs_llm = any(not item.get("explanation") for item in items)
    else:
        needs_llm = not args.explanation

    if needs_llm:
        client = anthropic.Anthropic()

    if args.groups_file:
        items = load_groups_file(args.groups_file)
        start_groups = len(db["groups"])
        start_words  = len(already_grouped(db))

        for i, item in enumerate(items, 1):
            words        = item.get("words", [])
            theme_zh     = item.get("theme") or item.get("title_zh") or ""
            title_en     = item.get("title") or item.get("title_en") or ""
            explanation  = item.get("explanation")
            if not theme_zh:
                print(f"  [{i}/{len(items)}] SKIP: 缺 theme", file=sys.stderr)
                continue
            try:
                commit_one(db, theme_zh, words, title_en, explanation, client)
                print(f"  [{i}/{len(items)}] OK: {theme_zh} → {words}", file=sys.stderr)
            except SystemExit as e:
                print(f"  [{i}/{len(items)}] SKIP: {theme_zh} → {e}", file=sys.stderr)

        save_db(db, JSON_PATH)
        batch_added = len(db["groups"]) - start_groups
        batch_words = len(already_grouped(db)) - start_words
        print_summary(db, batch_added, batch_words)
        return 0

    # 单组
    if not args.theme:
        raise SystemExit("❌ 需要 --theme")
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    commit_one(db, args.theme, words, args.title, args.explanation, client if not args.explanation else None)

    save_db(db, JSON_PATH)
    print_summary(db, 1, len(words))
    return 0


def parse_args() -> argparse.Namespace:
    """CLI 解析入口。"""
    p = argparse.ArgumentParser(description="CET-4 增量分组")

    p.add_argument("--words", default="", help="逗号分隔的单词列表，如 cheek,chin,jaw")
    p.add_argument("--theme", default="", help="组主题中文，用于 title_zh 和讲解 prompt")
    p.add_argument("--title", default="", help="组主题英文，可选")
    p.add_argument("--explanation", default=None, help="直接提供讲解文字，跳过 LLM 调用")
    p.add_argument("--groups-file", default=None, help="JSON 文件，含 [{theme, title, words, explanation?}, ...]")
    p.add_argument("--status", action="store_true", help="只打印当前概要，不修改 JSON")
    p.add_argument("--auto",    action="store_true", help="自动模式：从剩余词里挑一组并写讲解")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.status:
        db = load_db(JSON_PATH)
        print_summary(db, 0, 0)
        return 0

    if args.auto:
        return _run_auto()

    # 没传任何处理参数 → 默认走 auto
    if not any([args.words, args.groups_file, args.auto]):
        return _run_auto()

    # 单组 / 批处理 → 主库
    db = load_db(JSON_PATH)
    return _run_manual(db, args)


def print_summary(db: dict, batch_added: int, batch_words: int) -> None:
    grouped = already_grouped(db)
    vocab_sz = len(load_vocab())
    print("\n────────── 概要 ──────────")
    print(f"本次提取: {batch_words} 个单词 / {batch_added} 组")
    print(f"累计:     共 {len(db['groups'])} 组 / {len(grouped)} 个单词已分组")
    print(f"词表:     {vocab_sz} 个，还剩 {vocab_sz - len(grouped)} 个未分组")

# ---------------------------------------------------------------------------
# 批处理 JSON 格式
# ---------------------------------------------------------------------------

def load_groups_file(path: str) -> list[dict]:
    """加载 --groups-file 格式：[{theme, title?, words, explanation?}, ...]"""
    return json.loads(Path(path).read_text(encoding="utf-8"))

if __name__ == "__main__":
    raise SystemExit(main())
