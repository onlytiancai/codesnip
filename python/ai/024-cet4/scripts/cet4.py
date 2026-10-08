#!/usr/bin/env python3
"""CET4_2.json 单条记录查询 — 随机或按 wordHead。

CET4_2.json 是 JSONL 格式（每行一个 JSON 对象），脚本首次运行会构建一份
``wordHead -> 字节偏移`` 的索引缓存到 ``CET4_2.idx``，后续按 mtime/size 复用。
查找时 ``seek + readline`` 即可单行解析。
"""

from __future__ import annotations

import argparse
import json
import mmap
import random
import sys
from pathlib import Path

# scripts/cet4.py -> 仓库根
ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "vocab_all.json"  # 由 scripts/build-vocab.py 合并生成
INDEX_PATH = JSON_PATH.with_suffix(".idx")
INDEX_VERSION = 1  # 索引格式版本，未来字段调整时一并失效


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

def build_index(json_path: Path) -> dict:
    """遍历 JSONL，建 wordHead -> 字节偏移的索引。"""
    heads: list[str] = []
    offsets: dict[str, int] = {}
    with json_path.open("rb") as f:
        # mmap 一次性映射，遍历 newline 切分；省去手动 readinto 循环
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            line_start = 0
            while True:
                nl = mm.find(b"\n", line_start)
                if nl == -1:
                    chunk = mm[line_start:]
                    if chunk.strip():
                        _record_line(chunk, line_start, heads, offsets)
                    break
                chunk = mm[line_start:nl]
                if chunk.strip():
                    _record_line(chunk, line_start, heads, offsets)
                line_start = nl + 1
    return {
        "version": INDEX_VERSION,
        "mtime": json_path.stat().st_mtime,
        "size": json_path.stat().st_size,
        "heads": heads,
        "offsets": offsets,
    }


def _record_line(raw: bytes, offset: int, heads: list[str], offsets: dict[str, int]) -> None:
    row = json.loads(raw)
    head = row.get("content", {}).get("word", {}).get("wordHead")
    if not head:
        return
    if head in offsets:
        return  # JSONL 重复键时只保留首个，与 JS 版 BY_HEAD.set 行为一致
    offsets[head] = offset
    heads.append(head)


def load_index(json_path: Path, index_path: Path, verbose: bool = False) -> dict:
    """按 mtime/size 校验：命中则加载缓存，否则重建。"""
    stat = json_path.stat()
    if index_path.exists():
        try:
            cached = json.loads(index_path.read_text(encoding="utf-8"))
            if (
                cached.get("version") == INDEX_VERSION
                and cached.get("mtime") == stat.st_mtime
                and cached.get("size") == stat.st_size
            ):
                if verbose:
                    print(f"[index] reused: {len(cached['heads'])} entries", file=sys.stderr)
                return cached
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # 损坏就重建
    if verbose:
        print("[index] building…", file=sys.stderr)
    index = build_index(json_path)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[index] built: {len(index['heads'])} entries -> {index_path}", file=sys.stderr)
    return index


# ---------------------------------------------------------------------------
# 读取单行
# ---------------------------------------------------------------------------

def fetch_row(json_path: Path, offset: int) -> dict:
    """按偏移读出对应行并解析为顶层 JSON 对象。"""
    with json_path.open("rb") as f:
        f.seek(offset)
        raw = f.readline()
    if not raw.endswith(b"\n"):
        # 末行无换行：readline 仍能返回；这里保留尾部数据
        pass
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 详情格式化
# ---------------------------------------------------------------------------

def _safe(obj, *path, default=""):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def format_detail(row: dict) -> str:
    word_obj = _safe(row, "content", "word", default={})
    c = word_obj.get("content") or {}
    head = word_obj.get("wordHead", "?")
    usphone = c.get("usphone", "")
    ukphone = c.get("ukphone", "")
    phone = c.get("phone", "")
    word_id = word_obj.get("wordId", "")

    lines: list[str] = []
    # 标题行：词 + 音标
    title = head
    if ukphone or usphone or phone:
        phones = [f"/{p}/" for p in (ukphone, usphone, phone) if p]
        title = f"{head}  {'  '.join(phones)}"
    lines.append(title)
    if word_id:
        lines.append(f"# {word_id}")
    lines.append("─" * max(40, len(title)))

    # 释义（中英双语：中文释义 + 英文释义各占一行）
    trans = c.get("trans") or []
    if trans:
        lines.append("【释义】")
        for t in trans:
            pos = t.get("pos") or ""
            cn = t.get("tranCn") or ""
            other = t.get("tranOther") or ""
            head_txt = pos or ""
            if not (head_txt or cn or other):
                continue
            cn_line = f"  {head_txt} {cn}".rstrip() if cn else f"  {head_txt}".rstrip()
            lines.append(cn_line)
            if other:
                # 英文释义作为下一行，缩进对齐到中文释义的开头
                lines.append(f"    英 {other}")

    # 例句
    sentences = _safe(c, "sentence", "sentences", default=[]) or []
    if sentences:
        lines.append("【例句】")
        for i, s in enumerate(sentences, 1):
            en = s.get("sContent", "")
            cn = s.get("sCn", "")
            lines.append(f"  {i}. {en}")
            if cn:
                lines.append(f"     {cn}")

    # 真题例句
    exam = _safe(c, "realExamSentence", "sentences", default=[]) or []
    if exam:
        lines.append("【真题】")
        for s in exam:
            en = s.get("sContent", "")
            info = _safe(s, "sourceInfo", default={}) or {}
            tag = " · ".join(
                str(v) for v in (info.get("year"), info.get("paper"), info.get("type")) if v
            )
            if tag:
                lines.append(f"  [{tag}] {en}")
            else:
                lines.append(f"  {en}")

    # 短语
    phrases = _safe(c, "phrase", "phrases", default=[]) or []
    if phrases:
        lines.append("【短语】")
        for p in phrases:
            en = p.get("pContent", "")
            cn = p.get("pCn", "")
            lines.append(f"  {en} — {cn}")

    # 同义词
    synos = _safe(c, "syno", "synos", default=[]) or []
    if synos:
        lines.append("【同义词】")
        for s in synos:
            pos = s.get("pos", "")
            tran = s.get("tran", "")
            words = [h.get("w", "") for h in (s.get("hwds") or []) if h.get("w")]
            head = " ".join(filter(None, [pos, tran])).rstrip()
            tail = ", ".join(words)
            if head and tail:
                lines.append(f"  {head}: {tail}")
            elif tail:
                lines.append(f"  {tail}")

    # 反义词
    antos = _safe(c, "antos", "anto", default=[]) or []
    if antos:
        items = [a.get("hwd", "") for a in antos if a.get("hwd")]
        if items:
            lines.append("【反义词】 " + ", ".join(items))

    # 词根相关
    rels = _safe(c, "relWord", "rels", default=[]) or []
    if rels:
        lines.append("【词根相关】")
        for r in rels:
            pos = r.get("pos", "")
            words = r.get("words") or []
            parts = [f"{w.get('hwd', '')} ({w.get('tran', '')})".strip() for w in words if w.get("hwd")]
            if not parts:
                continue
            prefix = f"{pos}: " if pos else ""
            lines.append(f"  {prefix}" + "; ".join(parts))

    # 记忆法
    rem = _safe(c, "remMethod", "val", default="")
    if rem:
        lines.append("【记忆】")
        lines.append(f"  {rem}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 找不到时给候选 — 编辑距离 + 词形归一
# ---------------------------------------------------------------------------

# 按长度倒序，匹配时取第一个且剥后剩余长度 ≥ 2
_INFLECTION_SUFFIXES: tuple[str, ...] = (
    "ies", "ied", "ying", "ing", "ed", "er", "est", "es", "ly", "d", "s",
)


def _levenshtein(a: str, b: str) -> int:
    """经典 DP，两行滚动数组。词长都很小（≤ 30 字符），无需更花哨的算法。"""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _strip_suffix(word: str) -> str:
    """剥一个常见词形后缀；剥后剩余 < 2 字符则不动。"""
    for suf in _INFLECTION_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def _inflection_aware_similarity(query: str, head: str) -> float:
    """混合相似度（0..1），对常见词形变化友好。"""
    if query == head:
        return 1.0
    # 1. 各自剥后缀后 stem 相等 → 强匹配
    q_stem = _strip_suffix(query)
    h_stem = _strip_suffix(head)
    if q_stem and q_stem == h_stem:
        return 0.95
    # 2. query 是 head 的短前缀（典型截断场景：用户少打几个字母）
    #    长度差越大越像截断 → 给一点加成；这样 "abandon" 胜过 "a"
    if head and query and head.startswith(query):
        diff = len(head) - len(query)
        if 0 < diff <= 3:
            return 0.9 + 0.05 * diff / 3
    # 3. head 是 query 的短前缀（query 在 head 后面多带了尾巴：含形变 / 拼错尾）
    if query and head and query.startswith(head):
        diff = len(query) - len(head)
        if 0 < diff <= 3:
            return 0.9
    # 4. 兜底：归一化编辑距离
    dist = _levenshtein(query, head)
    return 1.0 - dist / max(len(query), len(head), 1)


def suggest_similar(
    query: str, heads: list[str], k: int = 10
) -> list[tuple[float, str]]:
    """返回按相似度降序、score > 0 的前 k 个 (score, head) 候选。"""
    q = query.lower()
    scored = [(_inflection_aware_similarity(q, h.lower()), h) for h in heads]
    scored.sort(key=lambda x: -x[0])
    return [(s, h) for s, h in scored[:k] if s > 0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 CET4_2.json 随机或按词查一个词条。",
    )
    parser.add_argument(
        "word",
        nargs="?",
        default=None,
        help="wordHead（如 abandon）。省略则随机抽取。",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="不打印索引构建/复用提示。",
    )
    args = parser.parse_args(argv)

    if not JSON_PATH.exists():
        print(f"error: {JSON_PATH} 不存在", file=sys.stderr)
        return 1

    index = load_index(JSON_PATH, INDEX_PATH, verbose=not args.quiet)
    heads: list[str] = index["heads"]
    offsets: dict[str, int] = index["offsets"]

    if args.word:
        key = args.word.strip()
        if key not in offsets:
            suggestions = suggest_similar(key, heads, k=10)
            if suggestions:
                print(
                    f"error: 找不到 wordHead={key!r}。你可能想找:",
                    file=sys.stderr,
                )
                for score, cand in suggestions:
                    print(f"  - {cand}  (score={score:.2f})", file=sys.stderr)
            else:
                preview = ", ".join(heads[:5])
                print(
                    f"error: 找不到 wordHead={key!r}。可选示例: {preview}…",
                    file=sys.stderr,
                )
            return 2
        head = key
    else:
        head = random.choice(heads)

    row = fetch_row(JSON_PATH, offsets[head])
    print(format_detail(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
