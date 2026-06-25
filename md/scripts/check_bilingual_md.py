#!/usr/bin/env python3
"""
检查双语 markdown 文档的完整性和正确性。

对应规范文档：docs/bilingual-md-guide.md

用法：
    python check_bilingual_md.py <file.md>            # 打印摘要
    python check_bilingual_md.py <file.md> --strict   # 不一致时退出码 1
    python check_bilingual_md.py <file.md> --json     # 输出 JSON 格式

输出：
    - 块结构统计（en / zh / 块外）
    - 各类 markdown 元素在 en / zh / 块外三栏的数量
    - 一致性检查（标题、段落、列表、引用、表格等）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# 块解析
# ---------------------------------------------------------------------------

def parse_blocks(content: str) -> list[dict]:
    """
    把 markdown 切成块。每块形如 {"lang": "en"|"zh"|None, "lines": [...]}
    - lang="en"/"zh"：包裹在 ::: en / ::: zh 里的内容
    - lang=None：块外内容（图片、代码块、文档级分割线、未包裹的元信息等）
    """
    blocks: list[dict] = []
    current: dict | None = None
    in_code = False
    code_fence: str | None = None

    for raw in content.split("\n"):
        stripped = raw.strip()

        # 围栏代码块（``` 或 ~~~）
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            if not in_code:
                in_code = True
                code_fence = fence
            elif code_fence and stripped.startswith(code_fence):
                in_code = False
                code_fence = None
            # 代码块内容归属当前块（无论 en/zh/neutral）
            _append_line(current, blocks, raw)
            continue

        if in_code:
            _append_line(current, blocks, raw)
            continue

        # 块标记
        if stripped == "::: en":
            _close(current, blocks)
            current = {"lang": "en", "lines": []}
            continue
        if stripped == "::: zh":
            _close(current, blocks)
            current = {"lang": "zh", "lines": []}
            continue
        if stripped == ":::":
            _close(current, blocks)
            current = None
            continue

        _append_line(current, blocks, raw)

    _close(current, blocks)
    return blocks


def _append_line(current: dict | None, blocks: list[dict], line: str) -> None:
    if current is not None:
        current["lines"].append(line)
    else:
        # 块外内容，每行独立成块（合并相邻同语言行更省事，但这里不必要）
        blocks.append({"lang": None, "lines": [line]})


def _close(current: dict | None, blocks: list[dict]) -> None:
    if current is not None:
        blocks.append(current)
        current = None


# ---------------------------------------------------------------------------
# 元素计数
# ---------------------------------------------------------------------------

def count_features(text: str) -> Counter:
    """
    统计文本里各种 markdown 元素的数量。
    代码块内部的内容不计入段落 / 引用 / 列表等，但单独计 code_lines。
    """
    counts: Counter = Counter()
    in_code = False
    code_fence: str | None = None
    in_table = False
    para_buf: list[str] = []

    def flush_para() -> None:
        # 一个段落 = 一段连续非空、非标题、非列表、非引用的文字行
        if para_buf:
            counts["paragraphs"] += 1
            para_buf.clear()

    for line in text.split("\n"):
        stripped = line.strip()

        # --- 代码块 ---
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            if not in_code:
                in_code = True
                code_fence = fence
                counts["code_blocks"] += 1
            elif code_fence and stripped.startswith(code_fence):
                in_code = False
                code_fence = None
                flush_para()
            continue

        if in_code:
            counts["code_lines"] += 1
            continue

        # --- 标题 ---
        m = re.match(r"^(#{1,6})\s+\S", stripped)
        if m:
            flush_para()
            counts[f"h{len(m.group(1))}"] += 1
            in_table = False
            continue

        # --- 表格 ---
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 2:
            if not in_table:
                counts["tables"] += 1
                in_table = True
            counts["table_rows"] += 1
            continue
        else:
            in_table = False

        # --- 引用 ---
        if re.match(r"^>+\s", stripped):
            flush_para()
            counts["blockquotes"] += 1
            continue

        # --- 列表 ---
        if re.match(r"^[-*+]\s+", stripped):
            flush_para()
            counts["unordered_list_items"] += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            counts["ordered_list_items"] += 1
            continue

        # --- 段落缓冲 ---
        if stripped:
            para_buf.append(stripped)
        else:
            flush_para()

    flush_para()
    return counts


# ---------------------------------------------------------------------------
# 一致性检查
# ---------------------------------------------------------------------------

def compare(en: Counter, zh: Counter) -> list[tuple[str, int, int]]:
    """对比 en/zh 同名元素，返回 (key, en_value, zh_value) 的差异列表。"""
    diffs: list[tuple[str, int, int]] = []
    keys = set(en) | set(zh)
    for k in sorted(keys):
        if en.get(k, 0) != zh.get(k, 0):
            diffs.append((k, en.get(k, 0), zh.get(k, 0)))
    return diffs


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

# 显示列顺序
ROW_KEYS = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    "paragraphs",
    "tables", "table_rows",
    "unordered_list_items", "ordered_list_items",
    "blockquotes",
    "images",
    "code_blocks", "code_lines",
    "links",
]

ROW_LABELS = {
    "h1": "H1 标题",
    "h2": "H2 标题",
    "h3": "H3 标题",
    "h4": "H4 标题",
    "h5": "H5 标题",
    "h6": "H6 标题",
    "paragraphs": "段落",
    "tables": "表格",
    "table_rows": "  └ 行数",
    "unordered_list_items": "无序列表项",
    "ordered_list_items": "有序列表项",
    "blockquotes": "引用块",
    "images": "图片",
    "code_blocks": "代码块",
    "code_lines": "  └ 行数",
    "links": "链接",
}


def print_human(path: Path, blocks: list[dict], counts: dict[str, Counter]) -> None:
    en_n = sum(1 for b in blocks if b["lang"] == "en")
    zh_n = sum(1 for b in blocks if b["lang"] == "zh")
    out_n = sum(1 for b in blocks if b["lang"] is None)

    print(f"📄 文件: {path}")
    print(f"📦 块结构: 英文={en_n}  中文={zh_n}  块外={out_n}")

    if en_n == zh_n:
        print(f"✅ en / zh 块数量平衡（{en_n} ↔ {zh_n}）")
    else:
        print(f"❌ en / zh 块数量不一致：英文 {en_n}，中文 {zh_n}")

    # 元素统计表
    print()
    header = f"{'元素':<18} {'英文':>8} {'中文':>8} {'块外':>8}"
    print(header)
    print("-" * len(header))
    for k in ROW_KEYS:
        v_en = counts["en"].get(k, 0)
        v_zh = counts["zh"].get(k, 0)
        v_out = counts["out"].get(k, 0)
        if v_en or v_zh or v_out:
            print(f"{ROW_LABELS[k]:<18} {v_en:>8} {v_zh:>8} {v_out:>8}")

    # 一致性检查
    print()
    print("🔍 一致性检查：")
    diffs = compare(counts["en"], counts["zh"])
    if not diffs:
        print("  ✅ 所有结构元素数量一致")
    else:
        for k, e, z in diffs:
            mark = "✅" if k in {"paragraphs"} and abs(e - z) <= 1 else "⚠️ "
            print(f"  {mark} {k}: 英文={e}, 中文={z}")


def to_json(path: Path, blocks: list[dict], counts: dict[str, Counter]) -> str:
    return json.dumps(
        {
            "file": str(path),
            "blocks": {
                "en": sum(1 for b in blocks if b["lang"] == "en"),
                "zh": sum(1 for b in blocks if b["lang"] == "zh"),
                "out": sum(1 for b in blocks if b["lang"] is None),
            },
            "counts": {k: dict(c) for k, c in counts.items()},
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="检查双语 markdown 文档的结构完整性。",
    )
    parser.add_argument("file", type=Path, help="待检查的 markdown 文件")
    parser.add_argument(
        "--strict", action="store_true",
        help="en/zh 块数量或关键元素不一致时退出码 1",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="以 JSON 格式输出",
    )
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
        return 2

    content = args.file.read_text()
    blocks = parse_blocks(content)

    en_text = "\n".join("\n".join(b["lines"]) for b in blocks if b["lang"] == "en")
    zh_text = "\n".join("\n".join(b["lines"]) for b in blocks if b["lang"] == "zh")
    out_text = "\n".join("\n".join(b["lines"]) for b in blocks if b["lang"] is None)

    counts = {
        "en": count_features(en_text),
        "zh": count_features(zh_text),
        "out": count_features(out_text),
    }

    if args.as_json:
        print(to_json(args.file, blocks, counts))
    else:
        print_human(args.file, blocks, counts)

    if args.strict:
        en_n = sum(1 for b in blocks if b["lang"] == "en")
        zh_n = sum(1 for b in blocks if b["lang"] == "zh")
        if en_n != zh_n:
            return 1
        # 关键元素必须完全一致
        for k in ("h1", "h2", "h3", "blockquotes", "tables",
                 "unordered_list_items", "ordered_list_items"):
            if counts["en"].get(k, 0) != counts["zh"].get(k, 0):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
