#!/usr/bin/env python3
"""把 CET4_2.json / GaoZhongluan_2.json / ChuZhongluan_2.json 按 wordHead 去重
合并到 vocab_all.json (JSONL)。

每行原始 JSON 整体保留（不动 schema），仅按 wordHead 跳过重复。优先级：
CET4 → 高中 → 初中，所以同 head 优先取 CET4 那条（释义/例句/短语最丰富）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 顺序即优先级：先到先得
SOURCES: list[tuple[str, Path]] = [
    ("CET4", ROOT / "CET4_2.json"),
    ("GaoZhong", ROOT / "GaoZhongluan_2.json"),
    ("ChuZhong", ROOT / "ChuZhongluan_2.json"),
]
OUTPUT = ROOT / "vocab_all.json"


def main() -> int:
    seen: set[str] = set()
    per_source_kept: dict[str, int] = {}
    per_source_total: dict[str, int] = {}
    duplicates: dict[str, list[str]] = {}  # head -> [bookIds that had it but lost]

    with OUTPUT.open("w", encoding="utf-8") as out:
        for label, path in SOURCES:
            kept = 0
            total = 0
            if not path.exists():
                print(f"warn: {path} 不存在，跳过 {label}", file=sys.stderr)
                per_source_total[label] = 0
                per_source_kept[label] = 0
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"warn: {label} 第 {total} 行 JSON 解析失败: {e}", file=sys.stderr)
                        continue
                    head = (
                        row.get("content", {})
                        .get("word", {})
                        .get("wordHead")
                    )
                    if not head:
                        continue
                    if head in seen:
                        duplicates.setdefault(head, []).append(label)
                        continue
                    seen.add(head)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
            per_source_total[label] = total
            per_source_kept[label] = kept

    # 报告
    total_in = sum(per_source_total.values())
    print(f"输入合计: {total_in} 条")
    for label, _ in SOURCES:
        t = per_source_total.get(label, 0)
        k = per_source_kept.get(label, 0)
        skip = t - k
        marker = f"  - 跳过重复 {skip}" if skip else ""
        print(f"  {label:>10}: 保留 {k:>4} / {t}{marker}")
    print(f"去重输出: {len(seen)} 条 -> {OUTPUT}")
    if duplicates:
        # 简单抽 5 个示例，方便核对
        sample = list(duplicates.items())[:5]
        print(f"被跳过的重复 head 示例: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
