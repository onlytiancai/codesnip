#!/usr/bin/env python3
r"""
verify_blocks.py — 校验 content/ 下所有 .md 文件的 ::: ... ::: 容器配平

【为什么不用正则贪婪/非贪婪】
markdown.js:28 的正则 `([\s\S]*?)\n:::[ \t]*$` 是非贪婪——一旦块未闭合，
正则会一路向前借下一个 `:::` 当作自己的闭合，把错误"消化"掉。
本脚本用**栈式解析**（line-by-line 扫描），未闭合块**必须**显式报错。

检测两类问题：
1. 细粒度：栈式扫描每个 ::: 块起点，如果它在下一个 ::: 开之前没有闭合，报错
2. 粗粒度：^::: （开）vs ^:::$（闭）数量必须相等（兜底包括 ::: details 等自定义容器）

用法：
    python3 scripts/verify_blocks.py            # 扫 content/ch*_{zh,en}.md
    python3 scripts/verify_blocks.py path1.md ...  # 自定义文件
退出码：有未配平返回 1，否则 0
"""
import re
import sys
from pathlib import Path

# 必须与 assets/js/markdown.js 的 BLOCK_TYPES 保持同步
BLOCK_TYPES = [
    "quiz", "chart", "graph", "network",
    "train-demo", "formula", "perceptron-playground",
]

OPEN_RE = re.compile(r"^::: (\S+)(.*)$")   # 任意 ::: 容器（用于栈扫描）
CLOSE_RE = re.compile(r"^:::[ \t]*$")       # 独立的闭合 :::


def stack_scan(text: str) -> list[dict]:
    """
    栈式解析：逐行扫描，对 OPEN_RE 命中就 push，CLOSE_RE 命中就 pop。
    返回未闭合块的列表 [{line, type, args, depth}]。
    """
    stack = []  # [{line, type, args}]
    unclosed = []
    for i, line in enumerate(text.splitlines(), start=1):
        if m := OPEN_RE.match(line):
            stack.append({"line": i, "type": m.group(1), "args": m.group(2).strip()})
        elif CLOSE_RE.match(line):
            if not stack:
                unclosed.append({"line": i, "type": "<close>", "args": "no matching open"})
                continue
            stack.pop()
    # 文件结束时栈里剩下的都是未闭合
    for entry in stack:
        unclosed.append(entry)
    return unclosed


def verify_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    open_count = sum(1 for line in text.splitlines() if OPEN_RE.match(line))
    close_count = sum(1 for line in text.splitlines() if CLOSE_RE.match(line))
    unclosed = stack_scan(text)
    return {
        "path": path,
        "open": open_count,
        "close": close_count,
        "balanced": open_count == close_count,
        "unclosed": unclosed,
    }


def main(argv: list[str]) -> int:
    if argv:
        files = [Path(p) for p in argv]
    else:
        root = Path(__file__).resolve().parent.parent / "content"
        files = sorted(root.glob("ch*_zh.md")) + sorted(root.glob("ch*_en.md"))

    if not files:
        print("no markdown files found", file=sys.stderr)
        return 1

    overall_ok = True
    for f in files:
        r = verify_file(f)
        rel = f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f
        status = "✓" if r["balanced"] and not r["unclosed"] else "✗"
        print(f"{status} {rel}: open={r['open']} close={r['close']}", end="")
        if r["unclosed"]:
            print(f"  [{len(r['unclosed'])} unclosed]")
            for u in r["unclosed"]:
                if u["type"] == "<close>":
                    print(f"    line {u['line']}: stray closing ::: with no matching open")
                else:
                    print(f"    line {u['line']}: ::: {u['type']} {u['args']}  ← no closing :::")
            overall_ok = False
        else:
            print()

    print()
    if overall_ok:
        print(f"all {len(files)} files OK")
        return 0
    print(f"FAILED — 修复 ::: 闭合后再跑一次")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
