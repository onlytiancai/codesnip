#!/usr/bin/env python3
"""Interactive translation REPL with suffix-level replacement suggestions.

跟 ``interactive_translate.py``(走 Anthropic Claude)流程一致,
但后端是本地 ollama —— 缺 ``ANTHROPIC_API_KEY`` 也能跑,只要你装了
``s2021008840/hy-mt2:1.8b``(或别的 CN→EN 翻译模型)并 ollama serve 起来。

REPL flow:
    > Chinese sentence                 -> translate, show sentence + numbered words
    Word # to replace: <n>             -> ask ollama for alternative suffixes
    Pick: <n>                          -> substitute the suffix, loop

Commands:
    /new   start a fresh session
    /exit  quit
    h      show help

逻辑都在 ``ollama_rephrase.py`` —— 这个文件只负责终端 UX:参数解析、spinner、
提示、命令分发。
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import List, Optional

from ollama_rephrase import OllamaTranslator
from translator import render_numbered_blocks, tokenize


# --- Spinner (与 interactive_translate.py 同实现) --------------------------


_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(message: str, *, debug: bool = False):
    """stdout 一行 spinner,debug 时直接打印 ``message``。"""
    if debug:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
        yield
        return

    stop = threading.Event()

    def run() -> None:
        for ch in itertools.cycle(_SPINNER_CHARS):
            if stop.is_set():
                break
            sys.stderr.write(f"\r{message} {ch}")
            sys.stderr.flush()
            for _ in range(8):
                if stop.is_set():
                    break
                time.sleep(0.01)
        sys.stderr.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stderr.flush()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=0.5)


# --- CLI -------------------------------------------------------------------


HELP_TEXT = (
    "Commands:\n"
    "  /new     start a fresh session\n"
    "  /exit    quit\n"
    "  h        show this help\n"
    "\n"
    "On the first prompt, type a Chinese sentence to translate.\n"
    "On the next prompts, type a word number to rewrite the suffix starting at\n"
    "  that word. The prefix before it stays fixed; the target word itself is always\n"
    "  swapped for a different word, and the words after it may be rephrased.\n"
    "An empty Enter reprints the current sentence.\n"
    "\n"
    "后端:本地 ollama (默认 s2021008840/hy-mt2:1.8b),不需要 ANTHROPIC_API_KEY。"
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="interactive_ollama_translate",
        description="Interactive translation with suffix-level replacement suggestions (ollama backend).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="把每次调 ollama 的 prompt + 原始回复打到 stderr,关闭 spinner。",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "s2021008840/hy-mt2:1.8b"),
        help="ollama 模型名。默认 s2021008840/hy-mt2:1.8b,可用 OLLAMA_MODEL 环境变量覆盖。",
    )
    p.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL"),
        help="ollama 服务地址,默认 http://127.0.0.1:11434。",
    )
    return p.parse_args(argv)


# --- REPL ------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print("Interactive translation (ollama backend). (h for help)")
    if args.debug:
        print(
            "(--debug: full LLM requests and responses will be dumped to stderr; "
            "spinner suppressed.)",
            file=sys.stderr,
        )
    print(f"(model={args.model})", file=sys.stderr)
    print()

    try:
        translator = OllamaTranslator(
            model=args.model,
            base_url=args.ollama_url,
            debug=args.debug,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: 无法初始化 ollama 后端: {e}", file=sys.stderr)
        return 1

    source_zh: Optional[str] = None
    current: Optional[str] = None

    def render_translation(label: str = "Translation") -> None:
        assert current is not None
        print(f"{label}: {current}")
        print()
        print(render_numbered_blocks(current))
        print()

    while True:
        # === STATE 1: ask for a Chinese sentence ===
        if current is None:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue
            if raw in ("/exit", "/quit"):
                break
            if raw == "/new":
                continue
            if raw in ("/help", "/h", "h"):
                print(HELP_TEXT)
                continue

            source_zh = raw
            try:
                with spinner("Translating...", debug=args.debug):
                    current = translator.translate(source_zh)
            except Exception as e:  # noqa: BLE001
                print(f"Translation failed: {e}")
                current = None
                continue

            print()
            print(f"Source: {source_zh}")
            render_translation("Translation")

        # === STATE 2: ask which word to replace ===
        try:
            raw = input("Word # to replace (Enter=keep, /new, /exit, h=help): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw in ("/exit", "/quit"):
            break
        if raw == "/new":
            current = None
            continue
        if raw in ("/help", "/h", "h"):
            print(HELP_TEXT)
            continue
        if not raw:
            print()
            render_translation("Translation")
            continue
        if not raw.isdigit():
            print(f"  (expected a number, got {raw!r})")
            continue

        idx = int(raw)
        words = tokenize(current)
        if idx < 1 or idx > len(words):
            print(f"  (out of range; this sentence has {len(words)} words)")
            continue

        target_word = words[idx - 1]
        prefix = " ".join(words[: idx - 1])

        try:
            with spinner("Rephrasing...", debug=args.debug):
                candidates = translator.rephrase(source_zh, current, prefix, target_word)
        except Exception as e:  # noqa: BLE001
            print(f"Lookup failed: {e}")
            continue

        if not candidates:
            print(
                f"\n  (no candidates from ollama for target={target_word!r} "
                f"at position #{idx}; try a different word.)"
            )
            print()
            continue

        # 各候选都是带 prefix 的完整句;原样打印即可,不用再 glue。
        print()
        print(f"Replacements starting at word #{idx} ({target_word!r}):")
        width = max(2, len(str(len(candidates))))
        for i, c in enumerate(candidates, 1):
            print(f"  {i:>{width}}. {c}")
        print(f"  0. (cancel, keep current)")
        print()

        # === STATE 3: pick a replacement ===
        try:
            pick = input("Pick: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if pick in ("/exit", "/quit"):
            break
        if pick == "/new":
            current = None
            continue
        if pick in ("/help", "/h", "h"):
            print(HELP_TEXT)
            continue
        if not pick:
            print("  (no selection; keeping current translation)")
            continue
        if not pick.isdigit():
            print(f"  (expected a number, got {pick!r})")
            continue

        pick_idx = int(pick)
        if pick_idx == 0:
            print("  (cancelled)")
            continue
        if pick_idx < 1 or pick_idx > len(candidates):
            print(f"  (out of range; 1-{len(candidates)} or 0)")
            continue

        # 候选已经是完整句(包含 prefix),直接赋值。
        current = candidates[pick_idx - 1]
        print()
        render_translation("Updated translation")

    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
