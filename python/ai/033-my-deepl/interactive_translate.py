#!/usr/bin/env python3
"""Interactive translation REPL with suffix-level replacement suggestions.

REPL flow:
    > Chinese sentence                 -> translate, show sentence + numbered words
    Word # to replace: <n>             -> ask LLM for alternative suffixes
    Pick: <n>                          -> substitute the suffix, loop

Replacing keeps the prefix (everything before the picked word) unchanged.
The target word is always swapped for a different word; the words after it
may be rephrased freely (or kept similar — no requirement either way).

Commands:
    /new   start a fresh session
    /exit  quit
    h      show help

The translate/rephrase logic lives in ``translator.py`` — this file is the
terminal UX: argument parsing, spinner, prompts, command dispatch.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager

from translator import (
    Translator,
    join_sentence,
    render_numbered_blocks,
    tokenize,
)


# --- Spinner ---------------------------------------------------------------


_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(message: str, *, debug: bool = False):
    """Print a one-line spinner to stderr while the body runs. Cleared on exit.

    With ``debug=True``, skip the spinner entirely: the Translator's request
    and response dumps already announce what is happening (and the per-frame
    TTY animation would clobber the dump lines as they scroll by). A single
    line is printed instead, mirroring what the spinner would show first.
    """
    if debug:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
        yield
        return

    stop = threading.Event()

    def run() -> None:
        # Write to stderr with \r so it overwrites the same line in place.
        for ch in itertools.cycle(_SPINNER_CHARS):
            if stop.is_set():
                break
            sys.stderr.write(f"\r{message} {ch}")
            sys.stderr.flush()
            # Sleep in small chunks so we react quickly to stop().
            for _ in range(8):
                if stop.is_set():
                    break
                time.sleep(0.01)
        # Clear the line: overwrite with spaces and move cursor back to col 0.
        sys.stderr.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stderr.flush()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=0.5)


# --- CLI ------------------------------------------------------------------


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
    "An empty Enter reprints the current sentence."
)


def parse_args(argv: list[str] | None = None) -> bool:
    """Parse sys.argv and return the --debug flag. Exits with usage on error."""
    parser = argparse.ArgumentParser(
        prog="interactive_translate",
        description="Interactive translation with suffix-level replacement suggestions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print every LLM request and response to stderr in full, and "
            "suppress the spinner (the dump replaces it visually)."
        ),
    )
    args = parser.parse_args(argv)
    return args.debug


# --- REPL ------------------------------------------------------------------


def main() -> int:
    debug = parse_args()

    # Enable line editing (arrow keys, Ctrl-A/E, history) for input(). Without
    # this, typing Chinese in the source prompt and trying to use arrow keys
    # to move the cursor prints the raw escape sequence (e.g. "[[C", "^[[D")
    # instead of editing the buffer. import readline patches the stdin TTY
    # mode for the lifetime of the process; the import has no effect on
    # Windows (readline is unavailable there) and is safe to no-op.
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print("Interactive translation. (h for help)")
    if debug:
        print(
            "(--debug: full LLM requests and responses will be dumped to stderr; "
            "spinner suppressed.)",
            file=sys.stderr,
        )
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Export it and try again.", file=sys.stderr)
        return 1

    try:
        translator = Translator(debug=debug)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not init translator: {e}", file=sys.stderr)
        return 1

    source_zh: str | None = None
    current: str | None = None

    def render_translation(label: str = "Translation") -> None:
        """Print the current sentence on one line, then 2-row numbered blocks below."""
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
                # nothing to clear
                continue
            if raw in ("/help", "/h", "h"):
                print(HELP_TEXT)
                continue

            source_zh = raw
            try:
                with spinner("Translating...", debug=debug):
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
            with spinner("Rephrasing...", debug=debug):
                candidates = translator.rephrase(source_zh, current, prefix, target_word)
        except Exception as e:  # noqa: BLE001
            print(f"Lookup failed: {e}")
            continue

        # Each candidate is a multi-word suffix; glue onto the fixed prefix for display.
        print()
        print(f"Replacements starting at word #{idx} ({target_word!r}):")
        width = max(2, len(str(len(candidates))))
        for i, c in enumerate(candidates, 1):
            new_sentence = join_sentence(prefix, c)
            print(f"  {i:>{width}}. {new_sentence}")
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

        current = join_sentence(prefix, candidates[pick_idx - 1])
        print()
        render_translation("Updated translation")

    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
