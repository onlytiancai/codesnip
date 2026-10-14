#!/usr/bin/env python3
"""Interactive translation with word-level replacement suggestions.

REPL flow:
    > Chinese sentence                 -> translate, show sentence + numbered words
    Word # to replace: <n>             -> ask LLM for single-word alternatives
    Pick: <n>                          -> substitute just that word, loop

Commands:
    /new   start a fresh session
    /exit  quit
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager

import anthropic  # type: ignore[import-not-found]


# --- Configuration ---------------------------------------------------------


MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


# --- LLM tool schemas ------------------------------------------------------


TRANSLATE_TOOL = {
    "name": "submit_translation",
    "description": "Submit a single natural English translation of the source.",
    "input_schema": {
        "type": "object",
        "properties": {
            "translation": {
                "type": "string",
                "minLength": 1,
                "description": "One complete, idiomatic English sentence.",
            }
        },
        "required": ["translation"],
    },
}

# Replace ONLY the target word — keep the prefix and the suffix after it as-is.
# Candidates are single words (one whitespace-separated token).
REPLACE_WORD_TOOL = {
    "name": "submit_replacements",
    "description": "Submit single-word alternatives to replace the target word in place.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "A SINGLE word (one whitespace-separated token) that could replace "
                        "the target word at its current position. Punctuation is allowed at "
                        "the end (e.g. contracted forms like 'isn't' are fine as one token)."
                    ),
                },
            }
        },
        "required": ["candidates"],
    },
}


TRANSLATE_SYSTEM = (
    "You are a translator. Translate the given Chinese sentence into a single, "
    "natural English sentence. Preserve the source's meaning faithfully. "
    "Reply by calling the submit_translation tool with the translation."
)

REPLACE_WORD_SYSTEM = (
    "You are helping the user refine a translation by substituting a SINGLE word.\n\n"
    "Given the Chinese source, the current English sentence, and the target word to "
    "replace, propose 5-8 single-word alternatives. The user wants to swap just that "
    "one word — the prefix before it and the suffix after it stay EXACTLY as they are.\n\n"
    "Rules:\n"
    "- Each candidate must be exactly ONE whitespace-separated token (e.g. 'is', "
    "  'isn't', 'remains', 'does'). Contractions like 'isn't' or 'wasn't' count as one.\n"
    "- Punctuation like a trailing period or comma is fine if the word naturally carries it.\n"
    "- Candidate #1 should be the original target word (so the user can keep it if they like).\n"
    "- Other candidates should differ in vocabulary or grammatical form but still be a "
    "  single, sensible English word at that position.\n"
    "- Do NOT include multi-word phrases, suffixes, or full clauses.\n"
    "- Use the Chinese source to keep the meaning consistent with the original.\n"
    "- Call submit_replacements with the candidate list."
)


# --- LLM client ------------------------------------------------------------


class Translator:
    """Thin Anthropic wrapper for translate and replace-word."""

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        kwargs: dict = {"api_key": api_key}
        if os.environ.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        self.client = anthropic.Anthropic(**kwargs)
        self.model = MODEL

    def translate(self, source_zh: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=TRANSLATE_SYSTEM,
            tools=[TRANSLATE_TOOL],
            tool_choice={"type": "tool", "name": "submit_translation"},
            messages=[{"role": "user", "content": f"Translate: {source_zh}"}],
            max_tokens=512,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", None) or {}
                if isinstance(data, dict) and data.get("translation"):
                    return str(data["translation"]).strip()
        raise RuntimeError("LLM did not return a translation.")

    def replace_word(
        self,
        source_zh: str,
        current: str,
        target_word: str,
    ) -> list[str]:
        user_msg = (
            f"Source (Chinese): {source_zh}\n"
            f"Current English sentence: {current}\n"
            f"Target word to replace: {target_word!r}\n\n"
            "Generate 5-8 single-word alternatives. Candidate #1 should be the original "
            "target word. Other candidates should differ in vocabulary or grammatical form."
        )
        resp = self.client.messages.create(
            model=self.model,
            system=REPLACE_WORD_SYSTEM,
            tools=[REPLACE_WORD_TOOL],
            tool_choice={"type": "tool", "name": "submit_replacements"},
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=256,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", None) or {}
                cands = data.get("candidates", []) if isinstance(data, dict) else []
                if isinstance(cands, list):
                    out = [str(c).strip() for c in cands if str(c).strip()]
                    if out:
                        return out
        raise RuntimeError("LLM did not return candidates.")


# --- Spinner ---------------------------------------------------------------


_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(message: str):
    """Print a one-line spinner to stderr while the body runs. Cleared on exit."""
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


# --- Helpers ---------------------------------------------------------------


def tokenize(sentence: str) -> list[str]:
    """Split into whitespace-separated tokens; punctuation stays attached to its token."""
    return sentence.split()


def render_numbered_blocks(
    sentence: str,
    sep: str = "  ",
    indent: str = "  ",
) -> str:
    """Render the sentence on a single line as '1.Word1  2.Word2  ...'.

    Each "unit" is `<index>.<word>`, units are separated by `sep`. Number and
    word are both LEFT-aligned (no padding). For typical short sentences
    (a single Chinese sentence translated to one English sentence) the entire
    layout fits on one terminal line below the Translation row.

    Example for "Elon Musk is not on LinkedIn.":
      1.Elon  2.Musk  3.is  4.not  5.on  6.LinkedIn.
    """
    words = tokenize(sentence)
    if not words:
        return f"{indent}(empty)"
    return indent + sep.join(f"{i}.{w}" for i, w in enumerate(words, 1))


def substitute_word(sentence: str, idx: int, new_word: str) -> str:
    """Replace the idx-th word (1-based) of `sentence` with `new_word`. Other words untouched."""
    words = tokenize(sentence)
    if idx < 1 or idx > len(words):
        raise ValueError(f"index {idx} out of range (sentence has {len(words)} words)")
    words[idx - 1] = new_word
    return " ".join(words)


# --- REPL ------------------------------------------------------------------


HELP_TEXT = (
    "Commands:\n"
    "  /new     start a fresh session\n"
    "  /exit    quit\n"
    "  h        show this help\n"
    "\n"
    "On the first prompt, type a Chinese sentence to translate.\n"
    "On the next prompts, type a word number to substitute that single word.\n"
    "An empty Enter reprints the current sentence."
)


def main() -> int:
    print("Interactive translation. (h for help)")
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Export it and try again.", file=sys.stderr)
        return 1

    try:
        translator = Translator()
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
                with spinner("Translating..."):
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

        try:
            with spinner("Finding alternatives..."):
                candidates = translator.replace_word(source_zh, current, target_word)
        except Exception as e:  # noqa: BLE001
            print(f"Lookup failed: {e}")
            continue

        # Each candidate is a single word; build the new sentence inline for clarity.
        print()
        print(f"Replacements for word #{idx} ({target_word!r}):")
        width = max(2, len(str(len(candidates))))
        for i, c in enumerate(candidates, 1):
            new_sentence = substitute_word(current, idx, c)
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

        current = substitute_word(current, idx, candidates[pick_idx - 1])
        print()
        render_translation("Updated translation")

    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
