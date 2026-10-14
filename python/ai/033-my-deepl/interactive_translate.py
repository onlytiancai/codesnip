#!/usr/bin/env python3
"""Interactive translation with suffix-level replacement suggestions.

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

# Replace the suffix starting at the target word. The prefix (everything
# before the target) is FIXED; the target word itself MUST be replaced with a
# different word, while the words after it are free to change (or stay).
# Candidates are full multi-word suffixes ending with sentence-final
# punctuation, and each candidate's first word must differ from the original
# target word.
REPHRASE_TOOL = {
    "name": "submit_replacements",
    "description": (
        "Submit a list of alternative suffixes that replace the segment "
        "starting at the target position. Each suffix's first word must "
        "DIFFER from the original target word."
    ),
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
                        "An alternative SUFFIX replacing the segment starting at the "
                        "target position and ending with sentence-final punctuation "
                        "(., !, ?). The prefix before the target position is fixed "
                        "and must NOT be included. The first word of the new suffix "
                        "must NOT equal the original target word."
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

REPHRASE_SYSTEM = (
    "You are helping the user refine a translation by replacing the suffix "
    "starting at a specific position.\n\n"
    "Given the Chinese source, the current English sentence, the fixed prefix, "
    "and the original target word, propose 4-8 alternative suffixes that "
    "rewrite the segment starting at the target position.\n\n"
    "Rules:\n"
    "- The PREFIX before the target position is FIXED. Do NOT include it in "
    "  the candidates — only return the suffix that replaces the target word "
    "  onward.\n"
    "- Each candidate's FIRST word MUST be a DIFFERENT word from the original "
    "  target word. The user explicitly wants the current word to be replaced "
    "  (no ties, no same-word rephrasing).\n"
    "- Mix two kinds of candidates so the user has a real choice:\n"
    "    a) 'Word-only' candidates: keep every word AFTER the target word "
    "       IDENTICAL to the current English sentence, and swap only the "
    "       target word itself. Joined with the prefix, only that one word "
    "       changes.\n"
    "    b) 'Full-rewrite' candidates: freely rephrase the target word AND "
    "       the words after it, producing noticeably different phrasing.\n"
    "  Aim for a roughly even split (e.g. half word-only, half full-rewrite). "
    "  Do NOT make every candidate a full-rewrite.\n"
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

    def rephrase(
        self,
        source_zh: str,
        current: str,
        prefix: str,
        target_word: str,
    ) -> list[str]:
        user_msg = (
            f"Source (Chinese): {source_zh}\n"
            f"Current English sentence: {current}\n"
            f"Prefix to KEEP unchanged: {prefix!r}\n"
            f"Original target word (each candidate's FIRST word must NOT equal this): {target_word!r}\n\n"
            "Generate 4-8 alternative suffixes starting at the target position. Each "
            "must begin with a word DIFFERENT from the original target word. Produce "
            "a MIX of two styles:\n"
            "  (a) 'word-only' candidates that keep every word after the target "
            "      IDENTICAL to the current English sentence and only swap the "
            "      target word itself,\n"
            "  (b) 'full-rewrite' candidates that rephrase the target word AND the "
            "      words after it.\n"
            "Aim for a roughly even split (e.g. half word-only, half full-rewrite). "
            "Do NOT make every candidate a full-rewrite."
        )
        resp = self.client.messages.create(
            model=self.model,
            system=REPHRASE_SYSTEM,
            tools=[REPHRASE_TOOL],
            tool_choice={"type": "tool", "name": "submit_replacements"},
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
        )
        raw: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", None) or {}
                cands = data.get("candidates", []) if isinstance(data, dict) else []
                if isinstance(cands, list):
                    raw = [str(c).strip() for c in cands if str(c).strip()]
                    if raw:
                        break
        if not raw:
            raise RuntimeError("LLM did not return candidates.")
        # Defensive: drop candidates whose first word still equals the original
        # target word. The user wants the current word replaced, so a candidate
        # that starts with the same head word violates that intent. If filtering
        # empties the list, fall back to the raw list so the caller still has
        # something to pick from (or cancel with 0).
        target_head = _first_word(target_word)
        filtered = [c for c in raw if _first_word(c) != target_head]
        return filtered or raw


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


def _first_word(s: str) -> str:
    """Lowercased first whitespace-separated token, with leading/trailing
    punctuation stripped. Used to compare the head of a candidate suffix with
    the original target word so we can tell whether the LLM actually swapped
    it. 'Directly,' / 'directly.' / 'directly' all match the same head word.
    """
    s = s.lstrip().lstrip(",.;:!?")
    if not s:
        return ""
    return s.split(maxsplit=1)[0].lower().rstrip(".,!?;:")


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


def join_sentence(prefix: str, suffix: str) -> str:
    """Join a fixed prefix with a new suffix, handling leading punctuation cleanly.

    `prefix` is everything before the target word (no trailing space). `suffix` is
    the alternative suffix that starts with the target word. If the suffix begins
    with a closing-glue punctuation (",", ".", "!", "?", etc.) we glue without a
    space; otherwise insert a single space.
    """
    if not prefix:
        return suffix
    if suffix and suffix[0] in (",", ".", "!", "?", ";", ":", ")", "]", "}"):
        return f"{prefix}{suffix}"
    return f"{prefix} {suffix}"


# --- REPL ------------------------------------------------------------------


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
        prefix = " ".join(words[: idx - 1])

        try:
            with spinner("Rephrasing..."):
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
