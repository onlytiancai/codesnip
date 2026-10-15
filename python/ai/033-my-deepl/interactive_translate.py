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

import argparse
import itertools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager

import anthropic  # type: ignore[import-not-found]


# --- Configuration ---------------------------------------------------------


MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


# Module-level debug flag. Set by parse_args() before Translator is constructed.
# When True, Translator dumps full request/response JSON to stderr around every
# API call and the spinner suppresses its TTY animation (since the dump itself
# flows to stderr and would otherwise be overwritten frame-by-frame).
_DEBUG = False


def _set_debug(flag: bool) -> None:
    global _DEBUG
    _DEBUG = flag


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

    def __init__(self, debug: bool = False) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        kwargs: dict = {"api_key": api_key}
        if os.environ.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        self.client = anthropic.Anthropic(**kwargs)
        self.model = MODEL
        self.debug = debug

    def translate(self, source_zh: str) -> str:
        request_kwargs = {
            "model": self.model,
            "system": TRANSLATE_SYSTEM,
            "tools": [TRANSLATE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_translation"},
            "messages": [{"role": "user", "content": f"Translate: {source_zh}"}],
            "max_tokens": 512,
        }
        _dump_request("translate", request_kwargs)
        resp = self.client.messages.create(**request_kwargs)
        _dump_response("translate", resp)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", None) or {}
                if isinstance(data, dict) and data.get("translation"):
                    result = str(data["translation"]).strip()
                    _dump_summary("translate", {"translation": result})
                    return result
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
        request_kwargs = {
            "model": self.model,
            "system": REPHRASE_SYSTEM,
            "tools": [REPHRASE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_replacements"},
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": 512,
        }
        _dump_request("rephrase", request_kwargs)
        resp = self.client.messages.create(**request_kwargs)
        _dump_response("rephrase", resp)
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
        result = filtered or raw
        _dump_summary("rephrase", {"count": len(result), "candidates": result})
        return result


# --- Debug helpers ---------------------------------------------------------


def _dump_request(name: str, fields: dict) -> None:
    """Pretty-print each request field to stderr as labeled JSON. No-op when off."""
    if not _DEBUG:
        return
    sys.stderr.write(f"\n[debug] ── {name} request ──\n")
    for k, v in fields.items():
        sys.stderr.write(f"  {k}:\n")
        for line in json.dumps(v, ensure_ascii=False, indent=2).splitlines():
            sys.stderr.write(f"    {line}\n")
    sys.stderr.flush()


def _dump_response(name: str, resp) -> None:
    """Pretty-print the API response to stderr. No-op when off."""
    if not _DEBUG:
        return
    sys.stderr.write(f"[debug] ── {name} response ──\n")
    sys.stderr.write(json.dumps(_resp_to_dict(resp), ensure_ascii=False, indent=2, default=str))
    sys.stderr.write("\n")
    sys.stderr.flush()


def _dump_summary(label: str, obj: dict) -> None:
    """Short one-line '✓' confirmation after a successful call. No-op when off."""
    if not _DEBUG:
        return
    sys.stderr.write(f"[debug] ✓ {label}: {json.dumps(obj, ensure_ascii=False, default=str)}\n")
    sys.stderr.flush()


def _resp_to_dict(resp) -> dict:
    """Serialize an anthropic Message response to a JSON-safe dict.

    Prefers pydantic v2 `model_dump()` when available, otherwise walks the
    response attributes manually so we stay compatible with older SDK shapes.
    """
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:  # noqa: BLE001
            pass
    blocks: list[dict] = []
    for b in (getattr(resp, "content", None) or []):
        d: dict = {"type": getattr(b, "type", None)}
        if d["type"] == "tool_use":
            d["id"] = getattr(b, "id", None)
            d["name"] = getattr(b, "name", None)
            d["input"] = getattr(b, "input", None)
        elif d["type"] == "text":
            d["text"] = getattr(b, "text", None)
        blocks.append(d)
    usage = getattr(resp, "usage", None)
    usage_dict: dict | None = None
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        }
    return {
        "id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "stop_reason": getattr(resp, "stop_reason", None),
        "content": blocks,
        "usage": usage_dict,
    }


# --- Spinner ---------------------------------------------------------------


_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(message: str):
    """Print a one-line spinner to stderr while the body runs. Cleared on exit.

    With --debug, skip the spinner entirely: the request/response dump already
    announces what is happening (and the per-frame TTY animation would clobber
    the dump lines as they scroll by). A single 'Translating...' line is printed
    instead, for parity with what the spinner would have shown first.
    """
    if _DEBUG:
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


# --- CLI -------------------------------------------------------------------


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
    debug = parse_args()
    _set_debug(debug)

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
