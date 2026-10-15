"""Translator library: Chinese→English translation and suffix rephrasing via
Anthropic's Messages API.

Public API
----------
- ``Translator(debug=False)``
    - ``.translate(source_zh) -> str``
    - ``.rephrase(source_zh, current, prefix, target_word) -> list[str]``
- ``tokenize(sentence) -> list[str]``
- ``render_numbered_blocks(sentence, ...) -> str``
- ``join_sentence(prefix, suffix) -> str``

Configuration (read at import time, except ``api_key`` which is checked at
``Translator.__init__``):

- ``ANTHROPIC_API_KEY``  (required)
- ``ANTHROPIC_MODEL``    (default: ``claude-haiku-4-5``)
- ``ANTHROPIC_BASE_URL`` (optional, for proxying)

Design notes
------------
The library owns no global mutable state. Debug-dumping is per-instance
(``Translator.debug``), so multiple Translators can coexist with different
verbosity without interfering with each other.
"""

from __future__ import annotations

import json
import os
import sys

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


# --- System prompts --------------------------------------------------------


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


# --- Helpers ---------------------------------------------------------------


def tokenize(sentence: str) -> list[str]:
    """Split into whitespace-separated tokens; punctuation stays attached."""
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


def join_sentence(prefix: str, suffix: str) -> str:
    """Join a fixed prefix with a new suffix, handling leading punctuation
    cleanly.

    ``prefix`` is everything before the target word (no trailing space).
    ``suffix`` is the alternative suffix starting at the target word. If the
    suffix begins with closing-glue punctuation (',.!?:)]}') we glue without
    a space; otherwise insert a single space.
    """
    if not prefix:
        return suffix
    if suffix and suffix[0] in (",", ".", "!", "?", ";", ":", ")", "]", "}"):
        return f"{prefix}{suffix}"
    return f"{prefix} {suffix}"


def render_numbered_blocks(
    sentence: str,
    sep: str = "  ",
    indent: str = "  ",
) -> str:
    """Render the sentence on a single line as '1.Word1  2.Word2  ...'.

    Each unit is ``<index>.<word>``, units separated by ``sep``. Number and
    word are both LEFT-aligned (no padding). For a typical short sentence
    (one Chinese sentence translated to one English sentence) the whole
    layout fits on one terminal line below the Translation row.

    Example::

        1.Elon  2.Musk  3.is  4.not  5.on  6.LinkedIn.
    """
    words = tokenize(sentence)
    if not words:
        return f"{indent}(empty)"
    return indent + sep.join(f"{i}.{w}" for i, w in enumerate(words, 1))


# --- Translator ------------------------------------------------------------


class Translator:
    """Thin Anthropic wrapper for translate and replace-word.

    Pass ``debug=True`` to dump every request and response to stderr as
    labeled JSON. Useful when an LLM call returns unexpected content or
    fails outright — you can see the exact prompt and raw response without
    rerunning with logging patched in.
    """

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

    # --- Debug helpers -----------------------------------------------------

    def _dump_request(self, name: str, fields: dict) -> None:
        """Pretty-print each request field to stderr as labeled JSON.
        No-op when ``self.debug`` is False."""
        if not self.debug:
            return
        sys.stderr.write(f"\n[debug] ── {name} request ──\n")
        for k, v in fields.items():
            sys.stderr.write(f"  {k}:\n")
            for line in json.dumps(v, ensure_ascii=False, indent=2).splitlines():
                sys.stderr.write(f"    {line}\n")
        sys.stderr.flush()

    def _dump_response(self, name: str, resp) -> None:
        """Pretty-print the API response to stderr. No-op when off."""
        if not self.debug:
            return
        sys.stderr.write(f"[debug] ── {name} response ──\n")
        sys.stderr.write(
            json.dumps(self._resp_to_dict(resp), ensure_ascii=False, indent=2, default=str)
        )
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _dump_summary(self, label: str, obj: dict) -> None:
        """Short one-line '✓' confirmation after a successful call.
        No-op when off."""
        if not self.debug:
            return
        sys.stderr.write(
            f"[debug] ✓ {label}: {json.dumps(obj, ensure_ascii=False, default=str)}\n"
        )
        sys.stderr.flush()

    @staticmethod
    def _resp_to_dict(resp) -> dict:
        """Serialize an Anthropic Message to a JSON-safe dict.

        Prefers pydantic v2 ``model_dump()`` when available; falls back to
        attribute walking so older SDK shapes still work.
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

    # --- Public API --------------------------------------------------------

    def translate(self, source_zh: str) -> str:
        """Translate a single Chinese sentence to one English sentence."""
        request_kwargs = {
            "model": self.model,
            "system": TRANSLATE_SYSTEM,
            "tools": [TRANSLATE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_translation"},
            "messages": [{"role": "user", "content": f"Translate: {source_zh}"}],
            "max_tokens": 512,
        }
        self._dump_request("translate", request_kwargs)
        resp = self.client.messages.create(**request_kwargs)
        self._dump_response("translate", resp)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = getattr(block, "input", None) or {}
                if isinstance(data, dict) and data.get("translation"):
                    result = str(data["translation"]).strip()
                    self._dump_summary("translate", {"translation": result})
                    return result
        raise RuntimeError("LLM did not return a translation.")

    def rephrase(
        self,
        source_zh: str,
        current: str,
        prefix: str,
        target_word: str,
    ) -> list[str]:
        """Propose alternative suffixes that replace the target word onward.

        ``prefix`` is the verbatim prefix before the target word (FIXED —
        the LLM must not repeat it). Each returned candidate is a multi-word
        suffix whose first word MUST differ from ``target_word``.

        Defensive: any candidate whose first head word still equals the
        target's head word is dropped; if that empties the list we fall back
        to the raw list so the caller still has options to pick from (or
        cancel with 0).
        """
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
        self._dump_request("rephrase", request_kwargs)
        resp = self.client.messages.create(**request_kwargs)
        self._dump_response("rephrase", resp)
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
        target_head = _first_word(target_word)
        filtered = [c for c in raw if _first_word(c) != target_head]
        result = filtered or raw
        self._dump_summary("rephrase", {"count": len(result), "candidates": result})
        return result
