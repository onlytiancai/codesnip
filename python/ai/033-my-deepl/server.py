"""FastAPI web wrapper around the `translator` library.

Endpoints
---------
- ``GET  /``                 → static/index.html (Vue 3 SPA)
- ``POST /api/translate``    → translate a Chinese sentence to English
- ``POST /api/rephrase``     → propose alternative suffixes for a word position

Run::

    ANTHROPIC_API_KEY=... uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from translator import Translator


# --- Configuration ---------------------------------------------------------

# Enable per-instance debug dumping in the Translator. Off by default; the
# CLI's `--debug` is a separate code path (interactive_translate.py).
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


# --- Translator singleton -------------------------------------------------


_translator: Translator | None = None


def get_translator() -> Translator:
    """Lazy-create the Translator on first call."""
    global _translator
    if _translator is None:
        _translator = Translator(debug=DEBUG)
    return _translator


# --- Lifespan -------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Eager-init the Translator at startup so the first request isn't slow.
    Missing API key is a warning, not a fatal error — the user may set the
    env var after launch. Each request still checks at call time.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "WARNING: ANTHROPIC_API_KEY is not set; /api/* calls will return "
            "503 until you restart uvicorn with the variable exported.",
            file=sys.stderr,
        )
    else:
        try:
            t = get_translator()
            print(f"Translator ready (model={t.model}, debug={DEBUG})", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"Translator init failed: {e}", file=sys.stderr)
    yield


# --- App ------------------------------------------------------------------


app = FastAPI(
    title="Interactive Translation (Web)",
    description="Chinese→English with suffix-level rewriting.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schemas --------------------------------------------------------------


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Chinese sentence to translate.")


class TranslateResponse(BaseModel):
    translation: str


class RephraseRequest(BaseModel):
    source: str = Field(..., description="Original Chinese sentence.")
    current: str = Field(..., description="Current English sentence.")
    prefix: str = Field(..., description="Verbatim prefix before the target word.")
    target_word: str = Field(..., description="Word at the picked position.")


class RephraseResponse(BaseModel):
    candidates: list[str]


# --- Routes ---------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the single-page Vue app."""
    if not INDEX_HTML.exists():
        # Surface the error to the user clearly instead of returning 404.
        raise HTTPException(
            status_code=500,
            detail=(
                f"Frontend bundle missing at {INDEX_HTML}. "
                "Make sure static/index.html exists next to server.py."
            ),
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Server is missing ANTHROPIC_API_KEY. Set it and restart uvicorn.",
        )
    try:
        result = get_translator().translate(req.text.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}") from e
    return TranslateResponse(translation=result)


@app.post("/api/rephrase", response_model=RephraseResponse)
async def rephrase(req: RephraseRequest) -> RephraseResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Server is missing ANTHROPIC_API_KEY. Set it and restart uvicorn.",
        )
    try:
        cands = get_translator().rephrase(
            source_zh=req.source,
            current=req.current,
            prefix=req.prefix,
            target_word=req.target_word,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Rephrase failed: {e}") from e
    return RephraseResponse(candidates=cands)
