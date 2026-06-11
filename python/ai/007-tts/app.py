"""
FastAPI Web Demo：Kokoro-82M 文本转语音。

接口：
  GET  /                      → 静态前端页面 (static/index.html)
  GET  /api/voices            → 返回所有可用音色
  GET  /api/langs             → 返回所有支持的语言
  POST /api/tts               → JSON 请求体，返回二进制音频
  GET  /api/tts               → URL 参数合成（便于浏览器直接 <audio src=...>）

启动：
  python app.py
  # 或  uvicorn app:app --reload --port 8765
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from tts_service import (
    LANG_LABELS,
    TTSService,
    all_voices,
    get_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts-web")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Kokoro TTS Web Demo",
    description="基于 Kokoro-82M 的多语言、多音色文本转语音 Web 演示",
    version="1.0.0",
)


# ---------- Pydantic 模型 ----------

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="要合成的文本")
    voice: str = Field("zf_xiaoni", description="音色名，如 zf_xiaoni / af_heart")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="语速，0.5-2.0")
    format: str = Field("wav", pattern="^(wav|mp3)$", description="输出格式")


# ---------- 静态资源 ----------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/static/{path:path}")
def static_file(path: str):
    """兼容 <link href='/static/...'> 的直接访问。"""
    file_path = STATIC_DIR / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(file_path)


# ---------- API：音色 / 语言 ----------

@app.get("/api/voices")
def api_voices() -> JSONResponse:
    return JSONResponse({"voices": all_voices()})


@app.get("/api/langs")
def api_langs() -> JSONResponse:
    return JSONResponse(
        {"langs": [{"code": k, "label": v} for k, v in LANG_LABELS.items()]}
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ts": int(time.time())}


# ---------- API：合成 ----------

def _do_synthesize(text: str, voice: str, speed: float, fmt: str):
    service: TTSService = get_service()
    try:
        audio_bytes, content_type, sample_rate = service.synthesize(
            text=text, voice=voice, speed=speed, output_format=fmt
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("synthesize failed")
        raise HTTPException(status_code=500, detail=f"synthesize error: {e}")

    ext = "mp3" if fmt.lower() == "mp3" else "wav"
    return audio_bytes, content_type, ext, sample_rate


@app.post("/api/tts")
def api_tts_post(req: TTSRequest) -> Response:
    audio_bytes, content_type, ext, sample_rate = _do_synthesize(
        req.text, req.voice, req.speed, req.format
    )

    # 写一份到 outputs/ 方便下载 / 二次使用
    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:6]}.{ext}"
    out_path = OUTPUT_DIR / fname
    out_path.write_bytes(audio_bytes)

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{fname}"',
            "X-Sample-Rate": str(sample_rate),
            "X-File-Name": fname,
        },
    )


@app.get("/api/tts")
def api_tts_get(
    text: str = Query(..., min_length=1, max_length=2000),
    voice: str = Query("zf_xiaoni"),
    speed: float = Query(1.0, ge=0.5, le=2.0),
    format: str = Query("wav", pattern="^(wav|mp3)$"),
    download: bool = Query(False),
) -> Response:
    """GET 接口：便于浏览器 <audio src='/api/tts?text=...&voice=...'> 直接播放。"""
    audio_bytes, content_type, ext, sample_rate = _do_synthesize(
        text, voice, speed, format
    )

    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:6]}.{ext}"
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{fname}"' if download
            else f'inline; filename="{fname}"'
        ),
        "X-Sample-Rate": str(sample_rate),
    }
    return Response(content=audio_bytes, media_type=content_type, headers=headers)


@app.get("/api/outputs/{fname}")
def download_output(fname: str):
    """下载/试听 outputs/ 目录里之前合成过的音频。"""
    # 防止路径穿越
    safe = (OUTPUT_DIR / fname).resolve()
    if OUTPUT_DIR.resolve() not in safe.parents and safe != OUTPUT_DIR.resolve():
        raise HTTPException(status_code=400, detail="invalid path")
    if not safe.is_file():
        raise HTTPException(status_code=404, detail="not found")

    ext = safe.suffix.lower()
    media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
    return FileResponse(safe, media_type=media_type, filename=safe.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )
