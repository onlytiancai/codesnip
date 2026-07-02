#!/usr/bin/env python3
"""
XOR 反向传播视频 · TTS 音频生成脚本。

读 project root 的 desc.json, 对每张卡的 tts_segments 调 MiniMax TTS,
把 mp3 写到 video/public/audio/, 同时回填到 desc.json:

  - tts_segments[i].audio_path
  - tts_segments[i].duration_ms
  - card.duration_sec   (= MAX(4s, ceil(总时长 / 1000) + 1))
  - 顶层 duration_sec / duration_frames

用法:
  export MINIMAX_API_KEY=...
  python pipeline/render_audio.py [--force]

协议: https://platform.minimaxi.com/docs/api-reference/speech-t2a-http
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent          # /013/
PIPELINE = ROOT / "pipeline"                           # /013/pipeline/
AUDIO_DIR = ROOT / "video" / "public" / "audio"        # /013/video/public/audio/
DESC_PATH = ROOT / "desc.json"

TTS_ENDPOINT = "https://api.minimaxi.com/v1/t2a_v2"
VOICE_ZH = "female-shaonv"     # 中文女声
MODEL = "speech-02-hd"

# 每张卡最小 4s, +1s padding(让音频自然衰减后还显示一会图片)
MIN_DURATION_SEC = 4
PADDING_SEC = 1
PAUSE_MS = 0   # 单语种(全中文),不需要段间停顿


def synthesize(text: str, voice_id: str = VOICE_ZH) -> tuple[bytes, int, int]:
    """调 MiniMax TTS,返回 (audio_bytes, duration_ms, sample_rate)。"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise SystemExit("❌ MINIMAX_API_KEY 未设置")

    body = json.dumps({
        "model": MODEL,
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        TTS_ENDPOINT, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    data = json.loads(raw)
    if data.get("base_resp", {}).get("status_code") != 0:
        raise SystemExit(f"❌ TTS 错误: {data.get('base_resp')}")
    audio_hex = data["data"]["audio"]
    # 嗅探 hex vs base64
    if all(c in "0123456789abcdefABCDEF" for c in audio_hex) and len(audio_hex) % 2 == 0:
        audio_bytes = bytes.fromhex(audio_hex)
    else:
        audio_bytes = __import__("base64").b64decode(audio_hex)
    extra = data.get("extra_info", {})
    # ⚠️ MiniMax extra_info.audio_length 单位是毫秒(整数),不要再 ×1000
    duration_ms = int(extra.get("audio_length", 0)) or 0
    sample_rate = int(extra.get("audio_sample_rate", 32000))
    if duration_ms == 0:
        # 兜底: 按字节估算时长(128kbps mp3)
        duration_ms = int(len(audio_bytes) * 8 / 128000 * 1000)
    return audio_bytes, duration_ms, sample_rate


def write_mp3(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def measure_mp3_ms(path: Path, fallback_text_len: int) -> int:
    """优先 ffprobe,其次按 4 字/秒 保守估算。"""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return int(float(out) * 1000)
    except Exception:
        return fallback_text_len * 250


def process(args: argparse.Namespace) -> None:
    if not DESC_PATH.exists():
        raise SystemExit(f"❌ desc.json 不存在: {DESC_PATH}")
    desc: dict[str, Any] = json.loads(DESC_PATH.read_text("utf-8"))
    fps = int(desc["fps"])

    print("=" * 60)
    print(f"🎤 TTS 阶段 · video {desc.get('id', '?')} · {len(desc['cards'])} 张卡")
    print("=" * 60)

    for card in desc["cards"]:
        print(f"\n   📌 card {card['index']} ({card['type']})")
        total_ms = 0
        audio_id = desc.get("id", "xor-bp")
        for i, seg in enumerate(card["tts_segments"]):
            out_name = f"{audio_id}-{card['index']}-{i}-{seg['lang']}.mp3"
            out_path = AUDIO_DIR / out_name
            # ⚠️ Remotion staticFile() 不要 public/ 前缀
            audio_relative = f"audio/{out_name}"

            if not args.force and out_path.exists():
                ms = measure_mp3_ms(out_path, len(seg["text"]))
                seg["audio_path"] = audio_relative
                seg["duration_ms"] = ms
                total_ms += ms
                print(f"      ↻ {out_name} · {ms / 1000:.2f}s")
                continue

            print(f"      🎤 seg {i} ({seg['lang']}): \"{seg['text'][:30]}...\"")
            audio_bytes, ms, sr = synthesize(seg["text"])
            write_mp3(out_path, audio_bytes)
            seg["audio_path"] = audio_relative
            seg["duration_ms"] = ms
            total_ms += ms
            print(f"         ✓ {len(audio_bytes) / 1024:.1f} KB · {ms / 1000:.2f}s @ {sr}Hz")

        pause_total_ms = max(0, len(card["tts_segments"]) - 1) * PAUSE_MS
        dur = max(MIN_DURATION_SEC,
                  (total_ms + pause_total_ms + 999) // 1000 + PADDING_SEC)
        card["duration_sec"] = dur
        print(f"      → duration_sec = {dur}s")

    # 顶层累加
    desc["duration_frames"] = sum(
        int(round(c["duration_sec"] * fps)) for c in desc["cards"]
    )
    desc["duration_sec"] = desc["duration_frames"] / fps
    print("\n   📐 总时长: "
          f"{desc['duration_sec']:.2f}s · {desc['duration_frames']} 帧 @ {fps}fps")

    DESC_PATH.write_text(json.dumps(desc, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print("✅ 回写 desc.json 完成")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XOR 视频 TTS 生成")
    p.add_argument("--force", action="store_true",
                   help="强制重新生成所有音频")
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
