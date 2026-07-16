#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_design_demo.py — 设计一个新音色，再用它朗读一段文本。

流程：
  1. POST /v1/voice_design        设计音色 → 拿到 voice_id + 试听音频
  2. POST /v1/t2a_v2              用该 voice_id 朗读任意文本 → mp3

依赖：仅 Python 3 标准库（urllib, json, os）。
环境变量：MINIMAX_API_KEY（必填）。
"""

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request

# ---- 端点 ----
DESIGN_URL = "https://api.minimaxi.com/v1/voice_design"
TTS_URL    = "https://api.minimaxi.com/v1/t2a_v2"

# ---- 设计参数（按需修改） ----
VOICE_PROMPT = (
    "一位中年男性的恐怖故事讲述者，声音低沉浑厚略带沙哑，"
    "语速缓慢阴森，咬字清晰，善用停顿营造诡异悬疑的气氛，"
    "仿佛在深夜的篝火旁对听众娓娓道来，带有压迫感和神秘感。"
)

PREVIEW_TEXT = (
    "那是民国年间一个深秋的夜晚，村里接连死了三头牛，"
    "脖子上的伤口整整齐齐，却找不到半点血迹。"
    "我爹说，那东西，又回来了。"
)

# ---- 朗读文本 ----
TARGET_TEXT = (
    "你们别笑，那事儿真真切切发生在我太爷爷身上。"
    "那年腊月二十九，他一个人走夜路经过乱葬岗，"
    "忽然听见背后有人叫他名字，回头一看，"
    "月亮底下空空荡荡，连个鬼影子都没有。"
    "可那声音清清楚楚，分明就是他死去三年的娘。"
)

OUTPUT_DIR = "output"
PREVIEW_MP3 = os.path.join(OUTPUT_DIR, "voice_design_preview.mp3")
TARGET_MP3   = os.path.join(OUTPUT_DIR, "voice_design_reading.mp3")
META_JSON    = os.path.join(OUTPUT_DIR, "voice_design_meta.json")


def _bearer_headers():
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        sys.exit(
            "环境变量 MINIMAX_API_KEY 未设置。\n"
            "  export MINIMAX_API_KEY=...\n"
            "若网络不通可挂代理：export HTTPS_PROXY=http://127.0.0.1:10808"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }


def _post(url, payload, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_bearer_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 {url}（{e.reason}）。"
            f"若需代理：export HTTPS_PROXY=http://127.0.0.1:10808"
        ) from e


def _decode_audio(s):
    """嗅探 hex / base64 编码的音频字符串，返回 bytes。"""
    if re.fullmatch(r"[0-9a-fA-F]+", s) and len(s) % 2 == 0:
        return bytes.fromhex(s)
    return base64.b64decode(s)


def _check_base_resp(body, where):
    base = body.get("base_resp", {}) or {}
    if base.get("status_code", 0) != 0:
        raise RuntimeError(
            f"[{where}] 返回错误：{base.get('status_msg')} "
            f"(code={base.get('status_code')})"
        )


def design_voice(prompt, preview_text, voice_id=None):
    """调用 /v1/voice_design，返回 (voice_id, trial_audio_bytes)。"""
    if len(preview_text) > 500:
        preview_text = preview_text[:500]
    payload = {
        "prompt":       prompt,
        "preview_text": preview_text,
    }
    if voice_id:
        payload["voice_id"] = voice_id

    print(f"[design] POST {DESIGN_URL}")
    print(f"  prompt      : {prompt}")
    print(f"  preview_text: {preview_text[:60]}…")
    body = _post(DESIGN_URL, payload)
    _check_base_resp(body, "design")

    new_voice_id = body.get("voice_id")
    trial_hex    = body.get("trial_audio", "")
    if not new_voice_id or not trial_hex:
        raise RuntimeError(f"响应缺少 voice_id / trial_audio：{body}")

    trial_bytes = _decode_audio(trial_hex)
    print(f"  → voice_id        = {new_voice_id}")
    print(f"  → trial_audio len = {len(trial_bytes)} bytes")
    return new_voice_id, trial_bytes


def tts_speak(text, voice_id, model="speech-02-hd",
              emotion="calm", vol=1.0, speed=1.0, pitch=0,
              audio_format="mp3"):
    """调用 /v1/t2a_v2，返回音频 bytes。"""
    payload = {
        "model":          model,
        "text":           text,
        "stream":         False,
        "language_boost": "Chinese",
        "voice_setting": {
            "voice_id": voice_id,
            "speed":    max(0.5, min(2.0, float(speed))),
            "vol":      max(0.0, min(10.0, float(vol))),
            "pitch":    max(-12, min(12, int(pitch))),
            "emotion":  emotion,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate":      128000,
            "format":       audio_format,
            "channel":      1,
        },
    }
    print(f"[tts] POST {TTS_URL}")
    print(f"  voice_id = {voice_id}  emotion = {emotion}")
    print(f"  text     = {text[:60]}…")
    body = _post(TTS_URL, payload)
    _check_base_resp(body, "tts")

    data  = body.get("data", {}) or {}
    extra = body.get("extra_info", {}) or {}
    audio_str = data.get("audio")
    if not audio_str:
        raise RuntimeError(f"响应缺少 data.audio：{body}")

    audio_bytes = _decode_audio(audio_str)
    print(f"  → audio_length = {extra.get('audio_length', 0)} ms")
    print(f"  → size         = {len(audio_bytes)} bytes")
    return audio_bytes, extra


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 设计音色
    voice_id, preview_bytes = design_voice(VOICE_PROMPT, PREVIEW_TEXT)
    with open(PREVIEW_MP3, "wb") as f:
        f.write(preview_bytes)

    # 2. 用该音色朗读目标文本
    target_bytes, extra = tts_speak(
        TARGET_TEXT, voice_id,
        emotion="calm", vol=1.0, speed=1.0, pitch=0,
    )
    with open(TARGET_MP3, "wb") as f:
        f.write(target_bytes)

    # 3. 落盘元数据
    meta = {
        "voice_id":     voice_id,
        "prompt":       VOICE_PROMPT,
        "preview_text": PREVIEW_TEXT,
        "target_text":  TARGET_TEXT,
        "preview_file": os.path.basename(PREVIEW_MP3),
        "target_file":  os.path.basename(TARGET_MP3),
        "preview_size_bytes": len(preview_bytes),
        "target_size_bytes":  len(target_bytes),
        "target_audio_length_ms": extra.get("audio_length", 0),
        "usage_characters":     extra.get("usage_characters", 0),
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print(f"✓ 试听音频   → {PREVIEW_MP3}")
    print(f"✓ 朗读音频   → {TARGET_MP3}")
    print(f"✓ 元数据     → {META_JSON}")
    print(f"✓ 音色 ID    : {voice_id}")


if __name__ == "__main__":
    main()