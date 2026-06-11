"""
TTS 服务封装：基于 Kokoro-82M 提供多语言、多音色的文本转语音能力。

- 内部缓存多个 lang_code 对应的 KPipeline，避免重复加载。
- 提供同步合成方法 `synthesize`，返回 (wav_bytes, sample_rate) 供 FastAPI 流式返回或落盘。
- 可选调用 ffmpeg 将 WAV 转码为 MP3（环境无 ffmpeg 时回退到 WAV）。
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceInfo:
    """音色元信息（用于前端下拉展示）。"""

    name: str
    lang_code: str
    gender: str  # "F" / "M"
    quality: str  # A / B / C / D / etc
    label: str  # 中文标签


# 精选音色清单（按 lang_code 分组）。
# 数据来源：Kokoro-82M 官方 VOICES.md。
VOICE_LIBRARY: dict[str, list[VoiceInfo]] = {
    "z": [
        VoiceInfo("zf_xiaobei",   "z", "F", "C", "小蓓 (女)"),
        VoiceInfo("zf_xiaoni",    "z", "F", "C", "小妮 (女)"),
        VoiceInfo("zf_xiaoxiao",  "z", "F", "C", "小晓 (女)"),
        VoiceInfo("zf_xiaoyi",    "z", "F", "C", "小怡 (女)"),
        VoiceInfo("zm_yunjian",   "z", "M", "C", "云健 (男)"),
        VoiceInfo("zm_yunxi",     "z", "M", "C", "云溪 (男)"),
        VoiceInfo("zm_yunxia",    "z", "M", "C", "云霞 (男)"),
        VoiceInfo("zm_yunyang",   "z", "M", "C", "云扬 (男)"),
    ],
    "a": [
        VoiceInfo("af_heart",   "a", "F", "A", "Heart (US-F)"),
        VoiceInfo("af_bella",   "a", "F", "A", "Bella (US-F)"),
        VoiceInfo("af_nicole",  "a", "F", "B", "Nicole (US-F)"),
        VoiceInfo("am_adam",    "a", "M", "D", "Adam (US-M)"),
        VoiceInfo("am_michael", "a", "M", "B", "Michael (US-M)"),
        VoiceInfo("am_puck",    "a", "M", "B", "Puck (US-M)"),
    ],
    "b": [
        VoiceInfo("bf_emma",    "b", "F", "B", "Emma (UK-F)"),
        VoiceInfo("bf_isabella","b", "F", "B", "Isabella (UK-F)"),
        VoiceInfo("bm_george",  "b", "M", "B", "George (UK-M)"),
        VoiceInfo("bm_lewis",   "b", "M", "C", "Lewis (UK-M)"),
    ],
    "j": [
        VoiceInfo("jf_alpha",       "j", "F", "B", "Alpha (JP-F)"),
        VoiceInfo("jf_gongitsune",  "j", "F", "B", "Gongitsune (JP-F)"),
        VoiceInfo("jm_kumo",        "j", "M", "C", "Kumo (JP-M)"),
    ],
    "e": [
        VoiceInfo("ef_dora",  "e", "F", "", "Dora (ES-F)"),
        VoiceInfo("em_alex",  "e", "M", "", "Alex (ES-M)"),
        VoiceInfo("em_santa", "e", "M", "", "Santa (ES-M)"),
    ],
    "f": [
        VoiceInfo("ff_siwis", "f", "F", "B", "Siwis (FR-F)"),
    ],
    "h": [
        VoiceInfo("hf_alpha", "h", "F", "B", "Alpha (HI-F)"),
        VoiceInfo("hm_omega", "h", "M", "B", "Omega (HI-M)"),
    ],
    "i": [
        VoiceInfo("if_sara",   "i", "F", "B", "Sara (IT-F)"),
        VoiceInfo("im_nicola", "i", "M", "B", "Nicola (IT-M)"),
    ],
    "p": [
        VoiceInfo("pf_dora",  "p", "F", "", "Dora (PT-F)"),
        VoiceInfo("pm_alex",  "p", "M", "", "Alex (PT-M)"),
        VoiceInfo("pm_santa", "p", "M", "", "Santa (PT-M)"),
    ],
}

LANG_LABELS: dict[str, str] = {
    "z": "🇨🇳 中文 Mandarin",
    "a": "🇺🇸 美式英语 American English",
    "b": "🇬🇧 英式英语 British English",
    "j": "🇯🇵 日语 Japanese",
    "e": "🇪🇸 西班牙语 Spanish",
    "f": "🇫🇷 法语 French",
    "h": "🇮🇳 印地语 Hindi",
    "i": "🇮🇹 意大利语 Italian",
    "p": "🇧🇷 巴西葡语 Brazilian Portuguese",
}


def all_voices() -> list[dict]:
    """展平 VOICE_LIBRARY，供前端 API 返回。"""
    out: list[dict] = []
    for lang_code, voices in VOICE_LIBRARY.items():
        for v in voices:
            out.append(
                {
                    "name": v.name,
                    "lang_code": v.lang_code,
                    "gender": v.gender,
                    "quality": v.quality,
                    "label": v.label,
                    "lang_label": LANG_LABELS.get(v.lang_code, v.lang_code),
                }
            )
    return out


def get_lang_code(voice: str) -> str:
    """根据 voice 名推断 lang_code（约定：voice 前两位是 lang+gender）。"""
    if not voice:
        return "z"
    prefix = voice[:2]
    # 形如 zf_xxx, zm_xxx, af_xxx, am_xxx, bf_xxx, bm_xxx, jf_xxx, jm_xxx, ...
    return prefix[0] if prefix[0] in VOICE_LIBRARY else "z"


class TTSService:
    """单例 TTS 服务：按 lang_code 懒加载并缓存 KPipeline。"""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self._lock = Lock()

    def _get_pipeline(self, lang_code: str):
        from kokoro import KPipeline  # 延迟导入，避免启动时下载

        with self._lock:
            if lang_code not in self._pipelines:
                logger.info("Loading KPipeline for lang_code=%s", lang_code)
                self._pipelines[lang_code] = KPipeline(
                    lang_code=lang_code,
                    repo_id="hexgrad/Kokoro-82M",
                )
            return self._pipelines[lang_code]

    @staticmethod
    def _concat_wav(audio_segments: list[tuple], sample_rate: int) -> bytes:
        """把多段 (gs, ps, audio) 中的 audio 拼接成单个 WAV bytes。"""
        import numpy as np

        if not audio_segments:
            raise ValueError("no audio segments produced")

        arrays = [np.asarray(seg[2], dtype=np.float32) for seg in audio_segments]
        merged = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]

        # 限幅，避免削顶爆音
        peak = float(max(abs(merged.max()), abs(merged.min()), 1e-9))
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        pcm16 = (merged * 32767.0).clip(-32768, 32767).astype("int16")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def synthesize(
        self,
        text: str,
        voice: str = "zf_xiaoni",
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> tuple[bytes, str, int]:
        """
        同步合成语音。

        返回: (audio_bytes, content_type, sample_rate)
        """
        if not text or not text.strip():
            raise ValueError("text is empty")

        text = text.strip()
        lang_code = get_lang_code(voice)
        pipeline = self._get_pipeline(lang_code)

        logger.info(
            "Synthesize start: voice=%s lang=%s speed=%s chars=%d",
            voice, lang_code, speed, len(text),
        )

        # Kokoro 生成器是 lazy 的；先消费完再合并。
        segments: list[tuple] = []
        sample_rate = 24000
        for i, result in enumerate(pipeline(text, voice=voice, speed=speed)):
            # result: (gs, ps, audio) — audio 是 numpy 1-D float32
            segments.append(result)
            # 从首个非空段推断采样率
            audio = result[2]
            try:
                from kokoro import KPipeline  # noqa: F401
                # 直接尝试 numpy 属性
                if hasattr(audio, "dtype"):
                    pass
            except Exception:
                pass
        if not segments:
            raise RuntimeError("Kokoro produced no audio")

        wav_bytes = self._concat_wav(segments, sample_rate)

        if output_format.lower() == "mp3":
            mp3_bytes = self._wav_to_mp3(wav_bytes)
            return mp3_bytes, "audio/mpeg", sample_rate

        return wav_bytes, "audio/wav", sample_rate

    @staticmethod
    def _wav_to_mp3(wav_bytes: bytes) -> bytes:
        """用系统 ffmpeg 把 WAV 转 MP3（libmp3lame -q:a 2）。"""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH; cannot convert to mp3")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(wav_bytes)
            wav_path = wf.name
        mp3_path = wav_path.replace(".wav", ".mp3")
        try:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", wav_path,
                "-c:a", "libmp3lame", "-q:a", "2",
                mp3_path,
            ]
            subprocess.run(cmd, check=True)
            with open(mp3_path, "rb") as f:
                return f.read()
        finally:
            Path(wav_path).unlink(missing_ok=True)
            Path(mp3_path).unlink(missing_ok=True)


# 进程级单例
_service: Optional[TTSService] = None


def get_service() -> TTSService:
    global _service
    if _service is None:
        _service = TTSService()
    return _service
