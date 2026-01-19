import asyncio
import numpy as np
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import whisperx
from faster_whisper import WhisperModel

# ================== 配置 ==================
TEXT = "Learning English requires daily practice and patience."
VOICE = "en-US-AriaNeural"

AUDIO_FILE = "audio.wav"
VIDEO_FILE = "output.mp4"

WIDTH, HEIGHT = 1280, 720
FONT_SIZE = 60
BG_COLOR = (30, 30, 30)
NORMAL_COLOR = (180, 180, 180)
HIGHLIGHT_COLOR = (255, 215, 0)
FONT_PATH = "/System/Library/Fonts/SFNS.ttf"



DEVICE = "cpu"
# ==========================================


# ---------- 1. 生成语音 ----------
async def generate_audio():
    print("▶ Generating audio...")
    tts = edge_tts.Communicate(TEXT, VOICE)
    await tts.save(AUDIO_FILE)


# ---------- 2. 转写 + 对齐（无 VAD） ----------
def align_words():
    print("▶ Aligning words with WhisperX (first time may be slow)...")

    # 2.1 使用 faster-whisper（无 VAD）
    whisper_model = WhisperModel(
        "base",
        device=DEVICE,
        compute_type="float32"
    )

    segments, _ = whisper_model.transcribe(AUDIO_FILE, language="en")

    whisper_segments = []
    for seg in segments:
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    # 2.2 WhisperX 对齐
    audio = whisperx.load_audio(AUDIO_FILE)

    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device=DEVICE
    )

    aligned = whisperx.align(
        whisper_segments,
        align_model,
        metadata,
        audio,
        device=DEVICE
    )

    words = []
    for seg in aligned["segments"]:
        for w in seg["words"]:
            if w.get("start") is not None:
                words.append((w["word"], w["start"], w["end"]))

    return words


# ---------- 3. 视频帧 ----------
def make_frame(t, words):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    LEFT_MARGIN = 100
    RIGHT_MARGIN = WIDTH - 100
    MAX_WIDTH = RIGHT_MARGIN
    LINE_HEIGHT = FONT_SIZE + 20
    space = 20

    x = LEFT_MARGIN
    y = HEIGHT // 2

    for word, start, end in words:
        color = HIGHLIGHT_COLOR if start <= t <= end else NORMAL_COLOR

        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]

        # ⭐ 自动换行
        if x + w > MAX_WIDTH:
            x = LEFT_MARGIN
            y += LINE_HEIGHT

        draw.text((x, y), word, font=font, fill=color)
        x += w + space

    return np.array(img)




# ---------- 4. 生成视频 ----------
def generate_video(words):
    audio = AudioFileClip(AUDIO_FILE)

    clip = VideoClip(
        lambda t: make_frame(t, words),
        duration=audio.duration
    )

    clip = clip.set_audio(audio)
    clip.write_videofile(
        VIDEO_FILE,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        audio_fps=44100
    )



# ---------- 主流程 ----------
async def main():
    await generate_audio()
    words = align_words()
    print("Aligned words:", words)
    generate_video(words)


if __name__ == "__main__":
    asyncio.run(main())
