import os
import subprocess
from paddlespeech.cli.tts import TTSExecutor

OUTPUT_DIR = "audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ⚠️ 这里仍然是“基础名”，不带 -en / -zh
ZH_AM  = "fastspeech2_csmsc"
ZH_VOC = "pwgan_csmsc"

EN_AM  = "fastspeech2_ljspeech"
EN_VOC = "pwgan_ljspeech"

words = [
    ['amusement', '娱乐；游戏'],
    ['amusement park', '游乐场'],
    ['apple', '苹果'],
]


def normalize_filename(text: str) -> str:
    return text.strip().replace(" ", "-").lower()


def wav_to_mp3(wav_path: str, mp3_path: str):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-codec:a", "libmp3lame",
            "-b:a", "128k",
            mp3_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


for en, cn in words:
    name = normalize_filename(en)

    # ---------- 英文 ----------
    en_wav = os.path.join(OUTPUT_DIR, f"{name}_en.wav")
    en_mp3 = os.path.join(OUTPUT_DIR, f"{name}_en.mp3")

    tts = TTSExecutor()
    tts(
        text=en,
        output=en_wav,
        am=EN_AM,
        voc=EN_VOC,
        lang="en"          # ⭐⭐⭐ 关键修复点
    )
    wav_to_mp3(en_wav, en_mp3)
    os.remove(en_wav)

    # ---------- 中文 ----------
    cn_wav = os.path.join(OUTPUT_DIR, f"{name}_cn.wav")
    cn_mp3 = os.path.join(OUTPUT_DIR, f"{name}_cn.mp3")

    tts = TTSExecutor()
    tts(
        text=cn,
        output=cn_wav,
        am=ZH_AM,
        voc=ZH_VOC,
        lang="zh",         # ⭐⭐⭐ 关键修复点
    )
    wav_to_mp3(cn_wav, cn_mp3)
    os.remove(cn_wav)

    print(f"✓ 生成完成: {name}")
