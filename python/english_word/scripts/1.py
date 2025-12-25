from paddlespeech.cli.tts import TTSExecutor
from pydub import AudioSegment
import os

tts = TTSExecutor()

text = "苹果，香蕉，哈密瓜"
parts = [p.strip() for p in text.split("，")]

segments = []
silence = AudioSegment.silent(duration=1000)  # 1 秒停顿

for i, part in enumerate(parts):
    wav_path = f"part_{i}.wav"

    tts(
        text=part,
        output=wav_path,
        am="fastspeech2_csmsc",
        voc="pwgan_csmsc"
    )

    audio = AudioSegment.from_wav(wav_path)
    segments.append(audio)

# 拼接：每段后加 1 秒静音（最后一段不加）
final_audio = AudioSegment.empty()
for i, seg in enumerate(segments):
    final_audio += seg
    if i != len(segments) - 1:
        final_audio += silence

final_audio.export("apple_zh.wav", format="wav")

# 可选：清理中间文件
for i in range(len(parts)):
    os.remove(f"part_{i}.wav")
