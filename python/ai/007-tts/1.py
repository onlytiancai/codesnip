from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(
    lang_code="z",
    repo_id="hexgrad/Kokoro-82M"
)

text = "今天我们来聊一个有趣的话题。"

generator = pipeline(
    text,
    voice="zf_xiaoni"
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"output_{i}.wav", audio, 24000)

print("done")