import os
import subprocess
import json
from paddlespeech.cli.tts import TTSExecutor

OUTPUT_DIR = "audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚠️ 这里仍然是“基础名”，不带 -en / -zh
ZH_AM  = "fastspeech2_csmsc"
ZH_VOC = "pwgan_csmsc"

EN_AM  = "fastspeech2_ljspeech"
EN_VOC = "pwgan_ljspeech"

JSON_FILE = "/Users/huhao/src/codesnip/python/english_word/scripts/8-1.json"

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


# 读取JSON文件
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ---------- 英文音频生成循环 ----------
print("开始生成英文音频...")
# 重新初始化TTS执行器
tts = TTSExecutor()
for unit, items in data.items():
    print(f"处理 {unit} 的英文...")
    for item in items:
        if 'word' in item:
            en = item['word']
        elif 'phrase' in item:
            en = item['phrase']
        else:
            continue
        
        # 标准化文件名
        name = normalize_filename(en)
        
        try:
            # 英文
            en_wav = os.path.join(OUTPUT_DIR, f"{name}_en.wav")
            en_mp3 = os.path.join(OUTPUT_DIR, f"{name}_en.mp3")
            
            tts(
                text=en,
                output=en_wav,
                am=EN_AM,
                voc=EN_VOC,
                lang="en"
            )
            wav_to_mp3(en_wav, en_mp3)
            os.remove(en_wav)
            
            print(f"✓ 英文生成完成: {en}")
        except Exception as e:
            print(f"✗ 英文生成失败: {en}, 错误: {str(e)}")

# ---------- 中文音频生成循环 ----------
print("\n开始生成中文音频...")
# 重新初始化TTS执行器
tts = TTSExecutor()
for unit, items in data.items():
    print(f"处理 {unit} 的中文...")
    for item in items:
        if 'word' in item or 'phrase' in item:
            cn = item['chinese']
            if 'word' in item:
                en = item['word']
            else:
                en = item['phrase']
        else:
            continue
        
        # 标准化文件名
        name = normalize_filename(en)
        
        try:
            # 中文
            cn_wav = os.path.join(OUTPUT_DIR, f"{name}_cn.wav")
            cn_mp3 = os.path.join(OUTPUT_DIR, f"{name}_cn.mp3")
            
            tts(
                text=cn,
                output=cn_wav,
                am=ZH_AM,
                voc=ZH_VOC,
                lang="zh"
            )
            wav_to_mp3(cn_wav, cn_mp3)
            os.remove(cn_wav)
            
            print(f"✓ 中文生成完成: {cn}")
        except Exception as e:
            print(f"✗ 中文生成失败: {cn}, 错误: {str(e)}")

print("\n所有音频生成完成！")
