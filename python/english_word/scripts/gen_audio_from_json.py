import os
import subprocess
import json
import sys
import re
from paddlespeech.cli.tts import TTSExecutor
from pydub import AudioSegment

OUTPUT_DIR = "/Users/huhao/src/codesnip/python/english_word/wawa-word-pod/public/audio/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚠️ 这里仍然是“基础名”，不带 -en / -zh
ZH_AM  = "fastspeech2_csmsc"
ZH_VOC = "pwgan_csmsc"

EN_AM  = "fastspeech2_ljspeech"
EN_VOC = "pwgan_ljspeech"

JSON_FILE = "/Users/huhao/src/codesnip/python/english_word/scripts/8-1.json"

def normalize_filename(text: str) -> str:
    # 替换所有非字母数字字符为横杠
    normalized = re.sub(r'[^a-zA-Z0-9]', '-', text.strip())
    # 移除连续的横杠
    normalized = re.sub(r'-+', '-', normalized)
    # 移除首尾的横杠
    normalized = normalized.strip('-')
    return normalized.lower()


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
        
        # 英文
        en_wav = os.path.join(OUTPUT_DIR, f"{name}_en.wav")
        en_mp3 = os.path.join(OUTPUT_DIR, f"{name}_en.mp3")
        
        # 检查mp3文件是否已存在，如果存在则跳过
        if os.path.exists(en_mp3):
            print(f"✓ 英文文件已存在，跳过: {en} -> {name}_en.mp3")
            continue
        
        try:
            # 处理英文文本，将非英文字符替换为空格
            tts_en_text = re.sub(r'[^a-zA-Z\s]', ' ', en)
            # 移除连续的空格
            tts_en_text = re.sub(r'\s+', ' ', tts_en_text).strip()
            
            tts(
                text=tts_en_text,
                output=en_wav,
                am=EN_AM,
                voc=EN_VOC,
                lang="en"
            )
            wav_to_mp3(en_wav, en_mp3)
            os.remove(en_wav)
            
            print(f"✓ 英文生成完成: {en} -> {name}_en.mp3")
        except Exception as e:
            print(f"✗ 英文生成失败: {en}, 错误: {str(e)}")
            sys.exit(1)

# ---------- 中文音频生成循环 ----------
print("\n开始生成中文音频...")
# 重新初始化TTS执行器
tts = TTSExecutor()
for unit, items in data.items():
    print(f"处理 {unit} 的中文...")
    for item in items:
        if 'word' in item:
            en = item['word']
            cn_parts = item['cn_mp3_txt']
        elif 'phrase' in item:
            en = item['phrase']
            cn_parts = item['cn_mp3_txt']
        else:
            continue
        
        # 标准化文件名
        name = normalize_filename(en)
        
        # 中文
        cn_wav = os.path.join(OUTPUT_DIR, f"{name}_cn.wav")
        cn_mp3 = os.path.join(OUTPUT_DIR, f"{name}_cn.mp3")
        
        # 检查mp3文件是否已存在，如果存在则跳过
        if os.path.exists(cn_mp3):
            print(f"✓ 中文文件已存在，跳过: {en} -> {name}_cn.mp3")
            continue
        
        try:
            segments = []
            silence = AudioSegment.silent(duration=500)  # 500毫秒停顿
            
            for i, part in enumerate(cn_parts):
                # 生成临时音频文件
                temp_wav = os.path.join(OUTPUT_DIR, f"temp_{name}_{i}.wav")
                
                tts(
                    text=part,
                    output=temp_wav,
                    am=ZH_AM,
                    voc=ZH_VOC,
                    lang="zh"
                )
                
                # 读取临时音频文件
                audio = AudioSegment.from_wav(temp_wav)
                segments.append(audio)
                
                # 删除临时音频文件
                os.remove(temp_wav)
            
            # 合并音频文件，在每个项之间添加500ms停顿
            final_audio = AudioSegment.empty()
            for i, seg in enumerate(segments):
                final_audio += seg
                if i != len(segments) - 1:
                    final_audio += silence
            
            # 导出合并后的音频文件
            final_audio.export(cn_wav, format="wav")
            
            # 转换为MP3格式
            wav_to_mp3(cn_wav, cn_mp3)
            os.remove(cn_wav)
            
            print(f"✓ 中文生成完成: {en} -> {name}_cn.mp3")
        except Exception as e:
            print(f"✗ 中文生成失败: {en}, 错误: {str(e)}")
            sys.exit(1)

print("\n所有音频生成完成！")
