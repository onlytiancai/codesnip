import asyncio
import numpy as np
import os
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import whisperx
from faster_whisper import WhisperModel

# ================== 配置 ==================
TEXT = """
The software industry sits at a strange inflection point. AI coding has evolved from autocomplete on steroids to agents that can autonomously execute development tasks. The economic boom that fueled tech’s hiring spree has given way to an efficiency mandate: companies now often favor profitability over growth, experienced hires over fresh graduates, and smaller teams armed with better tools.

Meanwhile, a new generation of developers is entering the workforce with a different calculus: pragmatic about career stability, skeptical of hustle culture, and raised on AI assistance from day one.

What happens next is genuinely uncertain. Below are five critical questions that may shape software engineering through 2026, with two contrasting scenarios for each. These aren’t really predictions, but lenses for preparation. The goal is a clear roadmap for handling what comes next, grounded in current data and tempered by the healthy skepticism this community is known for.

"""
VOICE = "en-US-AriaNeural"

AUDIO_FILE = "audio.wav"
VIDEO_FILE = "output.mp4"

WIDTH, HEIGHT = 720, 1280
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
# 用于平滑滚动的全局变量
current_scroll_offset = 0

def make_frame(t, words):
    # 声明全局变量
    global current_scroll_offset
    
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 竖版视频的边距设置
    LEFT_MARGIN = 50
    RIGHT_MARGIN = WIDTH - 50
    TOP_MARGIN = 100
    LINE_HEIGHT = FONT_SIZE + 16
    space = 18

    # 第一步：计算所有词的位置
    word_positions = []  # 存储每个词的 (x, y) 和行信息
    line_words = []     # 当前行的所有词
    x = LEFT_MARGIN
    y = TOP_MARGIN
    current_line = 0

    for word, start, end in words:
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]

        # 检查是否需要换行
        if x + w > RIGHT_MARGIN:
            # 当前行已满，换行
            word_positions.extend(line_words)
            line_words = []
            current_line += 1
            x = LEFT_MARGIN
            y += LINE_HEIGHT

        # 添加当前词到当前行
        line_words.append({
            'word': word,
            'start': start,
            'end': end,
            'x': x,
            'y': y,
            'width': w,
            'line': current_line
        })
        x += w + space

    # 添加最后一行的词
    if line_words:
        word_positions.extend(line_words)

    # 第二步：找到当前高亮的词和它的位置
    highlighted_word_idx = -1
    
    for i, word_info in enumerate(word_positions):
        if word_info['start'] <= t <= word_info['end']:
            highlighted_word_idx = i
            break
    
    # 第三步：实现滚动逻辑
    if highlighted_word_idx >= 0 and len(word_positions) > 0:
        # 获取当前高亮词和它的位置
        highlighted_word = word_positions[highlighted_word_idx]
        highlighted_line = highlighted_word['line']
        highlighted_y = highlighted_word['y']
        
        # 计算屏幕能显示的行数
        visible_lines = int((HEIGHT - TOP_MARGIN - 100) / LINE_HEIGHT)  # 减去底部边距
        middle_line = visible_lines // 2
        
        # 计算总共有多少行
        total_lines = max(w['line'] for w in word_positions) + 1
        
        # 计算最大可滚动距离（最后一行显示在屏幕底部时的偏移量）
        max_scroll_distance = max(0, (total_lines - visible_lines) * LINE_HEIGHT)
        
        # 计算目标滚动偏移量
        target_offset = 0
        
        # 如果一屏幕能放下所有文字，不需要滚屏
        if total_lines <= visible_lines:
            target_offset = 0
        else:
            # 当高亮文本到中间行的时候，屏幕向上滚动一行
            # 计算当前应该滚动的行数
            scroll_lines = max(0, highlighted_line - middle_line)
            
            # 计算对应的滚动偏移量
            target_offset = scroll_lines * LINE_HEIGHT
            
            # 确保滚动不超过最大可滚动距离（最后一行已经显示在屏幕上时停止滚动）
            target_offset = min(target_offset, max_scroll_distance)
        
        # 确保高亮词始终显示在屏幕上
        actual_highlight_y = highlighted_y - target_offset
        
        # 如果高亮词在屏幕外，调整滚动偏移量
        if actual_highlight_y < TOP_MARGIN:
            target_offset = highlighted_y - TOP_MARGIN
        elif actual_highlight_y > HEIGHT - 100:  # 确保在底部边距内
            target_offset = highlighted_y - (HEIGHT - 100)
        
        # 平滑滚动：使用插值让当前偏移量逐渐接近目标偏移量
        # 平滑系数控制滚动速度，值越小越平滑（0-1之间）
        smooth_factor = 0.1
        current_scroll_offset += (target_offset - current_scroll_offset) * smooth_factor
        
        # 确保滚动偏移量在有效范围内
        current_scroll_offset = max(0, min(current_scroll_offset, max_scroll_distance))
        
        # 使用平滑后的滚动偏移量
        scroll_offset = current_scroll_offset
    else:
        # 当前时间没有对应的高亮单词，可能是在句子之间的停顿
        # 保持当前的滚动偏移量不变
        scroll_offset = current_scroll_offset
    
    # 第四步：绘制所有词
    for word_info in word_positions:
        # 确定颜色
        if word_info['start'] <= t <= word_info['end']:
            color = HIGHLIGHT_COLOR
        else:
            color = NORMAL_COLOR

        # 应用滚动偏移（保持浮点数精度）
        draw_y = word_info['y'] - scroll_offset

        # 扩大渲染范围，避免边缘闪烁
        if -LINE_HEIGHT * 4 <= draw_y <= HEIGHT + LINE_HEIGHT * 3:
            # 使用更平滑的文本渲染
            draw.text(
                (word_info['x'], draw_y), 
                word_info['word'], 
                font=font, 
                fill=color,
                stroke_width=0  # 确保没有额外描边导致闪烁
            )

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
    # 检查音频文件是否已存在
    if os.path.exists(AUDIO_FILE):
        print(f"▶ Audio file {AUDIO_FILE} already exists, skipping TTS generation...")
    else:
        await generate_audio()
    
    words = align_words()
    print("Aligned words:", words)
    generate_video(words)


if __name__ == "__main__":
    asyncio.run(main())
