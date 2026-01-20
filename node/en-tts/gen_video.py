import asyncio
import numpy as np
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
FONT_SIZE = 32
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
    
    # 第三步：计算滚动偏移量
    scroll_offset = 0
    
    if highlighted_word_idx >= 0 and len(word_positions) > 0:
        # 计算当前高亮词所在行的中心位置
        highlighted_word = word_positions[highlighted_word_idx]
        current_line = highlighted_word['line']
        
        # 获取当前行的所有词，计算行的平均y坐标
        line_words_current = [w for w in word_positions if w['line'] == current_line]
        line_center_y = sum(w['y'] for w in line_words_current) / len(line_words_current)
        
        # 目标是将当前行放在屏幕中央
        target_y = HEIGHT // 2 - LINE_HEIGHT // 2
        desired_scroll_offset = line_center_y - target_y
        
        # 计算最大可滚动距离（文本总高度 - 屏幕高度 + 底部边距）
        total_text_height = max(w['y'] for w in word_positions) + LINE_HEIGHT
        max_scroll_offset = max(0, total_text_height - HEIGHT + 100)  # 100px底部边距
        
        # 如果已经滚动到底部，不再向上滚动
        current_scroll_offset = scroll_offset  # 保存当前滚动位置
        if current_scroll_offset >= max_scroll_offset:
            # 已经在底部，停止滚动
            scroll_offset = current_scroll_offset
        else:
            # 正常滚动，但限制不超过最大滚动距离
            scroll_offset = min(desired_scroll_offset, max_scroll_offset)
        
        # 应用顶部边界限制
        max_scroll_up = line_center_y - TOP_MARGIN
        if scroll_offset > max_scroll_up:
            scroll_offset = max_scroll_up
        
        # 确保不会滚动到顶部以下
        if scroll_offset < 0:
            scroll_offset = 0
        
        # 添加行之间的平滑过渡（当切换到新行时）
        if highlighted_word_idx > 0:
            prev_word = word_positions[highlighted_word_idx - 1]
            if prev_word['line'] < current_line:
                # 计算前一行和当前行之间的过渡
                prev_line_words = [w for w in word_positions if w['line'] == current_line - 1]
                if prev_line_words:
                    prev_line_center_y = sum(w['y'] for w in prev_line_words) / len(prev_line_words)
                    prev_desired_scroll_offset = prev_line_center_y - target_y
                    
                    # 限制前一行滚动也不超过最大滚动距离
                    prev_scroll_offset = min(prev_desired_scroll_offset, max_scroll_offset)
                    
                    # 基于时间在两个偏移量之间进行插值
                    # 过渡窗口：从上一个词结束到当前词开始
                    transition_start = prev_word['end']
                    transition_end = highlighted_word['start']
                    transition_duration = transition_end - transition_start
                    
                    if transition_duration > 0 and t >= transition_start:
                        # 计算插值因子（0到1）
                        if t <= transition_end:
                            transition_progress = (t - transition_start) / transition_duration
                        else:
                            transition_progress = 1.0
                        
                        # 使用缓动函数（ease-in-out）
                        if transition_progress < 0.5:
                            eased_progress = 2 * transition_progress * transition_progress
                        else:
                            eased_progress = 1 - 2 * (1 - transition_progress) * (1 - transition_progress)
                        
                        # 插值计算
                        scroll_offset = prev_scroll_offset + eased_progress * (scroll_offset - prev_scroll_offset)
    
    # 第四步：绘制所有词
    for word_info in word_positions:
        # 确定颜色
        if word_info['start'] <= t <= word_info['end']:
            color = HIGHLIGHT_COLOR
        else:
            color = NORMAL_COLOR

        # 应用滚动偏移
        draw_y = word_info['y'] - scroll_offset

        # 扩大渲染范围，避免边缘闪烁
        if -LINE_HEIGHT * 3 <= draw_y <= HEIGHT + LINE_HEIGHT * 2:
            draw.text((word_info['x'], draw_y), word_info['word'], font=font, fill=color)

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
