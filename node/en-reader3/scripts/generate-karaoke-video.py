#!/usr/bin/env python3
"""
Generate karaoke video from word timestamps using PIL for text rendering
and ffmpeg for video encoding (bypasses FFmpeg's ass filter issue).
"""

import subprocess
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def parse_word_timing(filename):
    """Parse section-X-words.txt into list of (start, end, text) tuples."""
    words = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            text = parts[2]
            words.append((start, end, text))
    return words


def get_current_screen_words(words, current_time):
    """Get the 4-word screen for the current time."""
    # Find which screen we're on based on current_time
    screen_size = 4
    for i in range(0, len(words), screen_size):
        screen_words = words[i:i + screen_size]
        if screen_words:
            screen_start = screen_words[0][0]
            screen_end = screen_words[-1][1]
            if screen_start <= current_time <= screen_end + 0.3:
                return screen_words, screen_start, screen_end
    return None, 0, 0


def render_frame(draw, width, height, screen_words, current_time, font_path=None):
    """Render a single frame with karaoke effect."""
    # White background
    draw.rectangle([(0, 0), (width, height)], fill=(255, 255, 255))

    if not screen_words:
        return

    # Calculate positions (simple center layout)
    font_size = 48
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Build text string
    text_parts = []
    for i, (start, end, text) in enumerate(screen_words):
        # Determine color based on timing
        if current_time < start:
            # Not yet reached - white text
            color = (200, 200, 200)
        elif current_time < end:
            # Currently speaking - highlight
            color = (255, 0, 0)
        else:
            # Completed - dark grey
            color = (80, 80, 80)
        text_parts.append((text, color))

    # Draw all words
    y_pos = height // 2 - font_size
    x_center = width // 2

    # Calculate total width to center text
    full_text = ''.join(t[0] for t in text_parts)
    bbox = draw.textbbox((0, 0), full_text, font=font)
    text_width = bbox[2] - bbox[0]
    x_start = (width - text_width) // 2

    x_pos = x_start
    for text, color in text_parts:
        draw.text((x_pos, y_pos), text, font=font, fill=color)
        bbox = draw.textbbox((0, 0), text, font=font)
        x_pos += bbox[2] - bbox[0]


def generate_video_from_frames(words, output_path, screen_size=(540, 960), fps=25):
    """Generate video by rendering frames with PIL and encoding with ffmpeg."""
    width, height = screen_size
    total_duration = max(w[1] for w in words) + 0.5
    total_frames = int(total_duration * fps)

    print(f"Rendering {total_frames} frames at {fps}fps...")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix='karaoke_')
    frame_pattern = os.path.join(temp_dir, 'frame_%06d.png')

    # Try to find a suitable font
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    font_path = None
    for fp in font_paths:
        if os.path.exists(fp):
            font_path = fp
            break

    print(f"Using font: {font_path or 'default'}")

    # Render frames in batches to show progress
    batch_size = 100
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Get current screen words
        screen_words, _, _ = get_current_screen_words(words, current_time)
        if screen_words:
            # Build text with spacing
            text_parts = []
            for i, (start, end, text) in enumerate(screen_words):
                if current_time < start:
                    color = (200, 200, 200)
                elif current_time < end:
                    color = (255, 0, 0)
                else:
                    color = (80, 80, 80)
                text_parts.append((text, color))

            # Draw text centered
            font_size = 48
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            y_pos = height // 2 - font_size
            full_text = ''.join(t[0] for t in text_parts)
            bbox = draw.textbbox((0, 0), full_text, font=font)
            text_width = bbox[2] - bbox[0]
            x_start = (width - text_width) // 2

            x_pos = x_start
            for text, color in text_parts:
                draw.text((x_pos, y_pos), text, font=font, fill=color)
                bbox = draw.textbbox((0, 0), text, font=font)
                x_pos += bbox[2] - bbox[0]

        # Save frame
        img.save(frame_pattern % frame_num)

        if (frame_num + 1) % batch_size == 0:
            print(f"  Rendered {frame_num + 1}/{total_frames} frames...")

    print(f"Encoding video with ffmpeg...")

    # Use ffmpeg to create video from frames
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-i', 'output/segments/intro/section-0.mp3',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        output_path
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    # Cleanup temp frames
    import shutil
    shutil.rmtree(temp_dir)

    if result.returncode == 0:
        print(f"Generated: {output_path}")
    else:
        print(f"FFmpeg error:\n{result.stderr}")
        return False

    return True


def main():
    base_dir = 'output/segments/intro'
    words_file = f'{base_dir}/section-0-words.txt'
    mp3_input = f'{base_dir}/section-0.mp3'
    mp4_output = f'{base_dir}/section-0-karaoke.mp4'

    if not os.path.exists(words_file):
        print(f"Error: {words_file} not found")
        return 1

    if not os.path.exists(mp3_input):
        print(f"Error: {mp3_input} not found")
        return 1

    print("Parsing word timing...")
    words = parse_word_timing(words_file)
    print(f"Loaded {len(words)} words, duration: {max(w[1] for w in words):.2f}s")

    print("\nGenerating MP4 with PIL+ffmpeg (540x960 portrait)...")
    success = generate_video_from_frames(words, mp4_output)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
