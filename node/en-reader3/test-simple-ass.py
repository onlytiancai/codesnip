#!/usr/bin/env python3
"""Simple ASS test without karaoke tags."""

import subprocess

def format_ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centis:02d}"

# Simple ASS without karaoke
ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 540
PlayResY: 960

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Heiti SC,28,&H00FFFFFF,&H000000FF,&H000000,&H00000000,0,0,0,0,0,100,100,0,0,1,2,0,2,10,10,960,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,00:00:00.00,00:00:02.00,Default,,0,0,0,,同学们
Dialogue: 0,00:00:02.00,00:00:04.00,Default,,0,0,0,,今天我们
Dialogue: 0,00:00:04.00,00:00:06.00,Default,,0,0,0,,要读一篇
"""

with open('output/segments/intro/test-simple.ass', 'w', encoding='utf-8') as f:
    f.write(ass_content)

print("Created test-simple.ass")

# Create video
cmd = [
    'ffmpeg', '-y',
    '-f', 'lavfi', '-i', 'color=c=red:s=540x960:d=10',
    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=10',
    '-vf', "ass='output/segments/intro/test-simple.ass'",
    '-shortest',
    'output/segments/intro/test-simple.mp4'
]

r = subprocess.run(cmd, capture_output=True, text=True)
print(f"Result: {r.returncode}")
if r.returncode != 0:
    print(r.stderr[-300:])
