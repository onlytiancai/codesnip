#!/usr/bin/env python3
"""Test with SRT format - FFmpeg handles this better."""

import subprocess

# SRT format (much simpler, no style issues)
srt_content = """1
00:00:00,000 --> 00:00:03,000
Hello World

2
00:00:03,000 --> 00:00:06,000
This is a test
"""

with open('output/segments/intro/test-simple.srt', 'w', encoding='utf-8') as f:
    f.write(srt_content)

print("Created test-simple.srt")

# Generate video using subtitles filter (not ass filter)
cmd = [
    'ffmpeg', '-y',
    '-f', 'lavfi', '-i', 'color=c=blue:s=640x150:d=10',
    '-i', 'output/segments/intro/section-0.mp3',
    '-vf', "subtitles=output/segments/intro/test-simple.srt",
    '-shortest',
    '-map', '0:v', '-map', '1:a',
    'output/segments/intro/test-srt.mp4'
]

print("Running FFmpeg with SRT...")
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"Result: {r.returncode}")
if r.stderr:
    for line in r.stderr.split('\n'):
        if 'subtitle' in line.lower() or 'error' in line.lower():
            print(line)

# Check output
r2 = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                      '-of', 'default=noprint_wrappers=1:nokey=1',
                      'output/segments/intro/test-srt.mp4'],
                    capture_output=True, text=True)
print(f"Output duration: {r2.stdout.strip()}")
