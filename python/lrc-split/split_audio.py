import csv
import os
import subprocess

# Paths
ffmpeg_path = r"D:\haohu\soft\ffmpeg\bin\ffmpeg.exe"
input_mp3_dir = r"D:\BaiduYunDownload\NCE2-英音-(MP3+LRC)"
output_dir = r"D:\BaiduYunDownload\splited_mp3"
csv_file = r"D:\haohu\github\codesnip\python\lrc-split\output.csv"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory ensured: {output_dir}")

# Read CSV and process each row
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for idx, row in enumerate(reader, start=1):
        lrc_file = row['File']
        start_time = row['Start']
        end_time = row['End']

        # Derive MP3 file path
        mp3_file = os.path.join(input_mp3_dir, os.path.splitext(lrc_file)[0] + ".mp3")
        output_file = os.path.join(output_dir, f"{idx}.mp3")

        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping row {idx}: Output file {output_file} already exists.")
            continue

        # Handle empty end time (use file's end)
        if not end_time:
            print(f"Row {idx}: End time is empty, using file's end.")
            end_time_option = []
        else:
            end_time_option = ['-to', end_time]

        # Debug logs
        print(f"Processing row {idx}:")
        print(f"  LRC File: {lrc_file}")
        print(f"  MP3 File: {mp3_file}")
        print(f"  Start Time: {start_time}, End Time: {end_time or 'file end'}")
        print(f"  Output File: {output_file}")

        # Run ffmpeg command
        command = [
            ffmpeg_path,
            '-i', mp3_file,
            '-ss', start_time,
            *end_time_option,
            '-c', 'copy',
            output_file
        ]
        print(f"  Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"  Finished processing row {idx}")
