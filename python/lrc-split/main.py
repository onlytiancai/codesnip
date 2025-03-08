import os
import glob
import re
import csv

def list_lrc_files(directory):
    lrc_files = glob.glob(os.path.join(directory, '*.lrc'))
    return lrc_files

def parse_lrc_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    sentences = []
    current_sentence = ""
    start_time = None

    for line in lines:
        match = re.match(r'\[(\d{2}:\d{2}\.\d{2})\](.*)', line)
        if match:
            time = match.group(1)
            text = match.group(2).strip()
            if start_time is None:
                start_time = time
            if current_sentence:
                current_sentence += " " + text
            else:
                current_sentence = text
            
            if text.endswith(','):
                continue
            else:
                sentences.append((start_time, time, current_sentence))
                current_sentence = ""
                start_time = None

    # Update end times to be the start time of the next sentence
    for i in range(len(sentences) - 1):
        sentences[i] = (sentences[i][0], sentences[i + 1][0], sentences[i][2])
    if sentences:
        sentences[-1] = (sentences[-1][0], "", sentences[-1][2])

    return sentences

directory = r'D:\BaiduYunDownload\temp'
lrc_files = list_lrc_files(directory)

output_file = 'output.csv'
with open(output_file, 'w', newline='', encoding='gbk') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['File', 'Start', 'End', 'Sentence'])

    for file in lrc_files:
        sentences = parse_lrc_file(file)
        filename = os.path.basename(file)
        for start_time, end_time, sentence in sentences:
            print(f"File: {filename}, Start: {start_time}, End: {end_time}, Sentence: {sentence}")
            csvwriter.writerow([filename, start_time, end_time, sentence])