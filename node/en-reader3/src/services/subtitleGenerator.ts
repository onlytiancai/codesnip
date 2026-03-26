import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { logger } from '../utils/logger.js';

interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

interface SegmentTimestamp {
  text: string;
  start: number;
  end: number;
}

interface SubtitleEntry {
  start: number;
  end: number;
  text: string;
}

/**
 * Generate SRT subtitles using faster-whisper ASR for accurate timing.
 * Uses ASR segment text directly and applies intelligent line splitting.
 */
export async function generateSubtitles(
  narrationScript: string,
  audioPath: string,
  outputDir: string,
  sectionId: number
): Promise<string> {
  logger.debug(`Generating subtitles with faster-whisper for section ${sectionId}`);

  const { words, segments } = await runWhisperASR(audioPath);

  if (segments.length === 0) {
    logger.warn(`No ASR results for section ${sectionId}, falling back to estimation`);
    return generateFallbackSubtitles(narrationScript, audioPath, outputDir, sectionId);
  }

  const entries = buildSubtitleEntriesFromSegments(segments);

  const srtContent = generateSRT(entries);

  const outputPath = join(outputDir, `section-${sectionId}.srt`);
  await writeFile(outputPath, srtContent, 'utf-8');

  logger.debug(`Subtitles saved to ${outputPath}`);

  return outputPath;
}

/**
 * Run faster-whisper ASR to get word and segment timestamps.
 */
async function runWhisperASR(audioPath: string): Promise<{ words: WordTimestamp[]; segments: SegmentTimestamp[] }> {
  const pythonScript = `
import sys
import json
from faster_whisper import WhisperModel

model_size = "medium"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("${audioPath.replace(/\\/g, '\\\\')}", word_timestamps=True)

word_list = []
segment_list = []

for segment in segments:
    seg_start = round(segment.start, 2)
    seg_end = round(segment.end, 2)
    seg_text = segment.text.strip()

    segment_list.append({
        "text": seg_text,
        "start": seg_start,
        "end": seg_end
    })

    if segment.words:
        for word in segment.words:
            word_list.append({
                "word": word.word,
                "start": round(word.start, 2),
                "end": round(word.end, 2)
            })

result = {"words": word_list, "segments": segment_list}
print(json.dumps(result))
`;

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', ['-c', pythonScript], {
      shell: false,
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          const words = result.words as WordTimestamp[];
          const segments = result.segments as SegmentTimestamp[];
          logger.debug(`ASR returned ${words.length} words, ${segments.length} segments`);
          resolve({ words, segments });
        } catch (err) {
          logger.error('Failed to parse ASR output:', err);
          reject(new Error(`Failed to parse ASR output: ${stderr}`));
        }
      } else {
        logger.error(`ASR failed: ${stderr}`);
        reject(new Error(`ASR failed: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

/**
 * Build subtitle entries from ASR segments.
 * Splits long segments into shorter lines at natural boundaries.
 */
function buildSubtitleEntriesFromSegments(segments: SegmentTimestamp[]): SubtitleEntry[] {
  const entries: SubtitleEntry[] = [];

  for (const segment of segments) {
    const lines = splitSegmentIntoLines(segment.text);

    if (lines.length === 1) {
      entries.push({
        start: segment.start,
        end: segment.end,
        text: lines[0],
      });
    } else {
      const segmentDuration = segment.end - segment.start;
      const lineDuration = segmentDuration / lines.length;

      for (let i = 0; i < lines.length; i++) {
        entries.push({
          start: segment.start + i * lineDuration,
          end: segment.start + (i + 1) * lineDuration,
          text: lines[i],
        });
      }
    }
  }

  return mergeShortEntries(entries);
}

/**
 * Split a segment text into lines.
 * Preserves spaces and breaks at natural word boundaries.
 */
function splitSegmentIntoLines(text: string): string[] {
  const lines: string[] = [];
  let currentLine = '';
  let currentLen = 0;
  const MAX_CHARS = 24;

  // Split into words while preserving spaces
  const tokens = text.split(/(\s+)/);

  for (const token of tokens) {
    if (token.trim().length === 0) {
      // Whitespace - add to current line
      currentLine += token;
      continue;
    }

    const tokenLen = token.trim().length;

    if (currentLen + tokenLen > MAX_CHARS && currentLen > 0) {
      lines.push(currentLine.trim());
      currentLine = token;
      currentLen = tokenLen;
    } else {
      currentLine += token;
      currentLen += tokenLen;
    }
  }

  if (currentLine.trim().length > 0) {
    lines.push(currentLine.trim());
  }

  return lines;
}

/**
 * Merge very short entries with neighbors.
 */
function mergeShortEntries(entries: SubtitleEntry[]): SubtitleEntry[] {
  const MIN_DURATION = 0.8;
  const MIN_CHARS = 8;

  const merged: SubtitleEntry[] = [];

  for (const entry of entries) {
    if (
      merged.length > 0 &&
      entry.text.length < MIN_CHARS &&
      entry.end - entry.start < MIN_DURATION
    ) {
      const prev = merged[merged.length - 1];
      prev.text += ' ' + entry.text;
      prev.end = entry.end;
    } else {
      merged.push(entry);
    }
  }

  return merged;
}

/**
 * Generate SRT format content.
 */
function generateSRT(entries: SubtitleEntry[]): string {
  let srtContent = '';

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const index = i + 1;

    const startTime = formatSRTTime(entry.start);
    const endTime = formatSRTTime(entry.end);

    srtContent += `${index}\n`;
    srtContent += `${startTime} --> ${endTime}\n`;
    srtContent += `${entry.text}\n`;
    srtContent += '\n';
  }

  return srtContent;
}

/**
 * Format time for SRT format (HH:MM:SS,mmm).
 */
function formatSRTTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);

  return `${pad(hours)}:${pad(minutes)}:${pad(secs)},${pad(millis, 3)}`;
}

function pad(num: number, length = 2): string {
  return num.toString().padStart(length, '0');
}

/**
 * Fallback subtitle generation.
 */
async function generateFallbackSubtitles(
  narrationScript: string,
  audioPath: string,
  outputDir: string,
  sectionId: number
): Promise<string> {
  logger.warn(`Using fallback subtitle generation for section ${sectionId}`);

  const duration = await getAudioDuration(audioPath);
  const entries = estimateSubtitleTiming(narrationScript, duration);
  const srtContent = generateSRT(entries);

  const outputPath = join(outputDir, `section-${sectionId}.srt`);
  await writeFile(outputPath, srtContent, 'utf-8');

  return outputPath;
}

/**
 * Get audio duration using ffprobe.
 */
async function getAudioDuration(audioPath: string): Promise<number> {
  const { spawn } = await import('child_process');

  return new Promise((resolve, reject) => {
    const args = [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      audioPath,
    ];

    const proc = spawn('ffprobe', args);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        const duration = parseFloat(stdout.trim());
        resolve(isNaN(duration) ? 0 : duration);
      } else {
        reject(new Error(`ffprobe failed: ${stderr}`));
      }
    });
  });
}

/**
 * Estimate subtitle timing.
 */
function estimateSubtitleTiming(text: string, audioDuration: number): SubtitleEntry[] {
  const entries: SubtitleEntry[] = [];

  const sentences = text.split(/([。！？；])/);

  let currentTime = 0;
  let currentText = '';

  for (let i = 0; i < sentences.length; i++) {
    const part = sentences[i];

    if (!part || /^[。！？；]+$/.test(part)) {
      if (currentText) {
        currentText += part;
      }
      continue;
    }

    currentText += part;

    const charCount = currentText.replace(/[。！？；]/g, '').length;
    const estimatedDuration = charCount / 4.5;

    if (estimatedDuration >= 2 || i === sentences.length - 1) {
      const endTime = Math.min(currentTime + estimatedDuration, audioDuration);

      entries.push({
        start: currentTime,
        end: endTime,
        text: currentText.trim(),
      });

      currentTime = endTime + 0.1;
      currentText = '';
    }
  }

  return entries;
}
