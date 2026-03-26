import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { logger } from '../utils/logger.js';

interface SubtitleEntry {
  start: number;
  end: number;
  text: string;
}

/**
 * Generate SRT subtitles using Edge TTS WordBoundary events.
 * WordBoundary provides precise word-level timing directly from TTS.
 */
export async function generateSubtitles(
  narrationScript: string,
  audioPath: string,
  outputDir: string,
  sectionId: number
): Promise<string> {
  logger.debug(`Generating subtitles with Edge TTS WordBoundary for section ${sectionId}`);

  try {
    const entries = await runEdgeTTSWordBoundary(narrationScript);

    if (entries.length === 0) {
      logger.warn(`No WordBoundary results for section ${sectionId}, falling back to estimation`);
      return generateFallbackSubtitles(narrationScript, audioPath, outputDir, sectionId);
    }

    const srtContent = generateSRT(entries);

    const outputPath = join(outputDir, `section-${sectionId}.srt`);
    await writeFile(outputPath, srtContent, 'utf-8');

    logger.debug(`Subtitles saved to ${outputPath}`);

    return outputPath;
  } catch (err) {
    logger.error(`Edge TTS WordBoundary failed for section ${sectionId}:`, err);
    return generateFallbackSubtitles(narrationScript, audioPath, outputDir, sectionId);
  }
}

/**
 * Run Edge TTS with WordBoundary to get word timestamps and generate SRT.
 */
async function runEdgeTTSWordBoundary(narrationScript: string): Promise<SubtitleEntry[]> {
  const pythonScript = `
import sys
import asyncio
import edge_tts

async def get_word_boundaries(text):
    submaker = edge_tts.SubMaker()
    communicate = edge_tts.Communicate(text, boundary="WordBoundary")
    async for chunk in communicate.stream():
        if chunk["type"] in ("WordBoundary", "SentenceBoundary"):
            submaker.feed(chunk)
    return submaker.get_srt()

if __name__ == "__main__":
    text = """${narrationScript.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\n/g, ' ')}"""
    srt_content = asyncio.run(get_word_boundaries(text))
    print(srt_content)
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
          const entries = parseSRTEntries(stdout);
          logger.debug(`Edge TTS WordBoundary returned ${entries.length} subtitle entries`);
          resolve(entries);
        } catch (err) {
          logger.error('Failed to parse Edge TTS output:', err);
          reject(new Error(`Failed to parse Edge TTS output: ${stderr}`));
        }
      } else {
        logger.error(`Edge TTS WordBoundary failed: ${stderr}`);
        reject(new Error(`Edge TTS WordBoundary failed: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

/**
 * Parse SRT content into subtitle entries.
 */
function parseSRTEntries(srtContent: string): SubtitleEntry[] {
  const entries: SubtitleEntry[] = [];
  const blocks = srtContent.trim().split(/\n\n+/);

  for (const block of blocks) {
    const lines = block.split('\n');
    if (lines.length < 3) continue;

    // Parse time line: "00:00:01,234 --> 00:00:03,456"
    const timeMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
    if (!timeMatch) continue;

    const start =
      parseInt(timeMatch[1]) * 3600 +
      parseInt(timeMatch[2]) * 60 +
      parseInt(timeMatch[3]) +
      parseInt(timeMatch[4]) / 1000;

    const end =
      parseInt(timeMatch[5]) * 3600 +
      parseInt(timeMatch[6]) * 60 +
      parseInt(timeMatch[7]) +
      parseInt(timeMatch[8]) / 1000;

    const text = lines.slice(2).join('\n');

    entries.push({ start, end, text });
  }

  return entries;
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
