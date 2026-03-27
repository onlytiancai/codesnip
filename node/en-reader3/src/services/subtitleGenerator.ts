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
    const wordEntries = await runEdgeTTSWordBoundary(narrationScript);

    if (wordEntries.length === 0) {
      logger.warn(`No WordBoundary results for section ${sectionId}, falling back to estimation`);
      return generateFallbackSubtitles(narrationScript, audioPath, outputDir, sectionId);
    }

    // Detect if text is primarily English (has more Latin characters than CJK)
    const isEnglish = detectLanguage(wordEntries.map(w => w.text).join(''));

    // Split original script into subtitle-ready segments
    const segments = splitIntoSegments(narrationScript, isEnglish);

    // Map segments to timing
    const timedSegments = mapTimingToSegments(wordEntries, segments, isEnglish);

    const assContent = generateASS(timedSegments, isEnglish);

    const outputPath = join(outputDir, `section-${sectionId}.ass`);
    await writeFile(outputPath, assContent, 'utf-8');

    logger.debug(`Subtitles saved to ${outputPath} (${timedSegments.length} entries from ${wordEntries.length} words)`);

    return outputPath;
  } catch (err) {
    logger.error(`Edge TTS WordBoundary failed for section ${sectionId}:`, err);
    return generateFallbackSubtitles(narrationScript, audioPath, outputDir, sectionId);
  }
}

/**
 * Detect if text is primarily English (Latin characters) vs Chinese (CJK characters).
 */
function detectLanguage(text: string): boolean {
  const cjkCount = (text.match(/[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]/g) || []).length;
  const latinCount = (text.match(/[a-zA-Z]/g) || []).length;
  return latinCount > cjkCount;
}

/**
 * Split script into subtitle segments based on punctuation.
 * Rules:
 * - Chinese: max 30 chars, split by comma (，); merge if result < 5 chars
 * - English: max 15 words, split by comma/period
 */
function splitIntoSegments(script: string, isEnglish: boolean): string[] {
  if (isEnglish) {
    return splitEnglish(script);
  } else {
    return splitChinese(script);
  }
}

/**
 * Split English text into subtitle segments.
 */
function splitEnglish(text: string): string[] {
  const MAX_WORDS = 15;
  const segments: string[] = [];

  // First split by sentence endings
  const sentences = text.split(/(?<=[.!?])\s+/);

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).filter(w => w.length > 0);

    if (words.length <= MAX_WORDS) {
      if (sentence.trim()) segments.push(sentence.trim());
    } else {
      // Split by commas within sentence
      const parts = splitByComma(sentence, MAX_WORDS);
      segments.push(...parts);
    }
  }

  return segments;
}

/**
 * Split English text by comma, respecting word limit.
 */
function splitByComma(text: string, maxWords: number): string[] {
  const parts: string[] = [];
  const commas = text.match(/,\s*/g) || [];
  const commaIndices = [...text.matchAll(/,\s*/g)].map(m => m.index!);

  if (commaIndices.length === 0) {
    // No commas, split by words
    const words = text.split(/\s+/);
    let current = '';
    let wordCount = 0;

    for (const word of words) {
      if (wordCount >= maxWords && current.length > 0) {
        parts.push(current.trim());
        current = word;
        wordCount = 1;
      } else {
        current += (current ? ' ' : '') + word;
        wordCount++;
      }
    }
    if (current.trim()) parts.push(current.trim());
    return parts;
  }

  // Split by comma positions
  let current = '';
  let wordCount = 0;
  let lastIndex = 0;

  for (const idx of commaIndices) {
    const part = text.substring(lastIndex, idx + 1).trim();
    const partWords = part.split(/\s+/).filter(w => w.length > 0);

    if (wordCount + partWords.length > maxWords && current.length > 0) {
      parts.push(current.trim());
      current = part;
      wordCount = partWords.length;
    } else {
      current += (current ? ' ' : '') + part;
      wordCount += partWords.length;
    }
    lastIndex = idx + 1;
  }

  // Remaining text
  const remaining = text.substring(lastIndex).trim();
  if (remaining) {
    const remainingWords = remaining.split(/\s+/).filter(w => w.length > 0);
    if (wordCount + remainingWords.length > maxWords && current.length > 0) {
      parts.push(current.trim());
      current = remaining;
    } else {
      current += (current ? ' ' : '') + remaining;
    }
  }

  if (current.trim()) parts.push(current.trim());
  return parts;
}

/**
 * Split Chinese text into subtitle segments.
 * Rules:
 * - Split by sentence endings (。！？)
 * - For each sentence > 30 chars, split by comma (，)
 * - If comma-split part has < 5 chars, merge with previous part
 * - If still > 30 chars after comma split, split by character count
 */
function splitChinese(text: string): string[] {
  const MAX_CHARS = 30;
  const MIN_CHARS = 5;
  const segments: string[] = [];

  // First split by sentence endings (。！？)
  const sentences = text.split(/(?<=[。！？])\s*/);

  for (const sentence of sentences) {
    if (!sentence.trim()) continue;

    // Get raw sentence length (with punctuation)
    const rawLength = sentence.trim().length;
    // Get content length (without punctuation for comparison)
    const contentLength = sentence.replace(/[，。、；：！？\s]/g, '').length;

    if (contentLength <= MAX_CHARS) {
      // Fits within limit, push as-is (preserving punctuation)
      segments.push(sentence.trim());
    } else {
      // Need to split - first try by commas
      const commaParts = splitChineseByCommaNew(sentence.trim(), MAX_CHARS, MIN_CHARS);
      segments.push(...commaParts);
    }
  }

  return segments.filter(s => s.length > 0);
}

/**
 * Split Chinese text by comma, merging short segments with previous.
 * Returns array of segments that respect MAX_CHARS and MIN_CHARS rules.
 */
function splitChineseByCommaNew(text: string, maxChars: number, minChars: number): string[] {
  const parts: string[] = [];

  // Find all comma positions (，。、；：)
  const commaRegex = /[，、；：]/g;
  const commaMatches = [...text.matchAll(commaRegex)];
  const commaIndices = commaMatches.map(m => m.index!);

  if (commaIndices.length === 0) {
    // No commas, split by character count
    return splitChineseByLength(text, maxChars);
  }

  // Build segments by splitting at commas
  let current = '';
  let lastIndex = 0;

  for (const idx of commaIndices) {
    // Extract part from lastIndex to (and including) comma
    const part = text.substring(lastIndex, idx + 1);

    if (part.length >= minChars) {
      // This part is big enough on its own
      if (current.length > 0) {
        // First merge any accumulated short parts with this one
        const merged = current + part;
        if (merged.length <= maxChars) {
          current = merged;
          lastIndex = idx + 1;
          continue;
        } else {
          // Current is full, push it and start new
          if (current.length > 0) parts.push(current);
          current = part;
          lastIndex = idx + 1;
          continue;
        }
      } else {
        // No accumulated short parts
        if (part.length <= maxChars) {
          current = part;
          lastIndex = idx + 1;
          continue;
        } else {
          // Part itself is too long, split it
          if (current.length > 0) parts.push(current);
          const subParts = splitChineseByLength(part, maxChars);
          parts.push(...subParts.slice(0, -1)); // Push all but last
          current = subParts[subParts.length - 1] || '';
          lastIndex = idx + 1;
          continue;
        }
      }
    } else {
      // Part is too short (< minChars), accumulate it
      current += part;
      lastIndex = idx + 1;
    }
  }

  // Handle remaining text after last comma
  const remaining = text.substring(lastIndex);
  if (remaining) {
    if (current.length + remaining.length <= maxChars) {
      // Can merge with current
      current += remaining;
    } else {
      // Current is already full or too big, push it
      if (current.length > 0) parts.push(current);
      // Check if remaining is too long
      if (remaining.length > maxChars) {
        const subParts = splitChineseByLength(remaining, maxChars);
        parts.push(...subParts.slice(0, -1));
        current = subParts[subParts.length - 1] || '';
      } else {
        current = remaining;
      }
    }
  }

  if (current.length > 0) parts.push(current);

  return parts.filter(s => s.length > 0);
}

/**
 * Split Chinese text by character count when no punctuation exists.
 */
function splitChineseByLength(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text];

  const parts: string[] = [];
  let current = '';

  for (const char of text) {
    current += char;

    if (current.length >= maxChars) {
      parts.push(current.trim());
      current = '';
    }
  }

  if (current.trim()) parts.push(current.trim());
  return parts;
}

/**
 * Map Edge TTS word timings to our segments.
 */
function mapTimingToSegments(
  wordEntries: SubtitleEntry[],
  segments: string[],
  isEnglish: boolean
): SubtitleEntry[] {
  if (segments.length === 0 || wordEntries.length === 0) {
    return [];
  }

  // Calculate total duration from word entries
  const totalDuration = wordEntries[wordEntries.length - 1].end - wordEntries[0].start;

  // Calculate total characters/words
  let totalContent = 0;
  for (const seg of segments) {
    if (isEnglish) {
      totalContent += seg.split(/\s+/).filter(w => w.length > 0).length;
    } else {
      totalContent += seg.replace(/[，。、；：！？\s]/g, '').length;
    }
  }

  // Build result with proportional timing
  const result: SubtitleEntry[] = [];
  let currentTime = wordEntries[0].start;
  let processedContent = 0;

  for (const segment of segments) {
    const segmentContent = isEnglish
      ? segment.split(/\s+/).filter(w => w.length > 0).length
      : segment.replace(/[，。、；：！？\s]/g, '').length;

    // Calculate duration proportionally
    const duration = (segmentContent / totalContent) * totalDuration;

    result.push({
      start: currentTime,
      end: currentTime + duration,
      text: segment,
    });

    currentTime += duration;
    processedContent += segmentContent;
  }

  return result;
}

/**
 * Run Edge TTS with WordBoundary to get word timestamps.
 */
async function runEdgeTTSWordBoundary(narrationScript: string): Promise<SubtitleEntry[]> {
  // Preprocess to remove [朗读] and [讲解] markers
  const cleanText = narrationScript
    .replace(/\[朗读\]/g, '')
    .replace(/\[讲解\]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

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
    text = """${cleanText.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\n/g, ' ')}"""
    srt_content = asyncio.run(get_word_boundaries(text))
    print(srt_content)
`;

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', ['-c', pythonScript], { shell: false });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

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

    proc.on('error', (err) => { reject(err); });
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
 * Wrap Chinese text with \N line breaks if too long.
 * Chinese text without spaces can't auto-wrap in ASS, so we need explicit breaks.
 */
function wrapChineseText(text: string, maxCharsPerLine: number = 18): string {
  // Check if text has Chinese characters
  const hasChinese = /[\u4e00-\u9fff]/.test(text);

  if (!hasChinese) {
    return text;
  }

  // Remove existing \N if any
  text = text.replace(/\N/g, '');

  // If text is short enough, no wrapping needed
  if (text.length <= maxCharsPerLine) {
    return text;
  }

  // Try to find a good break point (comma or roughly middle)
  const mid = Math.floor(text.length / 2);

  // Look for comma-like punctuation first
  const commaMatch = text.substring(0, mid).match(/[,，、；：][^,，、；：]*$/);
  if (commaMatch && commaMatch.index !== undefined) {
    const breakIdx = commaMatch.index + 1;
    return text.substring(0, breakIdx) + '\\N' + text.substring(breakIdx);
  }

  // Look for comma-like punctuation in second half
  const commaMatch2 = text.substring(mid).match(/^[^,，、；：]*[,，、；：]/);
  if (commaMatch2 && commaMatch2[0]) {
    const breakIdx = mid + commaMatch2[0].length - 1;
    return text.substring(0, breakIdx) + '\\N' + text.substring(breakIdx);
  }

  // Otherwise, break at a character boundary near middle
  // Don't break right after a Chinese character start
  let breakIdx = mid;
  while (breakIdx > mid - 5 && breakIdx < mid + 5 && breakIdx < text.length) {
    const char = text[breakIdx];
    // If it's a Chinese character, we might want to break after it
    if (/[\u4e00-\u9fff]/.test(char)) {
      break;
    }
    breakIdx++;
  }

  if (breakIdx >= text.length) {
    return text;
  }

  return text.substring(0, breakIdx) + '\\N' + text.substring(breakIdx);
}

/**
 * Generate ASS format content.
 */
function generateASS(entries: SubtitleEntry[], isEnglish: boolean): string {
  const ASS_HEADER = `[Script Info]
; Generated by en-reader3
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Heiti SC,45,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

  let assContent = ASS_HEADER;

  for (const entry of entries) {
    const startTime = formatASSTime(entry.start + 0.15);
    const endTime = formatASSTime(entry.end + 0.2);

    // For Chinese: remove spaces; For English: preserve spaces
    let text = entry.text;
    if (!isEnglish) {
      text = text.replace(/ |　/g, '');
      // Wrap Chinese text with explicit line breaks
      text = wrapChineseText(text);
    }

    // Only escape braces, not commas
    const escapedText = text.replace(/\{/g, '\\{').replace(/\}/g, '\\}');

    assContent += `Dialogue: 0,${startTime},${endTime},Default,,0,0,0,,${escapedText}\n`;
  }

  return assContent;
}

/**
 * Format time for ASS format (HH:MM:SS.cc).
 */
function formatASSTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const centis = Math.floor((seconds % 1) * 100);

  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}.${pad(centis, 2)}`;
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

  const totalDuration = await getAudioDuration(audioPath);
  const isEnglish = detectLanguage(narrationScript);
  const segments = splitIntoSegments(narrationScript, isEnglish);

  const totalChars = segments.reduce((sum, s) => sum + s.replace(/[，。、；：！？\s]/g, '').length, 0);
  const entries: SubtitleEntry[] = [];
  let currentTime = 0;

  for (const segment of segments) {
    const charCount = segment.replace(/[，。、；：！？\s]/g, '').length;
    const segDuration = totalChars > 0 ? (charCount / totalChars) * totalDuration : 0;

    entries.push({
      start: currentTime,
      end: currentTime + segDuration,
      text: segment,
    });

    currentTime += segDuration;
  }

  const assContent = generateASS(entries, isEnglish);
  const outputPath = join(outputDir, `section-${sectionId}.ass`);
  await writeFile(outputPath, assContent, 'utf-8');

  return outputPath;
}

/**
 * Get audio duration using ffprobe.
 */
export async function getAudioDuration(audioPath: string): Promise<number> {
  const { spawn } = await import('child_process');

  return new Promise((resolve, reject) => {
    const proc = spawn('ffprobe', [
      '-v', 'error',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      audioPath,
    ]);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

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
