import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { readFileSync } from 'fs';
import { join } from 'path';
import { logger } from '../utils/logger.js';
import { generateTTSWithWords } from './tts.js';
import { getAssMaxCjkChars } from '../config/index.js';

interface SubtitleEntry {
  start: number;
  end: number;
  text: string;
  wordTimings?: { word: string; start: number; end: number }[];
}

interface KaraokeLine {
  text: string;
  start: number;
  end: number;
  words: { word: string; start: number; end: number }[];
}

/**
 * Calculate display width of a string.
 * CJK characters: width = 2
 * Half-width (ASCII, numbers, punctuation): width = 1
 * Full-width punctuation (，。): width = 2
 */
function calculateDisplayWidth(text: string): number {
  let width = 0;
  for (const char of text) {
    const code = char.codePointAt(0);
    if (!code) continue;

    // Check if it's a CJK character (CJK Unified Ideographs, Hangul, etc.)
    // Ranges: 0x4E00-0x9FFF (CJK), 0x3400-0x4DBF (Extension A), 0xAC00-0xD7AF (Hangul)
    if (
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0x3400 && code <= 0x4dbf) ||
      (code >= 0xac00 && code <= 0xd7af)
    ) {
      width += 2;
    } else if (char === '，' || char === '。') {
      // Full-width punctuation
      width += 2;
    } else {
      // Half-width character
      width += 1;
    }
  }
  return width;
}

/**
 * Calculate effective CJK char count from display width.
 * Each CJK char contributes 2 width units, ASCII contributes 1.
 */
function getCjkCharCount(text: string): number {
  return Math.ceil(calculateDisplayWidth(text) / 2);
}

/**
 * Validate ASS file for correct segmentation:
 * 1. No segment should contain more than one comma (，、) or other sentence separators
 * 2. No segment should exceed 30 characters without \N line break
 * Throws error if validation fails.
 */
function validateASSFile(assPath: string): void {
  const content = readFileSync(assPath, 'utf-8');
  const lines = content.split('\n');

  const errors: string[] = [];

  for (const line of lines) {
    if (!line.startsWith('Dialogue:')) continue;

    // Extract the text part (after the first {\k...} tag)
    // ASS format: Dialogue: ...,Default,{\k123}actual text here
    const firstBraceIdx = line.indexOf('}');
    if (firstBraceIdx < 0) continue;
    const text = line.substring(firstBraceIdx + 1);

    // Count commas and other punctuation that should cause splits
    // For Chinese: ，、 (comma, enumeration comma)
    // For English: , (but we check for both since mixed content is possible)
    const commaCount = (text.match(/[，、,]/g) || []).length;
    const sentenceEndCount = (text.match(/[。！？.!?]/g) || []).length;
    const totalPunct = commaCount + sentenceEndCount;

    if (totalPunct > 1) {
      errors.push(`Segment has ${totalPunct} punctuation marks (should be split): ${text.substring(0, 50)}...`);
    }

    // Check if text is too long (> 30 chars without \N)
    const hasLineBreak = text.includes('\\N');
    // Remove k-tags like \k63} and then remove remaining braces
    // In regex: \\k\d+ means backslash, literal k, one or more digits
    const displayText = text.replace(/\\k\d+/g, '').replace(/[{}\\]/g, '');
    if (displayText.length > 30 && !hasLineBreak) {
      errors.push(`Segment too long (${displayText.length} chars) without \\N line break: ${displayText.substring(0, 50)}...`);
    }
  }

  if (errors.length > 0) {
    const errorMsg = `ASS validation failed:\n${errors.join('\n')}`;
    logger.error(errorMsg);
    throw new Error(errorMsg);
  }

  logger.debug(`ASS validation passed for ${assPath}`);
}

/**
 * Generate SRT subtitles using Edge TTS WordBoundary events.
 * WordBoundary provides precise word-level timing directly from TTS.
 * This function generates audio AND word timings in a SINGLE call.
 */
export async function generateSubtitles(
  narrationScript: string,
  audioPath: string,
  outputDir: string,
  sectionId: number
): Promise<string> {
  logger.debug(`Generating subtitles with Edge TTS WordBoundary for section ${sectionId}`);

  try {
    // Use generateTTSWithWords which generates audio AND returns word timings in one call
    // This ensures the word timings are perfectly aligned with the audio
    const ttsResult = await generateTTSWithWords(narrationScript, outputDir, sectionId);

    // ttsResult now contains:
    // - audioPath: path to generated audio
    // - duration: audio duration
    // - wordTimings: word-level timings aligned with the audio

    // Update the audioPath with the correct one from generateTTSWithWords
    const finalAudioPath = ttsResult.audioPath;

    // Get actual audio duration
    const audioDuration = ttsResult.duration;
    const wordEntries: SubtitleEntry[] = ttsResult.wordTimings.map(w => ({
      start: w.start,
      end: w.end,
      text: w.text,
    }));

    if (wordEntries.length === 0) {
      throw new Error(`Word timing extraction failed for section ${sectionId}. Cannot generate ASS without word timings.`);
    }

    logger.debug(`Audio duration: ${audioDuration}s, word entries span: ${wordEntries[wordEntries.length - 1].end - wordEntries[0].start}s`);

    // The word timings from generateTTSWithWords should already be aligned with the audio
    // No scaling needed since they were generated in the same call

    // Save word timing to debug file
    const wordsDebugPath = join(outputDir, `section-${sectionId}-words.txt`);
    const wordsDebugContent = wordEntries
      .map(w => `${w.start.toFixed(3)}:${w.end.toFixed(3)}:${w.text}`)
      .join('\n');
    await writeFile(wordsDebugPath, wordsDebugContent, 'utf-8');
    logger.debug(`Word timing saved to ${wordsDebugPath} (${wordEntries.length} words)`);

    // Detect if text is primarily English (has more Latin characters than CJK)
    const isEnglish = detectLanguage(wordEntries.map(w => w.text).join(''));

    // Split original script into subtitle-ready segments
    const segments = splitIntoSegments(narrationScript, isEnglish);

    // Map segments to timing
    const timedSegments = mapTimingToSegments(wordEntries, segments, isEnglish, audioDuration);

    // Generate ASS with karaoke-style highlighting using TTS word timings
    const assContent = generateASS(timedSegments, isEnglish, wordEntries);

    const outputPath = join(outputDir, `section-${sectionId}.ass`);
    await writeFile(outputPath, assContent, 'utf-8');

    logger.debug(`Subtitles saved to ${outputPath} (${timedSegments.length} entries from ${wordEntries.length} words)`);

    // Validate ASS file for correct segmentation
    validateASSFile(outputPath);

    return outputPath;
  } catch (err) {
    logger.error(`Edge TTS failed for section ${sectionId}:`, err);
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
  const MAX_CHARS = 30;
  const segments: string[] = [];

  // First split by sentence endings
  const sentences = text.split(/(?<=[.!?])\s+/);

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).filter(w => w.length > 0);

    if (words.length <= MAX_WORDS) {
      // Even if under word limit, split by commas if present
      // Check for both English and Chinese commas
      const hasComma = /[,，]/.test(sentence);
      // Use actual sentence length (with spaces) for char limit check
      if (hasComma) {
        const parts = splitByComma(sentence, MAX_WORDS);
        segments.push(...parts);
      } else if (sentence.length > MAX_CHARS) {
        // Too long without commas - split by words
        const parts = splitByComma(sentence, MAX_WORDS);
        segments.push(...parts);
      } else {
        if (sentence.trim()) segments.push(sentence.trim());
      }
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
function splitByComma(text: string, maxWords: number, maxChars: number = 30): string[] {
  const parts: string[] = [];
  // Match English AND Chinese commas
  const commaRegex = /[,，]\s*/g;
  const commas = text.match(commaRegex) || [];
  const commaIndices = [...text.matchAll(commaRegex)].map(m => m.index!);

  if (commaIndices.length === 0) {
    // No commas, split by words or chars
    const words = text.split(/\s+/);
    let current = '';
    let wordCount = 0;
    let charCount = 0;

    for (const word of words) {
      if ((wordCount >= maxWords || charCount + word.length > maxChars) && current.length > 0) {
        parts.push(current.trim());
        current = word;
        wordCount = 1;
        charCount = word.length;
      } else {
        current += (current ? ' ' : '') + word;
        wordCount++;
        charCount += word.length;
      }
    }
    if (current.trim()) parts.push(current.trim());
    return parts;
  }

  // Split by comma positions - ALWAYS split at each comma
  let lastIndex = 0;

  for (const idx of commaIndices) {
    const part = text.substring(lastIndex, idx + 1).trim();
    if (part) {
      // Check if this part still exceeds maxChars and needs further splitting
      // Remove trailing comma before recursing to avoid infinite loop
      if (part.length > maxChars) {
        const subParts = splitByComma(part.replace(/[,，]\s*$/, ''), maxWords, maxChars);
        parts.push(...subParts);
      } else {
        parts.push(part);
      }
    }
    lastIndex = idx + 1;
  }

  // Remaining text after last comma
  const remaining = text.substring(lastIndex).trim();
  if (remaining) {
    // Check if remaining part still exceeds maxChars and needs further splitting
    if (remaining.length > maxChars) {
      const subParts = splitByComma(remaining, maxWords, maxChars);
      parts.push(...subParts);
    } else {
      parts.push(remaining);
    }
  }

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

    // Check if there are commas - if so, split regardless of length
    const hasComma = sentence.includes('，') || sentence.includes('、');

    if (contentLength <= MAX_CHARS && !hasComma) {
      // Fits within limit and no commas, push as-is (preserving punctuation)
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
 * Split Chinese text by punctuation (，。！？、；：——).
 * Each punctuation-separated phrase becomes a segment.
 * Long segments are split by character count.
 */
function splitChineseByCommaNew(text: string, maxChars: number, _minChars: number): string[] {
  // First pass: split by punctuation character by character
  const rawSegments: string[] = [];
  let current = '';
  let i = 0;

  while (i < text.length) {
    const char = text[i];

    // Check for em-dash (——) - treat as a SEPARATOR (push current, push em-dash, start new)
    if (char === '—' && i + 1 < text.length && text[i + 1] === '—') {
      if (current.length > 0) {
        rawSegments.push(current);
      }
      rawSegments.push('——');
      current = '';
      i += 2;
      continue;
    }

    // Check for other punctuation - treat as end of segment
    if ('，。！？、；：'.includes(char)) {
      current += char;
      rawSegments.push(current);
      current = '';
      i++;
      continue;
    }

    current += char;
    i++;
  }

  if (current.length > 0) {
    rawSegments.push(current);
  }

  if (rawSegments.length === 0) {
    return [text];
  }

  // Second pass: split long segments
  // Em-dash segments (——) always cause a split
  const result: string[] = [];

  for (let i = 0; i < rawSegments.length; i++) {
    const seg = rawSegments[i];

    // Em-dash always causes a split
    if (seg === '——') {
      result.push('——');
      continue;
    }

    if (seg.length > maxChars) {
      // Too long - split by character count, preserving trailing punctuation
      const trailingPunct = seg.match(/[，。！？、；：]$/)?.[0] || '';
      const cleanSeg = trailingPunct ? seg.slice(0, -1) : seg;
      const subParts = splitChineseByLength(cleanSeg, maxChars);
      // Add trailing punctuation to last part
      if (subParts.length > 0 && trailingPunct) {
        subParts[subParts.length - 1] += trailingPunct;
      }
      result.push(...subParts);
    } else {
      result.push(seg);
    }
  }

  return result.filter(s => s.length > 0);
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
 * Map word-level timing to subtitle segments using position-based matching.
 * For each segment, finds the corresponding TTS words by text matching and
 * uses their actual timing (start of first word, end of last word).
 * Falls back to proportional allocation only when no match is found.
 */
function mapTimingToSegments(
  wordEntries: SubtitleEntry[],
  segments: string[],
  isEnglish: boolean,
  audioDuration: number
): SubtitleEntry[] {
  if (segments.length === 0 || wordEntries.length === 0) {
    return [];
  }

  const result: SubtitleEntry[] = [];
  const ttsDuration = wordEntries[wordEntries.length - 1].end - wordEntries[0].start;

  // Calculate total content for fallback proportional allocation
  let totalContent = 0;
  for (const seg of segments) {
    if (isEnglish) {
      totalContent += seg.split(/\s+/).filter(w => w.length > 0).length;
    } else {
      totalContent += seg.replace(/[，。、；：！？\s]/g, '').length;
    }
  }

  let currentTime = wordEntries[0].start;

  for (const segment of segments) {
    let matchedTiming: { start: number; end: number } | null = null;

    if (isEnglish) {
      // English: match by words
      const segmentWords = segment.split(/\s+/).filter(w => w.length > 0);
      if (segmentWords.length > 0) {
        matchedTiming = findEnglishSegmentTiming(wordEntries, segmentWords);
      }
    } else {
      // Chinese: match by characters (ignoring punctuation)
      const cleanSegment = segment.replace(/[，。、；：！？\s]/g, '');
      if (cleanSegment.length > 0) {
        matchedTiming = findChineseSegmentTiming(wordEntries, cleanSegment);
      }
    }

    if (matchedTiming) {
      // Use matched timing from actual TTS word positions
      // But constrain end time to audio duration
      const constrainedEnd = Math.min(matchedTiming.end, audioDuration);
      const constrainedStart = Math.min(matchedTiming.start, constrainedEnd - 0.1);

      result.push({
        start: constrainedStart,
        end: constrainedEnd,
        text: segment,
      });
      currentTime = constrainedEnd;
    } else {
      // Fallback: proportional allocation based on content length
      const segmentContent = isEnglish
        ? segment.split(/\s+/).filter(w => w.length > 0).length
        : segment.replace(/[，。、；：！？\s]/g, '').length;

      if (segmentContent > 0 && totalContent > 0) {
        // Use ratio of segment content to total content, scaled to remaining audio time
        const remainingTime = Math.max(audioDuration - currentTime, 0);
        const segmentRatio = segmentContent / totalContent;

        // Recalculate remaining content for more accurate proportional allocation
        const unprocessedContent = segments.slice(result.length).reduce((sum, s) => {
          return sum + (isEnglish
            ? s.split(/\s+/).filter(w => w.length > 0).length
            : s.replace(/[，。、；：！？\s]/g, '').length);
        }, 0);

        let duration: number;
        if (unprocessedContent > 0) {
          // Allocate based on proportion of remaining content
          duration = (segmentContent / unprocessedContent) * remainingTime;
        } else {
          duration = remainingTime;
        }

        // Ensure minimum duration of 0.5s for readability, but don't exceed remaining time
        duration = Math.max(Math.min(duration, remainingTime), 0.5);

        const endTime = Math.min(currentTime + duration, audioDuration);
        result.push({
          start: currentTime,
          end: endTime,
          text: segment,
        });
        currentTime = endTime;

        // If we've reached audio duration, stop adding segments
        if (currentTime >= audioDuration) {
          break;
        }
      }
    }
  }

  return result;
}

/**
 * Find timing for English segment by matching words to TTS word entries.
 * Uses flexible matching: word in segment can match TTS word if either contains the other.
 */
function findEnglishSegmentTiming(
  wordEntries: SubtitleEntry[],
  segmentWords: string[]
): { start: number; end: number } | null {
  if (wordEntries.length === 0 || segmentWords.length === 0) {
    return null;
  }

  // Try to find a sequence of matching words
  for (let startIdx = 0; startIdx <= wordEntries.length - segmentWords.length; startIdx++) {
    let matchCount = 0;

    // Check if segment words match wordEntries starting at startIdx
    for (let i = 0; i < segmentWords.length; i++) {
      const ttsWord = wordEntries[startIdx + i].text.toLowerCase();
      const segWord = segmentWords[i].toLowerCase();

      // Match if one contains the other (handles partial matches and punctuation differences)
      if (ttsWord.includes(segWord) || segWord.includes(ttsWord)) {
        matchCount++;
      } else {
        break;
      }
    }

    if (matchCount === segmentWords.length) {
      // Found a complete match
      return {
        start: wordEntries[startIdx].start,
        end: wordEntries[startIdx + segmentWords.length - 1].end,
      };
    }
  }

  return null;
}

/**
 * Find timing for Chinese segment by matching characters to TTS word entries.
 * Concatenates TTS word text and finds segment position within it.
 */
function findChineseSegmentTiming(
  wordEntries: SubtitleEntry[],
  cleanSegment: string
): { start: number; end: number } | null {
  if (wordEntries.length === 0 || cleanSegment.length === 0) {
    return null;
  }

  // Build concatenated TTS text and track character positions to word indices
  let ttsText = '';
  const charToWordIdx: number[] = [];

  for (let i = 0; i < wordEntries.length; i++) {
    const wordText = wordEntries[i].text;
    for (const char of wordText) {
      ttsText += char;
      charToWordIdx.push(i);
    }
  }

  // Find segment position in TTS text
  const segmentLen = cleanSegment.length;

  for (let ttsIdx = 0; ttsIdx <= ttsText.length - segmentLen; ttsIdx++) {
    const ttsSubstr = ttsText.substring(ttsIdx, ttsIdx + segmentLen);

    // Check if this substring matches our segment (allowing for small differences)
    if (stringsSimilar(ttsSubstr, cleanSegment, 0.8)) {
      const startWordIdx = charToWordIdx[ttsIdx];
      const endWordIdx = charToWordIdx[Math.min(ttsIdx + segmentLen - 1, ttsText.length - 1)];

      return {
        start: wordEntries[startWordIdx].start,
        end: wordEntries[endWordIdx].end,
      };
    }
  }

  return null;
}

/**
 * Check if two strings are similar within a threshold (0-1).
 * Uses simple character-level comparison.
 */
function stringsSimilar(s1: string, s2: string, threshold: number): boolean {
  if (s1 === s2) return true;
  if (s1.length !== s2.length) return false;

  let matchCount = 0;
  for (let i = 0; i < s1.length; i++) {
    if (s1[i] === s2[i]) {
      matchCount++;
    }
  }

  return matchCount / s1.length >= threshold;
}

/**
 * Run Edge TTS with WordBoundary to get word-level timestamps.
 * Returns precise timing for each word spoken by TTS.
 */
async function runEdgeTTSWordBoundary(narrationScript: string): Promise<SubtitleEntry[]> {
  // Preprocess to remove [朗读] and [讲解] markers
  const cleanText = narrationScript
    .replace(/\[朗读\]/g, '')
    .replace(/\[讲解\]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  // Python script to get word-level timing directly from Edge TTS
  const pythonScript = `
import sys
import asyncio
import edge_tts

async def get_word_timings(text):
    words = []
    communicate = edge_tts.Communicate(text, boundary="WordBoundary")
    async for chunk in communicate.stream():
        if chunk["type"] == "WordBoundary":
            # offset and duration are in 100-nanoseconds
            offset_seconds = chunk["offset"] / 10000000
            duration_seconds = chunk["duration"] / 10000000
            words.append({
                "text": chunk["text"],
                "start": offset_seconds,
                "end": offset_seconds + duration_seconds
            })
    return words

if __name__ == "__main__":
    text = """${cleanText.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\n/g, ' ')}"""
    words = asyncio.run(get_word_timings(text))
    # Output as start:end:text per line for easy parsing
    for w in words:
        sys.stdout.write(f"{w['start']:.3f}:{w['end']:.3f}:{w['text']}\\n")
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
          const entries = parseWordTimingOutput(stdout);
          logger.debug(`Edge TTS WordBoundary returned ${entries.length} word entries`);
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
 * Parse word timing output from Python script.
 * Format: start:end:text (one per line)
 */
function parseWordTimingOutput(output: string): SubtitleEntry[] {
  const entries: SubtitleEntry[] = [];
  const lines = output.trim().split('\n');

  for (const line of lines) {
    if (!line) continue;
    const parts = line.split(':');
    if (parts.length < 3) continue;

    const start = parseFloat(parts[0]);
    const end = parseFloat(parts[1]);
    const text = parts.slice(2).join(':');

    if (!isNaN(start) && !isNaN(end) && text) {
      entries.push({ start, end, text });
    }
  }

  return entries;
}

/**
 * Wrap Chinese text with \N line breaks if too long.
 * Chinese text without spaces can't auto-wrap in ASS, so we need explicit breaks.
 */
function wrapChineseText(text: string, maxCharsPerLine: number = 40): string {
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
 * Extract word timings for a segment by matching to TTS word entries.
 * Returns array of {word, start, end} for each word in the segment.
 */
function extractWordTimingsForSegment(
  segment: string,
  wordEntries: SubtitleEntry[],
  isEnglish: boolean
): { word: string; start: number; end: number }[] {
  if (wordEntries.length === 0 || !segment) {
    return [];
  }

  if (isEnglish) {
    return extractEnglishWordTimings(segment, wordEntries);
  } else {
    return extractChineseWordTimings(segment, wordEntries);
  }
}

/**
 * Extract word timings for English segment.
 */
function extractEnglishWordTimings(
  segment: string,
  wordEntries: SubtitleEntry[]
): { word: string; start: number; end: number }[] {
  const segmentWords = segment.split(/\s+/).filter(w => w.length > 0);
  if (segmentWords.length === 0) return [];

  // Try to find matching sequence in TTS word entries
  for (let startIdx = 0; startIdx <= wordEntries.length - segmentWords.length; startIdx++) {
    const matchedWords: { word: string; start: number; end: number }[] = [];

    for (let i = 0; i < segmentWords.length; i++) {
      const ttsWord = wordEntries[startIdx + i].text.toLowerCase().replace(/[.,!?;:'"]/g, '');
      const segWord = segmentWords[i].toLowerCase().replace(/[.,!?;:'"]/g, '');

      if (ttsWord.includes(segWord) || segWord.includes(ttsWord)) {
        matchedWords.push({
          word: segmentWords[i],
          start: wordEntries[startIdx + i].start,
          end: wordEntries[startIdx + i].end,
        });
      } else {
        matchedWords.length = 0;
        break;
      }
    }

    if (matchedWords.length === segmentWords.length) {
      return matchedWords;
    }
  }

  // Fallback: split segment duration equally among words
  return createEqualWordTimings(segment, segmentWords.length);
}

/**
 * Extract word timings for Chinese segment.
 * TTS returns word-level timing, but our segment may include punctuation.
 */
function extractChineseWordTimings(
  segment: string,
  wordEntries: SubtitleEntry[]
): { word: string; start: number; end: number }[] {
  // Build character-level mapping to TTS word indices
  let ttsText = '';
  const charToWordIdx: number[] = [];

  for (let i = 0; i < wordEntries.length; i++) {
    const w = wordEntries[i];
    for (const char of w.text) {
      charToWordIdx.push(i);
      ttsText += char;
    }
  }

  // Remove punctuation from segment for matching (including single quotes)
  const cleanSegment = segment.replace(/[，。、；：！？\s（）【】《》""''.,!?;:'"()]/g, '');
  if (!cleanSegment) return [];

  // Find exact match position in TTS text
  const ttsIdx = ttsText.indexOf(cleanSegment);

  if (ttsIdx >= 0) {
    // Found exact match - group characters by TTS word
    const result: { word: string; start: number; end: number }[] = [];
    let currentWordIdx = -1;
    let currentWord = '';
    let currentStart = 0;
    let currentEnd = 0;

    for (let i = 0; i < cleanSegment.length && (ttsIdx + i) < charToWordIdx.length; i++) {
      const wordIdx = charToWordIdx[ttsIdx + i];
      const ttsWord = wordEntries[wordIdx];

      if (wordIdx !== currentWordIdx) {
        // New word - save previous if exists
        if (currentWordIdx >= 0 && currentWord) {
          result.push({ word: currentWord, start: currentStart, end: currentEnd });
        }
        // Start new word
        currentWordIdx = wordIdx;
        currentWord = ttsWord.text[0]; // Take first char of the TTS word
        currentStart = ttsWord.start;
        currentEnd = ttsWord.end;
      } else {
        // Same word - append character (but only if it fits in the TTS word)
        if (currentWord.length < ttsWord.text.length) {
          currentWord += ttsWord.text[currentWord.length];
          currentEnd = ttsWord.end;
        }
      }
    }

    // Don't forget the last word
    if (currentWordIdx >= 0 && currentWord) {
      result.push({ word: currentWord, start: currentStart, end: currentEnd });
    }

    if (result.length > 0) {
      return result;
    }
  }

  // Fallback: equal timing
  const charCount = cleanSegment.length;
  return createEqualWordTimings(segment, Math.min(charCount, 10));
}

/**
 * Build word timings for Chinese by mapping segment punctuation splits to TTS words.
 */
function buildChineseWordTimingsFromMatch(
  segment: string,
  wordEntries: SubtitleEntry[],
  startWordIdx: number,
  endWordIdx: number
): { word: string; start: number; end: number }[] {
  // Bounds check
  if (!wordEntries[startWordIdx] || !wordEntries[endWordIdx]) {
    const cleanSegment = segment.replace(/[，。、；：！？\s]/g, '');
    return createEqualWordTimings(segment, Math.min(cleanSegment.length, 10));
  }

  // Split segment by punctuation to get natural word groups
  const parts: string[] = [];
  let current = '';

  for (const char of segment) {
    if ('，。、；：！？'.includes(char)) {
      if (current) parts.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  if (current) parts.push(current);

  if (parts.length === 0) {
    // No punctuation, treat each character as potential word
    const cleanSegment = segment.replace(/[，。、；：！？\s]/g, '');
    return createEqualWordTimings(segment, Math.min(cleanSegment.length, 10));
  }

  // Calculate timing range per character based on word entries
  const totalDuration = wordEntries[endWordIdx].end - wordEntries[startWordIdx].start;
  const totalChars = parts.join('').replace(/[，。、；：！？\s]/g, '').length;

  if (totalChars === 0) return [];

  const charsPerSecond = totalChars / totalDuration;
  let currentTime = wordEntries[startWordIdx].start;
  const result: { word: string; start: number; end: number }[] = [];

  for (const part of parts) {
    const cleanPart = part.replace(/[，。、；：！？\s]/g, '');
    if (!cleanPart) continue;

    const partDuration = cleanPart.length / charsPerSecond;
    result.push({
      word: cleanPart,
      start: currentTime,
      end: currentTime + partDuration,
    });
    currentTime += partDuration;
  }

  return result;
}

/**
 * Create equal word timings as fallback when TTS matching fails.
 */
function createEqualWordTimings(
  segment: string,
  wordCount: number
): { word: string; start: number; end: number }[] {
  if (wordCount <= 0) return [];

  // Check if segment contains spaces (English words) by looking for Latin letters
  const hasSpaces = /\s/.test(segment);

  if (hasSpaces) {
    // English: split by whitespace to get words
    const words = segment.split(/\s+/).filter(w => w.length > 0);
    return words.map(w => ({ word: w, start: 0, end: 0 }));
  }

  // For Chinese, wordCount is approximate character groups
  const cleanSegment = segment.replace(/[，。、；：！？\s]/g, '');
  const charsPerWord = Math.ceil(cleanSegment.length / wordCount);

  const words: string[] = [];
  for (let i = 0; i < wordCount; i++) {
    const start = i * charsPerWord;
    const word = cleanSegment.substring(start, start + charsPerWord);
    if (word) words.push(word);
  }

  // We'll set timing later based on segment duration
  return words.map(w => ({ word: w, start: 0, end: 0 }));
}

/**
 * Split karaoke lines while preserving word timings.
 * Returns array of KaraokeLine with properly distributed word timings.
 * Uses CJK-aware width calculation: CJK chars = 2 width units, ASCII = 1 width unit.
 */
function splitKaraokeLines(
  segment: string,
  wordTimings: { word: string; start: number; end: number }[],
  maxCharsPerLine: number = getAssMaxCjkChars()
): KaraokeLine[] {
  // maxCharsPerLine is now CJK char count (e.g., 13), max width = 13 * 2 = 26
  const maxLineWidth = maxCharsPerLine * 2;

  if (wordTimings.length === 0) {
    // No timing info - create words from segment and assign equal timing
    const words = createEqualWordTimings(segment, Math.ceil(segment.length / 4));
    if (words.length === 0) {
      return splitByCharsEqualTiming(segment, maxCharsPerLine);
    }
    // Assign equal timing to each word based on segment duration
    // Assume 2 seconds per line duration
    const totalDuration = 2.0;
    const durationPerWord = totalDuration / words.length;
    words.forEach((w, i) => {
      w.start = i * durationPerWord;
      w.end = (i + 1) * durationPerWord;
    });
    return splitKaraokeLines(segment, words, maxCharsPerLine);
  }

  // Calculate total duration from word timings
  const totalDuration = wordTimings[wordTimings.length - 1].end - wordTimings[0].start;
  const totalChars = wordTimings.reduce((sum, w) => sum + w.word.length, 0);

  if (totalChars === 0) return [];

  const secondsPerChar = totalDuration / totalChars;

  // Group words into lines based on CJK-aware display width
  const lines: KaraokeLine[] = [];
  let currentLine: { word: string; start: number; end: number }[] = [];
  let currentLineWidth = 0;
  let lineStartTime = wordTimings[0]?.start || 0;

  for (const w of wordTimings) {
    const wordWidth = calculateDisplayWidth(w.word);
    if (currentLineWidth + wordWidth > maxLineWidth && currentLine.length > 0) {
      // Push current line and start new one
      const lineEndTime = currentLine[currentLine.length - 1].end;
      lines.push({
        text: currentLine.map(w => w.word).join(''),
        start: lineStartTime,
        end: lineEndTime,
        words: currentLine,
      });

      currentLine = [w];
      currentLineWidth = wordWidth;
      lineStartTime = w.start;
    } else {
      currentLine.push(w);
      currentLineWidth += wordWidth;
    }
  }

  // Add remaining words as final line
  if (currentLine.length > 0) {
    lines.push({
      text: currentLine.map(w => w.word).join(''),
      start: lineStartTime,
      end: currentLine[currentLine.length - 1].end,
      words: currentLine,
    });
  }

  // Ensure we don't exceed 2 lines
  if (lines.length > 2) {
    // Merge last lines if exceeds 2
    while (lines.length > 2) {
      const secondLast = lines[lines.length - 2];
      const last = lines[lines.length - 1];
      secondLast.words.push(...last.words);
      secondLast.text = secondLast.words.map(w => w.word).join('');
      secondLast.end = last.end;
      lines.pop();
    }
  }

  return lines;
}

/**
 * Build karaoke lines preserving punctuation in display text.
 * Takes original text WITH punctuation and wordTimings from clean text.
 * Uses CJK-aware width calculation: CJK chars = 2 width units, ASCII = 1 width unit.
 */
function buildKaraokeLinesWithText(
  originalText: string,
  wordTimings: { word: string; start: number; end: number }[],
  maxCharsPerLine: number = getAssMaxCjkChars()
): KaraokeLine[] {
  if (wordTimings.length === 0) {
    return splitByCharsEqualTiming(originalText, maxCharsPerLine);
  }

  // Build clean text from wordTimings
  const cleanText = wordTimings.map(w => w.word).join('');

  // Use splitKaraokeLines to get proper line grouping with timings
  const linesWithCleanText = splitKaraokeLines(cleanText, wordTimings, maxCharsPerLine);

  // Punctuation characters to check
  const punctChars = "，。、；：！？-_—()[]{}<>" + '""' + "''" + " \t\n\r";

  function isPunct(char: string): boolean {
    return punctChars.includes(char);
  }

  // Now rebuild lines with originalText (preserving punctuation)
  const result: KaraokeLine[] = [];

  for (const line of linesWithCleanText) {
    if (line.words.length === 0) {
      result.push(line);
      continue;
    }

    // Find which wordTimings indices this line uses
    const firstWordText = line.words[0].word;
    const lastWordText = line.words[line.words.length - 1].word;

    const firstWordIdx = wordTimings.findIndex(w => w.word === firstWordText);
    const lastWordIdx = wordTimings.findIndex(w => w.word === lastWordText);

    if (firstWordIdx === -1 || lastWordIdx === -1) {
      result.push(line);
      continue;
    }

    // Count clean (non-punct) chars before firstWordIdx
    let cleanBeforeFirst = 0;
    for (let i = 0; i < firstWordIdx; i++) {
      cleanBeforeFirst += wordTimings[i].word.length;
    }

    // Total clean chars in this line
    const totalCleanChars = line.words.reduce((sum, w) => sum + w.word.length, 0);

    // Now find the corresponding position in originalText
    let origStart = -1;
    let origEnd = -1;
    let cleanCount = 0;

    for (let i = 0; i < originalText.length; i++) {
      const char = originalText[i];

      if (!isPunct(char)) {
        if (cleanCount === cleanBeforeFirst && origStart === -1) {
          origStart = i;
        }
        if (cleanCount === cleanBeforeFirst + totalCleanChars - 1) {
          origEnd = i;
          break;
        }
        cleanCount++;
      }
    }

    if (origStart === -1 || origEnd === -1) {
      result.push(line);
      continue;
    }

    // Extract the substring from originalText including any trailing punctuation
    let endIdx = origEnd;
    while (endIdx + 1 < originalText.length) {
      const nextChar = originalText[endIdx + 1];
      if (isPunct(nextChar)) {
        endIdx++;
      } else {
        break;
      }
    }

    const displayText = originalText.substring(origStart, endIdx + 1);

    result.push({
      text: displayText,
      start: line.start,
      end: line.end,
      words: line.words,
    });
  }

  return result;
}

/**
 * Fallback: split by characters with equal timing.
 * Uses CJK-aware width calculation: CJK chars = 2 width units, ASCII = 1 width unit.
 */
function splitByCharsEqualTiming(
  segment: string,
  maxCharsPerLine: number
): KaraokeLine[] {
  const cleanSegment = segment.replace(/[，。、；：！？\s]/g, '');
  if (!cleanSegment) return [];

  const maxLineWidth = maxCharsPerLine * 2;
  const lines: KaraokeLine[] = [];
  let currentLineChars: string[] = [];
  let currentLineWidth = 0;

  for (const char of cleanSegment) {
    const charWidth = calculateDisplayWidth(char);
    if (currentLineWidth + charWidth > maxLineWidth && currentLineChars.length > 0) {
      lines.push({
        text: currentLineChars.join(''),
        start: 0,
        end: 0,
        words: currentLineChars.map(c => ({ word: c, start: 0, end: 0 })),
      });
      currentLineChars = [char];
      currentLineWidth = charWidth;
    } else {
      currentLineChars.push(char);
      currentLineWidth += charWidth;
    }
  }

  // Add remaining chars as final line
  if (currentLineChars.length > 0) {
    lines.push({
      text: currentLineChars.join(''),
      start: 0,
      end: 0,
      words: currentLineChars.map(c => ({ word: c, start: 0, end: 0 })),
    });
  }

  // Keep only first 2 lines
  return lines.slice(0, 2);
}

/**
 * Format karaoke text with \k tags for ASS.
 * The \k tag duration is in centiseconds (1/100 second).
 * isEnglish: whether to add spaces between words (English needs spaces, Chinese doesn't)
 */
function formatKaraokeText(line: KaraokeLine, isFirstLine: boolean, isEnglish: boolean): string {
  if (line.words.length === 0) return '';

  // If words have timing info, use word-by-word karaoke with punctuation preserved
  const hasTiming = line.words.some(w => w.end > w.start);

  if (hasTiming) {
    // Use formatKaraokeWithTiming which handles punctuation correctly
    return formatKaraokeWithTiming(line);
  } else {
    return formatKaraokeEqualTiming(line, isEnglish);
  }
}

/**
 * Format karaoke text using actual word timings.
 * Now supports word-by-word karaoke with punctuation preserved.
 */
function formatKaraokeWithTiming(line: KaraokeLine): string {
  if (line.words.length === 0) return '';

  const result: string[] = [];
  const lineText = line.text;
  // Punctuation chars - but NOT spaces (spaces are word separators)
  const punctChars = "，。、；：！？-_—()[]{}<>" + '""' + "''" + "\t\n\r";

  // Find position of each word in lineText and extract any following punctuation
  let searchStart = 0;

  for (let i = 0; i < line.words.length; i++) {
    const w = line.words[i];
    const durationCs = Math.round((w.end - w.start) * 100);
    const safeDuration = Math.max(durationCs, 10);

    // Find the word in lineText starting from searchStart
    const wordIndex = lineText.indexOf(w.word, searchStart);
    let displayWord = w.word;

    if (wordIndex >= 0) {
      // Check if there's punctuation immediately after the word (but NOT spaces)
      const afterIndex = wordIndex + w.word.length;
      if (afterIndex < lineText.length) {
        const afterChar = lineText[afterIndex];
        // Don't consume spaces - they're just word separators
        if (punctChars.includes(afterChar) && afterChar !== ' ') {
          displayWord = w.word + afterChar;
          searchStart = afterIndex + 1; // Next word starts after the punctuation
        } else {
          searchStart = afterIndex;
        }
      }
    } else {
      // Word not found - use fallback
      searchStart += w.word.length;
    }

    const escapedWord = displayWord.replace(/\{/g, '\\{').replace(/\}/g, '\\}');
    result.push(`{\\k${safeDuration}}${escapedWord}`);
  }

  // Add space between words for English, no space for Chinese
  const separator = lineText.match(/[a-zA-Z]/) ? ' ' : '';
  return result.join(separator);
}

/**
 * Format karaoke text with equal timing per word.
 */
function formatKaraokeEqualTiming(line: KaraokeLine, isEnglish: boolean): string {
  if (line.words.length === 0) return '';

  // Assume total duration of 2 seconds per line, divided equally
  const durationPerWordCs = Math.round(2000 / line.words.length);

  const result = line.words.map(w => {
    const escapedWord = w.word.replace(/\{/g, '\\{').replace(/\}/g, '\\}');
    return `{\\k${durationPerWordCs}}${escapedWord}`;
  });

  // Add space between words for English, no space for Chinese
  return result.join(isEnglish ? ' ' : '');
}

/**
 * Generate ASS format content with karaoke-style word highlighting.
 * Fixed: removed PlayResX/PlayResY which caused subtitles not to display.
 * Simplified: use word timings directly without complex scaling.
 */
function generateASS(entries: SubtitleEntry[], isEnglish: boolean, wordEntries?: SubtitleEntry[]): string {
  // NOTE: PlayResX/PlayResY removed - they cause subtitles not to display on some ffmpeg builds
  // Simple style: white text, no outline/shadow, no blue
  const ASS_HEADER = `[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Heiti SC,10,&H00FFFFFF,&H00FF0000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Text
`;

  let assContent = ASS_HEADER;

  for (const entry of entries) {
    // Use entry timing directly
    const startTime = formatASSTime(entry.start);
    const endTime = formatASSTime(entry.end);

    // For Chinese: remove spaces
    let text = entry.text;
    if (!isEnglish) {
      text = text.replace(/ |　/g, '');
    }

    // Extract word timings for karaoke
    let karaokeLines: KaraokeLine[];

    if (wordEntries && wordEntries.length > 0) {
      // Extract word timings from TTS data
      // Use cleanText (without punctuation) for matching since TTS doesn't output punctuation
      // For English: only remove punctuation, KEEP spaces for word splitting
      // For Chinese: remove punctuation and spaces (no spaces in Chinese anyway)
      const cleanText = isEnglish
        ? text.replace(/[。，、；：！？\-\—()【】《》""''.,!?;:'"()]/g, '')
        : text.replace(/[，。、；：！？\-\—()【】《》""''.,!?;:'"()\s]/g, '');
      const wordTimings = extractWordTimingsForSegment(cleanText, wordEntries, isEnglish);

      if (wordTimings.length > 0 && wordTimings.some(w => w.end > w.start)) {
        // Use TTS word timings directly (no scaling)
        // For display text, use the ORIGINAL segment text WITH punctuation
        // but we need to map word timings to the correct positions in the segment
        karaokeLines = buildKaraokeLinesWithText(text, wordTimings);
      } else {
        // Fallback to equal timing
        karaokeLines = splitKaraokeLines(text, wordTimings.length > 0 ? wordTimings : []);
      }
    } else {
      // No TTS data, use equal timing
      karaokeLines = splitKaraokeLines(text, []);
    }

    // Format karaoke text
    let karaokeText: string;
    if (karaokeLines.length === 1) {
      karaokeText = formatKaraokeText(karaokeLines[0], true, isEnglish);
    } else if (karaokeLines.length === 2) {
      const line1 = formatKaraokeText(karaokeLines[0], true, isEnglish);
      const line2 = formatKaraokeText(karaokeLines[1], false, isEnglish);
      karaokeText = line1 + '\\N' + line2;
    } else {
      // Fallback to simple text
      karaokeText = text.replace(/\{/g, '\\{').replace(/\}/g, '\\}');
    }

    assContent += `Dialogue: 0,${startTime},${endTime},Default,${karaokeText}\n`;
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

  const audioDuration = await getAudioDuration(audioPath);
  const isEnglish = detectLanguage(narrationScript);
  const segments = splitIntoSegments(narrationScript, isEnglish);

  // Save placeholder word timing file
  const wordsDebugPath = join(outputDir, `section-${sectionId}-words.txt`);
  await writeFile(wordsDebugPath, `# Fallback: no word timing available\n# Audio duration: ${audioDuration}s\n`, 'utf-8');

  const totalChars = segments.reduce((sum, s) => sum + s.replace(/[，。、；：！？\s]/g, '').length, 0);
  const entries: SubtitleEntry[] = [];
  let currentTime = 0;

  for (const segment of segments) {
    const charCount = segment.replace(/[，。、；：！？\s]/g, '').length;
    const segDuration = totalChars > 0 ? (charCount / totalChars) * audioDuration : 0;

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
