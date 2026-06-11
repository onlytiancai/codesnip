import { readFile } from 'fs/promises';
import { logger } from '../utils/logger.js';

export interface WordTimingEntry {
  word: string;
  start: number;
  end: number;
}

export interface SentenceTimingEntry {
  start: number;
  end: number;
  words: WordTimingEntry[];
}

/**
 * Parse SRT content and extract word-level timing.
 * SRT format:
 * 1
 * 00:00:01,234 --> 00:00:03,456
 * word1 word2 word3
 */
export async function extractWordTimings(srtPath: string): Promise<WordTimingEntry[]> {
  const content = await readFile(srtPath, 'utf-8');
  return parseWordTimings(content);
}

/**
 * Parse SRT content string and extract word-level timing.
 */
export function parseWordTimings(srtContent: string): WordTimingEntry[] {
  const entries: WordTimingEntry[] = [];
  const blocks = srtContent.trim().split(/\n\n+/);

  for (const block of blocks) {
    const lines = block.split('\n');
    if (lines.length < 3) continue;

    // Parse time line: "00:00:01,234 --> 00:00:03,456"
    const timeMatch = lines[1].match(
      /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/
    );
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

    const textLine = lines.slice(2).join(' ');
    const words = textLine.trim().split(/\s+/);

    if (words.length === 0) continue;

    // Estimate duration per word based on total time
    const duration = end - start;
    const avgWordDuration = duration / words.length;

    let currentTime = start;
    for (const word of words) {
      if (!word) continue;

      // Use average duration for each word
      entries.push({
        word: word.replace(/[。！？；：、，""''【】（）]/g, ''), // Remove punctuation
        start: currentTime,
        end: currentTime + avgWordDuration,
      });
      currentTime += avgWordDuration;
    }
  }

  logger.debug(`Extracted ${entries.length} word timings from SRT`);
  return entries;
}

/**
 * Group word timings by sentences.
 */
export function groupWordsBySentence(wordTimings: WordTimingEntry[]): SentenceTimingEntry[] {
  const sentences: SentenceTimingEntry[] = [];
  let currentSentence: SentenceTimingEntry | null = null;

  for (const entry of wordTimings) {
    // Check if this is the start of a new sentence (ends with punctuation or is first word)
    const isSentenceEnd = /[。！？；]$/.test(entry.word);

    if (!currentSentence) {
      currentSentence = {
        start: entry.start,
        end: entry.end,
        words: [entry],
      };
    } else {
      currentSentence.words.push(entry);
      currentSentence.end = entry.end;
    }

    if (isSentenceEnd && currentSentence) {
      sentences.push(currentSentence);
      currentSentence = null;
    }
  }

  // Don't forget the last sentence
  if (currentSentence && currentSentence.words.length > 0) {
    sentences.push(currentSentence);
  }

  return sentences;
}
