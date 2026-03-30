import { readFile, readdir } from 'fs/promises';
import { join, dirname, relative } from 'path';
import { getAssMaxCjkChars } from '../config/index.js';

export interface CheckResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface KaraokeWord {
  word: string;
  duration: number; // in milliseconds
}

/**
 * Calculate display width of a string.
 * CJK characters: width = 2
 * Half-width (ASCII, numbers, punctuation): width = 1
 * Full-width punctuation (，。): width = 2
 */
export function calculateDisplayWidth(text: string): number {
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
 * Count punctuation marks in a line.
 * Returns { commas, periods } counts.
 */
function countPunctuation(line: string): { commas: number; periods: number } {
  let commas = 0;
  let periods = 0;
  for (const char of line) {
    if (char === ',') commas++;
    if (char === '.' || char === '。') periods++;
  }
  return { commas, periods };
}

/**
 * Extract karaoke timings from an ASS dialogue line.
 * Parses {\k100}word patterns to get word and duration.
 */
export function extractKaraokeTimings(dialogueLine: string): KaraokeWord[] {
  const results: KaraokeWord[] = [];
  // Match {\k\d+}word patterns
  const regex = /\{\\k(\d+)\}([^{]+)/g;
  let match;

  while ((match = regex.exec(dialogueLine)) !== null) {
    // \k value in ASS is in centiseconds (1/100 second), convert to milliseconds
    const duration = parseInt(match[1], 10) * 10;
    const word = match[2].trim();
    if (word) {
      results.push({ word, duration });
    }
  }

  return results;
}

/**
 * Parse a words.txt file and return word timings.
 * Format: start:end:word
 */
export function parseWordsTxt(content: string): KaraokeWord[] {
  const results: KaraokeWord[] = [];
  const lines = content.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const parts = trimmed.split(':');
    if (parts.length >= 3) {
      const start = parseFloat(parts[0]);
      const end = parseFloat(parts[1]);
      const word = parts.slice(2).join(':').trim();

      if (!isNaN(start) && !isNaN(end) && word) {
        results.push({
          word,
          duration: Math.round((end - start) * 1000),
        });
      }
    }
  }

  return results;
}

/**
 * Parse ASS timestamp to seconds.
 * Format: H:MM:SS.CS (centiseconds)
 */
function parseAssTimestamp(timestamp: string): number {
  // ASS timestamp format: H:MM:SS.CC
  const match = timestamp.match(/^(\d+):(\d+):(\d+)\.(\d+)$/);
  if (!match) return -1;
  const hours = parseInt(match[1], 10);
  const minutes = parseInt(match[2], 10);
  const seconds = parseInt(match[3], 10);
  const centiseconds = parseInt(match[4], 10);
  return hours * 3600 + minutes * 60 + seconds + centiseconds / 100;
}

/**
 * Check karaoke tags in a dialogue line for validity.
 * Returns error messages for invalid \k tags.
 */
function checkKaraokeTags(textPart: string, lineNumber: number): string[] {
  const errors: string[] = [];

  // Match \k tags: {\k123}
  const karaokeTagRegex = /\{\\k(\d+)\}/g;
  let match;
  let lastEndPos = -1;

  while ((match = karaokeTagRegex.exec(textPart)) !== null) {
    const kValue = parseInt(match[1], 10);
    const startPos = match.index;

    // Check for \k0 or negative
    if (kValue <= 0) {
      errors.push(`[ERROR] Line ${lineNumber} has \\k${kValue} (zero or negative karaoke duration)`);
    }

    // Check for suspiciously long \k values (>200cs = 2 seconds per word)
    if (kValue > 200) {
      errors.push(`[WARN] Line ${lineNumber} has \\k${kValue} (>200cs = 2s per word, may be incorrect)`);
    }

    // Check for overlapping or out-of-order tags (tags should not overlap)
    if (lastEndPos >= startPos) {
      errors.push(`[ERROR] Line ${lineNumber} has overlapping karaoke tags`);
    }
    lastEndPos = startPos + match[0].length;
  }

  return errors;
}

/**
 * Compare ASS karaoke timings with words.txt timings.
 * Returns validation result with any discrepancies.
 */
export function compareTimings(
  assTimings: KaraokeWord[],
  wordsTxtTimings: KaraokeWord[],
  toleranceMs: number = 50
): { valid: boolean; mismatches: { word: string; assDuration: number; wordsDuration: number; diff: number }[] } {
  const mismatches: { word: string; assDuration: number; wordsDuration: number; diff: number }[] = [];

  const minLen = Math.min(assTimings.length, wordsTxtTimings.length);
  for (let i = 0; i < minLen; i++) {
    const assWord = assTimings[i];
    const wordsWord = wordsTxtTimings[i];

    // Compare by word content
    if (assWord.word.toLowerCase() !== wordsWord.word.toLowerCase()) {
      continue; // Skip if words don't match
    }

    const diff = Math.abs(assWord.duration - wordsWord.duration);
    if (diff > toleranceMs) {
      mismatches.push({
        word: assWord.word,
        assDuration: assWord.duration,
        wordsDuration: wordsWord.duration,
        diff,
      });
    }
  }

  return {
    valid: mismatches.length === 0,
    mismatches,
  };
}

/**
 * Check punctuation in a dialogue line.
 * Returns error message if line has too many commas or periods.
 */
function checkPunctuation(line: string, lineNumber: number): string | null {
  const { commas, periods } = countPunctuation(line);
  if (commas > 1) {
    return `[ERROR] Line ${lineNumber} has ${commas} commas (should split into separate lines)`;
  }
  if (periods > 1) {
    return `[ERROR] Line ${lineNumber} has ${periods} sentence endings (should split into separate lines)`;
  }
  return null;
}

/**
 * Check display width of a dialogue line.
 * Returns error message if line exceeds max width.
 */
function checkLineWidth(line: string, lineNumber: number): string | null {
  const maxCjkChars = getAssMaxCjkChars();
  // CJK-aware: each CJK char counts as 2 width units, each ASCII counts as 1
  const width = calculateDisplayWidth(line);
  // Convert to "effective CJK chars" (width / 2 gives CJK char count, width gives ASCII count)
  const effectiveCjkChars = Math.ceil(width / 2);

  if (effectiveCjkChars > maxCjkChars) {
    return `[ERROR] Line ${lineNumber} exceeds max CJK chars (${effectiveCjkChars} > ${maxCjkChars})`;
  }

  return null;
}

/**
 * Check a single ASS file.
 * Returns CheckResult with errors and warnings.
 */
export async function checkASSFile(
  assPath: string,
  wordsTxtPath: string | null = null
): Promise<CheckResult> {
  const errors: string[] = [];
  const warnings: string[] = [];

  try {
    const assContent = await readFile(assPath, 'utf-8');
    const lines = assContent.split('\n');

    let dialogueLineNumber = 0;
    let lastStartTime = -1;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Only check dialogue lines
      if (!line.startsWith('Dialogue:')) continue;
      dialogueLineNumber++;

      // Extract the text part (after the 9th comma, which is the Style field)
      const parts = line.split(',');
      if (parts.length < 10) continue;

      // Parts[1] is Start time, Parts[2] is End time
      const startTime = parseAssTimestamp(parts[1].trim());
      const textPart = parts.slice(9).join(',');

      // Check 0: Time ordering - Start time should be >= previous Start time
      if (startTime >= 0 && lastStartTime >= 0 && startTime < lastStartTime) {
        errors.push(
          `[ERROR] Line ${dialogueLineNumber} has out-of-order timestamp: ${parts[1].trim()} < previous ${lastStartTime.toFixed(2)}s`
        );
      }
      if (startTime >= 0) {
        lastStartTime = startTime;
      }

      // Check 1: Multiple punctuation
      const punctError = checkPunctuation(textPart, dialogueLineNumber);
      if (punctError) {
        errors.push(punctError);
      }

      // Check 2: Line width
      const widthError = checkLineWidth(textPart, dialogueLineNumber);
      if (widthError) {
        errors.push(widthError);
      }

      // Check 3: Karaoke tag validity (\k0, negative, too long)
      const karaokeErrors = checkKaraokeTags(textPart, dialogueLineNumber);
      errors.push(...karaokeErrors.filter(e => e.startsWith('[ERROR]')));
      warnings.push(...karaokeErrors.filter(e => e.startsWith('[WARN]')));

      // Check 4: Karaoke timing validation (if words.txt exists)
      if (wordsTxtPath) {
        try {
          const wordsTxtContent = await readFile(wordsTxtPath, 'utf-8');
          const wordsTxtTimings = parseWordsTxt(wordsTxtContent);
          const assTimings = extractKaraokeTimings(textPart);

          if (assTimings.length > 0 && wordsTxtTimings.length > 0) {
            const timingResult = compareTimings(assTimings, wordsTxtTimings);
            for (const mismatch of timingResult.mismatches) {
              warnings.push(
                `[WARN] Line ${dialogueLineNumber} karaoke time mismatch for "${mismatch.word}": ` +
                  `ASS=${mismatch.assDuration}ms, words.txt=${mismatch.wordsDuration}ms (diff=${mismatch.diff}ms)`
              );
            }
          }
        } catch {
          // words.txt read error - skip timing check
        }
      }
    }
  } catch (error) {
    errors.push(`[ERROR] Failed to read ASS file: ${assPath}`);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Recursively find all ASS files in a directory.
 */
async function findAssFiles(dir: string): Promise<string[]> {
  const results: string[] = [];

  try {
    const entries = await readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(dir, entry.name);

      if (entry.isDirectory()) {
        const subResults = await findAssFiles(fullPath);
        results.push(...subResults);
      } else if (entry.isFile() && entry.name.endsWith('.ass')) {
        results.push(fullPath);
      }
    }
  } catch {
    // Directory read error - return empty
  }

  return results;
}

/**
 * Get the corresponding words.txt path for an ASS file.
 * ASS file: section-1.ass -> section-1-words.txt
 */
function getWordsTxtPath(assPath: string): string | null {
  if (assPath.endsWith('.ass')) {
    const wordsTxtPath = assPath.replace(/\.ass$/, '-words.txt');
    return wordsTxtPath;
  }
  return null;
}

/**
 * Check all ASS files in a directory recursively.
 */
export async function checkASSFiles(outputDir: string): Promise<{
  totalFiles: number;
  totalErrors: number;
  totalWarnings: number;
  results: { file: string; result: CheckResult }[];
}> {
  const assFiles = await findAssFiles(outputDir);

  const results: { file: string; result: CheckResult }[] = [];
  let totalErrors = 0;
  let totalWarnings = 0;

  for (const assFile of assFiles) {
    const wordsTxtPath = getWordsTxtPath(assFile);
    const result = await checkASSFile(assFile, wordsTxtPath);

    results.push({
      file: relative(outputDir, assFile),
      result,
    });

    totalErrors += result.errors.length;
    totalWarnings += result.warnings.length;
  }

  return {
    totalFiles: assFiles.length,
    totalErrors,
    totalWarnings,
    results,
  };
}

/**
 * Print ASS check results in a formatted way.
 */
export function printASSCheckResults(
  outputDir: string,
  checkResults: Awaited<ReturnType<typeof checkASSFiles>>
): void {
  console.log(`\nChecking ASS files in ${outputDir}...`);
  console.log(`Found ${checkResults.totalFiles} ASS files\n`);

  for (const { file, result } of checkResults.results) {
    // Print errors
    for (const error of result.errors) {
      console.log(`[ERROR] ${file}: ${error.replace('[ERROR] ', '')}`);
    }
    // Print warnings
    for (const warning of result.warnings) {
      console.log(`[WARN] ${file}: ${warning.replace('[WARN] ', '')}`);
    }
  }

  console.log(`\nSummary: ${checkResults.totalFiles} files checked, ` +
    `${checkResults.totalErrors} errors, ${checkResults.totalWarnings} warnings`);

  if (checkResults.totalErrors === 0 && checkResults.totalWarnings === 0) {
    console.log('All ASS files passed validation!');
  }
}
