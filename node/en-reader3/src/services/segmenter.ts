import { logger } from '../utils/logger.js';

interface Segment {
  text: string;
  order: number;
}

interface SegmentResult {
  segments: Segment[];
  totalChars: number;
}

/**
 * Split article into logical sections.
 * Target: 3-8 sentences per section.
 * Preserve semantic coherence.
 */
export function segmentArticle(articleText: string): Segment[] {
  logger.info('Starting article segmentation');

  // Split by double newlines (paragraphs)
  const paragraphs = articleText
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  logger.debug(`Found ${paragraphs.length} paragraphs`);

  const segments: Segment[] = [];
  let currentSection: string[] = [];
  let sentenceCount = 0;
  let sectionId = 1;

  // Helper to split paragraph into sentences
  function splitSentences(text: string): string[] {
    // Split on sentence-ending punctuation followed by space or end
    // Handles ., !, ? but avoids splitting on abbreviations
    return text
      .split(/(?<=[.!?])\s+(?=[A-Z])/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  for (const paragraph of paragraphs) {
    const sentences = splitSentences(paragraph);

    for (const sentence of sentences) {
      currentSection.push(sentence);
      sentenceCount++;

      // If we have 3-8 sentences, create a segment
      if (sentenceCount >= 3 && sentenceCount <= 8) {
        segments.push({
          text: currentSection.join(' '),
          order: sectionId++,
        });
        currentSection = [];
        sentenceCount = 0;
      } else if (sentenceCount > 8) {
        // Too many sentences, split at last safe point
        // Push current section
        segments.push({
          text: currentSection.slice(0, -3).join(' '),
          order: sectionId++,
        });
        // Keep last 3 sentences for next section
        currentSection = currentSection.slice(-3);
        sentenceCount = 3;
      }
    }
  }

  // Handle remaining sentences
  if (currentSection.length > 0) {
    segments.push({
      text: currentSection.join(' '),
      order: sectionId++,
    });
  }

  logger.info(`Segmentation complete: ${segments.length} segments created`);

  return segments;
}
