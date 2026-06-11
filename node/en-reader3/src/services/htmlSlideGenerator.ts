import { mkdir, writeFile } from 'fs/promises';
import { join } from 'path';
import { SectionData, PartType } from '../types/index.js';
import { WordTimingEntry } from './wordTimingExtractor.js';
import { renderTemplate } from './templateEngine.js';
import { logger } from '../utils/logger.js';

export interface HtmlSlideOptions {
  section: SectionData;
  partType?: PartType;
  wordTimings: WordTimingEntry[];
  audioPath: string;
  outputDir: string;
  partId?: number;
}

interface SentenceGroup {
  text: string;
  start: number;
  end: number;
  words: { word: string; start: number; end: number; index: number }[];
}

function groupWordsIntoSentences(wordTimings: WordTimingEntry[]): SentenceGroup[] {
  const sentences: SentenceGroup[] = [];
  let currentSentence: SentenceGroup | null = null;
  const MAX_SENTENCE_DURATION = 2.0;

  for (let i = 0; i < wordTimings.length; i++) {
    const entry = wordTimings[i];
    const wordText = entry.word.trim();

    const sentenceDuration = currentSentence ? entry.start - currentSentence.start : 0;

    if (!currentSentence || sentenceDuration > MAX_SENTENCE_DURATION) {
      if (currentSentence && currentSentence.words.length > 0) {
        sentences.push(currentSentence);
      }
      currentSentence = {
        text: wordText,
        start: entry.start,
        end: entry.end,
        words: [{ word: wordText, start: entry.start, end: entry.end, index: i }],
      };
    } else {
      currentSentence.text += ' ' + wordText;
      currentSentence.end = entry.end;
      currentSentence.words.push({ word: wordText, start: entry.start, end: entry.end, index: i });
    }
  }

  if (currentSentence && currentSentence.words.length > 0) {
    sentences.push(currentSentence);
  }

  return sentences;
}

function getPartLabel(partType: PartType | undefined, partId: number): string {
  if (!partType) return `Part ${partId}`;

  const labels: Record<PartType, string> = {
    [PartType.READING]: '📖 朗读',
    [PartType.TRANSLATION]: '📝 翻译讲解',
    [PartType.VOCABULARY]: '📚 词汇学习',
    [PartType.GRAMMAR]: '💡 语法解析',
    [PartType.EXPLANATION]: '💬 背景解读',
  };

  return labels[partType];
}

function getTemplateName(partType: PartType | undefined): string {
  if (!partType) return 'slide-original.html';

  const templates: Record<PartType, string> = {
    [PartType.READING]: 'slide-original.html',
    [PartType.TRANSLATION]: 'slide-translation.html',
    [PartType.VOCABULARY]: 'slide-vocabulary.html',
    [PartType.GRAMMAR]: 'slide-grammar.html',
    [PartType.EXPLANATION]: 'slide-explanation.html',
  };

  return templates[partType];
}

function buildVocabItemsHtml(section: SectionData): string {
  return section.vocabulary
    .slice(0, 5)
    .map((v) => `
    <div class="vocab-item">
      <div class="vocab-word">${escapeHtml(v.word)} <span class="phonetic">${escapeHtml(v.phonetic)}</span></div>
      <div class="vocab-def">${escapeHtml(v.definition)}</div>
      ${v.example ? `<div class="vocab-example">"${escapeHtml(v.example)}"</div>` : ''}
    </div>
  `)
    .join('');
}

function buildGrammarItemsHtml(section: SectionData): string {
  return section.grammarPoints
    .slice(0, 2)
    .map(
      (g) => `
    <div class="grammar-item">
      <div class="grammar-rule">${escapeHtml(g.rule)}</div>
      <div class="grammar-exp">${escapeHtml(g.explanation)}</div>
      ${g.example ? `<div class="grammar-example">例句: ${escapeHtml(g.example)}</div>` : ''}
    </div>
  `
    )
    .join('');
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Generate HTML slide using template engine.
 */
export async function generateHtmlSlide(options: HtmlSlideOptions): Promise<string> {
  const { section, partType, wordTimings, audioPath, outputDir, partId = 1 } = options;

  await mkdir(outputDir, { recursive: true });

  const outputPath = join(outputDir, `slide-${section.id}-${partId}.html`);
  const templateName = getTemplateName(partType);
  const partLabel = getPartLabel(partType, partId);

  logger.debug(`Generating HTML slide for section ${section.id}, part ${partId} using ${templateName}`);

  // Group word timings into sentences
  const sentences = groupWordsIntoSentences(wordTimings);
  const sentencesJson = JSON.stringify(sentences);

  // Build template data based on part type
  const templateData: Record<string, string | string[] | number | boolean | undefined> = {
    title: `Section ${section.id}`,
    partLabel,
    originalText: section.originalText,
    audioPath: audioPath.replace(/\\/g, '/'),
    sentencesJson,
    translationText: section.narrationScript || '',
    contextText: section.contextExplanation || '',
    vocabItemsHtml: buildVocabItemsHtml(section),
    grammarItemsHtml: buildGrammarItemsHtml(section),
  };

  const htmlContent = await renderTemplate(templateName, templateData);

  await writeFile(outputPath, htmlContent, 'utf-8');

  logger.debug(`HTML slide saved to ${outputPath}`);

  return outputPath;
}
