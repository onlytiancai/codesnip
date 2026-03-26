import sharp from 'sharp';
import { mkdir } from 'fs/promises';
import { join } from 'path';
import { SectionData } from '../types/index.js';
import { logger } from '../utils/logger.js';

/**
 * Generate slide image for a section.
 * Shows key teaching points (vocabulary, grammar, context) - NOT the full narration script.
 */
export async function generateSlide(
  section: SectionData,
  outputDir: string
): Promise<string> {
  await mkdir(outputDir, { recursive: true });

  const outputPath = join(outputDir, `slide-${section.id}.png`);

  logger.debug(`Generating slide for section ${section.id}`);

  // Build vocabulary list JSX
  const vocabItems = section.vocabulary
    .slice(0, 5)
    .map(
      (v) =>
        `${v.word} - ${v.definition}${v.example ? ` (${v.example})` : ''}`
    )
    .join('\n');

  // Build grammar points JSX
  const grammarItems = section.grammarPoints
    .map((g) => `${g.rule}: ${g.explanation}`)
    .join('\n');

  const slideHtml = `
    <div style="
      width: 1080px;
      height: 1920px;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      padding: 60px;
      font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
      color: #eaeaea;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
    ">
      <div style="
        font-size: 42px;
        font-weight: bold;
        color: #ffd700;
        margin-bottom: 40px;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
      ">
        Section ${section.id}
      </div>

      <div style="
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 40px;
        border-left: 6px solid #ffd700;
      ">
        <div style="
          font-size: 28px;
          color: #ffffff;
          line-height: 1.8;
        ">
          ${escapeHtml(section.originalText)}
        </div>
      </div>

      <div style="
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 30px;
      ">
        ${
          vocabItems
            ? `
        <div style="
          background: rgba(0,255,136,0.1);
          border-radius: 16px;
          padding: 24px;
          border: 2px solid rgba(0,255,136,0.3);
        ">
          <div style="
            font-size: 32px;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
          ">
            📝 Key Vocabulary
          </div>
          <div style="
            font-size: 24px;
            color: #cccccc;
            line-height: 2;
            white-space: pre-wrap;
          ">
            ${escapeHtml(vocabItems)}
          </div>
        </div>
        `
            : ''
        }

        ${
          grammarItems
            ? `
        <div style="
          background: rgba(255,136,0,0.1);
          border-radius: 16px;
          padding: 24px;
          border: 2px solid rgba(255,136,0,0.3);
        ">
          <div style="
            font-size: 32px;
            font-weight: bold;
            color: #ff8800;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
          ">
            💡 Grammar Points
          </div>
          <div style="
            font-size: 24px;
            color: #cccccc;
            line-height: 1.8;
            white-space: pre-wrap;
          ">
            ${escapeHtml(grammarItems)}
          </div>
        </div>
        `
            : ''
        }

        ${
          section.contextExplanation
            ? `
        <div style="
          background: rgba(136,136,255,0.1);
          border-radius: 16px;
          padding: 24px;
          border: 2px solid rgba(136,136,255,0.3);
        ">
          <div style="
            font-size: 32px;
            font-weight: bold;
            color: #8888ff;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
          ">
            💬 Context
          </div>
          <div style="
            font-size: 24px;
            color: #cccccc;
            line-height: 1.8;
          ">
            ${escapeHtml(section.contextExplanation)}
          </div>
        </div>
        `
            : ''
        }
      </div>
    </div>
  `;

  // Since we don't have a real font, we'll create a simple colored rectangle as fallback
  // In production, you would bundle a proper CJK font
  const svgBuffer = await createSimpleSlideSvg(section);

  await sharp(svgBuffer).png().toFile(outputPath);

  logger.debug(`Slide saved to ${outputPath}`);

  return outputPath;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
    .replace(/\n/g, '<br/>');
}

/**
 * Wrap text into lines with proper line breaks.
 * Returns an array of text lines (not SVG).
 */
function wrapText(text: string, maxCharsPerLine: number, indent: string = ''): string[] {
  const lines: string[] = [];
  const paragraphs = text.split('\n');

  for (const paragraph of paragraphs) {
    const words = paragraph.split(/\s+/);
    let currentLine = indent;

    for (const word of words) {
      if (currentLine.length + word.length + 1 > maxCharsPerLine && currentLine.length > indent.length) {
        lines.push(currentLine);
        currentLine = indent + word;
      } else {
        currentLine += (currentLine === indent ? '' : ' ') + word;
      }
    }

    if (currentLine.length > indent.length) {
      lines.push(currentLine);
    }
  }

  return lines;
}

/**
 * Convert array of text lines to SVG tspan elements.
 */
function linesToTspan(lines: string[], maxLines: number): string {
  const displayLines = lines.slice(0, maxLines);
  const hasMore = lines.length > maxLines;
  return displayLines
    .map((line, i) => `<tspan x="40" dy="${i === 0 ? 0 : 36}">${escapeXml(line)}</tspan>`)
    .join('') + (hasMore ? `<tspan dy="36">...</tspan>` : '');
}

/**
 * Create vocabulary section SVG with proper wrapping and phonetics.
 */
function createVocabSection(vocabulary: SectionData['vocabulary'], startY: number): string {
  if (!vocabulary || vocabulary.length === 0) {
    return '';
  }

  const vocabLines: string[] = [];

  for (const v of vocabulary.slice(0, 4)) {
    // Word with phonetic
    vocabLines.push(`<tspan x="40" dy="0" font-size="26" fill="#00ff88" font-weight="bold">${escapeXml(v.word)}</tspan>`);
    vocabLines.push(`<tspan x="45" dy="0" font-size="20" fill="#888888"> ${escapeXml(v.phonetic)}</tspan>`);
    // Definition on new line
    vocabLines.push(`<tspan x="50" dy="32" font-size="22" fill="#cccccc">${escapeXml(v.definition)}</tspan>`);
    // Example if exists
    if (v.example) {
      const exampleText = `"${escapeXml(v.example)}"`;
      vocabLines.push(`<tspan x="50" dy="28" font-size="18" fill="#888888" font-style="italic">${exampleText}</tspan>`);
    }
    vocabLines.push(`<tspan x="40" dy="36" font-size="18" fill="transparent">placeholder</tspan>`);
  }

  return `
    <text x="40" y="${startY}" font-size="28" fill="#00ff88" font-weight="bold">📝 Key Vocabulary</text>
    <text x="40" y="${startY + 45}" font-family="sans-serif">
      ${vocabLines.join('\n')}
    </text>
  `;
}

/**
 * Create grammar section SVG with proper wrapping.
 */
function createGrammarSection(grammarPoints: SectionData['grammarPoints'], startY: number): string {
  if (!grammarPoints || grammarPoints.length === 0) {
    return '';
  }

  const grammarLines: string[] = [];

  for (const g of grammarPoints.slice(0, 2)) {
    grammarLines.push(`<tspan x="40" dy="0" font-size="24" fill="#ff8800" font-weight="bold">${escapeXml(g.rule)}</tspan>`);
    grammarLines.push(`<tspan x="45" dy="30" font-size="20" fill="#cccccc">${escapeXml(g.explanation)}</tspan>`);
    grammarLines.push(`<tspan x="40" dy="30" font-size="18" fill="transparent">placeholder</tspan>`);
  }

  return `
    <text x="40" y="${startY}" font-size="28" fill="#ff8800" font-weight="bold">💡 Grammar Points</text>
    <text x="40" y="${startY + 40}" font-family="sans-serif">
      ${grammarLines.join('\n')}
    </text>
  `;
}

async function createSimpleSlideSvg(section: SectionData): Promise<Buffer> {
  const WIDTH = 1080;
  const HEIGHT = 1920;
  const PADDING = 40;

  // Calculate positions
  const titleY = 70;
  const originalTextY = 130;
  const originalTextHeight = 180;
  const sectionStartY = originalTextY + originalTextHeight + 50;

  // Create original text with wrapping
  const originalTextLines = wrapText(section.originalText, 55, '  ');
  const originalTextSvg = linesToTspan(originalTextLines, 6);

  // Build vocabulary and grammar sections
  let currentY = sectionStartY;
  const vocabSvg = createVocabSection(section.vocabulary, currentY);
  currentY += section.vocabulary.slice(0, 4).length * 120 + 60;

  const grammarSvg = createGrammarSection(section.grammarPoints, currentY);
  currentY += section.grammarPoints.slice(0, 2).length * 80 + 60;

  // Context section
  let contextSvg = '';
  if (section.contextExplanation) {
    const contextLines = wrapText(section.contextExplanation, 55, '  ');
    const displayLines = linesToTspan(contextLines, 4);
    contextSvg = `
      <text x="40" y="${currentY}" font-size="28" fill="#8888ff" font-weight="bold">💬 Context</text>
      <text x="40" y="${currentY + 45}" font-size="22" fill="#cccccc" font-family="sans-serif">
        ${displayLines}
      </text>
    `;
  }

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${WIDTH}" height="${HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1a2e"/>
      <stop offset="100%" style="stop-color:#16213e"/>
    </linearGradient>
    <style>
      text { font-family: 'Helvetica Neue', Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }
    </style>
  </defs>

  <!-- Background -->
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#bg)"/>

  <!-- Title -->
  <text x="540" y="${titleY}" text-anchor="middle" font-size="48" font-weight="bold" fill="#ffd700">
    Section ${section.id}
  </text>

  <!-- Original Text Card -->
  <rect x="${PADDING}" y="${originalTextY}" width="${WIDTH - PADDING * 2}" height="${originalTextHeight}" rx="20" fill="rgba(255,255,255,0.08)" stroke="#ffd700" stroke-width="2"/>
  <text x="${PADDING + 15}" y="${originalTextY + 40}" font-size="24" fill="#ffffff" font-weight="bold">📖 Original Text</text>
  <text x="${PADDING + 15}" y="${originalTextY + 80}" font-size="20" fill="#e0e0e0" font-family="sans-serif">
    ${originalTextSvg}
  </text>

  <!-- Teaching Content Sections -->
  ${vocabSvg}
  ${grammarSvg}
  ${contextSvg}
</svg>`;

  return Buffer.from(svg);
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}
