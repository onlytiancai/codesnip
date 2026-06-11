import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { ArticleScript } from '../types/index.js';
import { logger } from '../utils/logger.js';
import { config } from '../config/index.js';
import { loadPromptsConfig, interpolateTemplate } from './promptLoader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface EvaluationResult {
  overall: number;
  dimensions: {
    name: string;
    score: number;
    maxScore: number;
    issues: string[];
  }[];
  totalScripts: number;
  totalCharacters: number;
  suggestions: string[];
  rawResponse?: string;
}

interface AIScriptResponse {
  overall: number;
  languagePurity: { score: number; issues: string[] };
  naturalness: { score: number; issues: string[] };
  professionalTone: { score: number; issues: string[] };
  contentAccuracy: { score: number; issues: string[] };
  structureCoherence: { score: number; issues: string[] };
  conciseness: { score: number; issues: string[] };
  overallCoherence: { score: number; issues: string[] };
  suggestions: string[];
}

/**
 * Collect AI-generated scripts (exclude reading parts which are original text)
 */
function collectAIScripts(script: ArticleScript): {
  scripts: { type: string; content: string }[];
  totalCharacters: number;
  originalTexts: string[];
} {
  const scripts: { type: string; content: string }[] = [];
  const originalTexts: string[] = [];

  // Add intro (AI generated)
  scripts.push({ type: 'intro', content: script.intro.script });

  // Add segment parts (exclude reading - it's original English text)
  for (const segment of script.segments) {
    originalTexts.push(segment.originalText);
    for (const part of segment.parts) {
      // Skip reading parts - they are original English, not AI generated
      if (part.type === 'reading') continue;
      if (part.script) {
        scripts.push({ type: part.type, content: part.script });
      }
    }
  }

  // Add outro (AI generated)
  scripts.push({ type: 'outro', content: script.outro.script });

  const totalCharacters = scripts.reduce((sum, s) => sum + s.content.length, 0);

  return { scripts, totalCharacters, originalTexts };
}

/**
 * Evaluate the quality of article-script.json using AI
 */
export async function evaluateScript(scriptPath: string): Promise<EvaluationResult> {
  logger.info(`Evaluating script with AI: ${scriptPath}`);

  const content = await readFile(scriptPath, 'utf-8');
  const script: ArticleScript = JSON.parse(content);

  const { scripts, totalCharacters, originalTexts } = collectAIScripts(script);

  // Format scripts for AI
  const scriptsText = scripts.map(s => `[${s.type.toUpperCase()}]\n${s.content}`).join('\n\n---\n\n');

  // Format original texts
  const originalsText = originalTexts.map((t, i) => `[ORIGINAL ${i + 1}]\n${t}`).join('\n\n---\n\n');

  // Load evaluation template
  const promptsConfig = await loadPromptsConfig();
  const evaluatePrompt = promptsConfig.prompts.evaluate;
  if (!evaluatePrompt?.template) {
    throw new Error('Evaluate prompt template not found');
  }

  const prompt = interpolateTemplate(evaluatePrompt.template, {
    originalsText,
    scriptsText,
  });

  try {
    const response = await fetch(`${config.LLM_BASE_URL}/text/chatcompletion_v2`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.LLM_API_KEY}`,
      },
      body: JSON.stringify({
        model: config.LLM_MODEL,
        messages: [
          {
            role: 'system',
            content: 'You are a professional English teaching evaluation expert. Evaluate scripts objectively and provide constructive feedback in JSON format.',
          },
          { role: 'user', content: prompt },
        ],
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      throw new Error(`AI API error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      throw new Error('Empty response from AI');
    }

    logger.debug(`AI evaluation response: ${content.substring(0, 500)}...`);

    const parsed = extractJSON(content) as unknown as AIScriptResponse;

    // Convert to EvaluationResult format
    const dimensions: EvaluationResult['dimensions'] = [
      {
        name: '语言纯净度 (Language Purity)',
        score: parsed.languagePurity?.score ?? 80,
        maxScore: 100,
        issues: parsed.languagePurity?.issues ?? [],
      },
      {
        name: '自然流畅度 (Naturalness)',
        score: parsed.naturalness?.score ?? 80,
        maxScore: 100,
        issues: parsed.naturalness?.issues ?? [],
      },
      {
        name: '专业教师语气 (Professional Tone)',
        score: parsed.professionalTone?.score ?? 80,
        maxScore: 100,
        issues: parsed.professionalTone?.issues ?? [],
      },
      {
        name: '内容准确性 (Content Accuracy)',
        score: parsed.contentAccuracy?.score ?? 80,
        maxScore: 100,
        issues: parsed.contentAccuracy?.issues ?? [],
      },
      {
        name: '结构连贯性 (Structure Coherence)',
        score: parsed.structureCoherence?.score ?? 80,
        maxScore: 100,
        issues: parsed.structureCoherence?.issues ?? [],
      },
      {
        name: '简洁适度 (Conciseness)',
        score: parsed.conciseness?.score ?? 80,
        maxScore: 100,
        issues: parsed.conciseness?.issues ?? [],
      },
      {
        name: '整体连贯性 (Overall Coherence)',
        score: parsed.overallCoherence?.score ?? 80,
        maxScore: 100,
        issues: parsed.overallCoherence?.issues ?? [],
      },
    ];

    return {
      overall: parsed.overall ?? dimensions.reduce((sum, d) => sum + d.score, 0) / dimensions.length,
      dimensions,
      totalScripts: scripts.length,
      totalCharacters,
      suggestions: parsed.suggestions ?? [],
      rawResponse: content,
    };
  } catch (error) {
    logger.error('AI evaluation failed:', error);
    throw error;
  }
}

/**
 * Extract JSON from AI response
 */
function extractJSON(content: string): Record<string, unknown> {
  let cleaned = content.replace(/<think>[\s\S]*?<\/think>/gi, '');
  cleaned = cleaned.replace(/<think>[^<]*$/gm, '');

  const codeMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeMatch) {
    cleaned = codeMatch[1];
  }

  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');

  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    const jsonStr = cleaned.substring(firstBrace, lastBrace + 1);
    try {
      return JSON.parse(jsonStr);
    } catch (e) {
      logger.debug(`JSON parse error: ${jsonStr.substring(0, 200)}`);
    }
  }

  return {};
}

/**
 * Print evaluation results
 */
export function printEvaluationResult(result: EvaluationResult): void {
  console.log('\n' + '='.repeat(60));
  console.log('📊 Script Quality Evaluation Report (AI-Generated)');
  console.log('='.repeat(60));
  console.log(`\n🎯 Overall Score: ${result.overall.toFixed(1)}/100`);
  console.log(`📝 Total Scripts: ${result.totalScripts}`);
  console.log(`📄 Total Characters: ${result.totalCharacters}`);
  console.log('\n' + '-'.repeat(60));
  console.log('📋 Dimension Scores:');
  console.log('-'.repeat(60));

  for (const dim of result.dimensions) {
    const bar = '█'.repeat(Math.round(dim.score / 10)) + '░'.repeat(10 - Math.round(dim.score / 10));
    const status = dim.score >= 90 ? '✅' : dim.score >= 70 ? '⚠️' : '❌';
    console.log(`\n${status} ${dim.name}`);
    console.log(`   [${bar}] ${dim.score}/100`);
    if (dim.issues && dim.issues.length > 0) {
      for (const issue of dim.issues.slice(0, 3)) {
        console.log(`   • ${issue}`);
      }
    }
  }

  console.log('\n' + '-'.repeat(60));
  console.log('💡 Suggestions:');
  console.log('-'.repeat(60));
  if (result.suggestions && result.suggestions.length > 0) {
    for (const suggestion of result.suggestions) {
      console.log(`  • ${suggestion}`);
    }
  } else {
    console.log('  整体质量良好，无需重大调整');
  }

  console.log('\n' + '='.repeat(60) + '\n');
}
