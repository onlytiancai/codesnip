import { config } from '../config/index.js';
import { logger } from '../utils/logger.js';
import { loadPromptsConfig, interpolateTemplate } from './promptLoader.js';
import {
  ArticleSection,
  ArticleScript,
  SegmentPart,
  PartType,
  VocabularyItem,
  GrammarPoint,
  SegmentGenerationResponse,
  PromptsConfig,
} from '../types/index.js';

/**
 * Generate article script with intro, segments (each with 6 parts), and outro.
 * Intro/outro get full original text for context.
 */
export async function generateArticleScript(
  sections: { text: string; order: number }[],
  title: string,
  fullOriginalText: string
): Promise<ArticleScript> {
  logger.info(`Generating article script for "${title}" with ${sections.length} segments`);

  // Load prompts configuration
  const promptsConfig = await loadPromptsConfig();

  // Generate intro (with full text for context)
  const intro = await generateIntro(title, sections.length, fullOriginalText, promptsConfig);

  // Generate segments with 6 parts each (single AI call per segment)
  const segments = [];
  for (const section of sections) {
    const segmentScript = await generateSegmentAllParts(section.order, section.text, title, promptsConfig);
    segments.push(segmentScript);
  }

  // Generate outro (with full text for context)
  const outro = await generateOutro(title, fullOriginalText, promptsConfig);

  logger.info(`Article script generated: 1 intro, ${segments.length} segments, 1 outro`);

  return { intro, segments, outro };
}

/**
 * Generate intro content using configured prompts.
 */
async function generateIntro(
  title: string,
  segmentCount: number,
  fullOriginalText: string,
  promptsConfig: PromptsConfig
): Promise<{ title: string; script: string }> {
  const template = promptsConfig.prompts.intro.template!;
  const prompt = interpolateTemplate(template, {
    title,
    segmentCount,
    originalText: fullOriginalText,
  });

  try {
    const response = await callAI(promptsConfig.systemPrompt, prompt, promptsConfig.temperature);
    const parsed = extractJSON(response) as { title?: string; script?: string };

    return {
      title: parsed.title || title,
      script: parsed.script || `Welcome to today's English learning. Let's explore an article about ${title}.`,
    };
  } catch (error) {
    logger.error('Failed to generate intro:', error);
    return {
      title,
      script: `Welcome to today's English learning. Let's explore an article about ${title}.`,
    };
  }
}

/**
 * Generate outro content using configured prompts.
 */
async function generateOutro(
  title: string,
  fullOriginalText: string,
  promptsConfig: PromptsConfig
): Promise<{ script: string }> {
  const template = promptsConfig.prompts.outro.template!;
  const prompt = interpolateTemplate(template, {
    title,
    originalText: fullOriginalText,
  });

  try {
    const response = await callAI(promptsConfig.systemPrompt, prompt, promptsConfig.temperature);
    const parsed = extractJSON(response) as { script?: string };

    return { script: parsed.script || 'Thank you for watching. Hope this was helpful. See you next time!' };
  } catch (error) {
    logger.error('Failed to generate outro:', error);
    return { script: 'Thank you for watching. Hope this was helpful. See you next time!' };
  }
}

/**
 * Generate all 6 parts of a segment in ONE AI call for coherence.
 */
async function generateSegmentAllParts(
  segmentId: number,
  originalText: string,
  title: string,
  promptsConfig: PromptsConfig
): Promise<{ id: number; originalText: string; parts: SegmentPart[] }> {
  logger.info(`Generating all 6 parts for segment ${segmentId} in single AI call`);

  const template = promptsConfig.prompts.segment.template!;
  const prompt = interpolateTemplate(template, {
    title,
    originalText,
  });

  let generationResponse: SegmentGenerationResponse = {
    translation: '',
    vocabularyScript: '',
    grammarScript: '',
    contextScript: '',
    vocabulary: [],
    grammarPoints: [],
  };
  const maxRetries = 2;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await callAI(promptsConfig.systemPrompt, prompt, promptsConfig.temperature);
      const parsed = extractJSON(response);

      // Validate required fields exist
      if (parsed && typeof parsed.translation === 'string' && parsed.translation.length > 10) {
        generationResponse = parsed as unknown as SegmentGenerationResponse;
        break;
      }

      if (attempt < maxRetries) {
        logger.warn(`Segment ${segmentId} response validation failed, retrying (${attempt + 1}/${maxRetries})`);
        continue;
      }

      // All retries failed, use fallback
      throw new Error('Response validation failed after retries');
    } catch (error) {
      if (attempt === maxRetries) {
        logger.error(`Failed to generate segment ${segmentId} content after ${maxRetries + 1} attempts:`, error);
        generationResponse = {
          translation: `现在让我们来翻译这篇文章。${originalText}`,
          vocabularyScript: '这篇文章没有需要特别讲解的词汇。',
          grammarScript: '这篇文章没有需要特别讲解的语法点。',
          contextScript: '这篇文章的背景信息比较简单。',
          vocabulary: [],
          grammarPoints: [],
        };
      }
    }
  }

  const parts: SegmentPart[] = [];

  // Part 1: Reading (original English)
  parts.push({
    id: 1,
    type: PartType.READING,
    originalText,
    script: originalText,
  });

  // Part 2: Translation (pure Chinese narration)
  parts.push({
    id: 2,
    type: PartType.TRANSLATION,
    originalText,
    script: generationResponse.translation,
    translation: generationResponse.translation,
  });

  // Part 3: Vocabulary
  parts.push({
    id: 3,
    type: PartType.VOCABULARY,
    originalText,
    script: generationResponse.vocabularyScript,
    vocabulary: generationResponse.vocabulary || [],
  });

  // Part 4: Grammar
  parts.push({
    id: 4,
    type: PartType.GRAMMAR,
    originalText,
    script: generationResponse.grammarScript,
    grammarPoints: generationResponse.grammarPoints || [],
  });

  // Part 5: Context/Explanation
  parts.push({
    id: 5,
    type: PartType.EXPLANATION,
    originalText,
    script: generationResponse.contextScript,
    contextExplanation: generationResponse.contextScript,
  });

  // Part 6: Reading (repeat original English)
  parts.push({
    id: 6,
    type: PartType.READING,
    originalText,
    script: originalText,
  });

  return { id: segmentId, originalText, parts };
}

/**
 * Call AI API with prompt
 */
async function callAI(systemPrompt: string, userPrompt: string, temperature: number): Promise<string> {
  const response = await fetch(`${config.LLM_BASE_URL}/text/chatcompletion_v2`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.LLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: config.LLM_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature,
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

  return content;
}

/**
 * Extract JSON from AI response content.
 * Handles AI thinking tags by finding JSON before processing other patterns.
 */
function extractJSON(content: string): Record<string, unknown> {
  // First, remove all AI thinking tags (with closing tag)
  let cleaned = content.replace(/<think>[\s\S]*?<\/think>/gi, '');

  // Also remove thinking tags without closing tag (at end of content)
  cleaned = cleaned.replace(/<think>[^<]*$/gm, '');

  // Try to find JSON in code blocks first
  const codeMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeMatch) {
    cleaned = codeMatch[1];
  }

  // Find first { and last }
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
 * Generate teaching script for a single section using AI.
 * @deprecated Use generateArticleScript instead
 */
export async function generateScript(
  sectionId: number,
  originalText: string,
  title: string
): Promise<ArticleSection> {
  logger.info(`Generating script for section ${sectionId}`);

  const promptsConfig = await loadPromptsConfig();
  const template = promptsConfig.prompts.segment.template!;
  const prompt = interpolateTemplate(template, { title, originalText });

  try {
    logger.debug(`Calling AI API with model ${config.LLM_MODEL} at ${config.LLM_BASE_URL}`);

    const response = await callAI(promptsConfig.systemPrompt, prompt, promptsConfig.temperature);
    const parsed = extractJSON(response) as unknown as SegmentGenerationResponse;

    if (!parsed) {
      throw new Error('Failed to parse AI response');
    }

    logger.debug(`Script generated for section ${sectionId}`);

    return {
      id: sectionId,
      originalText,
      summary: parsed.translation?.substring(0, 100) || '',
      vocabulary: parsed.vocabulary || [],
      phrases: [],
      grammarPoints: parsed.grammarPoints || [],
      contextExplanation: parsed.contextScript || '',
      narrationScript: parsed.translation || '',
    };
  } catch (error) {
    if (error instanceof Error) {
      logger.error(`Error generating script for section ${sectionId}:`, error.message);
    } else {
      logger.error(`Failed to generate script for section ${sectionId}:`, error);
    }
    throw error;
  }
}

/**
 * Generate scripts for all sections.
 * @deprecated Use generateArticleScript instead
 */
export async function generateScripts(
  sections: { text: string; order: number }[],
  title: string
): Promise<ArticleSection[]> {
  logger.info(`Generating scripts for ${sections.length} sections`);

  const results: ArticleSection[] = [];

  for (const section of sections) {
    const script = await generateScript(section.order, section.text, title);
    results.push(script);
  }

  return results;
}
