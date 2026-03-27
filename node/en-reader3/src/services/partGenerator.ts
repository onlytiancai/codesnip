import { config } from '../config/index.js';
import { logger } from '../utils/logger.js';
import { PartType, SegmentPart, VocabularyItem, GrammarPoint } from '../types/index.js';

const RESPONSE_FORMAT = { type: 'text' as const };

/**
 * Generate a single part script using AI.
 */
export async function generatePartScript(
  segmentId: number,
  partType: PartType,
  originalText: string,
  context?: {
    vocabulary?: VocabularyItem[];
    grammarPoints?: GrammarPoint[];
    contextExplanation?: string;
  }
): Promise<string> {
  logger.info(`Generating ${partType} script for segment ${segmentId}`);

  const prompt = buildPartPrompt(partType, originalText, context);

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
            content: '你是一位专业的英语期刊领读老师，擅长生成教学内容和讲解逐字稿。',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: 0.7,
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

    // For reading parts, return original text directly
    if (partType === PartType.READING) {
      return originalText;
    }

    // Extract script from response
    const script = extractScript(content);
    logger.debug(`Generated ${partType} script: ${script.substring(0, 50)}...`);

    return script;
  } catch (error) {
    logger.error(`Failed to generate ${partType} script:`, error);
    throw error;
  }
}

/**
 * Build prompt for different part types.
 */
function buildPartPrompt(
  partType: PartType,
  originalText: string,
  context?: {
    vocabulary?: VocabularyItem[];
    grammarPoints?: GrammarPoint[];
    contextExplanation?: string;
  }
): string {
  const baseContext = `【原文】
${originalText}`;

  switch (partType) {
    case PartType.READING:
      return `${baseContext}

请直接返回原文，不需要任何修改。`;

    case PartType.TRANSLATION:
      return `${baseContext}

请将上述英文翻译成中文，生成一段200-300字的中文讲解稿。要求：
- 流畅自然，适合口头讲解
- 先朗读英文原文，再用中文详细讲解
- 结合上下文给出必要背景信息

请直接输出讲解稿，不需要JSON格式。`;

    case PartType.VOCABULARY:
      const vocabList = context?.vocabulary
        ?.slice(0, 5)
        .map((v) => `- ${v.word}: ${v.definition}`)
        .join('\n') || '无';

      return `${baseContext}

【重点词汇】
${vocabList}

请生成一段150-200字的词汇讲解稿，重点讲解这些词汇在本文中的含义和用法。

请直接输出讲解稿，不需要JSON格式。`;

    case PartType.GRAMMAR:
      const grammarList = context?.grammarPoints
        ?.slice(0, 2)
        .map((g) => `- ${g.rule}: ${g.explanation}`)
        .join('\n') || '无';

      return `${baseContext}

【语法重点】
${grammarList}

请生成一段150-200字的语法讲解稿，解释这些语法点在本文中的应用。

请直接输出讲解稿，不需要JSON格式。`;

    case PartType.EXPLANATION:
      return `${baseContext}

【背景介绍】
${context?.contextExplanation || '无'}

请生成一段150-200字的背景解读稿，介绍文章的写作背景、文化背景或相关知识。

请直接输出讲解稿，不需要JSON格式。`;

    default:
      return originalText;
  }
}

/**
 * Extract script from AI response.
 */
function extractScript(content: string): string {
  // Remove thinking tags
  let cleaned = content.replace(/<think>[\s\S]*?<\/think>/gi, '');

  // Try to find content in code blocks
  const codeMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeMatch) {
    cleaned = codeMatch[1];
  }

  // If it's JSON, try to extract the script field
  try {
    const parsed = JSON.parse(cleaned.trim());
    if (parsed.script) return parsed.script;
    if (parsed.text) return parsed.text;
    if (parsed.content) return parsed.content;
  } catch {
    // Not JSON, use as-is
  }

  return cleaned.trim();
}

/**
 * Generate intro script based on title and segment count.
 */
export async function generateIntroScript(title: string, segmentCount: number): Promise<string> {
  logger.info('Generating intro script');

  const prompt = `【文章标题】
${title}

【段落数量】
${segmentCount}

请根据文章标题生成一段100-150字的中文开场白，介绍今天要学习的文章内容，激发观众学习兴趣。

请直接输出开场白内容，不需要JSON格式。`;

  try {
    const response = await fetch(`${config.LLM_BASE_URL}/text/chatcompletion_v2`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.LLM_API_KEY}`,
      },
      body: JSON.stringify({
        model: config.LLM_MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
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

    return extractScript(content);
  } catch (error) {
    logger.error('Failed to generate intro script:', error);
    return `大家好，欢迎来到今天的英语学习。我们一起来学习一篇关于${title}的文章。`;
  }
}

/**
 * Generate outro script based on article content.
 */
export async function generateOutroScript(title: string): Promise<string> {
  logger.info('Generating outro script');

  const prompt = `【文章标题】
${title}

请根据文章内容生成一段100-150字的中文结束语，感谢观众观看，总结文章要点，鼓励观众继续学习。

请直接输出结束语内容，不需要JSON格式。`;

  try {
    const response = await fetch(`${config.LLM_BASE_URL}/text/chatcompletion_v2`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.LLM_API_KEY}`,
      },
      body: JSON.stringify({
        model: config.LLM_MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
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

    return extractScript(content);
  } catch (error) {
    logger.error('Failed to generate outro script:', error);
    return '感谢大家的观看，希望今天的分享对你们有所帮助。我们下期再见！';
  }
}

/**
 * Generate all 6 parts for a segment.
 */
export async function generateAllParts(
  segmentId: number,
  originalText: string,
  context?: {
    vocabulary?: VocabularyItem[];
    grammarPoints?: GrammarPoint[];
    contextExplanation?: string;
  }
): Promise<SegmentPart[]> {
  const partTypes = [
    PartType.READING,
    PartType.TRANSLATION,
    PartType.VOCABULARY,
    PartType.GRAMMAR,
    PartType.EXPLANATION,
    PartType.READING, // Repeat reading
  ];

  const parts: SegmentPart[] = [];

  for (let i = 0; i < partTypes.length; i++) {
    const partType = partTypes[i];
    const script = await generatePartScript(segmentId, partType, originalText, context);

    parts.push({
      id: i + 1,
      type: partType,
      originalText,
      script,
      translation: partType === PartType.TRANSLATION ? script : undefined,
      vocabulary: partType === PartType.VOCABULARY ? context?.vocabulary : undefined,
      grammarPoints: partType === PartType.GRAMMAR ? context?.grammarPoints : undefined,
      contextExplanation: partType === PartType.EXPLANATION ? context?.contextExplanation : undefined,
    });
  }

  return parts;
}
