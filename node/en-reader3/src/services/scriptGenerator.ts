import OpenAI from 'openai';
import { config } from '../config/index.js';
import { ArticleSection, VocabularyItem, PhraseItem, GrammarPoint } from '../types/index.js';
import { logger } from '../utils/logger.js';

const client = new OpenAI({
  baseURL: config.LLM_BASE_URL,
  apiKey: config.LLM_API_KEY,
});

const RESPONSE_FORMAT = {
  type: 'json_object',
} as const;

interface AIScriptResponse {
  summary: string;
  vocabulary: VocabularyItem[];
  phrases: PhraseItem[];
  grammarPoints: GrammarPoint[];
  contextExplanation: string;
  narrationScript: string;
}

/**
 * Extract JSON from content that may include thinking tags.
 * Handles responses like: <think>...thinking...```json{...}```</think>
 */
function extractJSON(content: string): string {
  // Remove thinking tags using string operations
  let cleaned = content;
  while (cleaned.includes('<think>')) {
    const start = cleaned.indexOf('<think>');
    const end = cleaned.indexOf('</think>');
    if (end > start) {
      cleaned = cleaned.substring(0, start) + cleaned.substring(end + 4);
    } else {
      break;
    }
  }

  // Try to find JSON in code blocks first
  const codeBlockMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    return codeBlockMatch[1].trim();
  }

  // Find the first { and last }
  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');

  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    let jsonCandidate = cleaned.substring(firstBrace, lastBrace + 1);

    // Try to fix common JSON issues
    jsonCandidate = fixJSON(jsonCandidate);

    try {
      JSON.parse(jsonCandidate);
      return jsonCandidate;
    } catch {
      // Try a more aggressive approach - find balanced braces
      const balanced = findBalancedJSON(cleaned);
      if (balanced) {
        return balanced;
      }
    }
  }

  // Return cleaned content as fallback
  return cleaned.trim();
}

/**
 * Fix common JSON issues in AI-generated content.
 */
function fixJSON(json: string): string {
  // Remove any trailing commas before closing braces/brackets
  let fixed = json.replace(/,(\s*[}\]])/g, '$1');

  // Remove any text after the closing brace
  const lastBrace = fixed.lastIndexOf('}');
  if (lastBrace !== -1) {
    fixed = fixed.substring(0, lastBrace + 1);
  }

  return fixed;
}

/**
 * Find a balanced JSON object in the content.
 */
function findBalancedJSON(content: string): string | null {
  let start = -1;
  let braceCount = 0;
  let inString = false;
  let escape = false;

  for (let i = 0; i < content.length; i++) {
    const char = content[i];

    if (escape) {
      escape = false;
      continue;
    }

    if (char === '\\') {
      escape = true;
      continue;
    }

    if (char === '"' && !escape) {
      inString = !inString;
      continue;
    }

    if (inString) continue;

    if (char === '{') {
      if (start === -1) start = i;
      braceCount++;
    } else if (char === '}') {
      braceCount--;
      if (braceCount === 0 && start !== -1) {
        return content.substring(start, i + 1);
      }
    }
  }

  return null;
}

/**
 * Parse JSON from content with retry on failure.
 */
async function parseJSONWithRetry(
  sectionId: number,
  content: string,
  retries: number
): Promise<AIScriptResponse | null> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    const jsonStr = extractJSON(content);
    try {
      const parsed = JSON.parse(jsonStr) as AIScriptResponse;
      return parsed;
    } catch (parseError) {
      logger.warn(`JSON parse attempt ${attempt}/${retries} failed for section ${sectionId}: ${parseError}`);

      if (attempt < retries) {
        // Wait a bit before retry
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));

        // Try to re-extract with a different approach - remove thinking tags using string ops
        let cleaned = content;
        while (cleaned.includes('<think>')) {
          const start = cleaned.indexOf('<think>');
          const end = cleaned.indexOf('</think>');
          if (end > start) {
            cleaned = cleaned.substring(0, start) + cleaned.substring(end + 4);
          } else {
            break;
          }
        }

        // Try to find JSON in code blocks
        const codeBlockMatch = cleaned.match(/```json\n([\s\S]*?)```/);
        if (codeBlockMatch) {
          content = codeBlockMatch[1];
        } else {
          // Try to find any JSON object
          const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            content = jsonMatch[0];
          }
        }
      }
    }
  }
  return null;
}

/**
 * Generate teaching script for a single section using AI.
 */
export async function generateScript(
  sectionId: number,
  originalText: string,
  title: string
): Promise<ArticleSection> {
  logger.info(`Generating script for section ${sectionId}`);

  const prompt = `你是一位专业的英语期刊领读老师。请为以下英文文章段落生成教学内容：

【文章标题】
${title}

【原文】
${originalText}

请为每个段落提供：
1. 简洁摘要（2-3句中文）
2. 重点单词（3-5个）：单词、音标、词性、释义、例句
3. 固定短语/搭配（2-3个）：短语、含义、例句
4. 重点语法（1-2个）：语法规则、解释、例句
5. 背景/上下文讲解（2-3句中文）
6. 讲解逐字稿（200-300字中文）：结构为"先朗读一遍英文原文，再用中文详细讲解，讲解完后再朗读一遍英文原文"。朗读原文时用"[朗读]"标记，讲解时用"[讲解]"标记。例如："[朗读]Long, long ago, there lived a beautiful princess.[讲解]很久很久以前，有一位美丽的公主。今天我们要学的重点是..."

重要提醒：
- 朗读原文部分要用英文，讲解部分用中文
- 逐字稿要自然流畅，适合口头讲解
- 词汇和语法要紧密围绕原文内容

请输出符合以下 JSON schema 的内容：
{
  "summary": "中文摘要",
  "vocabulary": [{"word": "单词", "phonetic": "/音标/", "partOfSpeech": "词性", "definition": "中文释义", "example": "英文例句"}],
  "phrases": [{"phrase": "短语", "meaning": "中文含义", "example": "英文例句"}],
  "grammarPoints": [{"rule": "语法规则", "explanation": "中文解释", "example": "英文例句"}],
  "contextExplanation": "背景/上下文讲解",
  "narrationScript": "讲解逐字稿，格式：[朗读]英文原文[讲解]中文讲解[朗读]英文原文。200-300字。"
}

`;

  try {
    logger.debug(`Calling AI API with model ${config.LLM_MODEL} at ${config.LLM_BASE_URL}`);

    const response = await client.chat.completions.create({
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
      response_format: RESPONSE_FORMAT,
      temperature: 0.7,
    });

    logger.debug(`AI response object keys: ${Object.keys(response).join(', ')}`);
    logger.debug(`AI response choices count: ${response.choices?.length}`);

    if (!response.choices || response.choices.length === 0) {
      logger.error(`No choices in response: ${JSON.stringify(response).substring(0, 1000)}`);
      throw new Error('No choices in AI response');
    }

    const choice = response.choices[0];
    logger.debug(`Choice finish_reason: ${choice.finish_reason}`);
    logger.debug(`Choice message keys: ${Object.keys(choice.message || {}).join(', ')}`);

    const content = choice.message?.content;

    if (!content) {
      // Log the full response for debugging
      logger.error(`Empty content in response: ${JSON.stringify(response).substring(0, 1000)}`);
      throw new Error('Empty response from AI');
    }

    logger.debug(`Raw AI response for section ${sectionId}: ${content.substring(0, 500)}...`);

    // Extract and parse JSON with retry on failure
    let parsed = await parseJSONWithRetry(sectionId, content, 3);

    if (!parsed) {
      throw new Error('Failed to parse AI response after retries');
    }

    logger.debug(`Script generated for section ${sectionId}: ${parsed.summary.substring(0, 50)}...`);

    return {
      id: sectionId,
      originalText,
      summary: parsed.summary,
      vocabulary: parsed.vocabulary || [],
      phrases: parsed.phrases || [],
      grammarPoints: parsed.grammarPoints || [],
      contextExplanation: parsed.contextExplanation,
      narrationScript: parsed.narrationScript,
    };
  } catch (error) {
    // Log detailed error info
    if (error instanceof Error) {
      logger.error(`Error generating script for section ${sectionId}:`);
      logger.error(`  Message: ${error.message}`);
      logger.error(`  Stack: ${error.stack}`);
    } else {
      logger.error(`Failed to generate script for section ${sectionId}:`, error);
    }
    throw error;
  }
}

/**
 * Generate scripts for all sections.
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
