/**
 * 逐句翻译 prompt + 解析。
 *
 * 设计：一句中文 → 一句英文。模型只需要直接输出翻译文本，不强求 JSON。
 * 响应中 zh 由前端传入（=入参），en 由模型输出。
 *
 * 语气是可选的：不传 tone 时 system prompt 与改动前逐字节相同，
 * 这样老测试和老行为都不受影响。
 */

import { cleanText } from './cleanText.js'
import { toneInstruction, type ToneId } from './tones.js'

export interface TranslationResult {
  translation: string
}

/**
 * system message：明确只要一句英文。
 */
export const TRANSLATE_SYSTEM_PROMPT =
  'You are a Chinese-to-English translator. You translate a single Chinese sentence into one natural English sentence. Output ONLY the English sentence — no quotes, no labels, no explanation.'

/**
 * 用户消息：简洁输入 + 触发输出。
 */
export function buildTranslatePrompt(text: string): string {
  return `Chinese: ${text}\nEnglish:`
}

/**
 * 完整 system message。带语气时在基础约束后追加一条 tone 指令。
 */
export function buildTranslateSystemPrompt(tone?: ToneId): string {
  const instruction = toneInstruction(tone)
  return instruction ? `${TRANSLATE_SYSTEM_PROMPT}\n${instruction}` : TRANSLATE_SYSTEM_PROMPT
}

/**
 * 解析模型响应为纯英文一句话。
 */
export function parseTranslation(raw: string): string {
  return cleanText(raw)
}
