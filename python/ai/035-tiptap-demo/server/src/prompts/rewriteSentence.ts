/**
 * 整句改写的枚举与解析器。
 *
 * 提示词本身已经搬到 server/src/config/defaults.ts（可配置、可版本化）。
 */

import { cleanText } from './cleanText.js'

export const SENTENCE_OPERATIONS = ['rewrite', 'improve', 'shorten', 'expand'] as const
export type SentenceOperation = (typeof SENTENCE_OPERATIONS)[number]

export function isSentenceOperation(v: unknown): v is SentenceOperation {
  return typeof v === 'string' && (SENTENCE_OPERATIONS as readonly string[]).includes(v)
}

export function parseRewrittenSentence(raw: string): string {
  return cleanText(raw)
}
