/**
 * 整句改写 prompt。
 *
 * 四种操作：
 * - rewrite  换一种说法（可带语气）
 * - improve  改好语法/搭配/地道度，但意思和长度基本不动
 * - shorten  更短，但不丢核心信息
 * - expand   更详细，补足读者需要的信息，但不注水
 *
 * 输出是纯文本一句（不走 JSON），由 cleanText 清洗。
 */

import { cleanText } from './cleanText.js'
import { toneInstruction, type ToneId } from './tones.js'

export const SENTENCE_OPERATIONS = ['rewrite', 'improve', 'shorten', 'expand'] as const
export type SentenceOperation = (typeof SENTENCE_OPERATIONS)[number]

export function isSentenceOperation(v: unknown): v is SentenceOperation {
  return typeof v === 'string' && (SENTENCE_OPERATIONS as readonly string[]).includes(v)
}

const OPERATION_GUIDANCE: Record<SentenceOperation, string> = {
  rewrite: 'Express the same meaning using different wording and sentence structure.',
  improve:
    'Fix grammar, awkward collocations, word choice, and idiomatic phrasing. Keep the meaning and roughly the same length.',
  shorten:
    'Make the sentence noticeably shorter. Cut filler, redundancy, and qualifiers — but keep the core claim intact.',
  expand:
    'Add the specifics a reader would need: supporting detail, a concrete example, or a stated consequence. Do not pad with empty words.',
}

export const REWRITE_SENTENCE_SYSTEM = [
  'You are an English writing editor. You rewrite a single English sentence according to the requested operation.',
  'Output ONLY the rewritten sentence — no quotes, no labels, no explanation, no before/after pairs.',
  'The rewrite must be grammatical, idiomatic English and must stay on topic with the surrounding context.',
].join(' ')

/**
 * system message。语气是硬约束，放 system 里比塞 user message 里听话得多。
 * 只有 rewrite 支持语气——improve/shorten/expand 各自的操作定义已经够明确了。
 */
export function buildRewriteSentenceSystem(
  operation: SentenceOperation,
  tone?: ToneId,
): string {
  const toneLine = operation === 'rewrite' ? toneInstruction(tone) : undefined
  const base = `${REWRITE_SENTENCE_SYSTEM}\nOperation: ${operation} — ${OPERATION_GUIDANCE[operation]}`
  return toneLine ? `${base}\n${toneLine}` : base
}

/** user message：只有这一次的输入，不重复 system 里的规则。 */
export function buildRewriteSentencePrompt(
  sentence: string,
  context?: string,
  sourceSentence?: string,
): string {
  const ctx = context?.trim()
  const src = sourceSentence?.trim()

  const parts = [
    src
      ? 'The English sentence was translated from the Chinese sentence below. Keep the meaning aligned with the Chinese.'
      : '',
    `Sentence: ${sentence}`,
    ctx ? `Surrounding paragraph: ${ctx}` : '',
    src ? `Chinese source: ${src}` : '',
    'Output:',
  ]

  return parts.filter(Boolean).join('\n').trim()
}

export function parseRewrittenSentence(raw: string): string {
  return cleanText(raw)
}
