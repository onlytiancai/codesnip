/**
 * 选区级改写 prompt：给选中的单词或短语找替换候选。
 *
 * word 与 phrase 的差别主要在 prompt：
 * - word  ：同词性近义词，禁止只做形态变化
 * - phrase：候选可以是多词短语，要求搭配正确、语法正确、与前后文连贯
 *
 * 输出是 JSON 字符串数组，由本文件底部的 parseCandidates 解析。
 */

export const SELECTION_KINDS = ['word', 'phrase'] as const
export type SelectionKind = (typeof SELECTION_KINDS)[number]

export function isSelectionKind(v: unknown): v is SelectionKind {
  return typeof v === 'string' && (SELECTION_KINDS as readonly string[]).includes(v)
}

export const REWRITE_SELECTION_SYSTEM =
  'You are an English writing assistant. Given a selection from an English sentence, you produce replacement candidates that fit that exact grammatical slot.'

const KIND_RULES: Record<SelectionKind, string[]> = {
  word: [
    '- Each candidate must keep the SAME part of speech as the input word.',
    '- Avoid mere morphological variants (e.g. for "big", do not return "bigger" or "biggest").',
  ],
  phrase: [
    '- Candidates MAY be multi-word phrases, but each one must fit the same grammatical slot as the original selection.',
    '- Get the collocation right: the candidate must sound natural in THIS sentence, not just in isolation.',
    '- Keep the surrounding text grammatical when the candidate is spliced in — check articles, prepositions and verb forms around it.',
    '- Prefer tighter alternatives over long multi-clause rewrites.',
  ],
}

/** system message：输出格式与该 kind 的专属约束。 */
export function buildRewriteSelectionSystem(kind: SelectionKind): string {
  return `${REWRITE_SELECTION_SYSTEM}

Rules:
- Output ONLY a JSON array of strings, nothing else.
- Do NOT include explanations, notes, or markdown fences.
${KIND_RULES[kind].join('\n')}
- Match the capitalization of the input: if input is "Very important", return "Essential", "Of great importance".`
}

/** user message：这一次的选区与上下文。 */
export function buildRewriteSelectionPrompt(
  text: string,
  context?: string,
  sourceSentence?: string,
): string {
  const ctx = context?.trim()
  const src = sourceSentence?.trim()
  return [
    src
      ? "The English sentence comes from translating the Chinese sentence below. Use the Chinese to disambiguate the selection's meaning before choosing candidates."
      : '',
    `Selection: ${text}`,
    ctx ? `Context: ${ctx}` : '',
    src ? `Chinese source: ${src}` : '',
    'Output:',
  ]
    .filter(Boolean)
    .join('\n')
}

/**
 * 解析模型响应为字符串数组。模型可能输出：
 * - 干净 JSON：`["great","promising"]`
 * - 带 ```json``` 围栏
 * - 带 ``` ``` 围栏
 * - 前后有零散文本（"Here are ...: [...]"）
 *
 * 一律抛 Error 让上层返回 500。
 */
export function parseCandidates(raw: string, max = 5): string[] {
  const text = raw.trim()
  if (!text) throw new Error('empty response')

  const candidates: string[] = [text]

  // 剥 ```json ... ``` 围栏
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenced) candidates.push(fenced[1].trim())

  // 找第一个 [...] 子串
  const bracket = text.match(/\[[\s\S]*\]/)
  if (bracket) candidates.push(bracket[0])

  // 记下「JSON 解开了但形状不对」的错误——那比笼统的 "cannot parse" 有用得多。
  let shapeError: Error | undefined

  for (const c of candidates) {
    let parsed: unknown
    try {
      parsed = JSON.parse(c)
    } catch {
      continue // 连 JSON 都不是，试下一个候选
    }
    try {
      return cleanCandidates(parsed, max)
    } catch (err) {
      shapeError ??= err instanceof Error ? err : new Error('bad shape')
    }
  }

  if (shapeError) throw shapeError
  throw new Error('cannot parse JSON array from model response')
}

function cleanCandidates(value: unknown, max: number): string[] {
  if (!Array.isArray(value)) {
    throw new Error('response is not an array')
  }
  const seen = new Set<string>()
  const result: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') continue
    const t = item.trim()
    if (!t) continue
    const key = t.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    result.push(t)
    if (result.length >= max) break
  }
  if (result.length === 0) {
    throw new Error('no valid candidates in response')
  }
  return result
}
