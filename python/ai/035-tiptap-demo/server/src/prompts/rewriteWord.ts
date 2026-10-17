/**
 * Prompt 设计要点：
 * - 强约束：只输出 JSON 数组，禁止任何解释/围栏/标点
 * - 数量：3-5 个候选（用户偏向 3）
 * - 大小写：模型输出应与输入保持一致
 * - 词性：与输入一致
 * - 形态：避免只是词形变换（如 big/bigger/biggest），优先同义或近义替换
 */

export const REWRITE_WORD_SYSTEM = `You are a thesaurus assistant. Given a single English word and optional context, output 3 to 5 synonyms or near-replacement words.`

export function buildRewritePrompt(word: string, context?: string): string {
  const ctx = context?.trim()
  return `${REWRITE_WORD_SYSTEM}

Rules:
- Output ONLY a JSON array of strings, nothing else.
- Do NOT include explanations, notes, or markdown fences.
- Each candidate must keep the SAME part of speech as the input word.
- Avoid mere morphological variants (e.g. for "big", do not return "bigger" or "biggest").
- Preserve capitalization: if input is "Good", every candidate must be capitalized like "Great".

Word: ${word}${ctx ? `\nContext: ${ctx}` : ''}

Output:`.trim()
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

  // 1. 直接尝试
  try {
    return cleanCandidates(JSON.parse(text), max)
  } catch {
    // 继续尝试
  }

  // 2. 剥 ```json ... ``` 围栏
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenced) {
    try {
      return cleanCandidates(JSON.parse(fenced[1].trim()), max)
    } catch {
      // 继续尝试
    }
  }

  // 3. 找第一个 [...] 子串
  const bracket = text.match(/\[[\s\S]*\]/)
  if (bracket) {
    try {
      return cleanCandidates(JSON.parse(bracket[0]), max)
    } catch {
      // 继续尝试
    }
  }

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