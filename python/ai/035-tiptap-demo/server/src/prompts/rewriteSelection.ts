/**
 * 选区级改写的解析器。
 *
 * 提示词本身已经搬到 server/src/config/defaults.ts（可配置、可版本化），
 * 这里只留下「怎么把模型输出变成候选数组」这段逻辑。
 */

export const SELECTION_KINDS = ['word', 'phrase'] as const
export type SelectionKind = (typeof SELECTION_KINDS)[number]

export function isSelectionKind(v: unknown): v is SelectionKind {
  return typeof v === 'string' && (SELECTION_KINDS as readonly string[]).includes(v)
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

export function cleanCandidates(value: unknown, max: number): string[] {
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
