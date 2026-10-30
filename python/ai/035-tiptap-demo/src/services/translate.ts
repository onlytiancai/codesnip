/**
 * 并发调 /api/translate，每个句子一次调用，返回 AlignedSentence[]。
 * 限并发 3，避免打爆本地 Ollama。
 */

import type { ToneId } from '../constants/tones'

export interface AlignedSentence {
  /** 源句 */
  source: string
  /** 译文 */
  target: string
}

export type TranslateLang = 'zh' | 'en'

interface RawTranslateResponse {
  translation: string
}

class TranslateHttpError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'TranslateHttpError'
  }
}

/** 简易 pMap：限并发执行 promise 列表，按输入顺序返回结果。 */
async function pLimit<T, R>(
  items: T[],
  fn: (item: T, idx: number) => Promise<R>,
  concurrency = 3,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let cursor = 0
  const worker = async () => {
    while (cursor < items.length) {
      const idx = cursor++
      results[idx] = await fn(items[idx], idx)
    }
  }
  const pool = Array.from({ length: Math.min(concurrency, items.length) }, worker)
  await Promise.all(pool)
  return results
}

export async function translateSentence(
  text: string,
  tone?: ToneId,
  signal?: AbortSignal,
  source: TranslateLang = 'zh',
  target: TranslateLang = 'en',
): Promise<AlignedSentence> {
  const res = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      source,
      target,
      // 不传 tone 时字段整个不出现，后端走默认 prompt，行为与改动前一致
      ...(tone ? { tone } : {}),
    }),
    signal,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body?.error) detail = body.error
    } catch {
      // ignore
    }
    throw new TranslateHttpError(detail || `HTTP ${res.status}`, res.status)
  }
  // 后端响应：{ translation: "..." } —— 一句源句对应一句译文
  const body = (await res.json()) as RawTranslateResponse
  const translated = (body.translation ?? '').trim()
  if (!translated) throw new TranslateHttpError('empty translation in response', 500)
  return { source: text, target: translated }
}

/**
 * 批量翻译：每个源句并发调一次 /api/translate。
 * 返回与输入顺序一致的 AlignedSentence[]。
 *
 * onProgress(done, total) 在每句完成时触发，便于 UI 更新"X / N 已完成"。
 */
export async function translateSentences(
  sentences: string[],
  onProgress?: (done: number, total: number) => void,
  tone?: ToneId,
  source: TranslateLang = 'zh',
  target: TranslateLang = 'en',
  signal?: AbortSignal,
): Promise<AlignedSentence[]> {
  const total = sentences.length
  let done = 0

  const wrapped = async (s: string): Promise<AlignedSentence> => {
    const r = await translateSentence(s, tone, signal, source, target)
    done++
    onProgress?.(done, total)
    return r
  }

  return pLimit(sentences, wrapped, 3)
}

export { TranslateHttpError }
