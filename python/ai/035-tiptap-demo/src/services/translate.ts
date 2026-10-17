/**
 * 并发调 /api/translate，每个句子一次调用，返回 AlignedSentence[]。
 * 限并发 3，避免打爆本地 Ollama。
 */

import type { ToneId } from '../constants/tones'

export interface AlignedSentence {
  zh: string
  en: string
}

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
  zh: string,
  tone?: ToneId,
  signal?: AbortSignal,
): Promise<AlignedSentence> {
  const res = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // 不传 tone 时字段整个不出现，后端走默认 prompt，行为与改动前一致
    body: JSON.stringify(tone ? { text: zh, tone } : { text: zh }),
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
  // 后端响应：{ translation: "..." } —— 一句中文对应一句英文
  const body = (await res.json()) as RawTranslateResponse
  const en = (body.translation ?? '').trim()
  if (!en) throw new TranslateHttpError('empty translation in response', 500)
  return { zh, en }
}

/**
 * 批量翻译：每个中文句子并发调一次 /api/translate。
 * 返回与输入顺序一致的 AlignedSentence[]。
 *
 * onProgress(done, total) 在每句完成时触发，便于 UI 更新"X / N 已完成"。
 */
export async function translateSentences(
  zhSentences: string[],
  onProgress?: (done: number, total: number) => void,
  tone?: ToneId,
  signal?: AbortSignal,
): Promise<AlignedSentence[]> {
  const total = zhSentences.length
  let done = 0

  const wrapped = async (zh: string): Promise<AlignedSentence> => {
    const r = await translateSentence(zh, tone, signal)
    done++
    onProgress?.(done, total)
    return r
  }

  return pLimit(zhSentences, wrapped, 3)
}

export { TranslateHttpError }