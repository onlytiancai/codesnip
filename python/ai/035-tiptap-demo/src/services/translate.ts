/**
 * 并发调 /api/translate，每个句子一次调用，返回 AlignedSentence[]。
 * 限并发 3，避免打爆本地 Ollama。
 */

import type { ToneId } from '../constants/tones'
import { markUnauthorized } from '../composables/useAuth'

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
    // 会话过期 → 清掉登录态（路由守卫会跳登录页）
    if (res.status === 401) markUnauthorized()
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

/**
 * 流式批量翻译的回调。
 *
 * index 是句子在**输入数组**里的下标 —— 调用方拿它去对号入座。
 */
export interface TranslateStreamHooks {
  /** 某句开始请求时触发（用来给它挂 loading） */
  onStart?: (index: number) => void
  /** 某句结束时触发：成功给 result，失败给 error */
  onSettled?: (index: number, result: AlignedSentence | null, error: unknown | null) => void
}

/**
 * 流式批量翻译 —— 跟 translateSentences 一样是「每句一次请求、并发 3」，
 * 区别是**每句一完成就回调一次**，让界面能一句一句地显示译文。
 *
 * 与 translateSentences 的另一个区别：**单句失败不会中断整批**。
 * 失败的句子结果为 null，错误交给 onSettled，由调用方决定怎么呈现。
 * （登录过期、配额用尽这种「整体失败」的情况，调用方可以在 onSettled 里 abort。）
 */
export async function translateSentencesStream(
  sentences: string[],
  tone?: ToneId,
  source: TranslateLang = 'zh',
  target: TranslateLang = 'en',
  signal?: AbortSignal,
  hooks?: TranslateStreamHooks,
): Promise<(AlignedSentence | null)[]> {
  const results: (AlignedSentence | null)[] = new Array(sentences.length).fill(null)
  let cursor = 0

  const worker = async () => {
    while (cursor < sentences.length) {
      // 中途 abort 了 → 不要再发起新请求
      if (signal?.aborted) return
      const idx = cursor++
      hooks?.onStart?.(idx)
      try {
        const r = await translateSentence(sentences[idx], tone, signal, source, target)
        results[idx] = r
        hooks?.onSettled?.(idx, r, null)
      } catch (err) {
        // abort 引起的失败不算「这句翻错了」，交给调用方按 aborted 判断
        hooks?.onSettled?.(idx, null, err)
        if (signal?.aborted) return
      }
    }
  }

  const pool = Array.from({ length: Math.min(3, sentences.length) }, worker)
  await Promise.all(pool)
  return results
}

export { TranslateHttpError }
