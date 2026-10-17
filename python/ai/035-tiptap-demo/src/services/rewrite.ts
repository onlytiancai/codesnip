/**
 * AI 改写相关的四个后端调用。
 * Vite dev server 已经把 /api 代理到 127.0.0.1:8787。
 *
 * 全部非流式；都吃 AbortSignal —— 菜单一关就把在飞的请求掐掉。
 */

import type { ToneId } from '../constants/tones'

export type { ToneId }

export class RewriteError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'RewriteError'
  }
}

/** 统一的 POST + 错误提取：把后端 `{error}` 字段捞出来，而不是干巴巴的 statusText。 */
async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const parsed = (await res.json()) as { error?: string }
      if (parsed?.error) detail = parsed.error
    } catch {
      // ignore
    }
    throw new RewriteError(detail || `HTTP ${res.status}`, res.status)
  }

  return (await res.json()) as T
}

// ---- 单词 / 短语 → 候选 ----

export interface SelectionCandidatesRequest {
  text: string
  kind: 'word' | 'phrase'
  context?: string
  sourceSentence?: string
}

export async function fetchSelectionCandidates(
  req: SelectionCandidatesRequest,
  signal?: AbortSignal,
): Promise<string[]> {
  const body = await post<{ candidates?: unknown }>(
    '/api/rewrite-selection',
    req,
    signal,
  )
  if (!Array.isArray(body.candidates)) {
    throw new RewriteError('invalid response shape', 500)
  }
  return body.candidates.filter((c): c is string => typeof c === 'string')
}

// ---- 整句改写 ----

export type SentenceOperation = 'rewrite' | 'improve' | 'shorten' | 'expand'

export interface SentenceRewriteRequest {
  sentence: string
  operation: SentenceOperation
  tone?: ToneId
  context?: string
  sourceSentence?: string
}

export async function fetchSentenceRewrite(
  req: SentenceRewriteRequest,
  signal?: AbortSignal,
): Promise<string> {
  const body = await post<{ text?: unknown }>('/api/rewrite-sentence', req, signal)
  if (typeof body.text !== 'string' || !body.text.trim()) {
    throw new RewriteError('invalid response shape', 500)
  }
  return body.text
}

// ---- 语法 / 标点检查 ----

/**
 * 后端把模型的回答原样透传。
 *
 * 后端连输出格式都没要求——本地默认模型 hy-mt2 只有 1.8B，让它吐结构化 JSON
 * 不可靠，解析只会把 "模型回了段自然语言"变成 "cannot parse JSON"。
 * 模型写的是一段中文说明，界面直接显示原文。
 */
export async function fetchGrammarText(
  sentence: string,
  signal?: AbortSignal,
): Promise<string> {
  const body = await post<{ text?: unknown }>('/api/grammar-check', { sentence }, signal)
  if (typeof body.text !== 'string') {
    throw new RewriteError('invalid response shape', 500)
  }
  return body.text
}

// ---- 自定义指令改写 ----

export interface InstructionRewriteRequest {
  text: string
  instruction: string
  context?: string
}

export async function fetchInstructionRewrite(
  req: InstructionRewriteRequest,
  signal?: AbortSignal,
): Promise<string> {
  const body = await post<{ text?: unknown }>('/api/rewrite-instruction', req, signal)
  if (typeof body.text !== 'string' || !body.text.trim()) {
    throw new RewriteError('invalid response shape', 500)
  }
  return body.text
}
