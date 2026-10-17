/**
 * 调用后端 /api/rewrite-word 拿候选词列表。
 * Vite dev server 已经把 /api 代理到 127.0.0.1:8787。
 */

export interface RewriteRequest {
  word: string
  context?: string
  /** 当前英文句子对应的中文原文，提升消歧精度 */
  sourceSentence?: string
}

export interface RewriteResponse {
  candidates: string[]
}

export class RewriteError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'RewriteError'
  }
}

export async function fetchCandidates(req: RewriteRequest): Promise<string[]> {
  const res = await fetch('/api/rewrite-word', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body?.error) detail = body.error
    } catch {
      // ignore
    }
    throw new RewriteError(detail || `HTTP ${res.status}`, res.status)
  }

  const body = (await res.json()) as RewriteResponse
  if (!Array.isArray(body.candidates)) {
    throw new RewriteError('invalid response shape', 500)
  }
  return body.candidates
}