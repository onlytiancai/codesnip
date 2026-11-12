/**
 * 统一的 fetch 封装：把后端的 `{error}` 字段捞出来，别让界面只显示 statusText。
 */

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const parsed = (await res.json()) as { error?: string }
      if (parsed?.error) detail = parsed.error
    } catch {
      // ignore
    }
    throw new ApiError(detail || `HTTP ${res.status}`, res.status)
  }

  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

/**
 * 把查询参数拼到 URL 上。
 *
 * undefined / 空串跳过 —— 免得拼出 `?status=` 这种后端要额外判空的参数。
 */
function withQuery(
  path: string,
  params?: Record<string, string | undefined>,
): string {
  if (!params) return path
  const q = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === '') continue
    q.set(key, value)
  }
  const query = q.toString()
  return query ? `${path}?${query}` : path
}

export const http = {
  get: <T>(
    path: string,
    signal?: AbortSignal,
    params?: Record<string, string | undefined>,
  ) => request<T>(withQuery(path, params), { method: 'GET', signal }),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}), signal }),
  put: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>(path, { method: 'PUT', body: JSON.stringify(body ?? {}), signal }),
  del: <T>(path: string, signal?: AbortSignal) =>
    request<T>(path, { method: 'DELETE', signal }),
}
