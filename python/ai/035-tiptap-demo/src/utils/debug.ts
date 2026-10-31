/**
 * 浏览器调试日志。
 *
 * 开关来自后端（server/.env 里的 DEBUG）—— 前端启动时调 GET /api/debug 拿，
 * 两边共用一个开关，不用在前端另配 .env。
 */

let enabled = false

export function setDebugEnabled(value: boolean): void {
  enabled = value
}

export function debugEnabled(): boolean {
  return enabled
}

/** 调试日志：开关没打开时什么都不做。 */
export function debugLog(...args: unknown[]): void {
  if (!enabled) return
  console.log(...args)
}

/**
 * 给 window.fetch 打个补丁，每个请求打一行。
 *
 * 必须在开关打开之后再装 —— 装上以后所有请求（含静态资源）都会被记。
 */
export function installFetchLogger(): void {
  if (!enabled) return
  if ((window as unknown as { __fetchLogged?: boolean }).__fetchLogged) return
  ;(window as unknown as { __fetchLogged?: boolean }).__fetchLogged = true

  const original = window.fetch.bind(window)
  window.fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url
    const method = init?.method ?? 'GET'
    const started = Date.now()
    try {
      const res = await original(input, init)
      debugLog(
        `[debug] ${method} ${url} → ${res.status} (${Date.now() - started}ms)`,
      )
      return res
    } catch (err) {
      debugLog(
        `[debug] ${method} ${url} → 失败（${err instanceof Error ? err.message : 'unknown'}）`,
      )
      throw err
    }
  }
}
