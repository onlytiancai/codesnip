/**
 * 调试日志开关。
 *
 * 由 server/.env 里的 `DEBUG=1` 控制（关掉就把那行删掉 / 置 0）。
 * 前端也用同一个开关 —— 见 GET /api/debug。
 *
 * 用法：`debugLog('[auth]', '登录成功', username)`
 */

const FALSY = new Set(['0', 'false', 'off', 'no'])

/** DEBUG 是否打开。读 process.env，每次都读 —— 改 .env 不用重启也能生效。 */
export function debugEnabled(): boolean {
  const v = process.env.DEBUG
  if (!v) return false
  return !FALSY.has(v.trim().toLowerCase())
}

/** 调试日志：DEBUG 没打开时什么都不做。 */
export function debugLog(...args: unknown[]): void {
  if (!debugEnabled()) return
  console.log(...args)
}
