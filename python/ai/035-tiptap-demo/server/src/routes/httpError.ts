/**
 * 路由层共享的错误分类。
 *
 * 三态：
 * - 503 服务端配置不全（没有可用 provider、缺提示词模板）
 * - 500 模型响应解析失败
 * - 502 上游（provider）调用失败
 *
 * 500 / 502 的区分沿用迁移前的关键词启发式 —— 老测试依赖这个行为。
 */

import type { Response } from 'express'
import {
  ConfigError,
  InvalidRequestError,
  NotFoundError,
} from '../config/store.js'

const PARSE_ERROR_PATTERNS = [
  'JSON',
  'parse',
  'array',
  'object',
  'missing',
  'empty',
  'no valid',
]

export function classifyError(err: unknown): number {
  if (err instanceof ConfigError) return 503
  if (err instanceof NotFoundError) return 404
  if (err instanceof InvalidRequestError) return 400
  const msg = err instanceof Error ? err.message : 'unknown error'
  if (PARSE_ERROR_PATTERNS.some((p) => msg.includes(p))) return 500
  return 502
}

/** 统一的错误响应：{ error: message } + 合适的 status。 */
export function fail(res: Response, err: unknown): Response {
  const msg = err instanceof Error ? err.message : 'unknown error'
  return res.status(classifyError(err)).json({ error: msg })
}
