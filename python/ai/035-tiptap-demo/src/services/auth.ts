/**
 * 登录 / 注册 / 当前用户。
 *
 * session cookie 是 httpOnly 的，浏览器自动带上 —— 前端不用管 token。
 */

import { http } from './http'

export interface CurrentUser {
  id: number
  username: string
}

export interface UsageTotals {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  calls: number
}

export interface MeInfo {
  user: CurrentUser
  today: UsageTotals
  month: UsageTotals
}

export const login = (username: string, password: string) =>
  http.post<{ user: CurrentUser }>('/api/auth/login', { username, password })

export const register = (username: string, password: string) =>
  http.post<{ user: CurrentUser }>('/api/auth/register', { username, password })

export const logout = () => http.post<{ ok: boolean }>('/api/auth/logout')

/** 当前登录用户 + 今日/本月 token 用量。没登录会抛 ApiError(401)。 */
export const fetchMe = (signal?: AbortSignal) =>
  http.get<MeInfo>('/api/auth/me', signal)

/** 后端的调试日志开关（server/.env 里的 DEBUG）。 */
export const fetchDebugFlag = (signal?: AbortSignal) =>
  http.get<{ enabled: boolean }>('/api/debug', signal)
