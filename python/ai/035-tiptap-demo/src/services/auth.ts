/**
 * 登录 / 注册 / 当前用户。
 *
 * session cookie 是 httpOnly 的，浏览器自动带上 —— 前端不用管 token。
 */

import { http } from './http'

/** 套餐（后端 config/plans.ts 的对外形状）。 */
export interface PlanInfo {
  id: string
  label: string
  /** 每日 token 上限（输入+输出）；0 = 不限 */
  dailyLimit: number
  /** 每月 token 上限；0 = 不限 */
  monthlyLimit: number
  /** 能不能提交申请升到这一档 */
  upgradeable: boolean
  description: string
}

/** 配额状态。 */
export interface QuotaInfo {
  planId: string
  dailyUsed: number
  dailyLimit: number
  monthlyUsed: number
  monthlyLimit: number
  dailyExceeded: boolean
  monthlyExceeded: boolean
}

export interface CurrentUser {
  id: number
  username: string
  role: 'user' | 'admin'
  planId: string
  plan: PlanInfo
  quota: QuotaInfo
  /** 账号被管理员停用了 */
  disabled: boolean
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
  /** 有没有待审核的升级申请 */
  pendingRequest: boolean
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
