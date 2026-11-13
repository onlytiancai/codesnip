/**
 * 管理端接口：用户管理 / 升级审核 / 邀请码池。
 *
 * 这些接口只有管理员能调（后端 requireAdmin 兜底）。
 */

import { http } from './http'
import type { PlanInfo, UsageTotals } from './auth'
import type { PlanRequestInfo } from './plans'

/** 用户列表的一行。 */
export interface AdminUser {
  id: number
  username: string
  role: 'user' | 'admin'
  planId: string
  planLabel: string
  disabled: boolean
  createdAt: string
  /** 邮箱；老用户可能没有 */
  email: string | null
  today: UsageTotals
  month: UsageTotals
}

/** 某天的用量（用户详情里的时间序列）。 */
export interface UsagePoint {
  day: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  calls: number
}

export interface AdminUserDetail {
  user: Omit<AdminUser, 'today' | 'month'>
  /** 近 30 天每日用量 */
  series: UsagePoint[]
}

/** 邀请码。 */
export interface InviteCode {
  code: string
  planId: string
  batch: string | null
  createdAt: string
  createdBy: number | null
  usedBy: number | null
  usedByName: string | null
  usedAt: string | null
  revokedAt: string | null
}

/** 用户列表。 */
export const fetchAdminUsers = () =>
  http.get<{ users: AdminUser[] }>('/api/admin/users')

/** 用户详情 + 近 30 天每日用量。 */
export const fetchAdminUser = (id: number) =>
  http.get<AdminUserDetail>(`/api/admin/users/${id}`)

/** 禁用 / 启用用户。 */
export const disableUser = (id: number) =>
  http.post<{ ok: boolean }>(`/api/admin/users/${id}/disable`)

export const enableUser = (id: number) =>
  http.post<{ ok: boolean }>(`/api/admin/users/${id}/enable`)

/** 管理员直接改套餐。 */
export const setUserPlan = (id: number, planId: string) =>
  http.post<{ ok: boolean; plan: PlanInfo }>(`/api/admin/users/${id}/plan`, {
    planId,
  })

/** 管理员给用户补 / 改邮箱（老用户要邮箱才能用「验证码重置密码」）。 */
export const setUserEmail = (id: number, email: string) =>
  http.post<{ ok: boolean; email: string }>(`/api/admin/users/${id}/email`, {
    email,
  })

/** 升级申请列表（status 可选：pending / approved / rejected）。 */
export const fetchPlanRequests = (status?: string) =>
  http.get<{ requests: PlanRequestInfo[]; plans: PlanInfo[] }>(
    '/api/admin/plan-requests',
    undefined,
    status ? { status } : undefined,
  )

/** 审核：通过 / 驳回。 */
export const approvePlanRequest = (id: number, note?: string) =>
  http.post<{ request: PlanRequestInfo }>(`/api/admin/plan-requests/${id}/approve`, {
    note,
  })

export const rejectPlanRequest = (id: number, note?: string) =>
  http.post<{ request: PlanRequestInfo }>(`/api/admin/plan-requests/${id}/reject`, {
    note,
  })

/** 邀请码池。 */
export const fetchInviteCodes = (params?: { batch?: string; status?: string }) =>
  http.get<{ codes: InviteCode[]; plans: PlanInfo[] }>(
    '/api/admin/invite-codes',
    undefined,
    params,
  )

/** 批量生成邀请码。 */
export const createInviteCodes = (planId: string, count: number, batch?: string) =>
  http.post<{ codes: InviteCode[]; plan: PlanInfo }>('/api/admin/invite-codes', {
    planId,
    count,
    batch,
  })

/** 作废一个邀请码。 */
export const revokeInviteCode = (code: string) =>
  http.post<{ ok: boolean }>('/api/admin/invite-codes/revoke', { code })
