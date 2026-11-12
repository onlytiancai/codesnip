/**
 * 套餐（用户侧）：看套餐、邀请码兑换、提交升级申请、我的申请记录。
 */

import { http } from './http'
import type { PlanInfo } from './auth'

/** 升级申请。 */
export interface PlanRequestInfo {
  id: number
  userId: number
  username: string
  fromPlan: string
  toPlan: string
  toPlanLabel: string
  status: 'pending' | 'approved' | 'rejected'
  note: string | null
  createdAt: string
  reviewedAt: string | null
  reviewNote: string | null
}

/** 全部套餐。 */
export const fetchPlans = () => http.get<{ plans: PlanInfo[] }>('/api/plans')

/** 邀请码兑换 —— 一次性把账号永久升到该码绑定的套餐。 */
export const redeemInvite = (code: string) =>
  http.post<{ ok: boolean; plan: PlanInfo }>('/api/plans/redeem', { code })

/** 提交升级申请（要管理员审核）。 */
export const requestPlanUpgrade = (toPlan: string, note?: string) =>
  http.post<{ request: PlanRequestInfo }>('/api/plans/request', { toPlan, note })

/** 我的申请记录。 */
export const fetchMyPlanRequests = () =>
  http.get<{ requests: PlanRequestInfo[] }>('/api/plans/requests')
