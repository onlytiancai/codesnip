/**
 * 配额检查。
 *
 * 配额单位是 **token**（输入 + 输出），和 token_usage 表的口径一致。
 * 超额 = 直接拒绝新请求（403），并给一句明确的文案。
 *
 * 管理员不限量。
 */

import { getPlan } from '../config/plans.js'
import type { User, UserStore } from '../config/userStore.js'

export interface QuotaStatus {
  planId: string
  /** 今日已用 token */
  dailyUsed: number
  /** 今日上限（0 = 不限） */
  dailyLimit: number
  /** 本月已用 token */
  monthlyUsed: number
  /** 本月上限（0 = 不限） */
  monthlyLimit: number
  dailyExceeded: boolean
  monthlyExceeded: boolean
}

/** 算一个用户的配额状态（侧栏 / 设置页 / /api/auth/me 都用它）。 */
export function quotaStatus(userStore: UserStore, user: User): QuotaStatus {
  const plan = getPlan(user.planId)
  const today = userStore.usageToday(user.id)
  const month = userStore.usageThisMonth(user.id)
  return {
    planId: plan.id,
    dailyUsed: today.totalTokens,
    dailyLimit: plan.dailyLimit,
    monthlyUsed: month.totalTokens,
    monthlyLimit: plan.monthlyLimit,
    dailyExceeded: plan.dailyLimit > 0 && today.totalTokens >= plan.dailyLimit,
    monthlyExceeded: plan.monthlyLimit > 0 && month.totalTokens >= plan.monthlyLimit,
  }
}

/**
 * 配额检查：超了返回一句错误文案，没超返回 null。
 *
 * 管理员不限量 —— 管理员要能随时试 provider / 查用量。
 */
export function quotaError(userStore: UserStore, user: User): string | null {
  if (user.role === 'admin') return null
  const q = quotaStatus(userStore, user)
  const label = getPlan(user.planId).label
  if (q.dailyExceeded) {
    return `${label}今日配额已用完（${fmt(q.dailyLimit)} token），请升级套餐`
  }
  if (q.monthlyExceeded) {
    return `${label}本月配额已用完（${fmt(q.monthlyLimit)} token），请升级套餐`
  }
  return null
}

function fmt(n: number): string {
  return n.toLocaleString('zh-CN')
}
