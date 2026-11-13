/**
 * 套餐定义。
 *
 * 套餐只有 4 档，**写死在代码里**（照 defaults.ts 里 TONE_IDS 的做法）——
 * 库里只存「用户属于哪一档」（users.plan_id），不存套餐本身。
 * 要改配额改这个文件就行，不用做套餐的增删改界面和迁移。
 *
 * 配额单位是 **token**（输入 + 输出），和 token_usage 表的口径一致。
 * 上限 0 = 不限。
 */

export interface Plan {
  id: string
  label: string
  /** 每日 token 上限（输入+输出）；0 = 不限 */
  dailyLimit: number
  /** 每月 token 上限（输入+输出）；0 = 不限 */
  monthlyLimit: number
  /** 用户能不能「提交申请」升级到这一档（不限量版只能管理员发邀请码） */
  upgradeable: boolean
  /** 前端展示用的一句话介绍 */
  description: string
}

/** 新注册用户的默认套餐。 */
export const FREE_PLAN_ID = 'free'

export const PLANS: Plan[] = [
  {
    id: 'free',
    label: '免费版',
    dailyLimit: 20_000,
    monthlyLimit: 100_000,
    upgradeable: false,
    description: '每日 2 万 token、每月 10 万 token。够日常试写改写。',
  },
  {
    id: 'basic',
    label: '基础版',
    dailyLimit: 100_000,
    monthlyLimit: 1_000_000,
    upgradeable: true,
    description: '每日 10 万 token、每月 100 万 token。日常写作够用。',
  },
  {
    id: 'pro',
    label: '专业版',
    dailyLimit: 500_000,
    monthlyLimit: 5_000_000,
    upgradeable: true,
    description: '每日 50 万 token、每月 500 万 token。长文档批量改写。',
  },
  {
    id: 'unlimited',
    label: '不限量',
    dailyLimit: 0,
    monthlyLimit: 0,
    upgradeable: false,
    description: '不限配额 —— 由管理员发放邀请码开通。',
  },
]

/** 按 id 找套餐；找不到（老数据 / 脏数据）一律当免费版。 */
export function getPlan(id: string | null | undefined): Plan {
  return PLANS.find((p) => p.id === id) ?? PLANS[0]
}

/**
 * 是不是付费套餐（免费版之外都算）。
 *
 * 用来卡「付费功能」—— 比如用户自定义指令。
 * 管理员不受这个限制（管理员本来就什么都能干）。
 */
export function isPaidPlan(id: string | null | undefined): boolean {
  return getPlan(id).id !== FREE_PLAN_ID
}

/** 套餐 id 是否存在 —— 管理员改套餐时要校验。 */
export function isPlanId(id: unknown): id is string {
  return typeof id === 'string' && PLANS.some((p) => p.id === id)
}

/** 套餐对外的形状（前端拿到的就是这个）。 */
export interface PlanInfo {
  id: string
  label: string
  dailyLimit: number
  monthlyLimit: number
  upgradeable: boolean
  description: string
}

/** 套餐 → 对外的形状。 */
export function publicPlan(plan: Plan): PlanInfo {
  return {
    id: plan.id,
    label: plan.label,
    dailyLimit: plan.dailyLimit,
    monthlyLimit: plan.monthlyLimit,
    upgradeable: plan.upgradeable,
    description: plan.description,
  }
}

import type { PlanRequest } from './userStore.js'

/** 升级申请对外的形状。 */
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

/** 升级申请 → 对外的形状。 */
export function publicPlanRequest(req: PlanRequest): PlanRequestInfo {
  return {
    id: req.id,
    userId: req.userId,
    username: req.username,
    fromPlan: req.fromPlan,
    toPlan: req.toPlan,
    toPlanLabel: getPlan(req.toPlan).label,
    status: req.status,
    note: req.note,
    createdAt: req.createdAt,
    reviewedAt: req.reviewedAt,
    reviewNote: req.reviewNote,
  }
}

/**
 * 能不能通过「提交申请」升到目标套餐。
 *
 * - 目标必须存在
 * - 目标得是 upgradeable 的（不限量版只能管理员发码）
 * - 不能申请比自己低的档（降级找管理员手动改）
 */
export function canRequestPlan(fromId: string, toId: string): boolean {
  const from = getPlan(fromId)
  const to = getPlan(toId)
  if (!to.upgradeable) return false
  return to.dailyLimit > from.dailyLimit || to.monthlyLimit > from.monthlyLimit
}
