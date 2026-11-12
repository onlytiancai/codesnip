/**
 * 套餐相关的路由（用户侧）。
 *
 *   GET  /api/plans            套餐列表
 *   POST /api/plans/redeem     邀请码兑换 → 立即升级（一次性、永久）
 *   POST /api/plans/request    提交升级申请 → 等管理员审核
 *   GET  /api/plans/requests   我的申请记录
 *
 * 这些接口**只要求登录，不查配额** —— 超配额的用户也得能点升级。
 * 也不给本地工具（curl / CLI）放行：这些操作必须有真人身份。
 */

import { Router } from 'express'
import {
  canRequestPlan,
  getPlan,
  isPlanId,
  PLANS,
  publicPlan,
  publicPlanRequest,
} from '../config/plans.js'
import type { UserStore } from '../config/userStore.js'
import { currentUser } from '../services/requestContext.js'
import { requireLogin } from './auth.js'
import { debugLog } from '../utils/debug.js'

export interface CreatePlansRouterOptions {
  userStore: UserStore
}

export function createPlansRouter(options: CreatePlansRouterOptions): Router {
  const router = Router()
  const { userStore } = options
  const login = requireLogin(userStore)

  /** 当前用户；拿不到就是没登录（守卫已经拦过了，这里只是兜底）。 */
  function me(): { id: number; username: string } | undefined {
    return currentUser()
  }

  router.get('/api/plans', login, (_req, res) => {
    res.json({ plans: PLANS.map(publicPlan) })
  })

  /** 邀请码兑换：一次性把账号永久升到该码绑定的套餐。 */
  router.post('/api/plans/redeem', login, (req, res) => {
    const body = (req.body ?? {}) as { code?: unknown }
    if (typeof body.code !== 'string' || !body.code.trim()) {
      return res.status(400).json({ error: '请输入邀请码' })
    }
    const user = me()
    if (!user) return res.status(401).json({ error: '需要登录' })

    const planId = userStore.redeemInviteCode(body.code, user.id)
    if (!planId) {
      return res
        .status(400)
        .json({ error: '邀请码无效（不存在 / 已被使用 / 已作废）' })
    }

    const plan = getPlan(planId)
    userStore.setUserPlan(user.id, plan.id)
    debugLog('[plan]', user.username, '用邀请码升级到', plan.label)
    res.json({ ok: true, plan: publicPlan(plan) })
  })

  /** 提交升级申请 → 管理员审核通过后生效。 */
  router.post('/api/plans/request', login, (req, res) => {
    const body = (req.body ?? {}) as { toPlan?: unknown; note?: unknown }
    if (!isPlanId(body.toPlan)) {
      return res.status(400).json({ error: '请选择要升级的套餐' })
    }
    const who = me()
    if (!who) return res.status(401).json({ error: '需要登录' })

    const user = userStore.findUserById(who.id)
    if (!user) return res.status(401).json({ error: '需要登录' })

    if (!canRequestPlan(user.planId, body.toPlan)) {
      return res.status(400).json({
        error: '这个套餐不能申请升级（已经是这一档了，或者只能由管理员发放）',
      })
    }
    if (userStore.hasPendingRequestFor(user.id, body.toPlan)) {
      return res
        .status(409)
        .json({ error: '你有一条针对这个套餐的待审核申请了，等管理员处理完再提交' })
    }

    const note =
      typeof body.note === 'string' && body.note.trim()
        ? body.note.trim().slice(0, 500)
        : null
    const request = userStore.createPlanRequest(user.id, body.toPlan, note)
    debugLog('[plan]', user.username, '申请升级到', body.toPlan)
    res.status(201).json({ request: publicPlanRequest(request) })
  })

  /** 我的申请记录。 */
  router.get('/api/plans/requests', login, (_req, res) => {
    const user = me()
    if (!user) return res.status(401).json({ error: '需要登录' })
    const mine = userStore
      .listPlanRequests()
      .filter((r) => r.userId === user.id)
      .map(publicPlanRequest)
    res.json({ requests: mine })
  })

  return router
}
