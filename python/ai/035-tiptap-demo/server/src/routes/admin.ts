/**
 * 管理端路由：用户管理 / 升级审核 / 邀请码池。
 *
 *   GET  /api/admin/users                       用户列表（带今日/本月用量）
 *   GET  /api/admin/users/:id                   用户详情 + 近 30 天每日用量
 *   POST /api/admin/users/:id/disable           禁用（顺手踢掉他的会话）
 *   POST /api/admin/users/:id/enable            启用
 *   POST /api/admin/users/:id/email             补 / 改邮箱（老用户要邮箱才能重置密码）
 *   POST /api/admin/users/:id/plan              直接改套餐（管理员手动兜底）
 *   GET  /api/admin/plan-requests               升级申请列表
 *   POST /api/admin/plan-requests/:id/approve   通过
 *   POST /api/admin/plan-requests/:id/reject    驳回
 *   GET  /api/admin/invite-codes                邀请码池
 *   POST /api/admin/invite-codes                批量生成
 *   POST /api/admin/invite-codes/revoke         作废
 *
 * 全部要管理员身份（requireAdmin）—— 本地工具（curl / CLI）不给放行。
 */

import { Router } from 'express'
import { getPlan, isPlanId, PLANS, publicPlan, publicPlanRequest } from '../config/plans.js'
import type { PlanRequestStatus, UserStore } from '../config/userStore.js'
import { validateEmail } from '../config/userStore.js'
import { currentUser } from '../services/requestContext.js'
import { requireAdmin } from './auth.js'
import { debugLog } from '../utils/debug.js'

export interface CreateAdminRouterOptions {
  userStore: UserStore
}

/** 一次生成多少个码的上限（防手滑填个 100000）。 */
const MAX_INVITE_BATCH = 200

export function createAdminRouter(options: CreateAdminRouterOptions): Router {
  const router = Router()
  const { userStore } = options
  const admin = requireAdmin(userStore)

  /** 当前管理员（守卫已经拦过了，这里只是兜底）。 */
  function me(): { id: number; username: string } | undefined {
    return currentUser()
  }

  /** 路径里的 :id —— 不是正整数就当没给。 */
  function userId(req: { params: Record<string, string> }): number | undefined {
    const n = Number(req.params.id)
    return Number.isInteger(n) && n > 0 ? n : undefined
  }

  // -------------------------------------------------------------------------
  // 用户管理
  // -------------------------------------------------------------------------

  router.get('/api/admin/users', admin, (_req, res) => {
    const users = userStore.listUsers().map((u) => ({
      id: u.id,
      username: u.username,
      role: u.role,
      planId: u.planId,
      planLabel: getPlan(u.planId).label,
      disabled: Boolean(u.disabledAt),
      createdAt: u.createdAt,
      email: u.email,
      today: u.today,
      month: u.month,
    }))
    res.json({ users })
  })

  router.get('/api/admin/users/:id', admin, (req, res) => {
    const id = userId(req)
    if (id === undefined) return res.status(400).json({ error: '用户 id 不合法' })
    const user = userStore.findUserById(id)
    if (!user) return res.status(404).json({ error: '用户不存在' })
    res.json({
      user: {
        id: user.id,
        username: user.username,
        role: user.role,
        planId: user.planId,
        planLabel: getPlan(user.planId).label,
        disabled: Boolean(user.disabledAt),
        createdAt: user.createdAt,
        email: user.email,
      },
      /** 近 30 天每日用量 */
      series: userStore.usageSeries(id, 30),
    })
  })

  router.post('/api/admin/users/:id/disable', admin, (req, res) => {
    const id = userId(req)
    if (id === undefined) return res.status(400).json({ error: '用户 id 不合法' })
    const user = userStore.findUserById(id)
    if (!user) return res.status(404).json({ error: '用户不存在' })
    if (user.role === 'admin') {
      return res.status(400).json({ error: '不能禁用管理员账号' })
    }
    userStore.setUserDisabled(id, true)
    debugLog('[admin]', '已禁用用户:', user.username)
    res.json({ ok: true })
  })

  router.post('/api/admin/users/:id/enable', admin, (req, res) => {
    const id = userId(req)
    if (id === undefined) return res.status(400).json({ error: '用户 id 不合法' })
    const user = userStore.findUserById(id)
    if (!user) return res.status(404).json({ error: '用户不存在' })
    userStore.setUserDisabled(id, false)
    debugLog('[admin]', '已启用用户:', user.username)
    res.json({ ok: true })
  })

  /**
   * 管理员给用户补 / 改邮箱。
   *
   * 老用户没有邮箱就没法用「邮箱验证码重置密码」，管理员在这儿补一个。
   */
  router.post('/api/admin/users/:id/email', admin, (req, res) => {
    const id = userId(req)
    if (id === undefined) return res.status(400).json({ error: '用户 id 不合法' })
    const body = (req.body ?? {}) as { email?: unknown }
    const invalid = validateEmail(body.email)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const user = userStore.findUserById(id)
    if (!user) return res.status(404).json({ error: '用户不存在' })
    const email = (body.email as string).trim().toLowerCase()
    try {
      userStore.setUserEmail(id, email)
    } catch (err) {
      return res.status(400).json({
        error: err instanceof Error ? err.message : '设置邮箱失败',
      })
    }
    debugLog('[admin]', '已设置用户邮箱:', user.username, '<', email, '>')
    res.json({ ok: true, email })
  })

  /** 管理员直接改套餐（不走审核流程）。 */
  router.post('/api/admin/users/:id/plan', admin, (req, res) => {
    const id = userId(req)
    if (id === undefined) return res.status(400).json({ error: '用户 id 不合法' })
    const body = (req.body ?? {}) as { planId?: unknown }
    if (!isPlanId(body.planId)) {
      return res.status(400).json({ error: '套餐 id 不合法' })
    }
    const user = userStore.findUserById(id)
    if (!user) return res.status(404).json({ error: '用户不存在' })
    userStore.setUserPlan(id, body.planId)
    debugLog('[admin]', `把 ${user.username} 的套餐改成 ${body.planId}`)
    res.json({ ok: true, plan: publicPlan(getPlan(body.planId)) })
  })

  // -------------------------------------------------------------------------
  // 升级审核
  // -------------------------------------------------------------------------

  router.get('/api/admin/plan-requests', admin, (req, res) => {
    const status = req.query.status
    const wanted: PlanRequestStatus | undefined =
      status === 'pending' || status === 'approved' || status === 'rejected'
        ? status
        : undefined
    const requests = userStore.listPlanRequests(wanted).map(publicPlanRequest)
    res.json({ requests, plans: PLANS.map(publicPlan) })
  })

  router.post('/api/admin/plan-requests/:id/approve', admin, (req, res) => {
    const id = Number(req.params.id)
    const body = (req.body ?? {}) as { note?: unknown }
    const reviewer = me()
    if (!reviewer) return res.status(401).json({ error: '需要登录' })
    const note = typeof body.note === 'string' && body.note.trim() ? body.note.trim() : null
    const request = userStore.reviewPlanRequest(id, true, reviewer.id, note)
    if (!request) return res.status(404).json({ error: '申请不存在或已经处理过了' })
    debugLog('[admin]', '通过升级申请:', request.username, '→', request.toPlan)
    res.json({ request: publicPlanRequest(request) })
  })

  router.post('/api/admin/plan-requests/:id/reject', admin, (req, res) => {
    const id = Number(req.params.id)
    const body = (req.body ?? {}) as { note?: unknown }
    const reviewer = me()
    if (!reviewer) return res.status(401).json({ error: '需要登录' })
    const note = typeof body.note === 'string' && body.note.trim() ? body.note.trim() : null
    const request = userStore.reviewPlanRequest(id, false, reviewer.id, note)
    if (!request) return res.status(404).json({ error: '申请不存在或已经处理过了' })
    debugLog('[admin]', '驳回升级申请:', request.username)
    res.json({ request: publicPlanRequest(request) })
  })

  // -------------------------------------------------------------------------
  // 邀请码池
  // -------------------------------------------------------------------------

  router.get('/api/admin/invite-codes', admin, (req, res) => {
    const batch = typeof req.query.batch === 'string' ? req.query.batch : undefined
    const status = typeof req.query.status === 'string' ? req.query.status : undefined
    const codes = userStore.listInviteCodes({ batch, status })
    res.json({ codes, plans: PLANS.map(publicPlan) })
  })

  router.post('/api/admin/invite-codes', admin, (req, res) => {
    const body = (req.body ?? {}) as {
      planId?: unknown
      count?: unknown
      batch?: unknown
    }
    if (!isPlanId(body.planId)) {
      return res.status(400).json({ error: '请选择邀请码对应的套餐' })
    }
    const count = Number(body.count)
    if (!Number.isInteger(count) || count < 1 || count > MAX_INVITE_BATCH) {
      return res.status(400).json({ error: `数量得是 1 ~ ${MAX_INVITE_BATCH} 的整数` })
    }
    const creator = me()
    const batch =
      typeof body.batch === 'string' && body.batch.trim()
        ? body.batch.trim().slice(0, 60)
        : null

    const codes = userStore.createInviteCodes(
      body.planId,
      count,
      batch,
      creator?.id ?? null,
    )
    debugLog('[admin]', `生成了 ${count} 个 ${body.planId} 邀请码`, batch ?? '')
    res.status(201).json({ codes, plan: publicPlan(getPlan(body.planId)) })
  })

  router.post('/api/admin/invite-codes/revoke', admin, (req, res) => {
    const body = (req.body ?? {}) as { code?: unknown }
    if (typeof body.code !== 'string' || !body.code.trim()) {
      return res.status(400).json({ error: '请填写要作废的邀请码' })
    }
    const ok = userStore.revokeInviteCode(body.code)
    if (!ok) {
      return res.status(400).json({ error: '这个码不存在 / 已被使用 / 已经作废过了' })
    }
    debugLog('[admin]', '作废邀请码:', body.code.trim().toUpperCase())
    res.json({ ok: true })
  })

  return router
}
