/**
 * 认证相关路由 + 「消耗 token 的接口」的登录守卫。
 *
 *   POST /api/auth/register   注册（顺带直接登录）
 *   POST /api/auth/login      登录
 *   POST /api/auth/logout     登出
 *   GET  /api/auth/me         当前用户 + 今日/本月 token 用量
 *
 * **会话**：token 是 32 字节随机数，服务端存 sessions 表，浏览器只拿 cookie
 * （HttpOnly / Path=/ / SameSite=Lax / 30 天）。
 *
 * **本地放行**：curl / CLI 这类「本地工具调用」不带 Origin、也不带 Sec-Fetch-Site，
 * 而且来自回环地址 —— 它们不需要登录，直接放行（但也不记账，因为没有用户）。
 * 浏览器一定会带 Sec-Fetch-Site（或 POST 的 Origin），所以浏览器必须登录。
 *
 * AUTH_LOCAL_BYPASS=0 可以把这个放行整个关掉（server/.env）。
 */

import { Router, type Request, type RequestHandler } from 'express'
import { SESSION_COOKIE, SESSION_TTL_MS, UserStore } from '../config/userStore.js'
import type { User } from '../config/userStore.js'
import { runAsUser } from '../services/requestContext.js'
import { getPlan, publicPlan } from '../config/plans.js'
import { quotaError, quotaStatus } from '../services/quota.js'
import { debugLog } from '../utils/debug.js'

export interface CreateAuthRouterOptions {
  userStore: UserStore
}

/** 读一个 cookie。手写解析，不引 cookie-parser。 */
export function readCookie(req: Request, name: string): string | undefined {
  const header = req.headers.cookie
  if (!header) return undefined
  for (const part of header.split(';')) {
    const eq = part.indexOf('=')
    if (eq < 0) continue
    if (part.slice(0, eq).trim() !== name) continue
    const value = part.slice(eq + 1).trim()
    try {
      return decodeURIComponent(value)
    } catch {
      return value
    }
  }
  return undefined
}

/** 请求是不是来自本机。 */
function isLoopback(req: Request): boolean {
  const addr = req.socket.remoteAddress ?? ''
  return addr === '127.0.0.1' || addr === '::1' || addr === '::ffff:127.0.0.1'
}

/**
 * 请求是不是浏览器发出来的。
 *
 * 浏览器一定会带 Sec-Fetch-Site（fetch/XHR 都带），POST 一定还会带 Origin。
 * curl / 脚本 / CLI 这两个都没有。
 */
function looksLikeBrowser(req: Request): boolean {
  return Boolean(req.headers.origin || req.headers['sec-fetch-site'])
}

/** 本地放行开关：默认开（server/.env 里 AUTH_LOCAL_BYPASS=0 可关）。 */
function localBypassEnabled(): boolean {
  const v = process.env.AUTH_LOCAL_BYPASS
  if (v === undefined) return true
  return !['0', 'false', 'off', 'no'].includes(v.trim().toLowerCase())
}

/**
 * 「本地工具调用」—— curl / CLI 这类。
 *
 * 放行它们，curl-tests.sh 才不用改；但它们没有用户身份，不会计入任何人的用量。
 */
export function isLocalToolRequest(req: Request): boolean {
  if (!localBypassEnabled()) return false
  return isLoopback(req) && !looksLikeBrowser(req)
}

/**
 * 需要登录的接口 —— 全是**会消耗 token** 的那些。
 *
 * 不花 token 的（/api/config、provider 的增删改、查词）不挂守卫：
 * 它们不返回明文 key（key 只在 .env 里），也没有归属问题。
 */
export const PROTECTED_API_PATHS = [
  '/api/translate',
  '/api/rewrite-selection',
  '/api/rewrite-sentence',
  '/api/grammar-check',
  '/api/rewrite-instruction',
  '/api/providers/:id/test',
  '/api/providers/:id/prompts/:capabilityId/dry-run',
]

/**
 * 登录守卫。
 *
 * - 有有效 session → 挂上用户身份（AsyncLocalStorage），放行
 * - 是本地工具调用 → 放行（不挂身份）
 * - 其它 → 401
 */
export function requireUser(userStore: UserStore): RequestHandler {
  return (req, res, next) => {
    const token = readCookie(req, SESSION_COOKIE)
    const user = token ? userStore.findSessionUser(token) : undefined

    if (user) {
      // 被禁用 → 403（不是 401：登录态是好的，只是账号被停了）
      if (user.disabledAt) {
        debugLog('[auth]', `${req.method} ${req.path}`, '→ 账号已禁用')
        res.status(403).json({ error: '账号已被禁用，请联系管理员' })
        return
      }
      // 配额：超了直接拒。管理员不限。
      const quota = quotaError(userStore, user)
      if (quota) {
        debugLog('[quota]', `${req.method} ${req.path}`, '→', quota)
        res.status(403).json({ error: quota })
        return
      }
      debugLog('[auth]', `${req.method} ${req.path}`, '→', user.username)
      runAsUser({ id: user.id, username: user.username }, next)
      return
    }

    if (isLocalToolRequest(req)) {
      debugLog('[auth]', `${req.method} ${req.path}`, '→ 本地工具调用，放行（不记账）')
      next()
      return
    }

    debugLog('[auth]', `${req.method} ${req.path}`, '→ 401 需要登录')
    res.status(401).json({ error: '需要登录' })
  }
}

/**
 * 只要登录，**不查配额**。
 *
 * 套餐申请 / 邀请码兑换这类接口用它 —— 超配额的用户也得能点升级。
 * 也不给本地工具放行：这些操作必须有真人身份。
 */
export function requireLogin(userStore: UserStore): RequestHandler {
  return (req, res, next) => {
    const token = readCookie(req, SESSION_COOKIE)
    const user = token ? userStore.findSessionUser(token) : undefined
    if (!user) {
      return res.status(401).json({ error: '需要登录' })
    }
    if (user.disabledAt) {
      return res.status(403).json({ error: '账号已被禁用，请联系管理员' })
    }
    runAsUser({ id: user.id, username: user.username }, next)
  }
}

/**
 * 管理员守卫：登录 + role = 'admin'。
 *
 * 本地工具（curl / CLI）不给放行 —— 管理操作必须有人。
 */
export function requireAdmin(userStore: UserStore): RequestHandler {
  return (req, res, next) => {
    const token = readCookie(req, SESSION_COOKIE)
    const user = token ? userStore.findSessionUser(token) : undefined
    if (!user) {
      return res.status(401).json({ error: '需要登录' })
    }
    if (user.disabledAt) {
      return res.status(403).json({ error: '账号已被禁用，请联系管理员' })
    }
    if (user.role !== 'admin') {
      debugLog('[auth]', `${req.method} ${req.path}`, '→ 不是管理员')
      return res.status(403).json({ error: '需要管理员权限' })
    }
    runAsUser({ id: user.id, username: user.username }, next)
  }
}

function publicUser(user: User): { id: number; username: string } {
  return { id: user.id, username: user.username }
}

/** 完整用户信息：角色 + 套餐 + 配额 + 禁用状态。 */
function fullUser(user: User, userStore: UserStore): Record<string, unknown> {
  const plan = getPlan(user.planId)
  return {
    ...publicUser(user),
    role: user.role,
    planId: plan.id,
    plan: publicPlan(plan),
    quota: quotaStatus(userStore, user),
    disabled: Boolean(user.disabledAt),
  }
}

export function createAuthRouter(options: CreateAuthRouterOptions): Router {
  const router = Router()
  const { userStore } = options

  /** 建 session + 下 cookie。 */
  function startSession(res: Parameters<RequestHandler>[1], user: User): void {
    const session = userStore.createSession(user.id, SESSION_TTL_MS)
    res.cookie(SESSION_COOKIE, session.token, {
      httpOnly: true,
      path: '/',
      sameSite: 'lax',
      maxAge: SESSION_TTL_MS,
    })
  }

  router.post('/api/auth/register', (req, res) => {
    const body = (req.body ?? {}) as { username?: unknown; password?: unknown }
    const invalid = UserStore.validateCredentials(body.username, body.password)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const username = (body.username as string).trim()
    const password = body.password as string

    let user: User
    try {
      user = userStore.createUser(username, password)
    } catch (err) {
      return res.status(400).json({
        error: err instanceof Error ? err.message : '注册失败',
      })
    }
    startSession(res, user)
    debugLog('[auth]', '注册成功:', username)
    res.status(201).json({ user: fullUser(user, userStore) })
  })

  router.post('/api/auth/login', (req, res) => {
    const body = (req.body ?? {}) as { username?: unknown; password?: unknown }
    if (typeof body.username !== 'string' || typeof body.password !== 'string') {
      return res.status(400).json({ error: 'username and password are required' })
    }
    const user = userStore.authenticate(body.username.trim(), body.password)
    if (!user) {
      // 用户不存在和密码错返回同一个提示 —— 不泄露用户是否存在
      debugLog('[auth]', '登录失败:', body.username.trim())
      return res.status(401).json({ error: '用户名或密码不对' })
    }
    if (user.disabledAt) {
      debugLog('[auth]', '登录被拒（账号已禁用）:', user.username)
      return res.status(403).json({ error: '账号已被禁用，请联系管理员' })
    }
    userStore.pruneExpiredSessions()
    startSession(res, user)
    debugLog('[auth]', '登录成功:', user.username)
    res.json({ user: fullUser(user, userStore) })
  })

  router.post('/api/auth/logout', (req, res) => {
    const token = readCookie(req, SESSION_COOKIE)
    if (token) userStore.deleteSession(token)
    res.clearCookie(SESSION_COOKIE, { path: '/' })
    debugLog('[auth]', '登出')
    res.json({ ok: true })
  })

  router.get('/api/auth/me', (req, res) => {
    const token = readCookie(req, SESSION_COOKIE)
    const user = token ? userStore.findSessionUser(token) : undefined
    if (!user) {
      return res.status(401).json({ error: '需要登录' })
    }
    res.json({
      user: fullUser(user, userStore),
      today: userStore.usageToday(user.id),
      month: userStore.usageThisMonth(user.id),
      /** 有没有待审核的升级申请 */
      pendingRequest: userStore.hasPendingRequest(user.id),
    })
  })

  return router
}
