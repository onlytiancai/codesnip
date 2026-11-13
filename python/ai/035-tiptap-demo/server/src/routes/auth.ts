/**
 * 认证相关路由 + 「消耗 token 的接口」的登录守卫。
 *
 *   POST /api/auth/register        注册（顺带直接登录）—— 要邮箱验证码
 *   POST /api/auth/register/code   发注册验证码
 *   POST /api/auth/login           登录（用户名或邮箱）
 *   POST /api/auth/logout          登出
 *   POST /api/auth/change-password 修改密码（要校验当前密码）
 *   POST /api/auth/password-reset       用邮箱验证码重置密码
 *   POST /api/auth/password-reset/code  发重置密码的验证码
 *   GET  /api/auth/me              当前用户 + 今日/本月 token 用量
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
import {
  SESSION_COOKIE,
  SESSION_TTL_MS,
  UserStore,
  validateEmail,
  type EmailCodePurpose,
  type User,
} from '../config/userStore.js'
import { currentUser, runAsUser } from '../services/requestContext.js'
import { getPlan, publicPlan } from '../config/plans.js'
import { quotaError, quotaStatus } from '../services/quota.js'
import { sendMail, verificationMail } from '../services/mailer.js'
import { debugLog } from '../utils/debug.js'

/** 验证码有效期：10 分钟。 */
const EMAIL_CODE_TTL_MS = 10 * 60 * 1000

/** 同一个邮箱多久内不能重复发码：60 秒。 */
const EMAIL_CODE_COOLDOWN_MS = 60 * 1000

/** 一个验证码最多试几次。 */
const EMAIL_CODE_MAX_ATTEMPTS = 5

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
      runAsUser(
      { id: user.id, username: user.username, role: user.role, planId: user.planId },
      next,
    )
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
    runAsUser(
      { id: user.id, username: user.username, role: user.role, planId: user.planId },
      next,
    )
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
    runAsUser(
      { id: user.id, username: user.username, role: user.role, planId: user.planId },
      next,
    )
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
    /** 邮箱（老用户可能没有） */
    email: user.email,
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

  /**
   * 发验证码。
   *
   * 两种用途：
   *   - `register`：注册码。邮箱**不能**已经注册过
   *   - `reset`：重置密码码。邮箱**必须**已经注册过（不然就是给陌生人发信）
   */
  async function sendCode(
    res: Parameters<RequestHandler>[1],
    email: string,
    purpose: EmailCodePurpose,
  ): Promise<void> {
    const last = userStore.lastEmailCodeAt(email, purpose)
    if (last) {
      const waited = Date.now() - Date.parse(last)
      if (Number.isFinite(waited) && waited < EMAIL_CODE_COOLDOWN_MS) {
        const left = Math.ceil((EMAIL_CODE_COOLDOWN_MS - waited) / 1000)
        res.status(429).json({ error: `发送太频繁，${left} 秒后再试` })
        return
      }
    }

    const { id, code } = userStore.createEmailCode(
      email,
      purpose,
      EMAIL_CODE_TTL_MS,
    )
    try {
      const mail = verificationMail({
        code,
        minutes: Math.round(EMAIL_CODE_TTL_MS / 60000),
        purpose,
      })
      await sendMail({ ...mail, to: email })
    } catch (err) {
      const message = err instanceof Error ? err.message : '邮件发送失败'
      debugLog('[mail]', '发码失败:', email, '-', message)
      // 码没发出去 —— 删掉，用户不用等冷却就能重试
      userStore.deleteEmailCode(id)
      res.status(500).json({ error: message })
      return
    }

    debugLog('[auth]', `已发${purpose === 'register' ? '注册' : '重置'}验证码:`, email)
    res.json({
      ok: true,
      /** 有效期（秒） */
      expiresIn: Math.round(EMAIL_CODE_TTL_MS / 1000),
      /** 冷却（秒）—— 前端据此倒计时 */
      cooldown: Math.round(EMAIL_CODE_COOLDOWN_MS / 1000),
    })
  }

  /** 发注册验证码。 */
  router.post('/api/auth/register/code', async (req, res) => {
    const body = (req.body ?? {}) as { email?: unknown }
    const invalid = validateEmail(body.email)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const email = (body.email as string).trim().toLowerCase()
    if (userStore.findUserByEmail(email)) {
      return res.status(400).json({ error: '这个邮箱已经注册过了，去登录吧' })
    }
    await sendCode(res, email, 'register')
  })

  router.post('/api/auth/register', (req, res) => {
    const body = (req.body ?? {}) as {
      username?: unknown
      password?: unknown
      email?: unknown
      code?: unknown
    }
    const invalid = UserStore.validateCredentials(body.username, body.password)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const emailInvalid = validateEmail(body.email)
    if (emailInvalid) {
      return res.status(400).json({ error: emailInvalid })
    }
    if (typeof body.code !== 'string' || !body.code.trim()) {
      return res.status(400).json({ error: '请输入邮箱验证码' })
    }
    const username = (body.username as string).trim()
    const password = body.password as string
    const email = (body.email as string).trim().toLowerCase()

    if (userStore.findUserByEmail(email)) {
      return res.status(400).json({ error: '这个邮箱已经注册过了，去登录吧' })
    }
    if (
      !userStore.consumeEmailCode(
        email,
        'register',
        body.code,
        EMAIL_CODE_MAX_ATTEMPTS,
      )
    ) {
      return res.status(400).json({ error: '验证码不对或已过期，重新发一个' })
    }

    let user: User
    try {
      user = userStore.createUser(username, password, { email })
    } catch (err) {
      return res.status(400).json({
        error: err instanceof Error ? err.message : '注册失败',
      })
    }
    startSession(res, user)
    debugLog('[auth]', '注册成功:', username, '<', email, '>')
    res.status(201).json({ user: fullUser(user, userStore) })
  })

  /** 发重置密码的验证码（邮箱必须已经注册）。 */
  router.post('/api/auth/password-reset/code', async (req, res) => {
    const body = (req.body ?? {}) as { email?: unknown }
    const invalid = validateEmail(body.email)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const email = (body.email as string).trim().toLowerCase()
    if (!userStore.findUserByEmail(email)) {
      return res.status(404).json({ error: '这个邮箱没有注册过账号' })
    }
    await sendCode(res, email, 'reset')
  })

  /**
   * 用邮箱验证码重置密码（不用记当前密码）。
   *
   * 重置成功 → 改密码 + **踢掉全部会话**（谁在用旧 cookie 谁都得重新登录）。
   */
  router.post('/api/auth/password-reset', (req, res) => {
    const body = (req.body ?? {}) as {
      email?: unknown
      code?: unknown
      newPassword?: unknown
    }
    const emailInvalid = validateEmail(body.email)
    if (emailInvalid) {
      return res.status(400).json({ error: emailInvalid })
    }
    const invalid = UserStore.validatePassword(body.newPassword)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    if (typeof body.code !== 'string' || !body.code.trim()) {
      return res.status(400).json({ error: '请输入邮箱验证码' })
    }
    const email = (body.email as string).trim().toLowerCase()
    const user = userStore.findUserByEmail(email)
    if (!user) {
      return res.status(404).json({ error: '这个邮箱没有注册过账号' })
    }
    if (
      !userStore.consumeEmailCode(email, 'reset', body.code, EMAIL_CODE_MAX_ATTEMPTS)
    ) {
      return res.status(400).json({ error: '验证码不对或已过期，重新发一个' })
    }

    userStore.setUserPassword(user.id, body.newPassword as string)
    userStore.deleteAllSessions(user.id)
    debugLog('[auth]', '密码已重置（邮件验证码）:', user.username)
    res.json({ ok: true })
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
      return res.status(401).json({ error: '用户名/邮箱或密码不对' })
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

  /**
   * 修改密码。
   *
   * - 要校验**当前密码** —— 拿到 cookie 就能改密码的话，cookie 泄漏等于密码泄漏
   * - 改完把**其它设备上的会话**作废（当前这个保留，不然自己被踢下线）
   * - 挂 requireLogin（不查配额）：超配额也得能改密码
   */
  router.post('/api/auth/change-password', requireLogin(userStore), (req, res) => {
    const body = (req.body ?? {}) as {
      currentPassword?: unknown
      newPassword?: unknown
    }
    if (typeof body.currentPassword !== 'string' || !body.currentPassword) {
      return res.status(400).json({ error: '请输入当前密码' })
    }
    const invalid = UserStore.validatePassword(body.newPassword)
    if (invalid) {
      return res.status(400).json({ error: invalid })
    }
    const newPassword = body.newPassword as string
    if (newPassword === body.currentPassword) {
      return res.status(400).json({ error: '新密码不能和当前密码一样' })
    }

    const who = currentUser()
    if (!who) return res.status(401).json({ error: '需要登录' })

    // 当前密码不对 → 和登录失败一样，不透露更多信息
    if (!userStore.authenticate(who.username, body.currentPassword)) {
      debugLog('[auth]', '修改密码失败（当前密码不对）:', who.username)
      return res.status(400).json({ error: '当前密码不对' })
    }

    userStore.setUserPassword(who.id, newPassword)
    const token = readCookie(req, SESSION_COOKIE) ?? ''
    userStore.deleteOtherSessions(who.id, token)
    debugLog('[auth]', '密码已修改:', who.username, '（其它设备会话已失效）')
    res.json({ ok: true })
  })

  return router
}
