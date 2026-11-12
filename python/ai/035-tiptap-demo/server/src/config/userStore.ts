/**
 * 用户 / 会话 / token 用量的存储。
 *
 * 和 ConfigStore 同级 —— 都挂在同一个 SQLite 库上。
 *
 * **密码哈希**：用 Node 内置的 `node:crypto` scrypt（零第三方依赖）：
 * 存 `saltHex:hashHex`，校验用 timingSafeEqual（防时序攻击）。
 *
 * **会话**：token 是 32 字节随机数（base64url），服务端存表、浏览器只拿 cookie。
 */

import { randomBytes, scryptSync, timingSafeEqual } from 'node:crypto'
import type { Db } from './db.js'
import { InvalidRequestError } from './store.js'
import { dayNumber, monthRange } from '../utils/day.js'

/** 角色：普通用户 / 管理员。 */
export type UserRole = 'user' | 'admin'

export interface User {
  id: number
  username: string
  /** 建号时间（ISO-8601） */
  createdAt: string
  role: UserRole
  /** 套餐 id（见 config/plans.ts） */
  planId: string
  /** 被禁用的时间；非 null = 已禁用 */
  disabledAt: string | null
}

/** 管理端用户列表的一行：用户 + 今日/本月用量。 */
export interface AdminUserRow extends User {
  today: UsageTotals
  month: UsageTotals
}

/** 邀请码（管理员批量生成，用户一次性兑换）。 */
export interface InviteCode {
  code: string
  planId: string
  batch: string | null
  createdAt: string
  createdBy: number | null
  usedBy: number | null
  /** 用掉这个码的用户名（列表里直接看） */
  usedByName: string | null
  usedAt: string | null
  revokedAt: string | null
}

/** 升级申请的状态。 */
export type PlanRequestStatus = 'pending' | 'approved' | 'rejected'

export interface PlanRequest {
  id: number
  userId: number
  /** 申请人的用户名 */
  username: string
  fromPlan: string
  toPlan: string
  status: PlanRequestStatus
  note: string | null
  createdAt: string
  reviewedAt: string | null
  reviewedBy: number | null
  reviewNote: string | null
}

export interface Session {
  token: string
  expiresAt: number
}

export interface UsageTotals {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  calls: number
}

/** 会话有效期：30 天。 */
export const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000

/** cookie 名。 */
export const SESSION_COOKIE = 'sid'

const USERNAME_RE = /^[A-Za-z0-9_-]{3,32}$/
const MIN_PASSWORD_LENGTH = 4

export interface UserStoreOptions {
  db: Db
}

export class UserStore {
  private readonly db: Db

  constructor(options: UserStoreOptions) {
    this.db = options.db
  }

  // -------------------------------------------------------------------------
  // 密码
  // -------------------------------------------------------------------------

  /** 生成 `saltHex:hashHex`。 */
  static hashPassword(password: string): string {
    const salt = randomBytes(16).toString('hex')
    const hash = scryptSync(password, salt, 64).toString('hex')
    return `${salt}:${hash}`
  }

  /** 校验密码。哈希格式不对 / 用户不存在都返回 false。 */
  static verifyPassword(password: string, stored: string): boolean {
    const [salt, hash] = stored.split(':')
    if (!salt || !hash) return false
    let expected: Buffer
    try {
      expected = Buffer.from(hash, 'hex')
    } catch {
      return false
    }
    const actual = scryptSync(password, salt, expected.length)
    if (actual.length !== expected.length) return false
    return timingSafeEqual(actual, expected)
  }

  // -------------------------------------------------------------------------
  // 用户
  // -------------------------------------------------------------------------

  /** 校验用户名 / 密码的格式。返回错误信息，合法则返回 undefined。 */
  static validateCredentials(
    username: unknown,
    password: unknown,
  ): string | undefined {
    if (typeof username !== 'string' || !USERNAME_RE.test(username.trim())) {
      return '用户名需为 3-32 位字母、数字、下划线或连字符'
    }
    if (typeof password !== 'string' || password.length < MIN_PASSWORD_LENGTH) {
      return `密码至少 ${MIN_PASSWORD_LENGTH} 位`
    }
    return undefined
  }

  /** 校验「新密码」的格式。合法返回 undefined，否则返回错误信息。 */
  static validatePassword(password: unknown): string | undefined {
    if (typeof password !== 'string' || password.length < MIN_PASSWORD_LENGTH) {
      return `密码至少 ${MIN_PASSWORD_LENGTH} 位`
    }
    if (password.length > 128) {
      return '密码太长了（最多 128 位）'
    }
    return undefined
  }

  countUsers(): number {
    const row = this.db.prepare('SELECT COUNT(*) AS n FROM users').get()
    return Number(row?.n ?? 0)
  }

  createUser(
    username: string,
    password: string,
    options: { role?: UserRole; planId?: string } = {},
  ): User {
    const name = username.trim()
    if (this.findUserByUsername(name)) {
      throw new InvalidRequestError(`用户名 "${name}" 已被占用`)
    }
    const createdAt = new Date().toISOString()
    const info = this.db
      .prepare(
        `INSERT INTO users (username, password_hash, created_at, role, plan_id)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(
        name,
        UserStore.hashPassword(password),
        createdAt,
        options.role ?? 'user',
        options.planId ?? 'free',
      )
    return {
      id: Number(info.lastInsertRowid),
      username: name,
      createdAt,
      role: options.role ?? 'user',
      planId: options.planId ?? 'free',
      disabledAt: null,
    }
  }

  findUserByUsername(username: string): User | undefined {
    const row = this.db
      .prepare(USER_COLUMNS + ' WHERE username = ?')
      .get(username.trim())
    return row ? toUser(row) : undefined
  }

  findUserById(id: number): User | undefined {
    const row = this.db
      .prepare(USER_COLUMNS + ' WHERE id = ?')
      .get(id)
    return row ? toUser(row) : undefined
  }

  /** 密码校验：用户不存在和密码错返回同一个结果（不泄露用户是否存在）。 */
  authenticate(username: string, password: string): User | undefined {
    const row = this.db
      .prepare(
        `SELECT id, username, password_hash, created_at, role, plan_id, disabled_at
           FROM users WHERE username = ?`,
      )
      .get(username.trim())
    if (!row) return undefined
    if (!UserStore.verifyPassword(password, String(row.password_hash))) {
      return undefined
    }
    return toUser(row)
  }

  // -------------------------------------------------------------------------
  // 会话
  // -------------------------------------------------------------------------

  createSession(userId: number, ttlMs: number = SESSION_TTL_MS): Session {
    const token = randomBytes(32).toString('base64url')
    const now = Date.now()
    const expiresAt = now + ttlMs
    this.db
      .prepare(
        'INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
      )
      .run(token, userId, new Date(now).toISOString(), expiresAt)
    return { token, expiresAt }
  }

  /** 按 token 找会话对应的用户；过期或不存在返回 undefined。 */
  findSessionUser(token: string): User | undefined {
    if (!token) return undefined
    const row = this.db
      .prepare(
        `SELECT u.id, u.username, u.created_at, u.role, u.plan_id, u.disabled_at
           FROM sessions s JOIN users u ON u.id = s.user_id
          WHERE s.token = ? AND s.expires_at > ?`,
      )
      .get(token, Date.now())
    return row ? toUser(row) : undefined
  }

  deleteSession(token: string): void {
    if (!token) return
    this.db.prepare('DELETE FROM sessions WHERE token = ?').run(token)
  }

  /** 清掉过期会话（登录时顺手做一次）。 */
  pruneExpiredSessions(): number {
    const info = this.db.prepare('DELETE FROM sessions WHERE expires_at <= ?').run(Date.now())
    return Number(info.changes)
  }

  // -------------------------------------------------------------------------
  // token 用量
  // -------------------------------------------------------------------------

  /** 记一次消耗（UPSERT 累加到「用户 × 日」那一行）。 */
  addUsage(userId: number, inputTokens: number, outputTokens: number): void {
    const day = dayNumber()
    this.db
      .prepare(
        `INSERT INTO token_usage (user_id, day, input_tokens, output_tokens, calls, updated_at)
         VALUES (?, ?, ?, ?, 1, ?)
         ON CONFLICT(user_id, day) DO UPDATE SET
           input_tokens  = input_tokens  + excluded.input_tokens,
           output_tokens = output_tokens + excluded.output_tokens,
           calls         = calls         + excluded.calls,
           updated_at    = excluded.updated_at`,
      )
      .run(userId, day, inputTokens, outputTokens, new Date().toISOString())
  }

  /** 某一天（day = 20261031）的用量。 */
  usageForDay(userId: number, day: number): UsageTotals {
    const row = this.db
      .prepare(
        'SELECT input_tokens, output_tokens, calls FROM token_usage WHERE user_id = ? AND day = ?',
      )
      .get(userId, day)
    return toTotals(row)
  }

  /** 今天的用量。 */
  usageToday(userId: number): UsageTotals {
    return this.usageForDay(userId, dayNumber())
  }

  /** 本月的用量（当月 1 号 ~ 下月 1 号，开区间）。 */
  usageThisMonth(userId: number): UsageTotals {
    const { from, to } = monthRange()
    const row = this.db
      .prepare(
        `SELECT SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                SUM(calls) AS calls
           FROM token_usage
          WHERE user_id = ? AND day >= ? AND day < ?`,
      )
      .get(userId, from, to)
    return toTotals(row)
  }

  // -------------------------------------------------------------------------
  // 账号状态：套餐 / 角色 / 禁用
  // -------------------------------------------------------------------------

  /** 禁用 / 启用用户。禁用时顺手踢掉他的所有会话（立即生效）。 */
  setUserDisabled(userId: number, disabled: boolean): void {
    if (disabled) {
      this.db
        .prepare('UPDATE users SET disabled_at = ? WHERE id = ?')
        .run(new Date().toISOString(), userId)
      this.db.prepare('DELETE FROM sessions WHERE user_id = ?').run(userId)
    } else {
      this.db.prepare('UPDATE users SET disabled_at = NULL WHERE id = ?').run(userId)
    }
  }

  /** 直接改套餐（管理员手动兜底）。 */
  setUserPlan(userId: number, planId: string): void {
    this.db.prepare('UPDATE users SET plan_id = ? WHERE id = ?').run(planId, userId)
  }

  /** 改角色。 */
  setUserRole(userId: number, role: UserRole): void {
    this.db.prepare('UPDATE users SET role = ? WHERE id = ?').run(role, userId)
  }

  /** 改密码（自己改 / 管理员重置都走这里）。 */
  setUserPassword(userId: number, password: string): void {
    this.db
      .prepare('UPDATE users SET password_hash = ? WHERE id = ?')
      .run(UserStore.hashPassword(password), userId)
  }

  /**
   * 踢掉某个用户的**其它**会话，保留 keepToken 那一个。
   *
   * 改密码之后调用 —— 密码变了，别的设备上的旧会话应该失效。
   */
  deleteOtherSessions(userId: number, keepToken: string): void {
    if (!keepToken) {
      this.db.prepare('DELETE FROM sessions WHERE user_id = ?').run(userId)
      return
    }
    this.db
      .prepare('DELETE FROM sessions WHERE user_id = ? AND token != ?')
      .run(userId, keepToken)
  }

  // -------------------------------------------------------------------------
  // 管理员：用户列表 / 用量明细
  // -------------------------------------------------------------------------

  /** 全部用户 + 今日/本月用量。 */
  listUsers(): AdminUserRow[] {
    const { from, to } = monthRange()
    const rows = this.db
      .prepare(
        `SELECT u.id, u.username, u.created_at, u.role, u.plan_id, u.disabled_at,
                COALESCE(td.input_tokens, 0)  AS t_in,
                COALESCE(td.output_tokens, 0) AS t_out,
                COALESCE(td.calls, 0)         AS t_calls,
                COALESCE(tm.input_tokens, 0)  AS m_in,
                COALESCE(tm.output_tokens, 0) AS m_out,
                COALESCE(tm.calls, 0)         AS m_calls
           FROM users u
           LEFT JOIN token_usage td
                  ON td.user_id = u.id AND td.day = ?
           LEFT JOIN (
                 SELECT user_id,
                        SUM(input_tokens)  AS input_tokens,
                        SUM(output_tokens) AS output_tokens,
                        SUM(calls)         AS calls
                   FROM token_usage
                  WHERE day >= ? AND day < ?
                  GROUP BY user_id
           ) tm ON tm.user_id = u.id
          ORDER BY u.id`,
      )
      .all(dayNumber(), from, to) as Array<Record<string, unknown>>

    return rows.map((row) => ({
      ...toUser(row),
      today: totalsFrom(row, 't_'),
      month: totalsFrom(row, 'm_'),
    }))
  }

  /** 某用户近 N 天的每日用量（没有记录的日期补 0）。 */
  usageSeries(userId: number, days: number): Array<{ day: number } & UsageTotals> {
    const today = new Date()
    const start = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() - (days - 1),
    )
    const rows = this.db
      .prepare(
        `SELECT day, input_tokens, output_tokens, calls
           FROM token_usage
          WHERE user_id = ? AND day >= ?
          ORDER BY day`,
      )
      .all(userId, dayNumber(start)) as Array<Record<string, unknown>>

    const byDay = new Map<number, UsageTotals>()
    for (const row of rows) byDay.set(Number(row.day), toTotals(row))

    const out: Array<{ day: number } & UsageTotals> = []
    for (let i = 0; i < days; i++) {
      const d = new Date(start.getFullYear(), start.getMonth(), start.getDate() + i)
      const key = dayNumber(d)
      out.push({ day: key, ...(byDay.get(key) ?? toTotals(undefined)) })
    }
    return out
  }

  // -------------------------------------------------------------------------
  // 邀请码池
  // -------------------------------------------------------------------------

  /**
   * 批量生成邀请码。
   *
   * 码是 12 位随机字符（去掉 0/O/1/I/l 这些容易看错的），分 3 段用 - 连起来。
   * 撞码就重试（概率极低，但还是查一下）。
   */
  createInviteCodes(
    planId: string,
    count: number,
    batch: string | null,
    createdBy: number | null,
  ): InviteCode[] {
    const createdAt = new Date().toISOString()
    const insert = this.db.prepare(
      `INSERT INTO invite_codes (code, plan_id, batch, created_at, created_by)
       VALUES (?, ?, ?, ?, ?)`,
    )
    const out: InviteCode[] = []
    for (let i = 0; i < count; i++) {
      let code = ''
      for (let attempt = 0; attempt < 5; attempt++) {
        const candidate = generateInviteCode()
        const exists = this.db
          .prepare('SELECT 1 FROM invite_codes WHERE code = ?')
          .get(candidate)
        if (!exists) {
          code = candidate
          break
        }
      }
      if (!code) throw new Error('邀请码生成失败（连续撞码）')
      insert.run(code, planId, batch, createdAt, createdBy)
      out.push({
        code,
        planId,
        batch,
        createdAt,
        createdBy,
        usedBy: null,
        usedByName: null,
        usedAt: null,
        revokedAt: null,
      })
    }
    return out
  }

  /** 邀请码池（可按批次 / 状态筛）。 */
  listInviteCodes(filter: { batch?: string; status?: string } = {}): InviteCode[] {
    const where: string[] = []
    const args: Array<string | number | null> = []
    if (filter.batch) {
      where.push('i.batch = ?')
      args.push(filter.batch)
    }
    if (filter.status === 'unused') {
      where.push('i.used_by IS NULL AND i.revoked_at IS NULL')
    } else if (filter.status === 'used') {
      where.push('i.used_by IS NOT NULL')
    } else if (filter.status === 'revoked') {
      where.push('i.revoked_at IS NOT NULL')
    }
    const sql = `SELECT i.code, i.plan_id, i.batch, i.created_at, i.created_by,
                        i.used_by, i.used_at, i.revoked_at, u.username AS used_by_name
                   FROM invite_codes i
                   LEFT JOIN users u ON u.id = i.used_by
                  ${where.length ? 'WHERE ' + where.join(' AND ') : ''}
                  ORDER BY i.created_at DESC, i.code`
    const rows = this.db.prepare(sql).all(...args) as Array<Record<string, unknown>>
    return rows.map(toInviteCode)
  }

  /**
   * 兑换邀请码 → 返回该码绑定的套餐 id。
   *
   * 抢码用一条带条件的 UPDATE（used_by IS NULL AND revoked_at IS NULL），
   * 条件不满足（不存在 / 已用 / 已作废）就返回 null。
   */
  redeemInviteCode(code: string, userId: number): string | null {
    const normalized = code.trim().toUpperCase()
    const row = this.db
      .prepare('SELECT plan_id FROM invite_codes WHERE code = ?')
      .get(normalized) as { plan_id?: string } | undefined
    if (!row) return null

    const info = this.db
      .prepare(
        `UPDATE invite_codes SET used_by = ?, used_at = ?
          WHERE code = ? AND used_by IS NULL AND revoked_at IS NULL`,
      )
      .run(userId, new Date().toISOString(), normalized)
    if (Number(info.changes) === 0) return null
    return row.plan_id ?? null
  }

  /** 作废一个码（只用掉之前能作废）。 */
  revokeInviteCode(code: string): boolean {
    const info = this.db
      .prepare(
        `UPDATE invite_codes SET revoked_at = ?
          WHERE code = ? AND used_by IS NULL AND revoked_at IS NULL`,
      )
      .run(new Date().toISOString(), code.trim().toUpperCase())
    return Number(info.changes) > 0
  }

  // -------------------------------------------------------------------------
  // 套餐升级申请
  // -------------------------------------------------------------------------

  /** 用户提交升级申请。 */
  createPlanRequest(
    userId: number,
    toPlan: string,
    note: string | null,
  ): PlanRequest {
    const user = this.findUserById(userId)
    const fromPlan = user?.planId ?? 'free'
    const info = this.db
      .prepare(
        `INSERT INTO plan_requests (user_id, from_plan, to_plan, status, note, created_at)
         VALUES (?, ?, ?, 'pending', ?, ?)`,
      )
      .run(userId, fromPlan, toPlan, note, new Date().toISOString())
    return this.getPlanRequest(Number(info.lastInsertRowid))!
  }

  getPlanRequest(id: number): PlanRequest | undefined {
    const row = this.db
      .prepare(
        `SELECT r.*, u.username FROM plan_requests r
           JOIN users u ON u.id = r.user_id
          WHERE r.id = ?`,
      )
      .get(id)
    return row ? toPlanRequest(row) : undefined
  }

  /** 申请列表（可按状态筛；待审的排最前）。 */
  listPlanRequests(status?: PlanRequestStatus): PlanRequest[] {
    const sql = `SELECT r.*, u.username FROM plan_requests r
                   JOIN users u ON u.id = r.user_id
                  ${status ? 'WHERE r.status = ?' : ''}
                  ORDER BY CASE r.status WHEN 'pending' THEN 0 ELSE 1 END,
                           r.created_at DESC`
    const rows = (status
      ? this.db.prepare(sql).all(status)
      : this.db.prepare(sql).all()) as Array<Record<string, unknown>>
    return rows.map(toPlanRequest)
  }

  /** 我有没有待审的申请。 */
  hasPendingRequest(userId: number): boolean {
    const row = this.db
      .prepare("SELECT 1 FROM plan_requests WHERE user_id = ? AND status = 'pending'")
      .get(userId)
    return Boolean(row)
  }

  /** 有没有针对某个套餐的待审申请（同一个人别重复提同一个档）。 */
  hasPendingRequestFor(userId: number, toPlan: string): boolean {
    const row = this.db
      .prepare(
        `SELECT 1 FROM plan_requests
          WHERE user_id = ? AND status = 'pending' AND to_plan = ?`,
      )
      .get(userId, toPlan)
    return Boolean(row)
  }

  /**
   * 审核升级申请：通过 → 顺手把用户的套餐改掉；驳回 → 只记状态。
   */
  reviewPlanRequest(
    id: number,
    approve: boolean,
    reviewerId: number,
    reviewNote: string | null,
  ): PlanRequest | null {
    const req = this.getPlanRequest(id)
    if (!req || req.status !== 'pending') return null

    const status: PlanRequestStatus = approve ? 'approved' : 'rejected'
    this.db
      .prepare(
        `UPDATE plan_requests
            SET status = ?, reviewed_at = ?, reviewed_by = ?, review_note = ?
          WHERE id = ? AND status = 'pending'`,
      )
      .run(status, new Date().toISOString(), reviewerId, reviewNote, id)

    if (approve) this.setUserPlan(req.userId, req.toPlan)
    return this.getPlanRequest(id) ?? null
  }
}

/** 用户表的公共列（改列的时候只改这一处）。 */
const USER_COLUMNS =
  'SELECT id, username, created_at, role, plan_id, disabled_at FROM users'

function toUser(row: Record<string, unknown>): User {
  return {
    id: Number(row.id),
    username: String(row.username),
    createdAt: String(row.created_at),
    role: row.role === 'admin' ? 'admin' : 'user',
    planId: String(row.plan_id ?? 'free'),
    disabledAt: row.disabled_at ? String(row.disabled_at) : null,
  }
}

function toTotals(row: Record<string, unknown> | undefined): UsageTotals {
  const input = Number(row?.input_tokens ?? 0)
  const output = Number(row?.output_tokens ?? 0)
  return {
    inputTokens: input,
    outputTokens: output,
    totalTokens: input + output,
    calls: Number(row?.calls ?? 0),
  }
}

/** 从行里拼出用量（前缀区分今天 t_ / 本月 m_）。 */
function totalsFrom(row: Record<string, unknown>, prefix: string): UsageTotals {
  const input = Number(row[`${prefix}in`] ?? 0)
  const output = Number(row[`${prefix}out`] ?? 0)
  return {
    inputTokens: input,
    outputTokens: output,
    totalTokens: input + output,
    calls: Number(row[`${prefix}calls`] ?? 0),
  }
}

/** 邀请码用的字符集：去掉了 0/O/1/I/l 这些容易看错的。 */
const INVITE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'

/** 生成 12 位邀请码，分 3 段（XXXX-XXXX-XXXX）。 */
function generateInviteCode(): string {
  const bytes = randomBytes(12)
  let out = ''
  for (let i = 0; i < 12; i++) {
    out += INVITE_ALPHABET[bytes[i] % INVITE_ALPHABET.length]
  }
  return out.match(/.{1,4}/g)!.join('-')
}

function toInviteCode(row: Record<string, unknown>): InviteCode {
  return {
    code: String(row.code),
    planId: String(row.plan_id),
    batch: row.batch ? String(row.batch) : null,
    createdAt: String(row.created_at),
    createdBy: row.created_by == null ? null : Number(row.created_by),
    usedBy: row.used_by == null ? null : Number(row.used_by),
    usedByName: row.used_by_name ? String(row.used_by_name) : null,
    usedAt: row.used_at ? String(row.used_at) : null,
    revokedAt: row.revoked_at ? String(row.revoked_at) : null,
  }
}

function toPlanRequest(row: Record<string, unknown>): PlanRequest {
  const status = String(row.status)
  return {
    id: Number(row.id),
    userId: Number(row.user_id),
    username: String(row.username ?? ''),
    fromPlan: String(row.from_plan),
    toPlan: String(row.to_plan),
    status:
      status === 'approved'
        ? 'approved'
        : status === 'rejected'
          ? 'rejected'
          : 'pending',
    note: row.note ? String(row.note) : null,
    createdAt: String(row.created_at),
    reviewedAt: row.reviewed_at ? String(row.reviewed_at) : null,
    reviewedBy: row.reviewed_by == null ? null : Number(row.reviewed_by),
    reviewNote: row.review_note ? String(row.review_note) : null,
  }
}
