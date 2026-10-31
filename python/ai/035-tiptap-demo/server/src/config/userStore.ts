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

export interface User {
  id: number
  username: string
  /** 建号时间（ISO-8601） */
  createdAt: string
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

  countUsers(): number {
    const row = this.db.prepare('SELECT COUNT(*) AS n FROM users').get()
    return Number(row?.n ?? 0)
  }

  createUser(username: string, password: string): User {
    const name = username.trim()
    if (this.findUserByUsername(name)) {
      throw new InvalidRequestError(`用户名 "${name}" 已被占用`)
    }
    const createdAt = new Date().toISOString()
    const info = this.db
      .prepare(
        'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
      )
      .run(name, UserStore.hashPassword(password), createdAt)
    return { id: Number(info.lastInsertRowid), username: name, createdAt }
  }

  findUserByUsername(username: string): User | undefined {
    const row = this.db
      .prepare('SELECT id, username, created_at FROM users WHERE username = ?')
      .get(username.trim())
    return row ? toUser(row) : undefined
  }

  findUserById(id: number): User | undefined {
    const row = this.db
      .prepare('SELECT id, username, created_at FROM users WHERE id = ?')
      .get(id)
    return row ? toUser(row) : undefined
  }

  /** 密码校验：用户不存在和密码错返回同一个结果（不泄露用户是否存在）。 */
  authenticate(username: string, password: string): User | undefined {
    const row = this.db
      .prepare('SELECT id, username, password_hash, created_at FROM users WHERE username = ?')
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
        `SELECT u.id, u.username, u.created_at
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
}

function toUser(row: Record<string, unknown>): User {
  return {
    id: Number(row.id),
    username: String(row.username),
    createdAt: String(row.created_at),
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
