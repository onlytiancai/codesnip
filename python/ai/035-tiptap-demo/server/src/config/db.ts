/**
 * SQLite 库：所有持久化（用户 / 会话 / token 用量 / 全局配置）都在这一个文件里。
 *
 * 用 Node 内置的 `node:sqlite`（DatabaseSync）—— 零第三方依赖，同步 API，
 * 和 ConfigStore 原来的同步风格一致。
 *
 * 注意：Node 24 里 node:sqlite 还会打一句 ExperimentalWarning，无害。
 * 要静音：NODE_OPTIONS=--no-warnings=ExperimentalWarning
 */

import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * node:sqlite 是 Node 内置模块，但 Vite 5 的内置模块列表里还没有它 ——
 * 直接 `import from 'node:sqlite'` 会被打包器当成 npm 包去解析
 * （"Failed to load url sqlite"）。走 createRequire 绕开打包器的解析。
 *
 * 类型用 `typeof import('node:sqlite')` 拿（类型导入会被擦除，不影响运行时）。
 */
const require = createRequire(import.meta.url)
const { DatabaseSync } = require('node:sqlite') as typeof import('node:sqlite')

/** 库的句柄类型（= node:sqlite 的 DatabaseSync）。 */
export type Db = InstanceType<typeof DatabaseSync>

/** 默认库路径：server/data/app.db（src/config/ 和 dist/config/ 深度一致，两边都指对）。 */
export const DEFAULT_DB_PATH = fileURLToPath(
  new URL('../../data/app.db', import.meta.url),
)

/**
 * 全部建表语句。
 *
 * - day / expires_at 用 INTEGER（见下面各表的注释）
 * - created_at / updated_at / saved_at 用 ISO-8601 文本：只读出来给人看，不参与比较
 */
const SCHEMA = `
-- ===== 用户 / 会话 =====
CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT    NOT NULL UNIQUE,
  password_hash TEXT    NOT NULL,
  created_at    TEXT    NOT NULL,
  role          TEXT    NOT NULL DEFAULT 'user',   -- 'user' | 'admin'
  plan_id       TEXT    NOT NULL DEFAULT 'free',    -- 套餐 id（见 config/plans.ts）
  disabled_at   TEXT                              -- 非空 = 已禁用
);

CREATE TABLE IF NOT EXISTS sessions (
  token      TEXT    PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT    NOT NULL,
  expires_at INTEGER NOT NULL          -- epoch 毫秒，直接和 Date.now() 比
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- ===== 每人每月的 token 用量 =====
-- day 是「日桶键」：INTEGER 的 YYYYMMDD（如 20261031，服务端本地日期）。
-- 用 INTEGER 而不是 TEXT：SQLite 没有真正的 DATE 类型（声明 DATE 只是给
-- NUMERIC affinity，底层还是文本），而这一列只做等值和范围比较 —— 真类型
-- 比较最干净，没有 affinity 歧义。
CREATE TABLE IF NOT EXISTS token_usage (
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  day           INTEGER NOT NULL,
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  calls         INTEGER NOT NULL DEFAULT 0,
  updated_at    TEXT    NOT NULL,
  PRIMARY KEY (user_id, day)
);

-- ===== 套餐升级申请 =====
-- 用户提交 → 管理员审核 → 通过就改 users.plan_id
CREATE TABLE IF NOT EXISTS plan_requests (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  from_plan   TEXT    NOT NULL,
  to_plan     TEXT    NOT NULL,
  status      TEXT    NOT NULL,        -- 'pending' | 'approved' | 'rejected'
  note        TEXT,                   -- 用户备注
  created_at  TEXT    NOT NULL,
  reviewed_at TEXT,
  reviewed_by INTEGER REFERENCES users(id),
  review_note TEXT                    -- 管理员的审核意见
);
CREATE INDEX IF NOT EXISTS idx_plan_requests_status ON plan_requests(status);
CREATE INDEX IF NOT EXISTS idx_plan_requests_user ON plan_requests(user_id);

-- ===== 邀请码池 =====
-- 管理员批量生成（绑定一个套餐），用户兑换 → 一次性永久升级
CREATE TABLE IF NOT EXISTS invite_codes (
  code       TEXT PRIMARY KEY,
  plan_id    TEXT    NOT NULL,
  batch      TEXT,                    -- 批次名，方便管理员认哪一批
  created_at TEXT    NOT NULL,
  created_by INTEGER REFERENCES users(id),
  used_by    INTEGER REFERENCES users(id),
  used_at    TEXT,
  revoked_at TEXT                    -- 非空 = 已作废
);
CREATE INDEX IF NOT EXISTS idx_invite_codes_batch ON invite_codes(batch);

-- ===== 全局配置（原来在 config.json / history.json）=====
CREATE TABLE IF NOT EXISTS providers (
  id      TEXT    PRIMARY KEY,
  ordinal INTEGER NOT NULL,
  config  TEXT    NOT NULL             -- JSON: ProviderConfig
);

CREATE TABLE IF NOT EXISTS app_state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL                  -- JSON
);

CREATE TABLE IF NOT EXISTS prompt_templates (
  provider_id   TEXT NOT NULL,
  capability_id TEXT NOT NULL,
  template      TEXT NOT NULL,         -- JSON: PromptTemplate
  PRIMARY KEY (provider_id, capability_id)
);

CREATE TABLE IF NOT EXISTS prompt_tones (
  provider_id TEXT NOT NULL,
  ordinal     INTEGER NOT NULL,
  tone        TEXT NOT NULL,           -- JSON: ToneConfig
  PRIMARY KEY (provider_id, ordinal)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
  provider_id   TEXT    NOT NULL,
  capability_id TEXT    NOT NULL,
  version       INTEGER NOT NULL,
  saved_at      TEXT    NOT NULL,
  note          TEXT,
  template      TEXT    NOT NULL,      -- JSON 快照
  PRIMARY KEY (provider_id, capability_id, version)
);
`

/**
 * 幂等地给已有表补列。
 *
 * ALTER TABLE ... ADD COLUMN 没有 IF NOT EXISTS —— 对已存在的列再跑一遍会报
 * "duplicate column name"。所以先查 pragma table_info，没有才加。
 */
function ensureColumns(db: Db): void {
  const columns = (table: string): Set<string> => {
    const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
      name: string
    }>
    return new Set(rows.map((r) => r.name))
  }

  const users = columns('users')
  if (!users.has('role')) {
    db.exec("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
  }
  if (!users.has('plan_id')) {
    db.exec("ALTER TABLE users ADD COLUMN plan_id TEXT NOT NULL DEFAULT 'free'")
  }
  if (!users.has('disabled_at')) {
    db.exec('ALTER TABLE users ADD COLUMN disabled_at TEXT')
  }
}

/**
 * 打开（或创建）库文件，并确保表结构是最新的。
 */
export function openDatabase(path: string): Db {
  mkdirSync(dirname(path), { recursive: true })
  const db = new DatabaseSync(path)
  // WAL：读写并发更稳（会产生 -wal / -shm 附属文件，记得一起 gitignore）
  db.exec('PRAGMA journal_mode = WAL')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(SCHEMA)
  ensureColumns(db)
  return db
}
