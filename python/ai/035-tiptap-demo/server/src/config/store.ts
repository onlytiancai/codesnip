/**
 * 配置存储：provider 列表 + 生效的 provider + 每个 provider 的提示词集 + 版本历史。
 *
 * **存在 SQLite 里**（和用户 / 会话 / token 用量同一个库文件）。
 * 表：providers / app_state / prompt_templates / prompt_tones / prompt_versions。
 *
 * 内存里的 this.config / this.history 继续当读路径（路由层已经在用），
 * 启动时从表里装载，写的时候同时更新内存和表。
 *
 * **API key 不在这里** —— 它在 server/.env 里，由 provider 的 apiKeyEnv 字段指明读哪个变量。
 * （以前是 secrets.json，现在统一收进 .env，见 config/env.ts。）
 *
 * **老配置导入**：历史上配置存在 server/data/config.json + history.json。
 * 库里没有 provider 时会自动把它们导进来（那两个文件在 git 里，调好的提示词不会丢），
 * 导入后不再回写 JSON 文件。
 */

import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Db } from './db.js'
import { clonePromptSet, defaultPromptSet, defaultProvider } from './defaults.js'
import { debugLog } from '../utils/debug.js'
import {
  CAPABILITY_IDS,
  type AppConfig,
  type CapabilityId,
  type PromptHistory,
  type PromptTemplate,
  type PromptVersion,
  type ProviderConfig,
  type ProviderPromptSet,
  type ToneConfig,
} from './types.js'

/** 默认数据目录：server/data/（src/config/ 和 dist/config/ 深度一致，两边都指对）。 */
export const DEFAULT_DATA_DIR = fileURLToPath(new URL('../../data', import.meta.url))

export interface ConfigStoreOptions {
  /** 共享的 SQLite 库（和 UserStore 用同一个文件）。 */
  db: Db
  /** 老 config.json / history.json 所在目录 —— 首次启动导入用。 */
  legacyDataDir?: string
}

function historyKey(providerId: string, capabilityId: CapabilityId): string {
  return `${providerId}/${capabilityId}`
}

/**
 * 服务端配置不全（没有可用 provider、缺提示词模板）。
 * 路由层靠它把这类错误判成 503 而不是 502。
 */
export class ConfigError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ConfigError'
  }
}

/** 找不到指定的 provider / 版本。 */
export class NotFoundError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NotFoundError'
  }
}

/** 请求本身不合法（比如删当前生效的 provider）。 */
export class InvalidRequestError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'InvalidRequestError'
  }
}

function clone<T>(value: T): T {
  return structuredClone(value)
}

export class ConfigStore {
  private readonly db: Db
  private readonly legacyDataDir: string

  private config: AppConfig
  private history: PromptHistory

  constructor(options: ConfigStoreOptions) {
    this.db = options.db
    this.legacyDataDir = options.legacyDataDir ?? DEFAULT_DATA_DIR

    this.config = this.loadConfig()
    this.history = this.loadHistory()
  }

  // -------------------------------------------------------------------------
  // 读
  // -------------------------------------------------------------------------

  /** 从表里读出一份原始 AppConfig 形状（还没 normalize）。 */
  private readConfigFromDb(): AppConfig {
    const providers: ProviderConfig[] = []
    const providerRows = this.db
      .prepare('SELECT config FROM providers ORDER BY ordinal')
      .all()
    for (const row of providerRows) {
      const parsed = parseJson<ProviderConfig>(row.config)
      if (parsed) providers.push(parsed)
    }

    const promptSets: Record<string, ProviderPromptSet> = {}
    for (const provider of providers) {
      const templates = {} as Record<CapabilityId, PromptTemplate>
      const templateRows = this.db
        .prepare('SELECT capability_id, template FROM prompt_templates WHERE provider_id = ?')
        .all(provider.id)
      for (const row of templateRows) {
        const parsed = parseJson<PromptTemplate>(row.template)
        if (parsed) templates[row.capability_id as CapabilityId] = parsed
      }

      const tones: ToneConfig[] = []
      const toneRows = this.db
        .prepare('SELECT tone FROM prompt_tones WHERE provider_id = ? ORDER BY ordinal')
        .all(provider.id)
      for (const row of toneRows) {
        const parsed = parseJson<ToneConfig>(row.tone)
        if (parsed) tones.push(parsed)
      }

      promptSets[provider.id] = { templates, tones }
    }

    const active = this.readAppState('activeProviderId')

    return {
      version: 1,
      activeProviderId: typeof active === 'string' ? active : '',
      providers,
      promptSets,
    }
  }

  private loadConfig(): AppConfig {
    const raw = this.readConfigFromDb()
    if (raw.providers.length > 0) {
      const parsed = normalizeConfig(raw)
      if (parsed) {
        // 补齐缺失的能力模板（老库里可能只有一部分）
        for (const provider of parsed.providers) {
          this.ensureRows(provider.id, parsed.promptSets[provider.id])
        }
        return parsed
      }
      console.warn('[config] 数据库里的配置不合法，已回退到默认配置')
    }

    // 库里没有 → 试试老 config.json
    const legacy = this.importLegacyConfig()
    if (legacy) {
      const parsed = normalizeConfig(legacy)
      if (parsed) {
        this.writeConfigToDb(parsed)
        debugLog('[config] 已从老的 config.json 导入配置')
        return parsed
      }
    }

    return this.seedConfig()
  }

  /** 读老 config.json（不存在 / 坏了都返回 undefined）。 */
  private importLegacyConfig(): AppConfig | undefined {
    const configPath = join(this.legacyDataDir, 'config.json')
    const raw = this.readJsonFile(configPath)
    if (!raw) return undefined
    const parsed = normalizeConfig(raw)
    if (!parsed) {
      console.warn(`[config] ${configPath} 内容不合法，已回退到默认配置`)
      return undefined
    }
    // 顺带把老的 history.json 也导进来
    const historyRaw = this.readJsonFile(join(this.legacyDataDir, 'history.json'))
    if (historyRaw && typeof historyRaw === 'object' && !Array.isArray(historyRaw)) {
      this.importHistoryToDb(historyRaw as PromptHistory)
    }
    return parsed
  }

  private loadHistory(): PromptHistory {
    const history: PromptHistory = {}
    const rows = this.db
      .prepare(
        'SELECT provider_id, capability_id, version, saved_at, note, template FROM prompt_versions ORDER BY provider_id, capability_id, version',
      )
      .all()
    for (const row of rows) {
      const template = parseJson<PromptTemplate>(row.template)
      if (!template) continue
      const key = `${String(row.provider_id)}/${String(row.capability_id)}`
      const list = history[key] ?? []
      list.push({
        providerId: String(row.provider_id),
        capabilityId: row.capability_id as CapabilityId,
        version: Number(row.version),
        savedAt: String(row.saved_at),
        note: row.note === null || row.note === undefined ? undefined : String(row.note),
        template,
      })
      history[key] = list
    }
    return history
  }

  // -------------------------------------------------------------------------
  // 写
  // -------------------------------------------------------------------------

  /** 整份写入（播种 / 导入老配置时用）。 */
  private writeConfigToDb(config: AppConfig): void {
    this.db.exec('DELETE FROM providers')
    this.db.exec('DELETE FROM prompt_templates')
    this.db.exec('DELETE FROM prompt_tones')
    config.providers.forEach((provider, ordinal) => {
      this.db
        .prepare('INSERT OR REPLACE INTO providers (id, ordinal, config) VALUES (?, ?, ?)')
        .run(provider.id, ordinal, JSON.stringify(provider))
      const set = config.promptSets[provider.id] ?? clonePromptSet(defaultPromptSet())
      this.writePromptSetToDb(provider.id, set)
    })
    this.writeAppState('activeProviderId', config.activeProviderId)
  }

  /** 写一个 provider 的行。 */
  private writeProviderToDb(providerId: string): void {
    const ordinal = this.config.providers.findIndex((p) => p.id === providerId)
    if (ordinal < 0) return
    const provider = this.config.providers[ordinal]
    this.db
      .prepare('INSERT OR REPLACE INTO providers (id, ordinal, config) VALUES (?, ?, ?)')
      .run(provider.id, ordinal, JSON.stringify(provider))
  }

  /** 写一个 provider 的整套提示词 + 语气表。 */
  private writePromptSetToDb(providerId: string, set: ProviderPromptSet): void {
    for (const capabilityId of CAPABILITY_IDS) {
      const template = set.templates[capabilityId]
      if (!template) continue
      this.db
        .prepare(
          'INSERT OR REPLACE INTO prompt_templates (provider_id, capability_id, template) VALUES (?, ?, ?)',
        )
        .run(providerId, capabilityId, JSON.stringify(template))
    }
    this.db.prepare('DELETE FROM prompt_tones WHERE provider_id = ?').run(providerId)
    set.tones.forEach((tone, ordinal) => {
      this.db
        .prepare('INSERT INTO prompt_tones (provider_id, ordinal, tone) VALUES (?, ?, ?)')
        .run(providerId, ordinal, JSON.stringify(tone))
    })
  }

  /** 只补缺失的行，不覆盖已有的。 */
  private ensureRows(providerId: string, set: ProviderPromptSet | undefined): void {
    if (!set) return
    const existing = new Set(
      this.db
        .prepare('SELECT capability_id FROM prompt_templates WHERE provider_id = ?')
        .all(providerId)
        .map((row) => String(row.capability_id)),
    )
    for (const capabilityId of CAPABILITY_IDS) {
      if (existing.has(capabilityId)) continue
      const template = set.templates[capabilityId]
      if (!template) continue
      this.db
        .prepare(
          'INSERT OR REPLACE INTO prompt_templates (provider_id, capability_id, template) VALUES (?, ?, ?)',
        )
        .run(providerId, capabilityId, JSON.stringify(template))
    }
    const toneCount = this.db
      .prepare('SELECT COUNT(*) AS n FROM prompt_tones WHERE provider_id = ?')
      .get(providerId)
    if (Number(toneCount?.n ?? 0) === 0 && set.tones.length > 0) {
      set.tones.forEach((tone, ordinal) => {
        this.db
          .prepare('INSERT INTO prompt_tones (provider_id, ordinal, tone) VALUES (?, ?, ?)')
          .run(providerId, ordinal, JSON.stringify(tone))
      })
    }
  }

  private writeAppState(key: string, value: unknown): void {
    this.db
      .prepare('INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)')
      .run(key, JSON.stringify(value))
  }

  private readAppState(key: string): unknown {
    const row = this.db.prepare('SELECT value FROM app_state WHERE key = ?').get(key)
    if (!row) return undefined
    return parseJson<unknown>(row.value)
  }

  /** 导入老 history.json 里的版本记录。 */
  private importHistoryToDb(history: PromptHistory): void {
    for (const [key, versions] of Object.entries(history)) {
      if (!Array.isArray(versions)) continue
      const [providerId, capabilityId] = key.split('/')
      if (!providerId || !capabilityId) continue
      for (const record of versions) {
        if (!record || typeof record.version !== 'number' || !record.template) continue
        this.db
          .prepare(
            'INSERT OR REPLACE INTO prompt_versions (provider_id, capability_id, version, saved_at, note, template) VALUES (?, ?, ?, ?, ?, ?)',
          )
          .run(
            providerId,
            capabilityId,
            record.version,
            record.savedAt ?? new Date().toISOString(),
            record.note ?? null,
            JSON.stringify(record.template),
          )
      }
    }
  }

  private readJsonFile(path: string): unknown {
    if (!existsSync(path)) return undefined
    try {
      return JSON.parse(readFileSync(path, 'utf8'))
    } catch (err) {
      console.warn(`[config] 读取 ${path} 失败：`, err)
      return undefined
    }
  }

  /** 首次运行：按内置默认值播种并落库。 */
  private seedConfig(): AppConfig {
    const provider = defaultProvider()
    const config: AppConfig = {
      version: 1,
      activeProviderId: provider.id,
      providers: [provider],
      promptSets: { [provider.id]: clonePromptSet(defaultPromptSet()) },
    }
    this.config = config
    this.writeConfigToDb(config)
    debugLog('[config] 已播种默认配置')
    return config
  }

  // -------------------------------------------------------------------------
  // 配置
  // -------------------------------------------------------------------------

  getConfig(): AppConfig {
    return this.config
  }

  listProviders(): ProviderConfig[] {
    return this.config.providers.map((p) => ({ ...p }))
  }

  getProvider(id: string): ProviderConfig | undefined {
    const found = this.config.providers.find((p) => p.id === id)
    return found ? { ...found } : undefined
  }

  /** 当前生效的 provider；配置坏了就退回第一个。 */
  getActiveProvider(): ProviderConfig {
    const found = this.config.providers.find(
      (p) => p.id === this.config.activeProviderId,
    )
    const provider = found ?? this.config.providers[0]
    if (!provider) {
      throw new ConfigError('没有可用的 provider —— 请先在设置里添加一个')
    }
    return { ...provider }
  }

  getActiveProviderId(): string {
    return this.getActiveProvider().id
  }

  /**
   * 新增 / 更新 provider。
   *
   * 新增时会用内置默认提示词集初始化它的一套独立提示词 —— 之后各 provider
   * 的提示词互不影响，可以单独改。
   *
   * API key 不在这里管 —— 它在 server/.env 里，provider 只记一个变量名（apiKeyEnv）。
   */
  upsertProvider(provider: ProviderConfig): ProviderConfig {
    const idx = this.config.providers.findIndex((p) => p.id === provider.id)
    const isNew = idx === -1

    if (isNew) {
      this.config.providers.push({ ...provider })
      this.config.promptSets[provider.id] = clonePromptSet(defaultPromptSet())
    } else {
      this.config.providers[idx] = { ...provider }
      // 老 provider 也要保证有提示词集（比如导入的旧数据）
      if (!this.config.promptSets[provider.id]) {
        this.config.promptSets[provider.id] = clonePromptSet(defaultPromptSet())
      }
    }

    this.writeProviderToDb(provider.id)
    this.writePromptSetToDb(provider.id, this.config.promptSets[provider.id])
    debugLog('[config] provider 变更:', provider.id, isNew ? '(新增)' : '(更新)')
    return { ...provider }
  }

  deleteProvider(id: string): void {
    if (this.config.activeProviderId === id) {
      throw new InvalidRequestError(
        '不能删除当前生效的 provider —— 请先切换到别的 provider',
      )
    }
    this.config.providers = this.config.providers.filter((p) => p.id !== id)
    delete this.config.promptSets[id]
    for (const key of Object.keys(this.history)) {
      if (key.startsWith(`${id}/`)) delete this.history[key]
    }

    this.db.prepare('DELETE FROM providers WHERE id = ?').run(id)
    this.db.prepare('DELETE FROM prompt_templates WHERE provider_id = ?').run(id)
    this.db.prepare('DELETE FROM prompt_tones WHERE provider_id = ?').run(id)
    this.db.prepare('DELETE FROM prompt_versions WHERE provider_id = ?').run(id)
    // 剩下的重新编号
    this.config.providers.forEach((provider) => this.writeProviderToDb(provider.id))
    debugLog('[config] 删除 provider:', id)
  }

  setActiveProvider(id: string): void {
    if (!this.config.providers.some((p) => p.id === id)) {
      throw new NotFoundError(`provider "${id}" 不存在`)
    }
    this.config.activeProviderId = id
    this.writeAppState('activeProviderId', id)
    debugLog('[config] 切换生效 provider:', id)
  }

  // -------------------------------------------------------------------------
  // 提示词
  // -------------------------------------------------------------------------

  getPromptSet(providerId: string): ProviderPromptSet {
    const set = this.config.promptSets[providerId]
    if (set) return set
    // 没记录就补一份默认的（不让调用方炸）
    const seeded = clonePromptSet(defaultPromptSet())
    this.config.promptSets[providerId] = seeded
    this.writePromptSetToDb(providerId, seeded)
    return seeded
  }

  getPrompt(providerId: string, capabilityId: CapabilityId): PromptTemplate {
    const set = this.getPromptSet(providerId)
    const template = set.templates[capabilityId]
    if (!template) {
      throw new ConfigError(
        `provider "${providerId}" 没有 "${capabilityId}" 的提示词模板`,
      )
    }
    return template
  }

  getTones(providerId: string): ToneConfig[] {
    return this.getPromptSet(providerId).tones.map((t) => ({ ...t }))
  }

  setTones(providerId: string, tones: ToneConfig[]): ToneConfig[] {
    const set = this.getPromptSet(providerId)
    set.tones = tones.map((t) => ({ ...t }))
    this.db.prepare('DELETE FROM prompt_tones WHERE provider_id = ?').run(providerId)
    set.tones.forEach((tone, ordinal) => {
      this.db
        .prepare('INSERT INTO prompt_tones (provider_id, ordinal, tone) VALUES (?, ?, ?)')
        .run(providerId, ordinal, JSON.stringify(tone))
    })
    debugLog('[config] 保存语气表:', providerId)
    return this.getTones(providerId)
  }

  /**
   * 保存提示词 → 产生一个新版本（append-only）。
   */
  savePrompt(
    providerId: string,
    capabilityId: CapabilityId,
    template: PromptTemplate,
    note?: string,
  ): PromptVersion {
    const set = this.getPromptSet(providerId)
    const key = historyKey(providerId, capabilityId)
    const list = this.history[key] ?? []
    const version = (list[list.length - 1]?.version ?? 0) + 1

    const record: PromptVersion = {
      providerId,
      capabilityId,
      version,
      savedAt: new Date().toISOString(),
      note,
      template: clone(template),
    }

    this.history[key] = [...list, record]
    set.templates[capabilityId] = clone(template)

    this.db
      .prepare(
        'INSERT OR REPLACE INTO prompt_templates (provider_id, capability_id, template) VALUES (?, ?, ?)',
      )
      .run(providerId, capabilityId, JSON.stringify(template))
    this.db
      .prepare(
        'INSERT OR REPLACE INTO prompt_versions (provider_id, capability_id, version, saved_at, note, template) VALUES (?, ?, ?, ?, ?, ?)',
      )
      .run(
        providerId,
        capabilityId,
        version,
        record.savedAt,
        note ?? null,
        JSON.stringify(template),
      )
    debugLog('[config] 保存提示词:', `${providerId}/${capabilityId}`, `v${version}`)
    return record
  }

  listVersions(
    providerId: string,
    capabilityId: CapabilityId,
  ): PromptVersion[] {
    return (this.history[historyKey(providerId, capabilityId)] ?? []).map((v) => ({
      ...v,
      template: clone(v.template),
    }))
  }

  getVersion(
    providerId: string,
    capabilityId: CapabilityId,
    version: number,
  ): PromptVersion | undefined {
    const found = this.listVersions(providerId, capabilityId).find(
      (v) => v.version === version,
    )
    return found
  }

  /**
   * 回滚到某个历史版本。
   *
   * **不改历史** —— 把旧内容复制成一个新版本再设为当前，
   * 这样版本号只增不减，回滚本身也是一条可追溯的记录。
   */
  rollback(
    providerId: string,
    capabilityId: CapabilityId,
    version: number,
  ): PromptVersion {
    const target = this.getVersion(providerId, capabilityId, version)
    if (!target) {
      throw new NotFoundError(`版本 v${version} 不存在`)
    }
    return this.savePrompt(
      providerId,
      capabilityId,
      target.template,
      `rollback to v${version}`,
    )
  }

  /**
   * 从另一个 provider 复制提示词。
   *
   * capabilityId 省略 = 复制整套；给了 = 只复制那一个能力的模板。
   * 复制走普通保存流程，因此也会留下一条版本记录。
   */
  copyPrompts(
    fromProviderId: string,
    toProviderId: string,
    capabilityId?: CapabilityId,
    note?: string,
  ): void {
    const from = this.getPromptSet(fromProviderId)
    const ids = capabilityId ? [capabilityId] : [...CAPABILITY_IDS]
    for (const id of ids) {
      const template = from.templates[id]
      if (!template) continue
      this.savePrompt(
        toProviderId,
        id,
        template,
        note ?? `copied from ${fromProviderId}`,
      )
    }
  }

  // -------------------------------------------------------------------------
  // 密钥
  // -------------------------------------------------------------------------

  /**
   * API key 只从环境变量读 —— provider 的 apiKeyEnv 字段指明读哪个变量名。
   *
   * key 本身写在 server/.env 里（见 config/env.ts），不进 git。
   */
  getApiKey(providerId: string): string | undefined {
    const provider = this.getProvider(providerId)
    const name = provider?.apiKeyEnv
    if (!name) return undefined
    const value = process.env[name]
    return value && value.trim() ? value : undefined
  }

  hasApiKey(providerId: string): boolean {
    return Boolean(this.getApiKey(providerId))
  }
}

/** 解析一行里的 JSON；坏了就返回 undefined（跳过这一行）。 */
function parseJson<T>(value: unknown): T | undefined {
  if (typeof value !== 'string') return undefined
  try {
    return JSON.parse(value) as T
  } catch {
    return undefined
  }
}

/**
 * 校正读回来的配置：补齐缺失字段、丢弃不认识的 provider 记录。
 * 返回 undefined 表示这份数据没法用。
 */
function normalizeConfig(raw: unknown): AppConfig | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const obj = raw as Record<string, unknown>

  const providers = Array.isArray(obj.providers)
    ? (obj.providers as ProviderConfig[]).filter(
        (p) => p && typeof p.id === 'string' && typeof p.kind === 'string',
      )
    : []
  if (providers.length === 0) return undefined

  const promptSetsRaw =
    obj.promptSets && typeof obj.promptSets === 'object'
      ? (obj.promptSets as Record<string, ProviderPromptSet>)
      : {}

  const promptSets: Record<string, ProviderPromptSet> = {}
  for (const provider of providers) {
    promptSets[provider.id] = promptSetsRaw[provider.id]
      ? normalizePromptSet(promptSetsRaw[provider.id])
      : clonePromptSet(defaultPromptSet())
  }

  const activeProviderId =
    typeof obj.activeProviderId === 'string' &&
    providers.some((p) => p.id === obj.activeProviderId)
      ? obj.activeProviderId
      : providers[0].id

  return { version: 1, activeProviderId, providers, promptSets }
}

function normalizePromptSet(raw: ProviderPromptSet): ProviderPromptSet {
  const fallback = defaultPromptSet()
  const templates = { ...fallback.templates }
  if (raw && typeof raw === 'object' && raw.templates) {
    for (const id of CAPABILITY_IDS) {
      const t = (raw.templates as Record<string, PromptTemplate>)[id]
      if (t && typeof t.system === 'string' && typeof t.user === 'string') {
        templates[id] = {
          ...t,
          responseMode: t.responseMode === 'json' ? 'json' : 'text',
        }
      }
    }
  }
  const tones =
    Array.isArray(raw?.tones) && raw.tones.length > 0
      ? raw.tones.map((t) => ({ ...t }))
      : fallback.tones.map((t) => ({ ...t }))
  return { templates, tones }
}
