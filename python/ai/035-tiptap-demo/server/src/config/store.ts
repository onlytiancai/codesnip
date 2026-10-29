/**
 * 配置存储：读写 config.json / history.json / secrets.json。
 *
 * 三个文件的分工：
 * - config.json   —— provider 列表 + 当前生效的 provider + 每个 provider 的提示词集（进 git）
 * - history.json  —— 提示词的 append-only 版本历史（进 git）
 * - secrets.json  —— API key（**不进 git**，见 server/.gitignore）
 *
 * 写入一律「先写 .tmp 再 rename」，避免半截文件。
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { clonePromptSet, defaultPromptSet, defaultProvider } from './defaults.js'
import {
  CAPABILITY_IDS,
  type AppConfig,
  type CapabilityId,
  type PromptHistory,
  type PromptTemplate,
  type PromptVersion,
  type ProviderConfig,
  type ProviderPromptSet,
  type Secrets,
  type ToneConfig,
} from './types.js'

/** 默认数据目录：server/data/（src/config/ 和 dist/config/ 深度一致，两边都指对）。 */
export const DEFAULT_DATA_DIR = fileURLToPath(new URL('../../data', import.meta.url))

export interface ConfigStoreOptions {
  /** 数据目录，默认 server/data */
  dataDir?: string
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
  private readonly dataDir: string
  private readonly configPath: string
  private readonly historyPath: string
  private readonly secretsPath: string

  private config: AppConfig
  private history: PromptHistory
  private secrets: Secrets

  constructor(options: ConfigStoreOptions = {}) {
    this.dataDir = options.dataDir ?? DEFAULT_DATA_DIR
    this.configPath = join(this.dataDir, 'config.json')
    this.historyPath = join(this.dataDir, 'history.json')
    this.secretsPath = join(this.dataDir, 'secrets.json')

    this.config = this.loadConfig()
    this.history = this.loadHistory()
    this.secrets = this.loadSecrets()

    // history.json 即使是空的也要存在 —— 它是要进 git 的文件
    if (!existsSync(this.historyPath)) {
      this.writeHistory()
    }
  }

  // -------------------------------------------------------------------------
  // 读写
  // -------------------------------------------------------------------------

  private loadConfig(): AppConfig {
    const raw = this.readJsonFile(this.configPath)
    if (raw) {
      const parsed = normalizeConfig(raw)
      if (parsed) return parsed
      console.warn(
        `[config] ${this.configPath} 内容不合法，已回退到默认配置（原文件保留，未覆盖）`,
      )
      return this.seedConfig()
    }
    return this.seedConfig()
  }

  /** 首次运行：按内置默认值播种并落盘。 */
  private seedConfig(): AppConfig {
    const provider = defaultProvider()
    const config: AppConfig = {
      version: 1,
      activeProviderId: provider.id,
      providers: [provider],
      promptSets: { [provider.id]: clonePromptSet(defaultPromptSet()) },
    }
    this.config = config
    this.writeConfig()
    return config
  }

  private loadHistory(): PromptHistory {
    const raw = this.readJsonFile(this.historyPath)
    if (!raw) return {}
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return {}
    return raw as PromptHistory
  }

  private loadSecrets(): Secrets {
    const raw = this.readJsonFile(this.secretsPath)
    if (!raw) return {}
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return {}
    return raw as Secrets
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

  private writeJsonFile(path: string, value: unknown, mode?: number): void {
    mkdirSync(this.dataDir, { recursive: true })
    const tmp = `${path}.tmp`
    writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: 'utf8',
      mode,
    })
    renameSync(tmp, path)
  }

  private writeConfig(): void {
    this.writeJsonFile(this.configPath, this.config)
  }

  private writeHistory(): void {
    this.writeJsonFile(this.historyPath, this.history)
  }

  private writeSecrets(): void {
    this.writeJsonFile(this.secretsPath, this.secrets, 0o600)
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
   */
  upsertProvider(provider: ProviderConfig, apiKey?: string): ProviderConfig {
    const idx = this.config.providers.findIndex((p) => p.id === provider.id)
    const isNew = idx === -1

    if (isNew) {
      this.config.providers.push({ ...provider })
      this.config.promptSets[provider.id] = clonePromptSet(defaultPromptSet())
    } else {
      this.config.providers[idx] = { ...provider }
      // 老 provider 也要保证有提示词集（比如手改过 config.json）
      if (!this.config.promptSets[provider.id]) {
        this.config.promptSets[provider.id] = clonePromptSet(defaultPromptSet())
      }
    }

    if (apiKey !== undefined) {
      this.setApiKey(provider.id, apiKey || undefined)
    }

    this.writeConfig()
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
    delete this.secrets[id]
    this.writeConfig()
    this.writeHistory()
    this.writeSecrets()
  }

  setActiveProvider(id: string): void {
    if (!this.config.providers.some((p) => p.id === id)) {
      throw new NotFoundError(`provider "${id}" 不存在`)
    }
    this.config.activeProviderId = id
    this.writeConfig()
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
    this.writeConfig()
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
    this.writeConfig()
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

    this.writeHistory()
    this.writeConfig()
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

  /** key 优先级：secrets.json > provider.apiKeyEnv 指向的环境变量。 */
  getApiKey(providerId: string): string | undefined {
    const fromFile = this.secrets[providerId]?.apiKey
    if (fromFile) return fromFile
    const provider = this.getProvider(providerId)
    if (provider?.apiKeyEnv) {
      const fromEnv = process.env[provider.apiKeyEnv]
      if (fromEnv) return fromEnv
    }
    return undefined
  }

  hasApiKey(providerId: string): boolean {
    return Boolean(this.getApiKey(providerId))
  }

  setApiKey(providerId: string, apiKey: string | undefined): void {
    if (apiKey) {
      this.secrets[providerId] = { apiKey }
    } else {
      delete this.secrets[providerId]
    }
    this.writeSecrets()
  }
}

/**
 * 校正从磁盘读回来的配置：补齐缺失字段、丢弃不认识的 provider 记录。
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
