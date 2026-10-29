/**
 * 配置相关路由：provider 管理 + 每个 provider 各自的提示词（含版本历史 / 回滚 / 试运行）。
 *
 * 提示词是**按 provider 分组**的 —— 不同模型需要不同措辞，所以每个 provider
 * 拥有一整套自己的模板 + 语气表，改一个不影响另一个。
 *
 * 现有的 5 个 AI 路由（/api/translate 等）契约不变，这个路由是新增的。
 */

import { Router } from 'express'
import {
  CAPABILITY_JSON_KEY,
  isToneId,
  TONE_IDS,
  type ToneId,
} from '../config/defaults.js'
import {
  CAPABILITY_SAMPLES,
  CAPABILITY_VARIABLES,
  buildContext,
  extractOutput,
  renderPrompt,
  type CapabilityInput,
} from '../config/capabilities.js'
import { ConfigError, NotFoundError, type ConfigStore } from '../config/store.js'
import {
  CAPABILITY_IDS,
  type CapabilityId,
  type JsonSchema,
  type PromptTemplate,
  type ProviderConfig,
  type ProviderKind,
  type StructuredOutputMode,
  type ToneConfig,
} from '../config/types.js'
import { isBalanced } from '../config/template.js'
import { ClientRegistry } from '../llm/registry.js'
import { isSelectionKind } from '../prompts/rewriteSelection.js'
import { isSentenceOperation } from '../prompts/rewriteSentence.js'
import { fail } from './httpError.js'

const PROVIDER_KINDS: ProviderKind[] = ['ollama', 'openai', 'anthropic']
const STRUCTURED_MODES: StructuredOutputMode[] = [
  'schema',
  'object',
  'prompt',
  'off',
]

export interface CreateSettingsRouterOptions {
  store: ConfigStore
  registry: ClientRegistry
}

/** provider 对外暴露的形状 —— 附带 hasKey，明文 key 永远不返回。 */
function publicProvider(
  provider: ProviderConfig,
  store: ConfigStore,
): ProviderConfig & { hasKey: boolean } {
  return { ...provider, hasKey: store.hasApiKey(provider.id) }
}

function isCapabilityId(v: unknown): v is CapabilityId {
  return typeof v === 'string' && (CAPABILITY_IDS as readonly string[]).includes(v)
}

// ---------------------------------------------------------------------------
// 入参解析
// ---------------------------------------------------------------------------

interface ParsedProvider {
  provider: ProviderConfig
  /** undefined = 不动现有 key；空串 = 删掉 key */
  apiKey?: string
}

function parseProvider(body: unknown): ParsedProvider | { error: string } {
  const b = (body ?? {}) as Record<string, unknown>
  const id = typeof b.id === 'string' ? b.id.trim() : ''
  if (!id) return { error: 'id is required' }
  if (!/^[A-Za-z0-9_.-]+$/.test(id)) {
    return { error: 'id 只能包含字母、数字、_ . -' }
  }
  const label = typeof b.label === 'string' ? b.label.trim() : ''
  if (!label) return { error: 'label is required' }
  const kind = b.kind as ProviderKind
  if (!PROVIDER_KINDS.includes(kind)) {
    return { error: `kind must be one of ${PROVIDER_KINDS.join(', ')}` }
  }
  const model = typeof b.model === 'string' ? b.model.trim() : ''
  if (!model) return { error: 'model is required' }

  const structuredOutput = (b.structuredOutput ?? 'schema') as StructuredOutputMode
  if (!STRUCTURED_MODES.includes(structuredOutput)) {
    return { error: `structuredOutput must be one of ${STRUCTURED_MODES.join(', ')}` }
  }

  const provider: ProviderConfig = {
    id,
    label,
    kind,
    baseUrl: typeof b.baseUrl === 'string' ? b.baseUrl.trim() : '',
    model,
    structuredOutput,
  }
  if (typeof b.apiKeyEnv === 'string' && b.apiKeyEnv.trim()) {
    provider.apiKeyEnv = b.apiKeyEnv.trim()
  }
  if (typeof b.maxTokens === 'number' && Number.isFinite(b.maxTokens)) {
    provider.maxTokens = Math.floor(b.maxTokens)
  }
  if (
    b.maxTokensParam === 'max_tokens' ||
    b.maxTokensParam === 'max_completion_tokens'
  ) {
    provider.maxTokensParam = b.maxTokensParam
  }
  if (typeof b.temperature === 'number' && Number.isFinite(b.temperature)) {
    provider.temperature = b.temperature
  }
  if (typeof b.timeoutMs === 'number' && Number.isFinite(b.timeoutMs)) {
    provider.timeoutMs = Math.floor(b.timeoutMs)
  }

  const parsed: ParsedProvider = { provider }
  if (typeof b.apiKey === 'string') parsed.apiKey = b.apiKey
  return parsed
}

interface ParsedTemplate {
  template: PromptTemplate
}

function parseTemplate(
  capabilityId: CapabilityId,
  body: unknown,
): ParsedTemplate | { error: string } {
  const b = (body ?? {}) as Record<string, unknown>
  const template = (b.template ?? b) as Record<string, unknown>

  const system = typeof template.system === 'string' ? template.system : ''
  const user = typeof template.user === 'string' ? template.user : ''
  if (!system.trim()) return { error: 'system is required' }
  if (!user.trim()) return { error: 'user is required' }
  if (!isBalanced(system)) return { error: 'system 模板的花括号没配平' }
  if (!isBalanced(user)) return { error: 'user 模板的花括号没配平' }

  const responseMode = template.responseMode === 'json' ? 'json' : 'text'

  const result: PromptTemplate = {
    id: capabilityId,
    label:
      typeof template.label === 'string' && template.label.trim()
        ? template.label.trim()
        : capabilityId,
    system,
    user,
    responseMode,
  }

  if (responseMode === 'json') {
    const schema = template.jsonSchema
    if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
      return { error: 'JSON 模式必须提供 jsonSchema 对象' }
    }
    const key = CAPABILITY_JSON_KEY[capabilityId]
    const props = (schema as { properties?: Record<string, { type?: unknown }> })
      .properties
    const prop = props?.[key]
    const expect = capabilityId === 'rewriteSelection' ? 'array' : 'string'
    if (!prop) {
      return { error: `jsonSchema 缺少 "${key}" 字段` }
    }
    if (prop.type !== expect) {
      return { error: `jsonSchema 的 "${key}" 字段类型必须是 ${expect}` }
    }
    result.jsonSchema = schema as JsonSchema
  }

  return { template: result }
}

/** 试运行用的输入：从裸变量组装成能力输入。 */
function buildDryRunInput(
  capabilityId: CapabilityId,
  raw: unknown,
): CapabilityInput | { error: string } {
  const v = (raw ?? {}) as Record<string, unknown>
  const str = (key: string): string | undefined => {
    const val = v[key]
    if (typeof val !== 'string') return undefined
    return val
  }

  switch (capabilityId) {
    case 'translate': {
      const text = str('text') ?? ''
      if (!text.trim()) return { error: '缺少 text' }
      const tone = str('tone')
      if (tone !== undefined && !isToneId(tone)) {
        return { error: `tone 必须是 ${TONE_IDS.join(' | ')}` }
      }
      return { text, tone: tone as ToneId | undefined }
    }
    case 'rewriteSelection': {
      const text = str('text') ?? ''
      if (!text.trim()) return { error: '缺少 text' }
      const kind = str('kind') ?? 'word'
      if (!isSelectionKind(kind)) {
        return { error: 'kind 必须是 word 或 phrase' }
      }
      return {
        text,
        kind,
        context: str('context'),
        sourceSentence: str('sourceSentence'),
      }
    }
    case 'rewriteSentence': {
      const sentence = str('sentence') ?? ''
      if (!sentence.trim()) return { error: '缺少 sentence' }
      const operation = str('operation') ?? 'rewrite'
      if (!isSentenceOperation(operation)) {
        return { error: 'operation 必须是 rewrite | improve | shorten | expand' }
      }
      const tone = str('tone')
      if (tone !== undefined && !isToneId(tone)) {
        return { error: `tone 必须是 ${TONE_IDS.join(' | ')}` }
      }
      return {
        sentence,
        operation,
        tone: tone as ToneId | undefined,
        context: str('context'),
        sourceSentence: str('sourceSentence'),
      }
    }
    case 'grammarCheck': {
      const sentence = str('sentence') ?? ''
      if (!sentence.trim()) return { error: '缺少 sentence' }
      return { sentence }
    }
    case 'rewriteInstruction': {
      const text = str('text') ?? ''
      const instruction = str('instruction') ?? ''
      if (!text.trim()) return { error: '缺少 text' }
      if (!instruction.trim()) return { error: '缺少 instruction' }
      return { text, instruction, context: str('context') }
    }
  }
}

// ---------------------------------------------------------------------------
// 路由
// ---------------------------------------------------------------------------

export function createSettingsRouter(options: CreateSettingsRouterOptions): Router {
  const { store, registry } = options
  const router = Router()

  // ---- 总览 ----

  router.get('/api/config', (_req, res) => {
    const activeId = store.getActiveProviderId()
    res.json({
      activeProviderId: activeId,
      providers: store.listProviders().map((p) => publicProvider(p, store)),
      tones: store.getTones(activeId),
      capabilities: CAPABILITY_IDS.map((id) => ({
        id,
        label: store.getPrompt(activeId, id).label,
      })),
      toneIds: TONE_IDS,
    })
  })

  // ---- provider ----

  router.get('/api/providers', (_req, res) => {
    res.json({
      activeProviderId: store.getActiveProviderId(),
      providers: store.listProviders().map((p) => publicProvider(p, store)),
    })
  })

  router.put('/api/providers', (req, res) => {
    const parsed = parseProvider(req.body)
    if ('error' in parsed) {
      return res.status(400).json({ error: parsed.error })
    }
    try {
      const saved = store.upsertProvider(parsed.provider, parsed.apiKey)
      res.json({ provider: publicProvider(saved, store) })
    } catch (err) {
      return fail(res, err)
    }
  })

  router.delete('/api/providers/:id', (req, res) => {
    try {
      store.deleteProvider(req.params.id)
      res.json({ ok: true })
    } catch (err) {
      return fail(res, err)
    }
  })

  router.post('/api/providers/active', (req, res) => {
    const id = (req.body as { id?: unknown })?.id
    if (typeof id !== 'string' || !id.trim()) {
      return res.status(400).json({ error: 'id is required' })
    }
    try {
      store.setActiveProvider(id.trim())
      res.json({ activeProviderId: store.getActiveProviderId() })
    } catch (err) {
      return fail(res, err)
    }
  })

  /** 探活：用该 provider 发一次最小请求。 */
  router.post('/api/providers/:id/test', async (req, res) => {
    const provider = store.getProvider(req.params.id)
    if (!provider) {
      return res.status(404).json({ error: `provider "${req.params.id}" 不存在` })
    }
    try {
      const client = registry.get(provider, store.getApiKey(provider.id))
      const result = await client.chat({
        system: 'Reply with the single word: OK',
        user: 'ping',
        jsonSchema: undefined,
      })
      res.json({
        ok: true,
        sample: result.content.trim().slice(0, 200),
        model: provider.model,
      })
    } catch (err) {
      return res.status(502).json({
        ok: false,
        error: err instanceof Error ? err.message : 'unknown error',
      })
    }
  })

  router.get('/api/providers/:id/models', async (req, res) => {
    const provider = store.getProvider(req.params.id)
    if (!provider) {
      return res.status(404).json({ error: `provider "${req.params.id}" 不存在` })
    }
    try {
      const client = registry.get(provider, store.getApiKey(provider.id))
      res.json({ models: await client.listModels() })
    } catch (err) {
      return res.status(502).json({
        error: err instanceof Error ? err.message : 'unknown error',
      })
    }
  })

  // ---- 提示词（按 provider）----

  /** 某个 provider 的整套提示词 + 语气表 + 每个模板的当前版本号。 */
  router.get('/api/providers/:id/prompts', (req, res) => {
    const providerId = req.params.id
    try {
      const set = store.getPromptSet(providerId)
      const templates = CAPABILITY_IDS.map((capabilityId) => {
        const versions = store.listVersions(providerId, capabilityId)
        return {
          ...set.templates[capabilityId],
          version: versions[versions.length - 1]?.version ?? 0,
          updatedAt: versions[versions.length - 1]?.savedAt,
        }
      })
      res.json({
        providerId,
        templates,
        tones: set.tones,
        variables: CAPABILITY_VARIABLES,
        samples: CAPABILITY_SAMPLES,
      })
    } catch (err) {
      return fail(res, err)
    }
  })

  /** 单个能力：模板 + 版本列表。 */
  router.get('/api/providers/:id/prompts/:capabilityId', (req, res) => {
    const { id, capabilityId } = req.params
    if (!isCapabilityId(capabilityId)) {
      return res.status(400).json({ error: `未知的 capability "${capabilityId}"` })
    }
    try {
      const versions = store.listVersions(id, capabilityId)
      res.json({
        providerId: id,
        capabilityId,
        template: store.getPrompt(id, capabilityId),
        version: versions[versions.length - 1]?.version ?? 0,
        versions: versions.map((v) => ({
          version: v.version,
          savedAt: v.savedAt,
          note: v.note,
        })),
      })
    } catch (err) {
      return fail(res, err)
    }
  })

  /** 保存 → 产生新版本。 */
  router.put('/api/providers/:id/prompts/:capabilityId', (req, res) => {
    const { id, capabilityId } = req.params
    if (!isCapabilityId(capabilityId)) {
      return res.status(400).json({ error: `未知的 capability "${capabilityId}"` })
    }
    const parsed = parseTemplate(capabilityId, req.body)
    if ('error' in parsed) {
      return res.status(400).json({ error: parsed.error })
    }
    const note = (req.body as { note?: unknown })?.note
    try {
      const record = store.savePrompt(
        id,
        capabilityId,
        parsed.template,
        typeof note === 'string' ? note : undefined,
      )
      res.json({
        prompt: record.template,
        version: record.version,
        savedAt: record.savedAt,
      })
    } catch (err) {
      return fail(res, err)
    }
  })

  router.get(
    '/api/providers/:id/prompts/:capabilityId/versions',
    (req, res) => {
      const { id, capabilityId } = req.params
      if (!isCapabilityId(capabilityId)) {
        return res
          .status(400)
          .json({ error: `未知的 capability "${capabilityId}"` })
      }
      res.json({
        versions: store
          .listVersions(id, capabilityId)
          .map((v) => ({ version: v.version, savedAt: v.savedAt, note: v.note })),
      })
    },
  )

  router.get(
    '/api/providers/:id/prompts/:capabilityId/versions/:version',
    (req, res) => {
      const { id, capabilityId } = req.params
      if (!isCapabilityId(capabilityId)) {
        return res
          .status(400)
          .json({ error: `未知的 capability "${capabilityId}"` })
      }
      const version = Number(req.params.version)
      if (!Number.isInteger(version)) {
        return res.status(400).json({ error: 'version 必须是整数' })
      }
      const found = store.getVersion(id, capabilityId, version)
      if (!found) {
        return res.status(404).json({ error: `版本 v${version} 不存在` })
      }
      res.json(found)
    },
  )

  /** 回滚：把旧版本复制成一个新版本（历史只增不减）。 */
  router.post('/api/providers/:id/prompts/:capabilityId/rollback', (req, res) => {
    const { id, capabilityId } = req.params
    if (!isCapabilityId(capabilityId)) {
      return res.status(400).json({ error: `未知的 capability "${capabilityId}"` })
    }
    const version = Number((req.body as { version?: unknown })?.version)
    if (!Number.isInteger(version)) {
      return res.status(400).json({ error: 'version 必须是整数' })
    }
    try {
      const record = store.rollback(id, capabilityId, version)
      res.json({
        prompt: record.template,
        version: record.version,
        savedAt: record.savedAt,
      })
    } catch (err) {
      return fail(res, err)
    }
  })

  /**
   * 试运行：用**被编辑的那个 provider** 真跑一次。
   *
   * 返回渲染后的 system / user、模型原始输出、解析后的结果。
   */
  router.post(
    '/api/providers/:id/prompts/:capabilityId/dry-run',
    async (req, res) => {
      const { id, capabilityId } = req.params
      if (!isCapabilityId(capabilityId)) {
        return res
          .status(400)
          .json({ error: `未知的 capability "${capabilityId}"` })
      }
      const provider = store.getProvider(id)
      if (!provider) {
        return res.status(404).json({ error: `provider "${id}" 不存在` })
      }

      // 模板：优先用请求里带的（未保存的编辑也能试），否则用已保存的
      let template: PromptTemplate
      const body = (req.body ?? {}) as Record<string, unknown>
      if (body.template && typeof body.template === 'object') {
        const parsed = parseTemplate(capabilityId, { template: body.template })
        if ('error' in parsed) {
          return res.status(400).json({ error: parsed.error })
        }
        template = parsed.template
      } else {
        try {
          template = store.getPrompt(id, capabilityId)
        } catch (err) {
          return fail(res, err)
        }
      }

      const input = buildDryRunInput(capabilityId, body.vars)
      if ('error' in input) {
        return res.status(400).json({ error: input.error })
      }

      const tones = store.getTones(id)
      const context = buildContext(capabilityId, input, tones)
      const { system, user } = renderPrompt(template, context)

      try {
        const client = registry.get(provider, store.getApiKey(provider.id))
        const result = await client.chat({
          system,
          user,
          jsonSchema:
            template.responseMode === 'json' ? template.jsonSchema : undefined,
        })
        let parsed: unknown
        let parseError: string | undefined
        try {
          parsed = extractOutput(capabilityId, result.content, template.responseMode)
        } catch (err) {
          parseError = err instanceof Error ? err.message : 'parse failed'
        }
        res.json({
          providerId: id,
          capabilityId,
          system: result.system,
          user,
          structured: result.structured,
          raw: result.content,
          parsed,
          parseError,
        })
      } catch (err) {
        res.status(502).json({
          providerId: id,
          capabilityId,
          system,
          user,
          error: err instanceof Error ? err.message : 'unknown error',
        })
      }
    },
  )

  /**
   * 从别的 provider 复制提示词。
   *
   * 必须注册在 `/prompts/:capabilityId` 前面 —— 否则 "copy" 会被当成 capabilityId。
   */
  router.post('/api/providers/:id/prompts/copy', (req, res) => {
    const toProviderId = req.params.id
    const body = (req.body ?? {}) as Record<string, unknown>
    const fromProviderId = typeof body.fromProviderId === 'string' ? body.fromProviderId : ''
    if (!fromProviderId) {
      return res.status(400).json({ error: 'fromProviderId is required' })
    }
    if (!store.getProvider(fromProviderId)) {
      return res.status(404).json({ error: `provider "${fromProviderId}" 不存在` })
    }
    const capabilityId = body.capabilityId
    if (capabilityId !== undefined && !isCapabilityId(capabilityId)) {
      return res.status(400).json({ error: `未知的 capability "${capabilityId}"` })
    }
    try {
      store.copyPrompts(fromProviderId, toProviderId, capabilityId)
      res.json({ ok: true, copied: capabilityId ?? CAPABILITY_IDS })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 语气表 ----

  router.get('/api/providers/:id/tones', (req, res) => {
    try {
      res.json({ tones: store.getTones(req.params.id) })
    } catch (err) {
      return fail(res, err)
    }
  })

  router.put('/api/providers/:id/tones', (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>
    const raw = Array.isArray(body.tones) ? body.tones : []
    const tones: ToneConfig[] = []
    for (const item of raw) {
      const t = (item ?? {}) as Record<string, unknown>
      const id = typeof t.id === 'string' ? t.id.trim() : ''
      if (!isToneId(id)) {
        return res.status(400).json({ error: `未知的语气 id "${id}"` })
      }
      const guidance = typeof t.guidance === 'string' ? t.guidance : ''
      if (!guidance.trim()) {
        return res.status(400).json({ error: `语气 "${id}" 的 guidance 不能为空` })
      }
      tones.push({
        id,
        label: typeof t.label === 'string' ? t.label.trim() : id,
        guidance,
      })
    }
    // id 集合必须完整
    const missing = TONE_IDS.filter(
      (id) => !tones.some((t) => t.id === id),
    )
    if (missing.length > 0) {
      return res
        .status(400)
        .json({ error: `缺少语气：${missing.join(' | ')}` })
    }
    try {
      res.json({ tones: store.setTones(req.params.id, tones) })
    } catch (err) {
      return fail(res, err)
    }
  })

  return router
}

export { ConfigError, NotFoundError }
