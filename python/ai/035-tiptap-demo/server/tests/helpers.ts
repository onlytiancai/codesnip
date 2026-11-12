import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { vi } from 'vitest'
import { createApp } from '../src/app.js'
import {
  buildContext,
  renderPrompt,
  type CapabilityInput,
} from '../src/config/capabilities.js'
import { defaultPromptSet } from '../src/config/defaults.js'
import { ConfigStore } from '../src/config/store.js'
import { openDatabase } from '../src/config/db.js'
import type { CapabilityId } from '../src/config/types.js'
import { ClientRegistry, type LlmClientFactory } from '../src/llm/registry.js'
import type { ChatRequest, LlmClient } from '../src/llm/types.js'
import type { AiService } from '../src/services/ai.js'
import { createAiService } from '../src/services/ai.js'
import type { TranslateService } from '../src/services/translate.js'
import { createTranslateService } from '../src/services/translate.js'

/**
 * 测试用的公共脚手架。
 */

/** 默认各能力返回值；传 overrides 覆盖单个方法。 */
export function buildAiService(overrides?: Partial<AiService>): AiService {
  return {
    rewriteSelection: vi.fn(async () => ['great', 'promising', 'excellent']),
    rewriteSentence: vi.fn(async () => 'Education is of great importance to children.'),
    grammarCheck: vi.fn(async () => '这句话没有语法或标点错误。'),
    rewriteInstruction: vi.fn(async () => 'Rewritten text.'),
    ...overrides,
  }
}

export function buildTranslateService(
  overrides?: Partial<TranslateService>,
): TranslateService {
  return {
    translate: vi.fn(async () => ({ translation: 'I went to the park yesterday.' })),
    ...overrides,
  }
}

export function buildApp(
  ai?: AiService,
  translate?: TranslateService,
): ReturnType<typeof createApp> {
  return createApp({
    aiService: ai ?? buildAiService(),
    translateService: translate ?? buildTranslateService(),
  })
}

// ---------------------------------------------------------------------------
// 提示词渲染
// ---------------------------------------------------------------------------

/** 用默认提示词集渲染某个能力，返回实际会发给模型的 system / user。 */
export function renderCapability(
  capability: CapabilityId,
  input: CapabilityInput,
): { system: string; user: string } {
  const set = defaultPromptSet()
  const template = set.templates[capability]
  const context = buildContext(capability, input, set.tones)
  return renderPrompt(template, context)
}

// ---------------------------------------------------------------------------
// LlmClient 桩
// ---------------------------------------------------------------------------

export interface StubClient {
  client: LlmClient
  chat: ReturnType<typeof vi.fn>
}

/** 一个只会回固定字符串的 LlmClient 桩，并记录它收到的请求。 */
export function stubLlmClient(reply: string): StubClient {
  const chat = vi.fn(async (req: ChatRequest) => ({
    content: reply,
    structured: false,
    system: req.system,
  }))
  const client: LlmClient = {
    kind: 'ollama',
    chat,
    listModels: vi.fn(async () => ['stub-model']),
  }
  return { client, chat }
}

/** 用桩 client 组装出两个 service。 */
export function buildServices(client: LlmClient): {
  ai: AiService
  translate: TranslateService
} {
  const promptSet = defaultPromptSet()
  const getActive = () => ({ client, promptSet })
  return {
    ai: createAiService({ getActive }),
    translate: createTranslateService({ getActive }),
  }
}

// ---------------------------------------------------------------------------
// 设置路由用的临时 store
// ---------------------------------------------------------------------------

export interface TempStore {
  store: ConfigStore
  dir: string
  cleanup: () => void
}

/**
 * 建一个落在临时目录里的 ConfigStore（不碰仓库里的 server/data）。
 *
 * 配置现在存在 SQLite 里 —— 所以这里建一个临时的库文件。
 * 注意：不传 legacyDataDir，所以**不会**导入仓库里的 config.json。
 */
export function buildTempStore(): TempStore {
  const dir = mkdtempSync(join(tmpdir(), 'tiptap-config-'))
  const store = new ConfigStore({
    db: openDatabase(join(dir, 'test.db')),
    legacyDataDir: dir,
  })
  return {
    store,
    dir,
    cleanup: () => {
      rmSync(dir, { recursive: true, force: true })
    },
  }
}

/** 用同一个库文件再开一个 ConfigStore（模拟重启服务）。 */
export function reopenStore(tmp: TempStore): ConfigStore {
  return new ConfigStore({
    db: openDatabase(join(tmp.dir, 'test.db')),
    legacyDataDir: tmp.dir,
  })
}

/** 一个不真的发请求的 client 工厂（设置路由测试用）。 */
export function stubClientFactory(reply = 'OK'): LlmClientFactory {
  return () => ({
    kind: 'ollama',
    chat: async (req: ChatRequest) => ({
      content: reply,
      structured: false,
      system: req.system,
    }),
    listModels: async () => ['stub-model'],
  })
}

export function buildRegistry(reply = 'OK'): ClientRegistry {
  return new ClientRegistry(stubClientFactory(reply))
}
