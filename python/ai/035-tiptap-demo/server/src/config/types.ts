/**
 * 配置相关的类型定义。
 *
 * 全部配置都落在 server/data/config.json 里，结构由这里定义。
 * 提示词按 provider 分组（ProviderPromptSet）——不同模型需要不同措辞，
 * 所以每个 provider 拥有一整套自己的提示词 + 语气表，互不影响。
 */

/** JSON Schema 的最小描述：够 OpenAI / Anthropic / Ollama 用了。 */
export type JsonSchema = Record<string, unknown>

export type ProviderKind = 'ollama' | 'openai' | 'anthropic'

/**
 * 结构化输出的强制程度：
 * - schema  ：API 层强制 JSON Schema（OpenAI json_schema / Anthropic tool-use / Ollama format）
 * - object  ：只强制"是个 JSON 对象"（OpenAI json_object），字段靠提示词
 * - prompt  ：完全靠提示词要求，API 层不加约束
 * - off     ：不要求 JSON
 */
export type StructuredOutputMode = 'schema' | 'object' | 'prompt' | 'off'

export interface ProviderConfig {
  id: string
  label: string
  kind: ProviderKind
  /** Ollama 用 host、OpenAI/Anthropic 用 baseURL */
  baseUrl: string
  model: string
  /** 走环境变量取 key 时填变量名；优先级低于 secrets.json */
  apiKeyEnv?: string
  structuredOutput: StructuredOutputMode
  maxTokens?: number
  /** 仅 OpenAI 用：max_tokens 兼容面广，max_completion_tokens 是 OpenAI 官方新写法 */
  maxTokensParam?: 'max_tokens' | 'max_completion_tokens'
  /** 仅显式设置才发送——Opus 5 / Sonnet 5 收到 temperature 会直接 400 */
  temperature?: number
  timeoutMs?: number
}

/** 一个能力（翻译 / 选区 / 整句 / 语法 / 指令）的提示词模板 */
export interface PromptTemplate {
  id: string
  label: string
  /** system message 模板 */
  system: string
  /** user message 模板 */
  user: string
  /** text = 模型输出纯文本；json = 模型输出符合 jsonSchema 的 JSON */
  responseMode: 'text' | 'json'
  /** responseMode==='json' 时必填 */
  jsonSchema?: JsonSchema
}

/** 语气：id 固定（代码里 TONE_IDS 是唯一来源），label / guidance 可按 provider 改 */
export interface ToneConfig {
  id: string
  /** 中文 label，界面下拉用 */
  label: string
  /** 英文 guidance，喂给模型 */
  guidance: string
}

/** 六个能力 id —— 提示词模板的 key */
export const CAPABILITY_IDS = [
  'translate',
  'translateReverse',
  'rewriteSelection',
  'rewriteSentence',
  'grammarCheck',
  'rewriteInstruction',
] as const

export type CapabilityId = (typeof CAPABILITY_IDS)[number]

/** 一个 provider 拥有的整套提示词材料 */
export interface ProviderPromptSet {
  templates: Record<CapabilityId, PromptTemplate>
  tones: ToneConfig[]
}

export interface AppConfig {
  version: 1
  /** 当前生效的 provider —— 所有 AI 请求都走它 */
  activeProviderId: string
  providers: ProviderConfig[]
  /** key = providerId */
  promptSets: Record<string, ProviderPromptSet>
}

/** 历史版本记录（append-only，回滚也只是追加一条） */
export interface PromptVersion {
  providerId: string
  capabilityId: CapabilityId
  version: number
  savedAt: string
  note?: string
  template: PromptTemplate
}

/** history.json 的结构：key = "<providerId>/<capabilityId>" */
export type PromptHistory = Record<string, PromptVersion[]>
