/**
 * 内置默认提示词集 —— 从迁移前的 server/src/prompts/*.ts **逐字**搬过来。
 *
 * 为什么要逐字：迁移的整个卖点是「首次运行行为和现在完全一样」。
 * server/tests/promptParity.test.ts 会拿这里的模板渲染结果和旧 builder
 * 的输出逐字节对比，改一个字符测试就红。
 *
 * 新建 provider 时会 clone 一份这份默认集（见 config/store.ts），
 * 之后各 provider 的提示词互不影响。
 */

import {
  CAPABILITY_IDS,
  type CapabilityId,
  type JsonSchema,
  type ProviderConfig,
  type ProviderPromptSet,
  type PromptTemplate,
  type ToneConfig,
} from './types.js'

/** 语气 id —— 固定集合，前端 src/constants/tones.ts 必须和这里对齐。 */
export const TONE_IDS = [
  'neutral',
  'formal',
  'casual',
  'professional',
  'friendly',
  'concise',
  'detailed',
  'persuasive',
] as const

export type ToneId = (typeof TONE_IDS)[number]

export function isToneId(v: unknown): v is ToneId {
  return typeof v === 'string' && (TONE_IDS as readonly string[]).includes(v)
}

/** 默认语气表（中文 label + 英文 guidance）。各 provider 可各自改。 */
export const DEFAULT_TONES: ToneConfig[] = [
  {
    id: 'neutral',
    label: '中性',
    guidance:
      'keep it neutral and matter-of-fact — no strong colouring, no obvious personality',
  },
  {
    id: 'formal',
    label: '正式',
    guidance:
      'write formally — precise wording, no contractions, no slang, avoid first-person asides',
  },
  {
    id: 'casual',
    label: '口语',
    guidance:
      'write casually and conversationally — natural spoken rhythm, contractions are fine',
  },
  {
    id: 'professional',
    label: '专业',
    guidance:
      'write professionally — like a domain expert addressing peers: technical where relevant, concrete and measured',
  },
  {
    id: 'friendly',
    label: '友好',
    guidance:
      'write in a friendly, warm voice — approachable and encouraging, but still clear',
  },
  {
    id: 'concise',
    label: '简洁',
    guidance:
      'write concisely — cut filler and redundant modifiers, keep every word load-bearing',
  },
  {
    id: 'detailed',
    label: '详细',
    guidance:
      'expand with detail — spell out the reasoning and add the specifics a reader would need, without padding',
  },
  {
    id: 'persuasive',
    label: '有说服力',
    guidance:
      'write persuasively — lead with the strongest point, use concrete language, build toward a convincing close',
  },
]

/**
 * 拼出喂给模型的一句 tone 指令。
 * 与迁移前 prompts/tones.ts 的 toneInstruction() 完全一致。
 */
export function toneInstruction(tones: ToneConfig[], tone: unknown): string | undefined {
  if (!isToneId(tone)) return undefined
  const found = tones.find((t) => t.id === tone)
  if (!found) return undefined
  return `Write in a ${found.id} tone: ${found.guidance}.`
}

// ---------------------------------------------------------------------------
// 提示词模板
// ---------------------------------------------------------------------------

/**
 * 翻译。
 *
 * 旧实现：system = TRANSLATE_SYSTEM_PROMPT (+ '\n' + tone 指令)，
 * user   = `Chinese: ${text}\nEnglish:`
 */
const TRANSLATE_SYSTEM = [
  'You are a Chinese-to-English translator. You translate a single Chinese sentence into one natural English sentence. Output ONLY the English sentence — no quotes, no labels, no explanation.',
  '{{#tone}}',
  '{{tone}}',
  '{{/tone}}',
].join('\n')

const TRANSLATE_USER = 'Chinese: {{text}}\nEnglish:'

/**
 * 选区级改写：给选中的单词或短语找替换候选。
 *
 * 旧实现的 system 是「基础句 + 空行 + Rules: + 两条固定规则 + 该 kind 的规则 + 大写规则」，
 * 这里用 {{#kindWord}} / {{#kindPhrase}} 两个 section 复现。
 */
const REWRITE_SELECTION_SYSTEM = [
  'You are an English writing assistant. Given a selection from an English sentence, you produce replacement candidates that fit that exact grammatical slot.',
  '',
  'Rules:',
  '- Output ONLY a JSON array of strings, nothing else.',
  '- Do NOT include explanations, notes, or markdown fences.',
  '{{#kindWord}}',
  '- Each candidate must keep the SAME part of speech as the input word.',
  '- Avoid mere morphological variants (e.g. for "big", do not return "bigger" or "biggest").',
  '{{/kindWord}}',
  '{{#kindPhrase}}',
  '- Candidates MAY be multi-word phrases, but each one must fit the same grammatical slot as the original selection.',
  '- Get the collocation right: the candidate must sound natural in THIS sentence, not just in isolation.',
  '- Keep the surrounding text grammatical when the candidate is spliced in — check articles, prepositions and verb forms around it.',
  '- Prefer tighter alternatives over long multi-clause rewrites.',
  '{{/kindPhrase}}',
  '- Match the capitalization of the input: if input is "Very important", return "Essential", "Of great importance".',
].join('\n')

const REWRITE_SELECTION_USER = [
  '{{#hasSource}}',
  'The English sentence comes from translating the Chinese sentence below. Use the Chinese to disambiguate the selection\'s meaning before choosing candidates.',
  '{{/hasSource}}',
  'Selection: {{text}}',
  '{{#context}}',
  'Context: {{context}}',
  '{{/context}}',
  '{{#hasSource}}',
  'Chinese source: {{sourceSentence}}',
  '{{/hasSource}}',
  'Output:',
].join('\n')

/**
 * 整句改写。
 *
 * 旧实现里三句 system 是用 [].join(' ') 拼的 —— 在同一行上，
 * 这里必须照抄，否则渲染结果和旧的不一致。
 */
const REWRITE_SENTENCE_SYSTEM = [
  'You are an English writing editor. You rewrite a single English sentence according to the requested operation.',
  'Output ONLY the rewritten sentence — no quotes, no labels, no explanation, no before/after pairs.',
  'The rewrite must be grammatical, idiomatic English and must stay on topic with the surrounding context.',
].join(' ')

const REWRITE_SENTENCE_SYSTEM_TEMPLATE = [
  REWRITE_SENTENCE_SYSTEM,
  'Operation: {{operation}} — {{operationGuidance}}',
  '{{#tone}}',
  '{{tone}}',
  '{{/tone}}',
].join('\n')

const REWRITE_SENTENCE_USER = [
  '{{#hasSource}}',
  'The English sentence was translated from the Chinese sentence below. Keep the meaning aligned with the Chinese.',
  '{{/hasSource}}',
  'Sentence: {{sentence}}',
  '{{#context}}',
  'Surrounding paragraph: {{context}}',
  '{{/context}}',
  '{{#hasSource}}',
  'Chinese source: {{sourceSentence}}',
  '{{/hasSource}}',
  'Output:',
].join('\n')

/**
 * 语法 / 标点检查。
 *
 * 旧实现：system = GRAMMAR_SYSTEM（多行），user 就是原句本身，
 * 模型写什么就原样透传给前端 —— 不解析、不结构化。
 */
const GRAMMAR_SYSTEM = [
  'You are an English proofreader helping a Chinese-speaking student.',
  '',
  'Check the sentence for grammar and punctuation errors only. Do not comment on style, tone, vocabulary, or word choice.',
  '',
  'Reply in Chinese. If the sentence is correct, just say: 这句话没有语法或标点错误。',
  'Otherwise, write one or two short sentences saying what is wrong and what to write instead. Point at the exact wrong words.',
].join('\n')

const GRAMMAR_USER = '{{sentence}}'

/** 自定义指令改写 —— 旧实现的 system 是 [].join(' ')，同样照抄。 */
const INSTRUCTION_SYSTEM = [
  'You are an English writing editor. You rewrite the given text according to the user\'s instruction.',
  'Output ONLY the rewritten text — no quotes, no labels, no explanation, no restating the instruction.',
  'If the instruction asks you to preserve the original, keep it unchanged. Otherwise return the improved version.',
].join(' ')

const INSTRUCTION_USER = [
  'Instruction: {{instruction}}',
  'Text to rewrite: {{text}}',
  '{{#context}}',
  'Surrounding context (for coherence; do not include it in the output): {{context}}',
  '{{/context}}',
  'Output:',
].join('\n')

// ---------------------------------------------------------------------------
// JSON 模式下的输出契约
// ---------------------------------------------------------------------------

/**
 * 各能力在 json 模式下的期望形状。
 *
 * 形状是固定的（前端契约不能变），但 schema 文本可以在 UI 里微调。
 * 提取器见 config/capabilities.ts。
 */
export const CAPABILITY_JSON_SCHEMAS: Record<CapabilityId, JsonSchema> = {
  translate: {
    type: 'object',
    properties: {
      translation: { type: 'string', description: 'The English translation.' },
    },
    required: ['translation'],
  },
  rewriteSelection: {
    type: 'object',
    properties: {
      candidates: {
        type: 'array',
        items: { type: 'string' },
        description: 'Replacement candidates for the selection.',
      },
    },
    required: ['candidates'],
  },
  rewriteSentence: {
    type: 'object',
    properties: {
      text: { type: 'string', description: 'The rewritten sentence.' },
    },
    required: ['text'],
  },
  grammarCheck: {
    type: 'object',
    properties: {
      text: {
        type: 'string',
        description: 'The proofreading feedback, written in Chinese.',
      },
    },
    required: ['text'],
  },
  rewriteInstruction: {
    type: 'object',
    properties: {
      text: { type: 'string', description: 'The rewritten text.' },
    },
    required: ['text'],
  },
}

/** 每个能力期望的 JSON 字段名 —— 提取器和 schema 校验都用它。 */
export const CAPABILITY_JSON_KEY: Record<CapabilityId, string> = {
  translate: 'translation',
  rewriteSelection: 'candidates',
  rewriteSentence: 'text',
  grammarCheck: 'text',
  rewriteInstruction: 'text',
}

// ---------------------------------------------------------------------------
// 默认模板表
// ---------------------------------------------------------------------------

const TEMPLATE_LABELS: Record<CapabilityId, string> = {
  translate: '翻译',
  rewriteSelection: '选区候选',
  rewriteSentence: '整句改写',
  grammarCheck: '语法检查',
  rewriteInstruction: '自定义指令',
}

function makeTemplates(): Record<CapabilityId, PromptTemplate> {
  return {
    translate: {
      id: 'translate',
      label: TEMPLATE_LABELS.translate,
      system: TRANSLATE_SYSTEM,
      user: TRANSLATE_USER,
      responseMode: 'text',
      jsonSchema: CAPABILITY_JSON_SCHEMAS.translate,
    },
    rewriteSelection: {
      id: 'rewriteSelection',
      label: TEMPLATE_LABELS.rewriteSelection,
      system: REWRITE_SELECTION_SYSTEM,
      user: REWRITE_SELECTION_USER,
      responseMode: 'text',
      jsonSchema: CAPABILITY_JSON_SCHEMAS.rewriteSelection,
    },
    rewriteSentence: {
      id: 'rewriteSentence',
      label: TEMPLATE_LABELS.rewriteSentence,
      system: REWRITE_SENTENCE_SYSTEM_TEMPLATE,
      user: REWRITE_SENTENCE_USER,
      responseMode: 'text',
      jsonSchema: CAPABILITY_JSON_SCHEMAS.rewriteSentence,
    },
    grammarCheck: {
      id: 'grammarCheck',
      label: TEMPLATE_LABELS.grammarCheck,
      system: GRAMMAR_SYSTEM,
      user: GRAMMAR_USER,
      responseMode: 'text',
      jsonSchema: CAPABILITY_JSON_SCHEMAS.grammarCheck,
    },
    rewriteInstruction: {
      id: 'rewriteInstruction',
      label: TEMPLATE_LABELS.rewriteInstruction,
      system: INSTRUCTION_SYSTEM,
      user: INSTRUCTION_USER,
      responseMode: 'text',
      jsonSchema: CAPABILITY_JSON_SCHEMAS.rewriteInstruction,
    },
  }
}

/**
 * 内置默认提示词集。
 *
 * ⚠️ 直接返回的对象会被 clone 后使用，别在运行时改它。
 */
export function defaultPromptSet(): ProviderPromptSet {
  return {
    templates: makeTemplates(),
    tones: DEFAULT_TONES.map((t) => ({ ...t })),
  }
}

/** 深拷贝（结构简单，手写比引依赖划算）。 */
export function clonePromptSet(set: ProviderPromptSet): ProviderPromptSet {
  const templates = {} as Record<CapabilityId, PromptTemplate>
  for (const id of CAPABILITY_IDS) {
    const t = set.templates[id]
    templates[id] = {
      ...t,
      system: t.system,
      user: t.user,
      jsonSchema: t.jsonSchema ? structuredClone(t.jsonSchema) : undefined,
    }
  }
  return {
    templates,
    tones: set.tones.map((t) => ({ ...t })),
  }
}

/** 默认 provider —— 本地 Ollama，行为和迁移前一致。 */
export function defaultProvider(): ProviderConfig {
  return {
    id: 'ollama-local',
    label: '本地 Ollama',
    kind: 'ollama',
    baseUrl: process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434',
    model: process.env.OLLAMA_MODEL ?? 's2021008840/hy-mt2:1.8b',
    structuredOutput: 'schema',
    maxTokens: 2048,
    timeoutMs: 120_000,
  }
}
