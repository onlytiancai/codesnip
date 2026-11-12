/**
 * 能力注册表：把「一个能力」需要的一切集中在一处。
 *
 * - 变量表：模板里能用哪些变量（UI 提示 + 一键插入）
 * - 示例输入：试运行时的默认值
 * - buildContext：把调用参数转成模板变量（含 section 的开关变量）
 * - extract：从模型原始输出里抽出最终值
 *
 * **extract 保证了对外 HTTP 响应形状不变**：
 *   翻译 → {translation}，选区 → {candidates}，其余 → {text}
 * 无论提示词是 text 模式还是 json 模式，前端拿到的都一样。
 */

import { cleanText } from '../prompts/cleanText.js'
import { cleanCandidates, parseCandidates } from '../prompts/rewriteSelection.js'
import { render } from './template.js'
import {
  CAPABILITY_JSON_KEY,
  CAPABILITY_JSON_SCHEMAS,
  toneInstruction,
  type ToneId,
} from './defaults.js'
import {
  CAPABILITY_IDS,
  type CapabilityId,
  type JsonSchema,
  type PromptTemplate,
  type ToneConfig,
} from './types.js'
import type { SelectionKind } from '../prompts/rewriteSelection.js'
import type { SentenceOperation } from '../prompts/rewriteSentence.js'

/** 旧 prompts/rewriteSentence.ts 里的操作说明，迁移成模板变量。 */
export const OPERATION_GUIDANCE: Record<SentenceOperation, string> = {
  rewrite: 'Express the same meaning using different wording and sentence structure.',
  improve:
    'Fix grammar, awkward collocations, word choice, and idiomatic phrasing. Keep the meaning and roughly the same length.',
  shorten:
    'Make the sentence noticeably shorter. Cut filler, redundancy, and qualifiers — but keep the core claim intact.',
  expand:
    'Add the specifics a reader would need: supporting detail, a concrete example, or a stated consequence. Do not pad with empty words.',
}

// ---------------------------------------------------------------------------
// 各能力的输入
// ---------------------------------------------------------------------------

export interface TranslateInput {
  text: string
  tone?: ToneId
}

export interface SelectionInput {
  text: string
  kind: SelectionKind
  context?: string
  sourceSentence?: string
}

export interface SentenceInput {
  sentence: string
  operation: SentenceOperation
  tone?: ToneId
  context?: string
  sourceSentence?: string
}

export interface GrammarInput {
  sentence: string
}

export interface InstructionInput {
  text: string
  instruction: string
  context?: string
}

export type CapabilityInput =
  | TranslateInput
  | SelectionInput
  | SentenceInput
  | GrammarInput
  | InstructionInput

// ---------------------------------------------------------------------------
// 模板变量
// ---------------------------------------------------------------------------

/** 各能力模板里可用的变量名 —— UI 用来显示提示 chip。 */
export const CAPABILITY_VARIABLES: Record<CapabilityId, string[]> = {
  translate: ['text', 'tone'],
  translateReverse: ['text', 'tone'],
  rewriteSelection: [
    'text',
    'context',
    'sourceSentence',
    'hasSource',
    'kindWord',
    'kindPhrase',
  ],
  rewriteSentence: [
    'sentence',
    'operation',
    'operationGuidance',
    'tone',
    'context',
    'sourceSentence',
    'hasSource',
  ],
  grammarCheck: ['sentence'],
  rewriteInstruction: ['text', 'instruction', 'context'],
}

/** 试运行时的示例输入。 */
export const CAPABILITY_SAMPLES: Record<CapabilityId, Record<string, string>> = {
  translate: { text: '今天天气很好，我们去公园散步吧。' },
  translateReverse: { text: 'Education is very important for children.' },
  rewriteSelection: {
    text: 'very important',
    kind: 'phrase',
    context: 'Education is very important for children.',
    sourceSentence: '教育对儿童来说非常重要。',
  },
  rewriteSentence: {
    sentence: 'Education is very important for children.',
    operation: 'rewrite',
    tone: 'formal',
    context: 'Education shapes a child’s future.',
    sourceSentence: '教育对儿童来说非常重要。',
  },
  grammarCheck: {
    sentence: 'This are a example sentence with a mistake.',
  },
  rewriteInstruction: {
    text: 'Education is very important for children.',
    instruction: 'Make it more academic.',
    context: 'Education shapes a child’s future.',
  },
}

// ---------------------------------------------------------------------------
// buildContext：调用参数 → 模板变量
// ---------------------------------------------------------------------------

/** 旧 builder 里 context / sourceSentence 都是先 trim 再用的 —— 这里保持一致。 */
function trimOrUndefined(v: string | undefined): string | undefined {
  if (typeof v !== 'string') return undefined
  const t = v.trim()
  return t.length > 0 ? t : undefined
}

/**
 * 把调用参数转成模板变量。
 *
 * `tone` 变量装的是**整行** tone 指令（`Write in a formal tone: ...`），
 * 没有语气时是 undefined —— 于是 `{{#tone}}` 整块被跳过。
 */
export function buildContext(
  capability: CapabilityId,
  input: CapabilityInput,
  tones: ToneConfig[],
): Record<string, unknown> {
  switch (capability) {
    case 'translate': {
      const i = input as TranslateInput
      return { text: i.text, tone: toneInstruction(tones, i.tone) }
    }
    case 'translateReverse': {
      const i = input as TranslateInput
      return { text: i.text, tone: toneInstruction(tones, i.tone) }
    }
    case 'rewriteSelection': {
      const i = input as SelectionInput
      const sourceSentence = trimOrUndefined(i.sourceSentence)
      return {
        text: i.text,
        context: trimOrUndefined(i.context),
        sourceSentence,
        hasSource: sourceSentence !== undefined,
        kindWord: i.kind === 'word',
        kindPhrase: i.kind === 'phrase',
      }
    }
    case 'rewriteSentence': {
      const i = input as SentenceInput
      const sourceSentence = trimOrUndefined(i.sourceSentence)
      return {
        sentence: i.sentence,
        operation: i.operation,
        operationGuidance: OPERATION_GUIDANCE[i.operation],
        // 和旧实现一致：只有 rewrite 支持语气
        tone:
          i.operation === 'rewrite' ? toneInstruction(tones, i.tone) : undefined,
        context: trimOrUndefined(i.context),
        sourceSentence,
        hasSource: sourceSentence !== undefined,
      }
    }
    case 'grammarCheck': {
      const i = input as GrammarInput
      // 旧 builder：buildGrammarCheckPrompt 返回 sentence.trim()
      return { sentence: i.sentence.trim() }
    }
    case 'rewriteInstruction': {
      const i = input as InstructionInput
      return {
        text: i.text,
        // 旧 builder：Instruction: ${instruction.trim()}
        instruction: i.instruction.trim(),
        context: trimOrUndefined(i.context),
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 渲染
// ---------------------------------------------------------------------------

export interface RenderedPrompt {
  system: string
  user: string
}

/**
 * 渲染模板。
 *
 * 尾部空白一律去掉：迁移前的 builder 输出都不带尾随换行，
 * 去掉之后两边才能逐字节相同（尾随空白对模型也没有意义）。
 */
export function renderPrompt(
  template: PromptTemplate,
  context: Record<string, unknown>,
): RenderedPrompt {
  return {
    system: render(template.system, context).trimEnd(),
    user: render(template.user, context).trimEnd(),
  }
}

// ---------------------------------------------------------------------------
// 提取：从模型输出里抽出最终值
// ---------------------------------------------------------------------------

/**
 * 宽松地解 JSON：直接解 → 剥 ``` 围栏 → 找第一个 {...}。
 */
export function parseJsonLoose(raw: string): unknown {
  const text = raw.trim()
  if (!text) throw new Error('empty response')

  const candidates: string[] = [text]
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenced) candidates.push(fenced[1].trim())
  const brace = text.match(/\{[\s\S]*\}/)
  if (brace) candidates.push(brace[0])

  let shapeError: Error | undefined
  for (const c of candidates) {
    let parsed: unknown
    try {
      parsed = JSON.parse(c)
    } catch {
      continue
    }
    if (parsed !== null && typeof parsed === 'object') return parsed
    shapeError ??= new Error('response is not a JSON object')
  }
  throw shapeError ?? new Error('cannot parse JSON object from model response')
}

/** json 模式下从对象里取出期望的那个字段。 */
function pickJsonField(
  raw: string,
  capability: CapabilityId,
  expect: 'string' | 'array',
): string | string[] {
  const parsed = parseJsonLoose(raw) as Record<string, unknown>
  const key = CAPABILITY_JSON_KEY[capability]
  const value = parsed[key]
  if (value === undefined || value === null) {
    throw new Error(`response is missing the "${key}" field`)
  }
  if (expect === 'string') {
    if (typeof value !== 'string') {
      throw new Error(`response field "${key}" must be a string`)
    }
    return value
  }
  if (!Array.isArray(value)) {
    throw new Error(`response field "${key}" must be an array`)
  }
  return value.filter((v): v is string => typeof v === 'string')
}

/**
 * 从模型输出抽出最终值。
 *
 * - text 模式：按各能力的老语义处理（语法检查原样透传，其余走 cleanText）
 * - json 模式：从 JSON 对象里取约定的字段
 */
export function extractOutput(
  capability: CapabilityId,
  raw: string,
  mode: 'text' | 'json',
  options?: { maxCandidates?: number },
): string | string[] {
  if (mode === 'json') {
    switch (capability) {
      case 'translate':
      case 'translateReverse':
        return pickJsonField(raw, capability, 'string') as string
      case 'rewriteSelection': {
        const list = pickJsonField(raw, capability, 'array') as string[]
        const max = options?.maxCandidates ?? list.length
        return cleanCandidates(list, max)
      }
      case 'rewriteSentence':
      case 'grammarCheck':
      case 'rewriteInstruction':
        return pickJsonField(raw, capability, 'string') as string
    }
  }

  switch (capability) {
    case 'translate':
    case 'translateReverse':
      return cleanText(raw)
    case 'rewriteSelection':
      return parseCandidates(raw, options?.maxCandidates ?? 5)
    case 'rewriteSentence':
      return cleanText(raw)
    case 'grammarCheck':
      // 老行为：模型写什么就原样透传，不做任何清洗
      return raw
    case 'rewriteInstruction':
      return cleanText(raw)
  }
}

/** 能力 id → 默认 JSON schema（新建模板时用）。 */
export function defaultJsonSchema(capability: CapabilityId): JsonSchema {
  return structuredClone(CAPABILITY_JSON_SCHEMAS[capability])
}

export { CAPABILITY_IDS }
export type { CapabilityId }
