/**
 * AI 改写服务：四个能力（选区候选 / 整句改写 / 语法检查 / 自定义指令）。
 *
 * 和迁移前相比，变化只有一处：**提示词和 provider 变成运行时可配的**。
 * 提示词从当前生效的 provider 的提示词集里取，provider 也是。
 * 对外的接口（AiService）完全没变，所以路由层和路由测试都不用动。
 */

import {
  buildContext,
  extractOutput,
  renderPrompt,
  type GrammarInput,
  type InstructionInput,
  type SelectionInput,
  type SentenceInput,
} from '../config/capabilities.js'
import type {
  CapabilityId,
  ProviderPromptSet,
} from '../config/types.js'
import type { LlmClient } from '../llm/types.js'
import type { SelectionKind } from '../prompts/rewriteSelection.js'
import type { SentenceOperation } from '../prompts/rewriteSentence.js'
import type { ToneId } from '../config/defaults.js'

export interface RewriteSelectionOptions {
  /** 选区所在段落/句子上下文，用于消歧 */
  context?: string
  /** 选区对应句子的中文原文，提升消歧精度 */
  sourceSentence?: string
  /** 候选数量上限，默认走 deps.defaultMaxCandidates */
  maxCandidates?: number
}

export interface RewriteSentenceOptions {
  operation: SentenceOperation
  tone?: ToneId
  context?: string
  sourceSentence?: string
}

export interface InstructionOptions {
  instruction: string
  context?: string
}

/** 一次请求用到的 provider 快照 —— 保证 client 和提示词来自同一个 provider。 */
export interface ActiveProvider {
  client: LlmClient
  promptSet: ProviderPromptSet
}

export interface CreateAiServiceOptions {
  /**
   * 取当前生效的 provider。
   *
   * 每次请求都调一次 —— 这样在 UI 里切换 provider 立刻生效，不用重启服务。
   */
  getActive: () => ActiveProvider
  /** 默认候选上限 */
  defaultMaxCandidates?: number
}

export interface AiService {
  rewriteSelection(
    text: string,
    kind: SelectionKind,
    options?: RewriteSelectionOptions,
  ): Promise<string[]>
  rewriteSentence(sentence: string, options: RewriteSentenceOptions): Promise<string>
  /** 返回模型原文（错误清单或 NONE），不做解析 */
  grammarCheck(sentence: string): Promise<string>
  rewriteInstruction(text: string, options: InstructionOptions): Promise<string>
}

/**
 * 构造一个 AiService。
 *
 * 提示词渲染 + 调用模型 + 解析输出，三步都在这里；
 * 四个能力共用同一套流程，只是模板和提取方式不同。
 */
export function createAiService(deps: CreateAiServiceOptions): AiService {
  const defaultMax = deps.defaultMaxCandidates ?? 5

  /** 跑一个能力：渲染提示词 → 调模型 → 按模板的输出模式提取结果。 */
  async function run(
    capability: CapabilityId,
    input: unknown,
    options?: { maxCandidates?: number },
  ): Promise<string | string[]> {
    const { client, promptSet } = deps.getActive()
    const template = promptSet.templates[capability]
    if (!template) {
      throw new Error(`没有 "${capability}" 的提示词模板`)
    }

    const context = buildContext(
      capability,
      input as Parameters<typeof buildContext>[1],
      promptSet.tones,
    )
    const { system, user } = renderPrompt(template, context)

    const mode = template.responseMode
    const result = await client.chat({
      system,
      user,
      jsonSchema: mode === 'json' ? template.jsonSchema : undefined,
    })

    return extractOutput(capability, result.content, mode, {
      maxCandidates: options?.maxCandidates ?? defaultMax,
    })
  }

  return {
    async rewriteSelection(text, kind, options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const input: SelectionInput = {
        text: text.trim(),
        kind,
        context: options?.context,
        sourceSentence: options?.sourceSentence,
      }
      const out = await run('rewriteSelection', input, {
        maxCandidates: options?.maxCandidates ?? defaultMax,
      })
      return out as string[]
    },

    async rewriteSentence(sentence, options) {
      if (!sentence || !sentence.trim()) {
        throw new Error('sentence must be non-empty')
      }
      const input: SentenceInput = {
        sentence: sentence.trim(),
        operation: options.operation,
        tone: options.tone,
        context: options.context,
        sourceSentence: options.sourceSentence,
      }
      return (await run('rewriteSentence', input)) as string
    },

    async grammarCheck(sentence) {
      if (!sentence || !sentence.trim()) {
        throw new Error('sentence must be non-empty')
      }
      const input: GrammarInput = { sentence: sentence.trim() }
      return (await run('grammarCheck', input)) as string
    },

    async rewriteInstruction(text, options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const input: InstructionInput = {
        text: text.trim(),
        instruction: options.instruction,
        context: options.context,
      }
      return (await run('rewriteInstruction', input)) as string
    },
  }
}
