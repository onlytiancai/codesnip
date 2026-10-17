import ollama, { type Ollama } from 'ollama'
import {
  buildRewriteSelectionPrompt,
  buildRewriteSelectionSystem,
  parseCandidates,
  type SelectionKind,
} from '../prompts/rewriteSelection.js'
import {
  buildRewriteSentencePrompt,
  buildRewriteSentenceSystem,
  parseRewrittenSentence,
  type SentenceOperation,
} from '../prompts/rewriteSentence.js'
import {
  buildGrammarCheckPrompt,
  GRAMMAR_SYSTEM,
} from '../prompts/grammarCheck.js'
import {
  buildInstructionPrompt,
  INSTRUCTION_SYSTEM,
  parseInstructionResult,
} from '../prompts/rewriteInstruction.js'
import { type ToneId } from '../prompts/tones.js'

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

export interface CreateOllamaServiceOptions {
  /** 测试时可注入 mock；生产留空走 default import */
  client?: Ollama
  /** 必填，Ollama 模型名 */
  model: string
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
 * 构造一个 AiService。生产代码只传 model；测试里传 client 做桩。
 */
export function createOllamaService(deps: CreateOllamaServiceOptions): AiService {
  const client = deps.client ?? ollama
  const defaultMax = deps.defaultMaxCandidates ?? 5

  /** 单轮 chat，返回模型文本。四个能力都走这里，统一错误来源。 */
  async function chat(system: string, user: string): Promise<string> {
    const response = await client.chat({
      model: deps.model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    })
    return response.message?.content ?? ''
  }

  return {
    async rewriteSelection(text, kind, options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const content = await chat(
        buildRewriteSelectionSystem(kind),
        buildRewriteSelectionPrompt(
          text.trim(),
          options?.context,
          options?.sourceSentence,
        ),
      )
      return parseCandidates(content, options?.maxCandidates ?? defaultMax)
    },

    async rewriteSentence(sentence, options) {
      if (!sentence || !sentence.trim()) {
        throw new Error('sentence must be non-empty')
      }
      const content = await chat(
        buildRewriteSentenceSystem(options.operation, options.tone),
        buildRewriteSentencePrompt(
          sentence.trim(),
          options.context,
          options.sourceSentence,
        ),
      )
      return parseRewrittenSentence(content)
    },

    async grammarCheck(sentence) {
      if (!sentence || !sentence.trim()) {
        throw new Error('sentence must be non-empty')
      }
      // 原样返回，不解析——界面直接显示模型写的文字
      return chat(GRAMMAR_SYSTEM, buildGrammarCheckPrompt(sentence.trim()))
    },

    async rewriteInstruction(text, options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const content = await chat(
        INSTRUCTION_SYSTEM,
        buildInstructionPrompt(text.trim(), options.instruction, options.context),
      )
      return parseInstructionResult(content)
    },
  }
}
