/**
 * 翻译服务：一句中文 → 一句英文，或一句英文 → 一句中文。
 *
 * 走和 AI 改写同一套流程（渲染提示词 → 调模型 → 提取），
 * 只是能力不同、模板不同。两个方向各占一个 capability id，
 * 所以每个 provider 可以给两个方向各调一套措辞。
 */

import {
  buildContext,
  extractOutput,
  renderPrompt,
  type TranslateInput,
} from '../config/capabilities.js'
import type { ToneId } from '../config/defaults.js'
import type { LlmClient } from '../llm/types.js'
import type { CapabilityId, ProviderPromptSet } from '../config/types.js'

export interface TranslationResult {
  translation: string
}

/** 翻译方向：中文→英文 / 英文→中文。 */
export type TranslateDirection = 'zh2en' | 'en2zh'

export interface TranslateOptions {
  /** 源语言，默认 zh */
  source?: 'zh' | 'en'
  /** 目标语言，默认 en */
  target?: 'zh' | 'en'
  /** 可选语气；不传则用中性默认 */
  tone?: ToneId
}

/** 方向 → 用哪个能力（= 用哪个 provider 提示词模板）。 */
export function directionOf(
  source: 'zh' | 'en' = 'zh',
  target: 'zh' | 'en' = 'en',
): TranslateDirection {
  if (source === 'en' && target === 'zh') return 'en2zh'
  if (source === 'zh' && target === 'en') return 'zh2en'
  throw new Error(`不支持的翻译方向：${source} → ${target}`)
}

/** 一次请求用到的 provider 快照。 */
export interface ActiveProvider {
  client: LlmClient
  promptSet: ProviderPromptSet
}

export interface CreateTranslateServiceOptions {
  /** 取当前生效的 provider —— 每次请求都调，切换立即生效 */
  getActive: () => ActiveProvider
}

export interface TranslateService {
  translate(text: string, options?: TranslateOptions): Promise<TranslationResult>
}

/**
 * 构造一个 TranslateService。
 */
export function createTranslateService(
  deps: CreateTranslateServiceOptions,
): TranslateService {
  return {
    async translate(text, options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const direction = directionOf(options?.source, options?.target)
      const capability: CapabilityId =
        direction === 'en2zh' ? 'translateReverse' : 'translate'

      const { client, promptSet } = deps.getActive()
      const template = promptSet.templates[capability]
      if (!template) {
        throw new Error(`没有 "${capability}" 的提示词模板`)
      }

      const input: TranslateInput = { text: text.trim(), tone: options?.tone }
      const context = buildContext(capability, input, promptSet.tones)
      const { system, user } = renderPrompt(template, context)

      const mode = template.responseMode
      const result = await client.chat({
        system,
        user,
        jsonSchema: mode === 'json' ? template.jsonSchema : undefined,
      })

      const translation = extractOutput(capability, result.content, mode)
      return { translation: translation as string }
    },
  }
}
