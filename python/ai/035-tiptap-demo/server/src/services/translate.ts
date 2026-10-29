/**
 * 翻译服务：一句中文 → 一句英文。
 *
 * 走和 AI 改写同一套流程（渲染提示词 → 调模型 → 提取），
 * 只是能力不同、模板不同。
 */

import {
  buildContext,
  extractOutput,
  renderPrompt,
  type TranslateInput,
} from '../config/capabilities.js'
import type { ToneId } from '../config/defaults.js'
import type { LlmClient } from '../llm/types.js'
import type { ProviderPromptSet } from '../config/types.js'

export interface TranslationResult {
  translation: string
}

export interface TranslateOptions {
  /** 暂固定 zh，未来扩展 */
  source?: 'zh'
  /** 暂固定 en，未来扩展 */
  target?: 'en'
  /** 可选语气；不传则用中性默认 */
  tone?: ToneId
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
      const { client, promptSet } = deps.getActive()
      const template = promptSet.templates.translate
      if (!template) {
        throw new Error('没有 "translate" 的提示词模板')
      }

      const input: TranslateInput = { text: text.trim(), tone: options?.tone }
      const context = buildContext('translate', input, promptSet.tones)
      const { system, user } = renderPrompt(template, context)

      const mode = template.responseMode
      const result = await client.chat({
        system,
        user,
        jsonSchema: mode === 'json' ? template.jsonSchema : undefined,
      })

      const translation = extractOutput('translate', result.content, mode)
      return { translation: translation as string }
    },
  }
}
