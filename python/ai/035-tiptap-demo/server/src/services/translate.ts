import ollama, { type Ollama } from 'ollama'
import {
  buildTranslatePrompt,
  parseTranslationResult,
  type TranslationResult,
} from '../prompts/translate.js'

export interface TranslateOptions {
  /** 暂固定 zh，未来扩展 */
  source?: 'zh'
  /** 暂固定 en，未来扩展 */
  target?: 'en'
}

export interface CreateTranslateServiceOptions {
  /** 测试时可注入 mock；生产留空走 default import */
  client?: Ollama
  /** 必填，Ollama 模型名 */
  model: string
}

export interface TranslateService {
  translate(text: string, options?: TranslateOptions): Promise<TranslationResult>
}

/**
 * 构造一个 TranslateService。生产代码只传 model；测试里传 client 做桩。
 */
export function createTranslateService(deps: CreateTranslateServiceOptions): TranslateService {
  const client = deps.client ?? ollama

  return {
    async translate(text, _options) {
      if (!text || !text.trim()) {
        throw new Error('text must be non-empty')
      }
      const prompt = buildTranslatePrompt(text)
      const response = await client.chat({
        model: deps.model,
        messages: [{ role: 'user', content: prompt }],
      })
      return parseTranslationResult(response.message?.content ?? '')
    },
  }
}