import ollama, { type Ollama } from 'ollama'
import { buildRewritePrompt, parseCandidates } from '../prompts/rewriteWord.js'

export interface RewriteOptions {
  /** 当前句/段落上下文，可选；用于消歧 */
  context?: string
  /** 候选数量上限，默认走 deps.defaultMaxCandidates */
  maxCandidates?: number
}

export interface CreateOllamaServiceOptions {
  /** 测试时可注入 mock；生产留空走 default import */
  client?: Ollama
  /** 必填，Ollama 模型名 */
  model: string
  /** 默认候选上限 */
  defaultMaxCandidates?: number
}

export interface OllamaService {
  rewriteWord(word: string, options?: RewriteOptions): Promise<string[]>
}

/**
 * 构造一个 OllamaService。生产代码只传 model；测试里传 client 做桩。
 */
export function createOllamaService(deps: CreateOllamaServiceOptions): OllamaService {
  const client = deps.client ?? ollama
  const defaultMax = deps.defaultMaxCandidates ?? 5

  return {
    async rewriteWord(word, options) {
      if (!word || !word.trim()) {
        throw new Error('word must be non-empty')
      }

      const prompt = buildRewritePrompt(word, options?.context)
      const response = await client.chat({
        model: deps.model,
        messages: [{ role: 'user', content: prompt }],
      })

      const content = response.message?.content ?? ''
      return parseCandidates(content, options?.maxCandidates ?? defaultMax)
    },
  }
}