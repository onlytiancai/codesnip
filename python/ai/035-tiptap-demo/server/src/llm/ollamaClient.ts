/**
 * Ollama 客户端。
 *
 * Ollama 的 `format` 字段接受 `string | object` —— 传 JSON Schema 对象时
 * 它会用语法约束输出，天然就是结构化输出。
 */

import { Ollama } from 'ollama'
import type { ProviderConfig } from '../config/types.js'
import {
  appendJsonContract,
  errorMessage,
  stripReasoning,
  type ChatRequest,
  type ChatResult,
  type LlmClient,
} from './types.js'

export function createOllamaClient(
  config: ProviderConfig,
  _apiKey?: string,
): LlmClient {
  const client = new Ollama({ host: config.baseUrl || undefined })

  return {
    kind: 'ollama',

    async chat(req: ChatRequest): Promise<ChatResult> {
      const enforceSchema = config.structuredOutput === 'schema'
      const wantsJson = Boolean(req.jsonSchema)
      // API 层能强制 schema 时就不用追加契约了（模型看到的和编辑器里一样）
      const system =
        wantsJson && config.structuredOutput !== 'schema' && config.structuredOutput !== 'off'
          ? appendJsonContract(req.system, req.jsonSchema)
          : req.system

      try {
        const res = await client.chat({
          model: config.model,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: req.user },
          ],
          // format 传 schema 对象 → Ollama 用语法约束输出
          format: enforceSchema ? req.jsonSchema : undefined,
          options:
            config.temperature !== undefined
              ? { temperature: config.temperature }
              : undefined,
        })
        return {
          content: stripReasoning(res.message?.content ?? ''),
          structured: enforceSchema && wantsJson,
          system,
        }
      } catch (err) {
        throw new Error(errorMessage(err))
      }
    },

    async listModels(): Promise<string[]> {
      const res = await client.list()
      return res.models.map((m) => m.name ?? m.model).filter(Boolean) as string[]
    },
  }
}
