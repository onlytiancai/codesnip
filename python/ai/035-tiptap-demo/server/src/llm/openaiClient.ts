/**
 * OpenAI 客户端（也兼容 DeepSeek / Moonshot / Qwen / vLLM 这类 OpenAI 兼容端点）。
 *
 * 结构化输出的三级降级：
 *   1. response_format = json_schema（OpenAI 官方支持 schema 强制）
 *   2. response_format = json_object（很多兼容端点只认这个）
 *   3. 干脆不带 response_format，靠 system 里追加的 JSON 契约
 *
 * 第 1 → 2 级的降级条件是 400（端点不认这个参数）。
 * 第 2 → 3 级同理。
 */

import OpenAI from 'openai'
import type { ProviderConfig } from '../config/types.js'
import {
  appendJsonContract,
  errorMessage,
  isBadRequest,
  stripReasoning,
  type ChatRequest,
  type ChatResult,
  type LlmClient,
} from './types.js'

type ChatParams = OpenAI.ChatCompletionCreateParamsNonStreaming
type Enforce = 'schema' | 'object' | 'none'

export function createOpenAiClient(
  config: ProviderConfig,
  apiKey?: string,
): LlmClient {
  const client = new OpenAI({
    apiKey: apiKey ?? null,
    baseURL: config.baseUrl || undefined,
    timeout: config.timeoutMs ?? 120_000,
    maxRetries: 2,
  })

  /** max_tokens 在 OpenAI v7 里已标记 deprecated，但兼容端点只认它 —— 所以可配。 */
  const tokenField =
    config.maxTokensParam === 'max_completion_tokens'
      ? 'max_completion_tokens'
      : 'max_tokens'

  function buildParams(
    req: ChatRequest,
    system: string,
    enforce: Enforce,
  ): ChatParams {
    const params: ChatParams = {
      model: config.model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: req.user },
      ],
    }
    if (config.maxTokens !== undefined) {
      if (tokenField === 'max_completion_tokens') {
        params.max_completion_tokens = config.maxTokens
      } else {
        params.max_tokens = config.maxTokens
      }
    }
    if (config.temperature !== undefined) params.temperature = config.temperature

    if (enforce === 'schema' && req.jsonSchema) {
      params.response_format = {
        type: 'json_schema',
        json_schema: {
          name: 'result',
          schema: req.jsonSchema,
          // strict:true 只支持 JSON Schema 子集（属性全 required +
          // additionalProperties:false），用户手写的 schema 很容易不满足 → 400
          strict: false,
        },
      }
    } else if (enforce === 'object') {
      params.response_format = { type: 'json_object' }
    }

    return params
  }

  async function call(
    req: ChatRequest,
    enforce: Enforce,
    appendContract: boolean,
  ): Promise<ChatResult> {
    const system = appendContract
      ? appendJsonContract(req.system, req.jsonSchema)
      : req.system
    const res = await client.chat.completions.create(
      buildParams(req, system, enforce),
    )
    return {
      content: stripReasoning(res.choices[0]?.message?.content ?? ''),
      structured: enforce !== 'none',
      system,
    }
  }

  return {
    kind: 'openai',

    async chat(req: ChatRequest): Promise<ChatResult> {
      const mode = config.structuredOutput
      const wantsJson = Boolean(req.jsonSchema)

      // 降级链：按顺序试，只有 400 才继续往下走
      const attempts: Array<{ enforce: Enforce; contract: boolean }> = []
      if (!wantsJson || mode === 'off') {
        // 不需要 JSON（或用户明确关掉了）→ 原样发
        attempts.push({ enforce: 'none', contract: false })
      } else if (mode === 'schema') {
        attempts.push({ enforce: 'schema', contract: false })
        attempts.push({ enforce: 'object', contract: true })
        attempts.push({ enforce: 'none', contract: true })
      } else if (mode === 'object') {
        attempts.push({ enforce: 'object', contract: true })
        attempts.push({ enforce: 'none', contract: true })
      } else {
        // mode === 'prompt'
        attempts.push({ enforce: 'none', contract: true })
      }

      let lastErr: unknown
      for (const attempt of attempts) {
        try {
          return await call(req, attempt.enforce, attempt.contract)
        } catch (err) {
          lastErr = err
          // 只有端点不认这个参数（400）才降级；其它错误直接抛
          if (!isBadRequest(err)) throw new Error(errorMessage(err))
        }
      }
      throw new Error(errorMessage(lastErr))
    },

    async listModels(): Promise<string[]> {
      const res = await client.models.list()
      return res.data.map((m) => m.id)
    },
  }
}
