/**
 * Anthropic 客户端。
 *
 * 结构化输出用 **tool-use 强制**：给一个名为 result 的工具，input_schema
 * 就是我们想要的 JSON Schema，再用 tool_choice 逼模型必须调用它。
 * 拿到 tool_use block 后把 input 原样 JSON.stringify 出去。
 *
 * 为什么不用 output_config.format：它要求 additionalProperties:false、
 * 不支持递归/数值约束，而且只有 Fable 5 / Opus 5 / Opus 4.8 / Sonnet 5 /
 * Haiku 4.5 支持。tool-use 全模型可用，更保险。
 *
 * 两个必须注意的点：
 * - max_tokens 是**必填**的，而且它是「思考 + 回复」的共同上限
 *   （Opus 5 默认开 adaptive thinking），给小了会截断 → 默认给 16000
 * - **采样参数（temperature / top_p / top_k）已经没有温度了** —— SDK v1 的
 *   MessageCreateParams 上根本没有这些字段，API 侧已经移除。所以 provider
 *   配置里的 temperature 对 Anthropic 无效，这里直接不读它。
 *   想要「换个说法」的差异，请改提示词。
 */

import Anthropic from '@anthropic-ai/sdk'
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

const RESULT_TOOL_NAME = 'result'

export function createAnthropicClient(
  config: ProviderConfig,
  apiKey?: string,
): LlmClient {
  const client = new Anthropic({
    apiKey: apiKey ?? 'not-set',
    baseURL: config.baseUrl || undefined,
    timeout: config.timeoutMs ?? 120_000,
    maxRetries: 2,
  })

  function buildParams(
    req: ChatRequest,
    system: string,
    useTool: boolean,
  ): Anthropic.MessageCreateParamsNonStreaming {
    const params: Anthropic.MessageCreateParamsNonStreaming = {
      model: config.model,
      max_tokens: config.maxTokens ?? 16000,
      system,
      messages: [{ role: 'user', content: req.user }],
    }
    // 注意：不发 temperature —— Anthropic API 已移除采样参数
    if (useTool && req.jsonSchema) {
      params.tools = [
        {
          name: RESULT_TOOL_NAME,
          description: 'Return the result as JSON matching the input schema.',
          input_schema: req.jsonSchema as Anthropic.Tool.InputSchema,
        },
      ]
      params.tool_choice = { type: 'tool', name: RESULT_TOOL_NAME }
    }
    return params
  }

  /** 从响应里取结果：优先 tool_use block，其次 text block。 */
  function extractContent(res: Anthropic.Message): string {
    const toolUse = res.content.find(
      (block): block is Anthropic.ToolUseBlock => block.type === 'tool_use',
    )
    if (toolUse) return JSON.stringify(toolUse.input)
    return stripReasoning(
      res.content
        .filter((block): block is Anthropic.TextBlock => block.type === 'text')
        .map((block) => block.text)
        .join(''),
    )
  }

  return {
    kind: 'anthropic',

    async chat(req: ChatRequest): Promise<ChatResult> {
      const wantsJson = Boolean(req.jsonSchema)
      const useTool = wantsJson && config.structuredOutput === 'schema'
      // API 层用 tool-use 强制时就不用追加契约了
      const system =
        wantsJson && config.structuredOutput !== 'schema' && config.structuredOutput !== 'off'
          ? appendJsonContract(req.system, req.jsonSchema)
          : req.system

      try {
        try {
          const res = await client.messages.create(buildParams(req, system, useTool))
          return { content: extractContent(res), structured: useTool, system }
        } catch (err) {
          // 部分代理端点不支持 tools → 降到纯提示词
          if (!useTool || !isBadRequest(err)) throw err
          const fallbackSystem = appendJsonContract(req.system, req.jsonSchema)
          const res = await client.messages.create(
            buildParams(req, fallbackSystem, false),
          )
          return { content: extractContent(res), structured: false, system: fallbackSystem }
        }
      } catch (err) {
        throw new Error(errorMessage(err))
      }
    },

    async listModels(): Promise<string[]> {
      const res = await client.models.list()
      return res.data.map((m) => m.id)
    },
  }
}
