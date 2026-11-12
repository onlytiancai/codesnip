/**
 * Provider 配置 → LlmClient。
 *
 * client 按配置内容缓存：改了 model / baseUrl / key 就会拿到新实例，
 * 没改就复用同一个（SDK 客户端内部有连接池，别每次请求都新建）。
 */

import type { ProviderConfig } from '../config/types.js'
import { createAnthropicClient } from './anthropicClient.js'
import { createOllamaClient } from './ollamaClient.js'
import { createOpenAiClient } from './openaiClient.js'
import type { LlmClient } from './types.js'

export type LlmClientFactory = (
  config: ProviderConfig,
  apiKey?: string,
) => LlmClient

export const defaultLlmClientFactory: LlmClientFactory = (config, apiKey) => {
  switch (config.kind) {
    case 'ollama':
      return createOllamaClient(config, apiKey)
    case 'openai':
      return createOpenAiClient(config, apiKey)
    case 'anthropic':
      return createAnthropicClient(config, apiKey)
    default:
      throw new Error(`未知的 provider 类型：${(config as ProviderConfig).kind}`)
  }
}

export class ClientRegistry {
  private readonly cache = new Map<string, LlmClient>()

  constructor(
    private readonly factory: LlmClientFactory = defaultLlmClientFactory,
  ) {}

  get(config: ProviderConfig, apiKey?: string): LlmClient {
    const key = `${config.id}|${JSON.stringify(config)}|${apiKey ?? ''}`
    const cached = this.cache.get(key)
    if (cached) return cached
    const client = this.factory(config, apiKey)
    this.cache.set(key, client)
    return client
  }

  clear(): void {
    this.cache.clear()
  }
}
