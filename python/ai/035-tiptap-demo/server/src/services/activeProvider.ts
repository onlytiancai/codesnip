/**
 * 把 ConfigStore + ClientRegistry 拼成「当前生效的 provider」快照。
 *
 * 服务层每次请求都问一次这个函数 —— 所以在 UI 里切换 provider
 * 立刻生效，不用重启服务。
 */

import type { ConfigStore } from '../config/store.js'
import type { ClientRegistry } from '../llm/registry.js'
import type { ActiveProvider } from './ai.js'

export function createActiveProviderGetter(
  store: ConfigStore,
  registry: ClientRegistry,
): () => ActiveProvider {
  return () => {
    const provider = store.getActiveProvider()
    const apiKey = store.getApiKey(provider.id)
    const client = registry.get(provider, apiKey)
    const promptSet = store.getPromptSet(provider.id)
    return { client, promptSet, provider }
  }
}
