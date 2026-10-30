/**
 * 设置相关的后端调用：provider 管理 + 每个 provider 各自的提示词。
 *
 * 提示词是**按 provider 分组**的 —— 不同模型需要不同措辞，所以每个 provider
 * 有一整套自己的模板 + 语气表。
 */

import { http } from './http'

export type ProviderKind = 'ollama' | 'openai' | 'anthropic'
export type StructuredOutputMode = 'schema' | 'object' | 'prompt' | 'off'
export type ResponseMode = 'text' | 'json'

export interface ProviderConfig {
  id: string
  label: string
  kind: ProviderKind
  baseUrl: string
  model: string
  apiKeyEnv?: string
  structuredOutput: StructuredOutputMode
  maxTokens?: number
  maxTokensParam?: 'max_tokens' | 'max_completion_tokens'
  temperature?: number
  timeoutMs?: number
  /** 后端告诉前端「这个 provider 存了 key」——明文 key 永远不会传过来 */
  hasKey?: boolean
}

export interface PromptTemplate {
  id: string
  label: string
  system: string
  user: string
  responseMode: ResponseMode
  jsonSchema?: Record<string, unknown>
  /** 当前版本号（0 = 还没改过，用的是内置默认） */
  version?: number
  updatedAt?: string
}

export interface ToneConfig {
  id: string
  label: string
  guidance: string
}

export interface CapabilityMeta {
  id: string
  label: string
}

export interface AppConfig {
  activeProviderId: string
  providers: ProviderConfig[]
  tones: ToneConfig[]
  capabilities: CapabilityMeta[]
  toneIds: string[]
}

export interface PromptSet {
  providerId: string
  templates: PromptTemplate[]
  tones: ToneConfig[]
  variables: Record<string, string[]>
  samples: Record<string, Record<string, string>>
}

export interface PromptVersionMeta {
  version: number
  savedAt: string
  note?: string
}

export interface PromptVersion extends PromptVersionMeta {
  providerId: string
  capabilityId: string
  template: PromptTemplate
}

export interface DryRunResult {
  providerId: string
  capabilityId: string
  system: string
  user: string
  structured: boolean
  raw?: string
  parsed?: string | string[]
  parseError?: string
  error?: string
}

// ---- 总览 ----

export const fetchConfig = (signal?: AbortSignal) =>
  http.get<AppConfig>('/api/config', signal)

// ---- provider ----

export const fetchProviders = (signal?: AbortSignal) =>
  http.get<{ activeProviderId: string; providers: ProviderConfig[] }>(
    '/api/providers',
    signal,
  )

export const saveProvider = (
  provider: Partial<ProviderConfig> & { id: string; label: string; kind: ProviderKind },
) => http.put<{ provider: ProviderConfig }>('/api/providers', provider)

export const deleteProvider = (id: string) =>
  http.del<{ ok: boolean }>(`/api/providers/${encodeURIComponent(id)}`)

export const setActiveProvider = (id: string) =>
  http.post<{ activeProviderId: string }>('/api/providers/active', { id })

export const testProvider = (id: string) =>
  http.post<{ ok: boolean; sample?: string; error?: string; model?: string }>(
    `/api/providers/${encodeURIComponent(id)}/test`,
  )

export const fetchModels = (id: string) =>
  http.get<{ models: string[] }>(`/api/providers/${encodeURIComponent(id)}/models`)

// ---- 提示词（按 provider）----

export const fetchPromptSet = (providerId: string) =>
  http.get<PromptSet>(`/api/providers/${encodeURIComponent(providerId)}/prompts`)

export const savePrompt = (
  providerId: string,
  capabilityId: string,
  template: PromptTemplate,
  note?: string,
) =>
  http.put<{ prompt: PromptTemplate; version: number; savedAt: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/${capabilityId}`,
    { template, note },
  )

export const fetchVersions = (providerId: string, capabilityId: string) =>
  http.get<{ versions: PromptVersionMeta[] }>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/${capabilityId}/versions`,
  )

export const fetchVersion = (
  providerId: string,
  capabilityId: string,
  version: number,
) =>
  http.get<PromptVersion>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/${capabilityId}/versions/${version}`,
  )

export const rollbackPrompt = (
  providerId: string,
  capabilityId: string,
  version: number,
) =>
  http.post<{ prompt: PromptTemplate; version: number; savedAt: string }>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/${capabilityId}/rollback`,
    { version },
  )

export const dryRunPrompt = (
  providerId: string,
  capabilityId: string,
  payload: { template?: Partial<PromptTemplate>; vars?: Record<string, unknown> },
) =>
  http.post<DryRunResult>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/${capabilityId}/dry-run`,
    payload,
  )

export const copyPrompts = (
  providerId: string,
  fromProviderId: string,
  capabilityId?: string,
) =>
  http.post<{ ok: boolean }>(
    `/api/providers/${encodeURIComponent(providerId)}/prompts/copy`,
    { fromProviderId, capabilityId },
  )

// ---- 语气表 ----

export const fetchTones = (providerId: string) =>
  http.get<{ tones: ToneConfig[] }>(
    `/api/providers/${encodeURIComponent(providerId)}/tones`,
  )

export const saveTones = (providerId: string, tones: ToneConfig[]) =>
  http.put<{ tones: ToneConfig[] }>(
    `/api/providers/${encodeURIComponent(providerId)}/tones`,
    { tones },
  )
