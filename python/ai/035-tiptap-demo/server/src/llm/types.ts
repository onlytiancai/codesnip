/**
 * LLM 抽象层：ollama / openai / anthropic 三家客户端的统一接口。
 *
 * 关键设计：**结构化输出在 API 层强制 + 提示词兜底**。
 *
 * - provider 支持 schema 强制（OpenAI json_schema / Anthropic tool-use /
 *   Ollama format 传 schema 对象）→ 直接用 API 的约束能力
 * - provider 不支持（很多 OpenAI 兼容端点只认 json_object）→ 降级
 * - 端点连 json_object 都不认 → 再降级成纯提示词要求
 *
 * 降级路径全都还能拿到 JSON，靠的是：只要这次请求带了 jsonSchema，
 * system 后面就会追加一段标准 JSON 契约（除非 API 层已经在强制了）。
 */

import type { JsonSchema, ProviderKind } from '../config/types.js'

export type { ProviderKind }

export interface ChatRequest {
  system: string
  user: string
  /** 非空 = 要求模型按此 schema 返回 JSON */
  jsonSchema?: JsonSchema
}

export interface ChatResult {
  /** 模型输出文本；API 层强制了 JSON 时这里是 JSON 字符串 */
  content: string
  /** API 层是否强制了 JSON 结构 */
  structured: boolean
  /** 实际发出去的 system 消息（可能追加了 JSON 契约） */
  system: string
}

export interface LlmClient {
  readonly kind: ProviderKind
  chat(req: ChatRequest): Promise<ChatResult>
  /** 列出该 provider 可用的模型 id */
  listModels(): Promise<string[]>
}

/**
 * JSON 契约文本。API 层没法强制 schema 时，靠这段话让模型输出 JSON。
 */
export function jsonContract(schema: JsonSchema): string {
  return [
    'Respond with a single JSON object that matches the following JSON Schema.',
    'Output JSON only — no prose, no explanations, no markdown fences.',
    '',
    JSON.stringify(schema, null, 2),
  ].join('\n')
}

/**
 * 算出真正发给模型的 system。
 *
 * API 层没法强制 schema 时（object / prompt 模式），靠追加这段契约
 * 让模型输出 JSON —— 这样降级路径也拿得到可解析的结果。
 */
export function appendJsonContract(
  system: string,
  schema: JsonSchema | undefined,
): string {
  if (!schema) return system
  return `${system}\n\n${jsonContract(schema)}`
}

/**
 * 剥掉模型在正文里吐的推理块。
 *
 * MiniMax-M3 走 **OpenAI 兼容端点**时，会把 `<think>…</think>` 直接塞进
 * content；走 Anthropic 兼容端点则不会（它单独返回一个 thinking block）。
 *
 * 不剥的话后果很具体：语法检查会把整段思考原样显示给用户（它本来是
 * 原文透传），翻译 / 改写的解析会直接失败，选区候选的 JSON 数组也会被
 * 思考里的方括号带偏。
 *
 * 没有 think 块时**原样返回** —— 语法检查「模型写什么就透传什么」的
 * 语义不受影响。
 */
export function stripReasoning(content: string): string {
  if (!/<\/?think/i.test(content)) return content
  return content
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*$/i, '')
    .trim()
}

/** SDK 抛错时判断是不是 400（用于触发降级重试）。 */
export function isBadRequest(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false
  const e = err as { status?: unknown; statusCode?: unknown; message?: unknown }
  if (e.status === 400 || e.statusCode === 400) return true
  return typeof e.message === 'string' && /\b400\b/.test(e.message)
}

/** 错误消息：SDK 的错误对象 message 通常已经在里面了。 */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  if (err && typeof err === 'object') {
    const e = err as { message?: unknown }
    if (typeof e.message === 'string') return e.message
  }
  return 'unknown error'
}
