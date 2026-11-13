/**
 * 用户自定义指令（用户提示词）的后端调用。
 *
 * 指令跟着 provider 走 —— 每个 provider 下是自己那套。
 * 只有本人能看 / 改 / 删 / 跑；新建还需要付费套餐（管理员不限）。
 */

import { http } from './http'
import type { PromptScope } from '../constants/customPrompts'

export interface UserPrompt {
  id: string
  providerId: string
  /** 名字（工具栏上显示的） */
  label: string
  /** 图标（emoji） */
  icon: string
  /** 作用范围：当前句 / 当前段 / 所有正文 */
  scope: PromptScope
  /** 用户写的那段指令文本 */
  instruction: string
  /** 要不要在英文编辑器的工具栏上显示 */
  showInToolbar: boolean
  ordinal: number
  createdAt: string
  updatedAt: string
}

export interface CustomPromptInput {
  label: string
  icon: string
  scope: PromptScope
  instruction: string
  showInToolbar: boolean
  ordinal?: number
}

/** 当前生效 provider 下的（自己的）指令。 */
export const fetchActiveCustomPrompts = (signal?: AbortSignal) =>
  http.get<{ activeProviderId: string; prompts: UserPrompt[] }>(
    '/api/custom-prompts',
    signal,
  )

/** 某个 provider 下的（自己的）指令。 */
export const fetchCustomPrompts = (providerId: string, signal?: AbortSignal) =>
  http.get<{ prompts: UserPrompt[] }>(
    `/api/providers/${encodeURIComponent(providerId)}/custom-prompts`,
    signal,
  )

/** 在当前生效的 provider 下新建一条。 */
export const createCustomPrompt = (input: CustomPromptInput) =>
  http.post<{ prompt: UserPrompt }>('/api/custom-prompts', input)

/** 在指定 provider 下新建一条。 */
export const createCustomPromptFor = (
  providerId: string,
  input: CustomPromptInput,
) =>
  http.post<{ prompt: UserPrompt }>(
    `/api/providers/${encodeURIComponent(providerId)}/custom-prompts`,
    input,
  )

export const updateCustomPrompt = (id: string, input: CustomPromptInput) =>
  http.put<{ prompt: UserPrompt }>(
    `/api/custom-prompts/${encodeURIComponent(id)}`,
    input,
  )

export const deleteCustomPrompt = (id: string) =>
  http.del<{ ok: boolean }>(`/api/custom-prompts/${encodeURIComponent(id)}`)

/**
 * 跑一条指令。
 *
 * text 由前端按「作用范围」取好（当前句 / 当前段 / 所有正文），
 * context 也是前端自动填的上下文；指令本身从库里取，不信客户端。
 */
export const runCustomPrompt = (
  id: string,
  payload: { text: string; context?: string; sourceSentence?: string },
  signal?: AbortSignal,
) =>
  http.post<{ text: string }>(
    `/api/custom-prompts/${encodeURIComponent(id)}/run`,
    payload,
    signal,
  )
