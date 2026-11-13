/**
 * 用户自定义指令（用户提示词）的模块级状态。
 *
 * 用单例而不是 Pinia —— 侧栏（App.vue）、编辑器（TiptapEditor.vue）、
 * 设置页（SettingsPage.vue）互相不是父子关系，状态得放在模块上（照 useMode.ts 的写法）。
 *
 * 指令是跟着**当前生效的 provider** 走的 —— 换 provider 就是另一套。
 */

import { computed, ref } from 'vue'
import {
  fetchActiveCustomPrompts,
  type UserPrompt,
} from '../services/customPrompts'
import { user } from './useAuth'
import { MAX_TOOLBAR_PROMPTS } from '../constants/customPrompts'

const prompts = ref<UserPrompt[]>([])
/** 这些指令属于哪个 provider（当前生效的那个） */
const activeProviderId = ref('')
const loading = ref(false)
/** 已经拉过一次了 —— 避免每个组件都去请求 */
let loaded = false

/**
 * 拉当前用户的指令。
 *
 * 没登录 / 后端没起来 → 当作没有（不报错，编辑器里就不显示「我的指令」）。
 */
export async function loadCustomPrompts(force = false): Promise<void> {
  if (loading.value) return
  if (loaded && !force) return
  loading.value = true
  try {
    const res = await fetchActiveCustomPrompts()
    prompts.value = res.prompts
    activeProviderId.value = res.activeProviderId
    loaded = true
  } catch {
    prompts.value = []
    activeProviderId.value = ''
  } finally {
    loading.value = false
  }
}

/** 清掉缓存（登出 / 切 provider 后调用）。 */
export function resetCustomPrompts(): void {
  prompts.value = []
  activeProviderId.value = ''
  loaded = false
}

/**
 * 能不能自己加指令（新建 / 修改）。
 *
 * 付费用户可以；管理员不限；免费版不行（服务端也会挡一道）。
 */
export function canCustomizePrompts(): boolean {
  const u = user.value
  if (!u) return false
  return u.role === 'admin' || u.planId !== 'free'
}

export function useCustomPrompts() {
  return {
    prompts,
    activeProviderId,
    loading,
    reload: () => loadCustomPrompts(true),
    /** 钉在工具栏上的那些（最多 MAX_TOOLBAR_PROMPTS 个） */
    toolbarPrompts: computed(() =>
      prompts.value.filter((p) => p.showInToolbar).slice(0, MAX_TOOLBAR_PROMPTS),
    ),
  }
}
