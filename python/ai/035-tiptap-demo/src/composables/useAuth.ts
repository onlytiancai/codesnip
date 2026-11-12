/**
 * 当前登录用户 + 今天的 token 用量。
 *
 * 模块级单例（照 src/composables/useMode.ts 的写法）——
 * 侧栏在 App.vue（外壳）里，登录页是独立路由，两边不是父子关系，
 * 所以状态得放在模块上。
 *
 * **不存 localStorage** —— 登录态由 httpOnly cookie 决定，
 * 刷新后靠 loadAuth() 重新问后端。
 */

import { ref } from 'vue'
import { ApiError } from '../services/http'
import {
  fetchMe,
  login as loginApi,
  logout as logoutApi,
  register as registerApi,
  type CurrentUser,
  type UsageTotals,
} from '../services/auth'
import { debugLog } from '../utils/debug'

export const user = ref<CurrentUser | null>(null)
export const usage = ref<{ today: UsageTotals; month: UsageTotals } | null>(null)
/** 已经问过一次后端了 —— 路由守卫靠它避免反复请求。 */
const loaded = ref(false)

export function useAuth() {
  return { user, usage, loaded }
}

/** 问后端「我是谁」。没登录就是 null（不算错误）。 */
export async function loadAuth(): Promise<void> {
  if (loaded.value) return
  try {
    const me = await fetchMe()
    user.value = me.user
    usage.value = { today: me.today, month: me.month }
    debugLog('[auth] 已登录:', me.user.username)
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      user.value = null
      usage.value = null
    } else {
      // 后端没起来之类 —— 也当没登录，让用户去登录页看到错误
      user.value = null
      usage.value = null
      debugLog('[auth] 拉取当前用户失败:', err instanceof Error ? err.message : err)
    }
  } finally {
    loaded.value = true
  }
}

/** 只刷新用量（侧栏定时用）。 */
export async function refreshUsage(): Promise<void> {
  if (!user.value) return
  try {
    const me = await fetchMe()
    user.value = me.user
    usage.value = { today: me.today, month: me.month }
  } catch {
    // 忽略
  }
}

export async function doLogin(username: string, password: string): Promise<void> {
  const res = await loginApi(username, password)
  user.value = res.user
  loaded.value = true
  await refreshUsage()
  debugLog('[auth] 登录成功:', res.user.username)
}

export async function doRegister(
  username: string,
  password: string,
): Promise<void> {
  const res = await registerApi(username, password)
  user.value = res.user
  loaded.value = true
  await refreshUsage()
  debugLog('[auth] 注册成功:', res.user.username)
}

export async function doLogout(): Promise<void> {
  try {
    await logoutApi()
  } catch {
    // 忽略
  }
  user.value = null
  usage.value = null
  debugLog('[auth] 已登出')
}

/** 接口返回 401 时调 —— 清掉登录态（路由守卫会跳登录页）。 */
export function markUnauthorized(): void {
  user.value = null
  usage.value = null
  loaded.value = true
}
