import { watch } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import EditorPage from './pages/EditorPage.vue'
import SettingsPage from './pages/SettingsPage.vue'
import LoginPage from './pages/LoginPage.vue'
import { loadAuth, user } from './composables/useAuth'
import { debugLog } from './utils/debug'

/**
 * 三个页面：
 *   /login       登录（顶层路由 —— 不套 AppSidebar 外壳）
 *   /            编辑器（中英对照 + AI 助手）
 *   /settings    设置（LLM provider + 每个 provider 的提示词 + 词库）
 *
 * 侧栏在 / 和 /settings 都在 —— 所以设置页里也能直接切翻译方向。
 *
 * 登录守卫：没登录一律跳 /login。
 * 注意 /login 要放在 '/' **之前** —— 不然会被父路由吃掉。
 */
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: LoginPage },
    {
      path: '/',
      component: App,
      children: [
        { path: '', name: 'editor', component: EditorPage },
        { path: 'settings', name: 'settings', component: SettingsPage },
      ],
    },
  ],
})

// 没登录 → 跳登录页
router.beforeEach(async (to) => {
  if (to.name === 'login') return true
  await loadAuth()
  if (!user.value) {
    debugLog('[router] 未登录 → 跳 /login')
    return { name: 'login' }
  }
  return true
})

// 接口返回 401（会话过期）→ 清掉登录态，跳登录页
watch(user, (next) => {
  if (next === null && router.currentRoute.value.name !== 'login') {
    void router.push({ name: 'login' })
  }
})
