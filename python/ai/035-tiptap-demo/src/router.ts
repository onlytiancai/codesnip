import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import SettingsPage from './pages/SettingsPage.vue'

/**
 * 两个页面：
 *   /            编辑器（中英对照 + AI 助手）
 *   /settings    设置（LLM provider + 每个 provider 的提示词）
 */
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'editor', component: App },
    { path: '/settings', name: 'settings', component: SettingsPage },
  ],
})
