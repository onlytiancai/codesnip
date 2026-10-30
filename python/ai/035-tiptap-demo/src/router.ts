import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import EditorPage from './pages/EditorPage.vue'
import SettingsPage from './pages/SettingsPage.vue'

/**
 * 两个页面，挂在同一个外壳（App.vue = 侧栏 + 内容区）下面：
 *   /            编辑器（中英对照 + AI 助手）
 *   /settings    设置（LLM provider + 每个 provider 的提示词 + 词库）
 *
 * 侧栏在两个页面都在 —— 所以设置页里也能直接切翻译方向。
 */
export const router = createRouter({
  history: createWebHistory(),
  routes: [
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
