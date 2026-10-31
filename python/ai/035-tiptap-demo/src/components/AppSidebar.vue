<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { MODES, setMode, useMode, type TranslationMode } from '../composables/useMode'
import { doLogout, refreshUsage, useAuth } from '../composables/useAuth'
import { debugLog } from '../utils/debug'

/**
 * 左侧栏：可收缩，收纳翻译方向切换，底部钉着「⚙️ 设置」。
 *
 * 收缩状态存 localStorage —— 刷新后还是收着的。
 *
 * 设置项**上方**是用户信息块：用户名 + 今日/本月 token 消耗。
 */

const STORAGE_KEY = 'tiptap-demo.sidebar-collapsed'

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

const collapsed = ref(readCollapsed())
const { mode } = useMode()
const route = useRoute()
const router = useRouter()
const { user, usage } = useAuth()

watch(collapsed, (v) => {
  try {
    localStorage.setItem(STORAGE_KEY, v ? '1' : '0')
  } catch {
    // 存不了就算了
  }
})

function toggle() {
  collapsed.value = !collapsed.value
}

const ORDER: TranslationMode[] = ['zh2en', 'en2zh']

/** 每 20 秒刷新一次用量 —— 翻译完侧栏的数字会跟着涨。 */
const USAGE_POLL_MS = 20_000
let timer: number | undefined

onMounted(() => {
  void refreshUsage()
  timer = window.setInterval(() => {
    void refreshUsage()
  }, USAGE_POLL_MS)
})

onUnmounted(() => {
  if (timer !== undefined) window.clearInterval(timer)
})

async function logout() {
  debugLog('[auth] 侧栏点击退出登录')
  await doLogout()
  await router.push({ name: 'login' })
}

/** 大数字加千分位，界面上好看点。 */
function fmt(n: number | undefined): string {
  if (n === undefined) return '—'
  return n.toLocaleString('zh-CN')
}
</script>

<template>
  <aside class="sidebar" :class="{ 'sidebar--collapsed': collapsed }">
    <!-- 顶部：收缩按钮 + 标题 -->
    <div class="sidebar__top">
      <button
        type="button"
        class="sidebar__toggle"
        :title="collapsed ? '展开侧边栏' : '收起侧边栏'"
        :aria-label="collapsed ? '展开侧边栏' : '收起侧边栏'"
        @click="toggle"
      >
        {{ collapsed ? '☰' : '«' }}
      </button>
      <span v-if="!collapsed" class="sidebar__brand">翻译工作台</span>
    </div>

    <!-- 中部：方向切换 -->
    <nav class="sidebar__nav">
      <button
        v-for="m in ORDER"
        :key="m"
        type="button"
        class="sidebar__item"
        :class="{ 'sidebar__item--on': mode === m }"
        :title="collapsed ? MODES[m].label : undefined"
        @click="setMode(m)"
      >
        <span class="sidebar__icon">{{ MODES[m].icon }}</span>
        <span v-if="!collapsed" class="sidebar__label">{{ MODES[m].label }}</span>
      </button>
    </nav>

    <!-- 用户信息 + token 用量（贴在设置项上方） -->
    <div class="sidebar__user">
      <div
        class="sidebar__user-head"
        :title="
          collapsed
            ? `${user?.username ?? ''}｜今日 ${fmt(usage?.today.totalTokens)} token`
            : undefined
        "
      >
        <span class="sidebar__icon">👤</span>
        <span v-if="!collapsed" class="sidebar__user-name">
          {{ user?.username ?? '未登录' }}
        </span>
      </div>

      <div v-if="!collapsed" class="sidebar__usage">
        <div class="sidebar__usage-row">
          <span class="sidebar__usage-tag">今日</span>
          <span class="sidebar__usage-num">
            入 {{ fmt(usage?.today.inputTokens) }} · 出
            {{ fmt(usage?.today.outputTokens) }}
          </span>
        </div>
        <div class="sidebar__usage-row">
          <span class="sidebar__usage-tag">本月</span>
          <span class="sidebar__usage-num">
            入 {{ fmt(usage?.month.inputTokens) }} · 出
            {{ fmt(usage?.month.outputTokens) }}
          </span>
        </div>
      </div>

      <button
        v-if="!collapsed"
        type="button"
        class="sidebar__logout"
        @click="logout"
      >
        退出登录
      </button>
    </div>

    <!-- 底部（左下角）：设置 -->
    <div class="sidebar__foot">
      <RouterLink
        to="/settings"
        class="sidebar__item"
        :class="{ 'sidebar__item--on': route.path.startsWith('/settings') }"
        :title="collapsed ? '设置' : undefined"
      >
        <span class="sidebar__icon">⚙️</span>
        <span v-if="!collapsed" class="sidebar__label">设置</span>
      </RouterLink>
    </div>
  </aside>
</template>
