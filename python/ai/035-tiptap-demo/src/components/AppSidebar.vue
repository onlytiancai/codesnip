<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { MODES, setMode, useMode, type TranslationMode } from '../composables/useMode'
import { doLogout, refreshUsage, useAuth } from '../composables/useAuth'
import { debugLog } from '../utils/debug'

/**
 * 左侧栏：可收缩，收纳翻译方向切换。
 *
 * 收缩状态存 localStorage —— 刷新后还是收着的。
 *
 * **左下角是用户块**：用户名 + 套餐徽标，点它弹出菜单（设置 / 退出登录）。
 * 用户块**上面**是用量（今日配额条 + 今日/本月 token）。
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

// ---------------------------------------------------------------------------
// 用户菜单（点用户名弹出来）
// ---------------------------------------------------------------------------

const menuOpen = ref(false)
/** 菜单所在的容器 —— 点它外面要关菜单。 */
const menuWrap = ref<HTMLElement | null>(null)

function toggleMenu() {
  menuOpen.value = !menuOpen.value
}

/** 点菜单外面 → 关。 */
function onDocMousedown(e: MouseEvent) {
  if (!menuOpen.value) return
  const el = menuWrap.value
  if (el && !el.contains(e.target as Node)) {
    menuOpen.value = false
  }
}

/** Esc → 关。 */
function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') menuOpen.value = false
}

/** 进设置页（和以前左下角那个「⚙️ 设置」一样）。 */
async function goSettings() {
  menuOpen.value = false
  debugLog('[auth] 侧栏菜单点击设置')
  await router.push('/settings')
}

/** 退出登录。 */
async function logout() {
  menuOpen.value = false
  debugLog('[auth] 侧栏菜单点击退出登录')
  await doLogout()
  await router.push({ name: 'login' })
}

// 路由一变（比如点了设置）就把菜单收起来
watch(() => route.path, () => {
  menuOpen.value = false
})

onMounted(() => {
  void refreshUsage()
  timer = window.setInterval(() => {
    void refreshUsage()
  }, USAGE_POLL_MS)
  document.addEventListener('mousedown', onDocMousedown)
  document.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  if (timer !== undefined) window.clearInterval(timer)
  document.removeEventListener('mousedown', onDocMousedown)
  document.removeEventListener('keydown', onKeydown)
})

/** 大数字加千分位，界面上好看点。 */
function fmt(n: number | undefined): string {
  if (n === undefined) return '—'
  return n.toLocaleString('zh-CN')
}

/** 配额条百分比（0~100）。上限 0 = 不限 → 0。 */
function percent(used: number | undefined, limit: number | undefined): number {
  if (!used || !limit || limit <= 0) return 0
  return Math.min(100, Math.round((used / limit) * 100))
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

    <!-- 底部：用量（上）+ 用户（下，点用户名弹菜单） -->
    <div class="sidebar__bottom">
      <!-- 用量：今日配额条 + 今日/本月 token -->
      <div v-if="!collapsed && user" class="sidebar__usage-block">
        <div class="sidebar__quota">
          <div class="sidebar__bar">
            <div
              class="sidebar__bar-fill"
              :class="{ 'sidebar__bar-fill--over': user.quota.dailyExceeded }"
              :style="{
                width: `${percent(user.quota.dailyUsed, user.quota.dailyLimit)}%`,
              }"
            />
          </div>
          <span class="sidebar__quota-num">
            今日 {{ fmt(user.quota.dailyUsed) }} /
            {{ user.quota.dailyLimit > 0 ? fmt(user.quota.dailyLimit) : '不限' }}
          </span>
        </div>

        <div class="sidebar__usage">
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
      </div>

      <!-- 用户：用户名 + 套餐徽标，点开菜单 -->
      <div ref="menuWrap" class="sidebar__user-wrap">
        <button
          type="button"
          class="sidebar__user"
          :title="collapsed ? user?.username ?? '未登录' : '点击打开菜单'"
          :aria-haspopup="'menu'"
          :aria-expanded="menuOpen"
          @click="toggleMenu"
        >
          <span class="sidebar__icon">👤</span>
          <span v-if="!collapsed" class="sidebar__user-name">
            {{ user?.username ?? '未登录' }}
          </span>
          <span v-if="!collapsed && user" class="sidebar__plan">
            {{ user.plan.label }}
          </span>
          <span v-if="!collapsed" class="sidebar__caret">{{
            menuOpen ? '▴' : '▾'
          }}</span>
        </button>

        <!-- 菜单：设置 / 退出登录（向上弹，因为贴着底边） -->
        <div v-if="menuOpen" class="sidebar__menu" role="menu">
          <button
            type="button"
            class="sidebar__menu-item"
            role="menuitem"
            @click="goSettings"
          >
            <span class="sidebar__icon">⚙️</span>
            <span v-if="!collapsed">设置</span>
          </button>
          <button
            type="button"
            class="sidebar__menu-item"
            role="menuitem"
            @click="logout"
          >
            <span class="sidebar__icon">⏻</span>
            <span v-if="!collapsed">退出登录</span>
          </button>
        </div>
      </div>
    </div>
  </aside>
</template>
