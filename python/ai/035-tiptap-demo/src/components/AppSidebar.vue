<script setup lang="ts">
import { ref, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { MODES, setMode, useMode, type TranslationMode } from '../composables/useMode'

/**
 * 左侧栏：可收缩，收纳翻译方向切换，底部钉着「⚙️ 设置」。
 *
 * 收缩状态存 localStorage —— 刷新后还是收着的。
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
