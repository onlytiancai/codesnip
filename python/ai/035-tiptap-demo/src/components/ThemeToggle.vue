<script setup lang="ts">
/**
 * 主题切换按钮（亮色 ↔ 暗色）。
 *
 * 挂在 `Root.vue` 里 —— 登录页和登录后的页面都有（路由是顶层/嵌套两套，
 * 只有 Root 是两套都经过的）。
 */
import AppIcon from './AppIcon.vue'
import { useTheme } from '../composables/useTheme'

const { theme, toggleTheme } = useTheme()
</script>

<template>
  <button
    type="button"
    class="theme-toggle"
    :data-tip="theme === 'dark' ? '切换到亮色' : '切换到暗色'"
    :aria-label="theme === 'dark' ? '切换到亮色' : '切换到暗色'"
    :aria-pressed="theme === 'dark'"
    @click="toggleTheme"
  >
    <AppIcon :name="theme === 'dark' ? 'sun' : 'moon'" :size="16" />
  </button>
</template>
