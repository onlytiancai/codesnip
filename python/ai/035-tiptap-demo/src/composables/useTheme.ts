/**
 * 主题（亮色 / 暗色）—— 整站一套。
 *
 * 做法是给 `<html>` 加 `data-theme="light|dark"`，CSS 变量在
 * `:root[data-theme='dark']` 里覆盖一遍。变量清单在 `src/style.css` 顶部。
 *
 * 存 localStorage —— 刷新后还是上次的选择。
 * ⚠️ **localStorage 的 key 是 `tiptap-demo.theme`** —— `index.html` 里有一段
 * 内联脚本要在首屏渲染前就把主题设上（不然会闪一下白），它用的是同一个 key，
 * 改名字要两处一起改。
 */

import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark'

export const THEME_STORAGE_KEY = 'tiptap-demo.theme'

function systemTheme(): Theme {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function readTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    // 存不了就算了
  }
  // 没设过 → 跟系统走
  return systemTheme()
}

const theme = ref<Theme>(readTheme())

/** 把主题写到 `<html>` 上（CSS 变量靠这个属性切换） */
function applyTheme(t: Theme) {
  const root = document.documentElement
  root.dataset.theme = t
  root.style.colorScheme = t
}

watch(
  theme,
  (t) => {
    applyTheme(t)
    try {
      localStorage.setItem(THEME_STORAGE_KEY, t)
    } catch {
      // 存不了就算了，内存里的状态照样能用
    }
  },
  { immediate: true },
)

export function useTheme() {
  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }
  return { theme, toggleTheme }
}
