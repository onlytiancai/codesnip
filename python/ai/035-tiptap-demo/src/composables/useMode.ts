/**
 * 翻译方向 —— 侧栏菜单项和编辑器页共用一份。
 *
 * 模块级单例（照 src/composables/useTones.ts 的写法）：
 * 侧栏在 App.vue（外壳）里，编辑器在 EditorPage.vue 里，
 * 两边不是父子关系，所以状态得放在模块上。
 *
 * 存 localStorage —— 刷新后还停在上次的方向。
 */

import { ref } from 'vue'

export type TranslationMode = 'zh2en' | 'en2zh'

const STORAGE_KEY = 'tiptap-demo.mode'

function readStoredMode(): TranslationMode {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'en2zh' ? 'en2zh' : 'zh2en'
  } catch {
    // 隐私模式 / 禁用了 localStorage —— 用默认值就行
    return 'zh2en'
  }
}

const mode = ref<TranslationMode>(readStoredMode())

/** 切方向并记住。 */
export function setMode(next: TranslationMode): void {
  mode.value = next
  try {
    localStorage.setItem(STORAGE_KEY, next)
  } catch {
    // 存不了就算了，内存里的状态照样能用
  }
}

export function useMode() {
  return { mode, setMode }
}

/** 每个方向的展示信息 —— 侧栏菜单和编辑器标题共用。 */
export const MODES: Record<TranslationMode, { label: string; icon: string; title: string }> = {
  zh2en: { label: '中译英', icon: '中→英', title: '中英翻译写作台' },
  en2zh: { label: '英译中', icon: '英→中', title: '英中翻译写作台' },
}
