/**
 * 编辑器的行为开关 —— 右侧工具栏上那几个开关的状态。
 *
 * 都是「要不要做某件事」的行为开关（高亮哪几层、双击查词、单击候选词），
 * 不是界面外观设置，所以归到 composable 而不是组件里。
 *
 * 模块级单例：工具栏和两个编辑器都要读同一份状态。
 * 存在 localStorage —— 刷新后还是上次的选择。
 */

import { ref, watch } from 'vue'

export interface EditorSettings {
  /** 高亮光标所在的单词 */
  highlightWord: boolean
  /** 高亮光标所在的句子 */
  highlightSentence: boolean
  /** 高亮光标所在的段落 */
  highlightParagraph: boolean
  /** 双击单词 → 直接查词 */
  doubleClickLookup: boolean
  /** 单击单词 → 弹出其它候选词 */
  clickCandidates: boolean
}

const STORAGE_KEY = 'tiptap-demo.editorSettings'

const DEFAULTS: EditorSettings = {
  highlightWord: true,
  highlightSentence: true,
  highlightParagraph: true,
  doubleClickLookup: true,
  clickCandidates: true,
}

function readSettings(): EditorSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<EditorSettings>
    // 只认布尔字段 —— 存坏了就退回默认值
    const out = { ...DEFAULTS }
    for (const key of Object.keys(DEFAULTS) as (keyof EditorSettings)[]) {
      if (typeof parsed[key] === 'boolean') out[key] = parsed[key]
    }
    return out
  } catch {
    return { ...DEFAULTS }
  }
}

const settings = ref<EditorSettings>(readSettings())

watch(
  settings,
  (v) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(v))
    } catch {
      // 存不了就算了，内存里的状态照样能用
    }
  },
  { deep: true },
)

export function useEditorSettings() {
  return { settings }
}
