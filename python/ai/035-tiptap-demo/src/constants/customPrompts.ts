/**
 * 用户自定义指令（用户提示词）的常量。
 *
 * 和「系统提示词」的区别见 README：
 *  - 系统提示词 = 管理员维护的全局配置（翻译 / 选区 / 整句 / 语法 / 指令 5 个能力 + 语气表）
 *  - 用户指令 = 用户自己写的「一段话」，每个 provider 一套，跑在英文编辑器里
 */

/** 作用范围：当前句 / 当前段 / 所有正文。 */
export type PromptScope = 'sentence' | 'paragraph' | 'full'

export interface ScopeOption {
  value: PromptScope
  label: string
  /** 这个范围会取哪段文本、上下文是什么 */
  hint: string
}

export const SCOPE_OPTIONS: ScopeOption[] = [
  {
    value: 'sentence',
    label: '当前句',
    hint: '改光标所在的那一句；上下文自动带上它所在的段落',
  },
  {
    value: 'paragraph',
    label: '当前段',
    hint: '改光标所在的整段；上下文自动带上全文摘要',
  },
  {
    value: 'full',
    label: '所有正文',
    hint: '改整篇正文（只取正文文字，不含工具栏之类的）',
  },
]

export const SCOPE_LABEL: Record<PromptScope, string> = {
  sentence: '当前句',
  paragraph: '当前段',
  full: '所有正文',
}

/**
 * 工具栏最多钉几个指令。
 *
 * 工具栏就那么宽 —— 钉多了会把「复制 / 清空」挤出去。
 * 超出的指令仍然能在「✨ AI ▾ → 我的指令」里点到。
 */
export const MAX_TOOLBAR_PROMPTS = 5

/** 可选的图标（新建指令时挑一个）。 */
export const PROMPT_ICONS = [
  '✨', '🎯', '📝', '🔧', '🌐', '💡', '📚', '🧩',
  '🗣️', '🎓', '⚡', '🔍', '✂️', '🧠', '📊', '🛠️',
]
