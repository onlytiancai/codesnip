/**
 * 选区菜单 / 工具栏按钮能触发的动作。
 * 单列一份，菜单项和工具栏按钮都从这里取，避免两边漂移。
 */
export type AiAction =
  | 'candidates'
  | 'tone'
  | 'improve'
  | 'shorten'
  | 'expand'
  | 'grammar'
  | 'instruction'
